"""Core data, modelling, persistence and decision-engine functions for the Explainable IDSS."""
from __future__ import annotations
import io, json, os, re, sqlite3, hashlib, warnings, urllib.request, zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             matthews_corrcoef, cohen_kappa_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
import joblib

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
ART_DIR = APP_DIR / "artifacts"
REPORT_DIR = APP_DIR / "reports"
PBI_DIR = APP_DIR / "powerbi_exports"
for _p in (DATA_DIR, SOURCE_DIR, ART_DIR, REPORT_DIR, PBI_DIR):
    _p.mkdir(parents=True, exist_ok=True)
DB_PATH = APP_DIR / "idss_audit.sqlite"
RANDOM_STATE = 42


class _Prog:
    """Progress adapter so training can report to any UI (or none)."""
    def __init__(self, cb): self.cb = cb
    def progress(self, frac, text=""):
        if self.cb: self.cb(frac, text)
    def empty(self): pass


# ---------- Persistent storage ----------
def db_conn():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con

def init_db():
    with db_conn() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL, dataset_key TEXT NOT NULL, row_key TEXT NOT NULL,
            original_prediction TEXT, corrected_label TEXT NOT NULL, reason TEXT NOT NULL,
            reviewer TEXT, approved INTEGER NOT NULL DEFAULT 0,
            model_version TEXT, created_at TEXT NOT NULL, retrained_at TEXT
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            registry_id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL,
            model_version TEXT NOT NULL, model_name TEXT NOT NULL, status TEXT NOT NULL,
            metrics_json TEXT NOT NULL, artifact_path TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL,
            dataset_key TEXT NOT NULL, row_key TEXT NOT NULL, model_version TEXT NOT NULL,
            prediction TEXT NOT NULL, probabilities_json TEXT, created_at TEXT NOT NULL
        );
        ''')
init_db()

# ---------- Dataset handling ----------
def load_uci_hospital():
    """Fetch UCI Diabetes 130-US Hospitals through ucimlrepo and save a local copy."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise RuntimeError("Install ucimlrepo first: pip install ucimlrepo") from e
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        target = y.iloc[:, 0]
    else:
        target = y
    df = X.copy()
    df["readmitted"] = target
    # Binary research target: <30 means readmitted within 30 days; >30 and NO are not early readmission.
    df["readmitted"] = df["readmitted"].astype(str).str.strip().map({"<30": "Readmitted within 30 days", ">30": "No readmission within 30 days", "NO": "No readmission within 30 days"})
    df = df.dropna(subset=["readmitted"])
    # Remove direct identifiers to reduce memorisation and leakage.
    df = df.drop(columns=[c for c in ["encounter_id", "patient_nbr"] if c in df.columns], errors="ignore")
    path = DATA_DIR / "hospital_readmission_uci.csv"
    df.to_csv(path, index=False)
    return df, str(path)


def load_uci_itsm():
    """Download the public UCI ITSM event log, reduce repeated event snapshots to one earliest row per incident, and save it."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00498/incident_event_log.zip"
    zip_path = DATA_DIR / "uci_itsm_incident_event_log.zip"
    csv_path = DATA_DIR / "incident_event_log.csv"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("incident_event_log.csv") or m.lower().endswith(".csv")]
            if not members:
                raise RuntimeError("No CSV file was found inside the UCI ZIP archive.")
            with zf.open(members[0]) as src, open(csv_path, "wb") as dst:
                dst.write(src.read())
    except Exception as e:
        raise RuntimeError(f"Could not download the UCI ITSM ZIP. Download it manually from https://archive.ics.uci.edu/dataset/498/incident%2Bmanagement%2Bprocess%2Benriched%2Bevent%2Blog and upload incident_event_log.csv. Details: {e}") from e
    raw = pd.read_csv(csv_path, low_memory=False)
    raw = raw.replace(r"^\s*\?\s*$", np.nan, regex=True)
    # The source contains repeated event snapshots per incident. Keep the earliest available creation snapshot.
    if "number" in raw.columns:
        sort_col = "sys_created_at" if "sys_created_at" in raw.columns else ("opened_at" if "opened_at" in raw.columns else None)
        if sort_col:
            raw["__sort_timestamp__"] = pd.to_datetime(raw[sort_col], dayfirst=True, errors="coerce")
            raw = raw.sort_values("__sort_timestamp__", na_position="last", kind="stable")
            raw = raw.drop(columns=["__sort_timestamp__"])
        raw = raw.drop_duplicates(subset=["number"], keep="first").reset_index(drop=True)
    saved_path = DATA_DIR / "itsm_incident_first_snapshot.csv"
    raw.to_csv(saved_path, index=False)
    return raw, str(saved_path)

def dataset_key(domain: str, df: pd.DataFrame, target: str) -> str:
    return hashlib.sha256((domain + "|" + str(df.shape) + "|" + target + "|" + "|".join(map(str, df.columns))).encode()).hexdigest()[:12]

def clean_frame(df: pd.DataFrame, target: str):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = str(target).strip()
    if target not in df.columns: raise ValueError(f"Target '{target}' is not in the dataset.")
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[target])
    df = df.drop_duplicates().reset_index(drop=True)
    # Stable row keys are persisted for human feedback.
    df["__row_key__"] = [f"row_{i:07d}" for i in range(len(df))]
    return df, target

def split_data(df: pd.DataFrame, target: str, id_cols: list[str], domain_name: str = ""):
    y = df[target].astype(str).reset_index(drop=True)
    X = df.drop(columns=[target] + [c for c in id_cols if c in df.columns] + ["__row_key__"], errors="ignore").reset_index(drop=True)
    # Prevent target/outcome leakage in the UCI ITSM priority task: priority is derived from impact and urgency.
    if domain_name == "ITSM Incident Priority":
        itsm_exclusions = ["impact", "urgency", "made_sla", "resolved_at", "closed_at", "closed_code", "resolved_by", "sys_updated_at", "opened_at", "sys_created_at", "problem_id", "rfc", "caused_by"]
        X = X.drop(columns=[c for c in itsm_exclusions if c in X.columns], errors="ignore")
    # Exclude very high-cardinality direct identifiers if they were not manually selected as IDs.
    drop_high = [c for c in X.columns if X[c].nunique(dropna=True) > max(5000, int(0.5 * len(X)))]
    X = X.drop(columns=drop_high, errors="ignore")
    if y.nunique() < 2: raise ValueError("The target must contain at least two classes after cleaning.")
    if y.value_counts().min() < 4: raise ValueError("Each target class needs at least four records for train/review/validation/test splitting.")
    X_dev, X_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_dev, y_dev, idx_dev, test_size=0.4375, random_state=RANDOM_STATE, stratify=y_dev)
    # proportions overall: 45% train, 15% review, 20% validation, 20% test
    X_review, X_val, y_review, y_val, idx_review, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=(20/35), random_state=RANDOM_STATE, stratify=y_temp)
    return {
        "X_train": X_train.reset_index(drop=True), "y_train": y_train.reset_index(drop=True),
        "X_review": X_review.reset_index(drop=True), "y_review": y_review.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True), "y_val": y_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True), "y_test": y_test.reset_index(drop=True),
        "idx_review": np.asarray(idx_review), "idx_val": np.asarray(idx_val), "idx_test": np.asarray(idx_test),
        "X_all": X.reset_index(drop=True), "y_all": y.reset_index(drop=True),
        "row_keys": df["__row_key__"].astype(str).reset_index(drop=True), "dropped_high_cardinality": drop_high,
    }

def _to_object(X):
    """Cast to plain object (pandas 3 string / bool dtypes confuse the imputer) keeping missing as NaN."""
    d = pd.DataFrame(X)
    return d.astype(object).where(d.notna(), np.nan)


def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if (
        pd.api.types.is_object_dtype(X[c].dtype)
        or isinstance(X[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(X[c].dtype)
        or isinstance(X[c].dtype, pd.StringDtype)
    )]
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler(with_mean=False))])
    categorical = Pipeline([("cast", FunctionTransformer(_to_object, feature_names_out="one-to-one")), ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)], remainder="drop", verbose_feature_names_out=False)

def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=160, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "SGD Logistic": SGDClassifier(loss="log_loss", max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Dummy Baseline": DummyClassifier(strategy="prior"),
    }

def ovr_auc_table(model, X, y) -> pd.DataFrame:
    """One-vs-rest ROC-AUC per class for multi-class models. Classes missing from y (or covering every row)
    have no defined AUC and are skipped; the macro average uses only the classes that are scored."""
    proba = model.predict_proba(X)
    ya = np.asarray(y).astype(str)
    rows = []
    for i, cls in enumerate(model.classes_):
        pos = ya == str(cls)
        n_pos = int(pos.sum())
        if 0 < n_pos < len(ya):
            rows.append({"class": str(cls), "records": n_pos, "auc": float(roc_auc_score(pos.astype(int), proba[:, i]))})
    return pd.DataFrame(rows, columns=["class", "records", "auc"])


def metrics_for(model, X, y):
    pred = model.predict(X)
    labels = list(model.classes_)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
    }
    out["roc_auc"] = None
    if hasattr(model, "predict_proba"):
        try:
            if len(labels) == 2:
                out["roc_auc"] = float(roc_auc_score(y, model.predict_proba(X)[:, 1], labels=labels))
            else:
                per = ovr_auc_table(model, X, y)
                out["roc_auc"] = float(per["auc"].mean()) if len(per) else None
        except Exception:
            out["roc_auc"] = None
    return out

def train_models(bundle, selected_models=None, progress=None, max_train=None):
    bundle = dict(bundle)
    if max_train and len(bundle["X_train"]) > max_train:
        keep = np.sort(np.random.RandomState(RANDOM_STATE).choice(len(bundle["X_train"]), size=int(max_train), replace=False))
        bundle["X_train"] = bundle["X_train"].iloc[keep].reset_index(drop=True)
        bundle["y_train"] = bundle["y_train"].iloc[keep].reset_index(drop=True)
    pre = build_preprocessor(bundle["X_train"])
    rows, fitted = [], {}
    prog = _Prog(progress)
    models = candidate_models()
    if selected_models is not None:
        models = {name: est for name, est in models.items() if name in selected_models}
    if not models:
        raise ValueError("Select at least one compatible model.")
    for i, (name, estimator) in enumerate(models.items()):
        pipe = Pipeline([("preprocess", clone(pre)), ("model", estimator)])
        try:
            y_fit = bundle["y_train"].copy()
            # Keep target labels homogeneous and avoid string-vs-numeric class-weight key mismatches.
            if pd.api.types.is_numeric_dtype(y_fit):
                y_fit = pd.to_numeric(y_fit, errors="raise")
            pipe.fit(bundle["X_train"], y_fit)
            m = metrics_for(pipe, bundle["X_val"], bundle["y_val"])
            rows.append({"model": name, "status": "OK", "error": "", **m})
            fitted[name] = pipe
        except Exception as exc:
            rows.append({"model": name, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "accuracy": np.nan, "balanced_accuracy": np.nan, "precision_macro": np.nan, "recall_macro": np.nan, "f1_macro": np.nan, "f1_weighted": np.nan})
        prog.progress((i+1)/len(models), f"Processed {name}")
    prog.empty()
    table = pd.DataFrame(rows)
    valid = table[table["model"].isin(fitted)].sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
    if valid.empty:
        raise RuntimeError("All candidate models failed. See the error column in the comparison table/logs and verify the target column and class labels.")
    table = pd.concat([valid, table[~table["model"].isin(fitted)]], ignore_index=True)
    best_name = valid.iloc[0]["model"]
    return fitted, table, best_name

def save_model_bundle(domain, model_name, model, metrics, key, version):
    path = ART_DIR / f"{domain.lower().replace(' ', '_')}_{version}.joblib"
    joblib.dump({"domain": domain, "model_name": model_name, "model": model, "metrics": metrics,
                 "dataset_key": key, "version": version, "created_at": datetime.now(timezone.utc).isoformat()}, path)
    with db_conn() as con:
        con.execute("UPDATE model_registry SET status='superseded' WHERE domain=? AND status='active'", (domain,))
        con.execute("INSERT INTO model_registry(domain, model_version, model_name, status, metrics_json, artifact_path, created_at) VALUES (?,?,?,?,?,?,?)",
                    (domain, version, model_name, "active", json.dumps(metrics), str(path), datetime.now(timezone.utc).isoformat()))
    return path

def save_candidate_models(domain, models, metrics_table, key, version):
    """Persist every trained candidate separately, not just the selected model."""
    saved = []
    safe_domain = domain.lower().replace(" ", "_")
    metric_lookup = {str(r["model"]): {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                                          for k, v in r.items() if k != "model"}
                     for _, r in metrics_table.iterrows()}
    for name, model in models.items():
        safe_name = "_".join(name.lower().split())
        path = ART_DIR / f"{safe_domain}_{safe_name}_{version}.joblib"
        joblib.dump({"domain": domain, "model_name": name, "model": model,
                     "metrics": metric_lookup.get(name, {}), "dataset_key": key,
                     "version": version, "status": "candidate", "created_at": datetime.now(timezone.utc).isoformat()}, path)
        saved.append(path)
    return saved

def get_active_bundle(domain, key):
    path = ART_DIR / f"{domain.lower().replace(' ', '_')}_active.joblib"
    if path.exists():
        try:
            b = joblib.load(path)
            if b.get("dataset_key") == key: return b, path
        except Exception: pass
    return None, path

def write_active_bundle(domain, bundle, model_name, model, metrics, key, version):
    active_path = ART_DIR / f"{domain.lower().replace(' ', '_')}_active.joblib"
    payload = {"domain": domain, "model_name": model_name, "model": model, "metrics": metrics,
               "dataset_key": key, "version": version, "created_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump(payload, active_path)
    version_path = save_model_bundle(domain, model_name, model, metrics, key, version)
    return active_path, version_path

def get_explanations(model, X_train, row, max_features=12):
    """Explain one row. Prefer model-agnostic permutation SHAP on raw inputs to avoid
    TreeExplainer additivity failures with one-hot encoded hospital data / sklearn versions.
    Never suppress additivity errors and present an unverified explanation."""
    out = {"shap": None, "lime": None, "errors": []}
    # Use the estimator on numeric transformed features with permutation SHAP. This avoids
    # TreeExplainer's known additivity incompatibilities across sklearn/SHAP versions and
    # keeps SHAP's output on the same probability scale as predict_proba.
    try:
        import shap
        prep = model.named_steps["preprocess"]
        est = model.named_steps["model"]
        background = prep.transform(X_train.sample(n=min(20, len(X_train)), random_state=RANDOM_STATE))
        x_trans = prep.transform(row)
        if hasattr(background, "toarray"): background = background.toarray()
        if hasattr(x_trans, "toarray"): x_trans = x_trans.toarray()
        background = np.asarray(background, dtype=np.float64)
        x_trans = np.asarray(x_trans, dtype=np.float64)
        if not np.isfinite(background).all() or not np.isfinite(x_trans).all():
            raise ValueError("Transformed feature matrix contains NaN or infinite values.")
        feat_names = np.asarray(prep.get_feature_names_out(), dtype=str)
        if background.shape[1] != x_trans.shape[1] or x_trans.shape[1] != len(feat_names):
            raise ValueError(f"Feature dimension mismatch: background={background.shape}, row={x_trans.shape}, names={len(feat_names)}")
        if not hasattr(est, "predict_proba"):
            raise ValueError("Selected estimator has no predict_proba method for probability-based SHAP.")
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(est.predict_proba, masker, algorithm="permutation",
                                   output_names=[str(c) for c in est.classes_])
        explanation = explainer(x_trans, max_evals=max(2 * x_trans.shape[1] + 1, 101))
        values = np.asarray(explanation.values)
        pred = model.predict(row)[0]
        class_idx = list(est.classes_).index(pred)
        if values.ndim == 3:
            vals_one = values[0, :, class_idx]
        elif values.ndim == 2:
            vals_one = values[0]
        else:
            raise ValueError(f"Unexpected SHAP output shape {values.shape}; no plot was produced.")
        vals_one = np.asarray(vals_one, dtype=float)
        if not np.isfinite(vals_one).all():
            raise ValueError("SHAP returned non-finite values; no plot was produced.")
        n = min(max_features, len(vals_one), len(feat_names))
        ix = np.argsort(np.abs(vals_one))[-n:][::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(feat_names[ix][::-1], vals_one[ix][::-1])
        ax.set_title(f"Permutation SHAP — contribution to {pred}")
        ax.set_xlabel("Contribution to predicted class probability (not causation)")
        fig.tight_layout()
        out["shap"] = fig
    except Exception as e:
        out["errors"].append(f"SHAP: {type(e).__name__}: {e}. Permutation SHAP uses the numeric transformed features; the explanation was not displayed because validation failed.")
    # LIME runs against the fitted preprocessor and estimator, using the same transformed feature space.
    try:
        from lime.lime_tabular import LimeTabularExplainer
        prep = model.named_steps["preprocess"]
        est = model.named_steps["model"]
        Xt = prep.transform(X_train.iloc[:min(300, len(X_train))])
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt, dtype=np.float64)
        xr = prep.transform(row)
        if hasattr(xr, "toarray"):
            xr = xr.toarray()
        xr = np.asarray(xr, dtype=np.float64)
        feature_names = prep.get_feature_names_out().tolist()
        explainer = LimeTabularExplainer(Xt, feature_names=feature_names,
                 class_names=[str(x) for x in est.classes_], mode="classification", discretize_continuous=True,
                 random_state=RANDOM_STATE)
        if not hasattr(est, "predict_proba"):
            out["errors"].append("LIME requires a model exposing predict_proba; unavailable for this estimator.")
        else:
            explanation = explainer.explain_instance(xr[0], est.predict_proba,
                num_features=min(max_features, len(feature_names)), num_samples=1000)
            fig2 = explanation.as_pyplot_figure()
            fig2.tight_layout()
            out["lime"] = fig2
    except Exception as e:
        out["errors"].append(f"LIME: {type(e).__name__}: {e}")
    return out



def repair_credit_csv_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Repair common UCI Default of Credit Card Clients CSVs whose first data row contains headers."""
    if raw.empty:
        return raw
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())
    expected = {"defaultpaymentnextmonth", "limitbal", "sex", "education", "marriage", "age", "pay0", "pay1"}
    current = {norm(c) for c in raw.columns}
    first = [str(v).strip() for v in raw.iloc[0].tolist()]
    first_norm = {norm(v) for v in first}
    # Typical bad parse: columns are X1..X25 or Y/X1..., and the real header is row 0.
    generic = all((norm(c).startswith("x") and norm(c)[1:].isdigit()) or norm(c) in {"y", "unnamed0", "index"} for c in raw.columns)
    if ("defaultpaymentnextmonth" in first_norm and (generic or "defaultpaymentnextmonth" not in current)):
        raw = raw.copy()
        raw.columns = first
        raw = raw.iloc[1:].reset_index(drop=True)
    # Normalize known target variants while preserving original feature names otherwise.
    aliases = {norm(c): c for c in raw.columns}
    target_aliases = ["defaultpaymentnextmonth", "defaultpaymentnextmonthy", "y"]
    target = next((aliases[a] for a in target_aliases if a in aliases), None)
    if target and target != "default payment next month":
        raw = raw.rename(columns={target: "default payment next month"})
    # Convert target to numeric when it consists of 0/1 strings/numbers.
    if "default payment next month" in raw.columns:
        vals = pd.to_numeric(raw["default payment next month"], errors="coerce")
        if vals.notna().mean() > 0.98:
            raw["default payment next month"] = vals
    return raw



# ---------- Domain constants ----------
DOMAINS = ["Hospital Readmission", "ITSM Incident Priority", "Credit Risk Assessment"]
SLUG = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

DOMAIN_META = {
    "Hospital Readmission": {
        "entity": "encounter", "entities": "encounters",
        "target_default": "readmitted",
        "adverse_label": "30-day readmission",
        "bundled": "diabetic_data.csv",
        "tiers": ["Priority clinical review", "Standard follow-up", "Routine discharge pathway"],
        "objective": "Which patients should be reviewed first for early-readmission risk?",
    },
    "ITSM Incident Priority": {
        "entity": "incident", "entities": "incidents",
        "target_default": "priority",
        "adverse_label": "high-priority incident",
        "bundled": "incident_event_log.csv",
        "tiers": ["Escalate now", "Standard handling", "Routine queue"],
        "objective": "Which incidents should the service desk escalate first?",
    },
    "Credit Risk Assessment": {
        "entity": "account", "entities": "accounts",
        "target_default": "default payment next month",
        "adverse_label": "payment default",
        "bundled": "default_of_credit_card_clients.csv",
        "tiers": ["Enhanced review", "Standard review", "Fast-track eligible"],
        "objective": "Which accounts need an enhanced credit review first?",
    },
}


def find_source_file(domain: str):
    """Locate a bundled/local copy of the domain's dataset (data/source first, then data/)."""
    name = DOMAIN_META[domain]["bundled"]
    for folder in (SOURCE_DIR, DATA_DIR, APP_DIR):
        p = folder / name
        if p.exists():
            return p
    return None


def load_bundled(domain: str):
    """Load the domain dataset from the local source file and apply domain-specific preparation."""
    path = find_source_file(domain)
    if path is None:
        raise FileNotFoundError(f"{DOMAIN_META[domain]['bundled']} was not found in {SOURCE_DIR}.")
    raw = pd.read_csv(path, low_memory=False)
    return prepare_domain_frame(domain, raw), str(path)


def prepare_domain_frame(domain: str, raw: pd.DataFrame):
    """Domain-specific preparation applied to every loaded dataset (bundled, UCI or uploaded)."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if domain == "Credit Risk Assessment":
        df = repair_credit_csv_header(df)
        df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() in {"id", "unnamed: 0", "customer_id", "client_id"}], errors="ignore")
        for c in df.columns:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.98:
                df[c] = conv
    elif domain == "Hospital Readmission":
        if "readmitted" in df.columns:
            m = {"<30": "Readmitted within 30 days", ">30": "No readmission within 30 days", "NO": "No readmission within 30 days"}
            df["readmitted"] = df["readmitted"].astype(str).str.strip().map(m).fillna(df["readmitted"].astype(str))
        df = df.drop(columns=[c for c in ["encounter_id", "patient_nbr"] if c in df.columns], errors="ignore")
    elif domain == "ITSM Incident Priority":
        df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)
        if "number" in df.columns:
            sort_col = "sys_created_at" if "sys_created_at" in df.columns else ("opened_at" if "opened_at" in df.columns else None)
            if sort_col:
                order = pd.to_datetime(df[sort_col], dayfirst=True, errors="coerce")
                df = df.assign(__t=order).sort_values("__t", na_position="last", kind="stable").drop(columns="__t")
            df = df.drop_duplicates(subset=["number"], keep="first").reset_index(drop=True)
    return df


# ---------- Adverse-outcome definition and ML risk ----------
def adverse_classes(domain: str, classes, y_train: pd.Series | None = None) -> list[str]:
    """Classes that represent the adverse / high-attention outcome for the domain."""
    classes = [str(c) for c in classes]
    if domain == "Hospital Readmission":
        hit = [c for c in classes if c.lower().startswith("readmitted within")]
    elif domain == "ITSM Incident Priority":
        hit = [c for c in classes if re.match(r"^\s*[12]\b", c)]
    elif domain == "Credit Risk Assessment":
        hit = [c for c in classes if c in {"1", "1.0", "default", "Default", "bad", "Charged Off"}]
    else:
        hit = []
    if not hit and y_train is not None and len(classes) >= 2:
        hit = [str(y_train.astype(str).value_counts().idxmin())]
    return hit or classes[-1:]


def risk_scores(model, X: pd.DataFrame, adverse: list[str]) -> np.ndarray:
    """Probability that a record belongs to an adverse class."""
    classes = [str(c) for c in model.classes_]
    idx = [i for i, c in enumerate(classes) if c in adverse]
    if hasattr(model, "predict_proba") and idx:
        return model.predict_proba(X)[:, idx].sum(axis=1)
    pred = model.predict(X).astype(str)
    return np.isin(pred, adverse).astype(float)


def local_drivers(model, X_train: pd.DataFrame, row: pd.DataFrame, adverse: list[str]) -> pd.DataFrame:
    """Field-level drivers for one record: change in adverse-outcome probability when the field is
    replaced by the typical (median / most frequent) training value. Works on original input fields."""
    base = {}
    for c in X_train.columns:
        s = X_train[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            base[c] = s.median()
        else:
            m = s.mode(dropna=True)
            base[c] = m.iloc[0] if len(m) else np.nan
    variants = [row.iloc[0].copy()]
    cols = list(row.columns)
    for c in cols:
        r = row.iloc[0].copy()
        r[c] = base[c]
        variants.append(r)
    batch = pd.DataFrame(variants).reset_index(drop=True)
    for c in cols:
        batch[c] = batch[c].astype(X_train[c].dtype, errors="ignore") if pd.api.types.is_numeric_dtype(X_train[c]) else batch[c]
    p = risk_scores(model, batch, adverse)
    out = pd.DataFrame({"field": cols, "value": [row.iloc[0][c] for c in cols],
                        "typical": [base[c] for c in cols], "effect": p[0] - p[1:]})
    out = out[~((out["value"].astype(str) == out["typical"].astype(str)))]
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True), float(p[0])


def feature_row_for_display(row: pd.DataFrame) -> pd.DataFrame:
    """Two-column field/value table with stable string values (avoids Arrow mixed-type errors)."""
    if row is None or row.empty:
        return pd.DataFrame(columns=["Field", "Value"])
    series = row.iloc[0]
    def disp(v):
        if pd.isna(v): return "(missing)"
        if isinstance(v, (np.integer, np.floating, np.bool_)): return str(v.item())
        return str(v)
    return pd.DataFrame({"Field": [str(c) for c in series.index], "Value": [disp(v) for v in series.values]})
