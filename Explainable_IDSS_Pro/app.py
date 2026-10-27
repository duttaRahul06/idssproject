from __future__ import annotations
import json, hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from idss import core as C, criteria as K, narrative as N, theme as T

st.set_page_config(page_title="Explainable IDSS", page_icon=":material/analytics:", layout="wide", initial_sidebar_state="expanded")
T.inject()
C.init_db()

if "project" not in st.session_state:
    st.session_state.project = {}
state = st.session_state.project
SELECTION_NOTE = "Records are ranked by review priority: a higher score means the record should be looked at sooner."


# ============================ Project loading ============================
def set_project(df: pd.DataFrame, domain: str, target: str, path: str):
    cleaned, target_clean = C.clean_frame(df, target)
    id_defaults = [c for c in cleaned.columns if c.lower() in {"encounter_id", "patient_nbr", "number", "incident_id", "sys_id", "id"}]
    key = C.dataset_key(domain, cleaned, target_clean)
    cleaned_path = C.DATA_DIR / f"{C.SLUG(domain)}_{key}_cleaned.csv"
    cleaned.to_csv(cleaned_path, index=False)
    bundle = C.split_data(cleaned, target_clean, id_defaults, domain)
    mcdm_X = cleaned.drop(columns=[target_clean] + [c for c in id_defaults if c in cleaned.columns] + ["__row_key__"], errors="ignore").reset_index(drop=True)
    mcdm_X = K.derive_columns(domain, mcdm_X)
    state.clear()
    state.update({"df": cleaned, "domain": domain, "target": target_clean, "source_path": str(path), "cleaned_path": str(cleaned_path),
                  "key": key, "bundle": bundle, "mcdm_X": mcdm_X, "id_cols": id_defaults, "trained": False, "models": {},
                  "metrics_table": None, "best_name": None, "active_model": None, "active_name": None, "active_version": None,
                  "model_metrics": None, "cfg": None, "decisions": {}, "adverse": None})
    saved, _ = C.get_active_bundle(domain, key)
    if saved is not None:
        state.update({"trained": True, "active_model": saved.get("model"), "active_name": saved.get("model_name"),
                      "best_name": saved.get("model_name"), "active_version": saved.get("version", "restored"),
                      "model_metrics": saved.get("metrics")})
        state["adverse"] = C.adverse_classes(domain, saved["model"].classes_, bundle["y_train"])


st.sidebar.markdown('<div class="brand">Explainable IDSS</div><div class="brand-sub">Intelligent decision support</div>', unsafe_allow_html=True)
domain = st.sidebar.selectbox("Decision domain", C.DOMAINS)
meta = C.DOMAIN_META[domain]

with st.sidebar.expander("Dataset", expanded=("df" not in state or state.get("domain") != domain)):
    src = C.find_source_file(domain)
    if src is not None:
        st.caption(f"Local file: {src.name}")
        if st.button("Load dataset", type="primary", key=f"load_local_{domain}", width="stretch"):
            with st.spinner("Preparing dataset and data partitions..."):
                try:
                    df0, p = C.load_bundled(domain)
                    set_project(df0, domain, meta["target_default"], p)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load the dataset: {e}")
    if domain != "Credit Risk Assessment":
        if st.button("Download from UCI repository", key=f"load_uci_{domain}", width="stretch"):
            with st.spinner("Downloading from the UCI repository..."):
                try:
                    if domain == "Hospital Readmission":
                        df0, p = C.load_uci_hospital()
                    else:
                        raw0, p = C.load_uci_itsm()
                        df0 = C.prepare_domain_frame(domain, raw0)
                    set_project(df0, domain, meta["target_default"], p)
                    st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")
    up = st.file_uploader("Upload a CSV", type=["csv"], key=f"upload_{domain}")
    if up is not None:
        try:
            raw = C.prepare_domain_frame(domain, pd.read_csv(up, low_memory=False))
            low = {str(c).strip().lower(): c for c in raw.columns}
            tgt_default = low.get(meta["target_default"].lower(), raw.columns[-1])
            tgt = st.selectbox("Target column", raw.columns.tolist(), index=raw.columns.tolist().index(tgt_default), key=f"target_{domain}")
            if st.button("Use uploaded dataset", key=f"use_{domain}", width="stretch"):
                p = C.DATA_DIR / f"{C.SLUG(domain)}_uploaded_original.csv"
                raw.to_csv(p, index=False)
                set_project(raw, domain, tgt, p)
                st.rerun()
        except Exception as e:
            st.error(f"Unable to read the file: {e}")

has_project = bool(state) and state.get("domain") == domain
if has_project:
    bundle, df, target, key = state["bundle"], state["df"], state["target"], state["key"]
    st.sidebar.caption(f"{len(df):,} records · target: {target}")


# ============================ Shared helpers ============================
def scope_data(scope: str) -> dict:
    if scope == "Human review pool":
        ids, X, y = bundle["idx_review"], bundle["X_review"], bundle["y_review"]
    elif scope == "Validation set":
        ids, X, y = bundle["idx_val"], bundle["X_val"], bundle["y_val"]
    else:
        ids, X, y = bundle["idx_test"], bundle["X_test"], bundle["y_test"]
    ids = np.asarray(ids, dtype=int)
    return dict(ids=ids, X=X.reset_index(drop=True), y=y.reset_index(drop=True),
                mrows=state["mcdm_X"].iloc[ids].reset_index(drop=True), keys=bundle["row_keys"].iloc[ids].reset_index(drop=True))


SCOPES = ["Human review pool", "Validation set", "Final test set"]


def build_default_cfg() -> dict:
    mx = state["mcdm_X"]
    crit = {}
    for col, spec in K.candidate_columns(domain, mx, target).items():
        vals = None if spec.get("categorical") else K.recode_numeric(mx[col], spec.get("recode"))
        crit[col] = {"spec": spec, "choice": K.default_choice(spec, vals)}
    return {"criteria": crit, "weights": None, "weight_method": "Equal weights", "method": "AHP + TOPSIS", "mix": 0.5, "cr": None}


def get_cfg() -> dict:
    if state.get("cfg") is None:
        state["cfg"] = build_default_cfg()
    return state["cfg"]


def active_criteria(cfg) -> list[str]:
    return [c for c, it in cfg["criteria"].items() if it["choice"]["treatment"] != "exclude"]


def active_weights(cfg) -> dict:
    act = active_criteria(cfg)
    w = cfg.get("weights")
    if not w or set(w) != set(act):
        return {c: 1 / len(act) for c in act}
    return w


def cfg_signature(cfg) -> str:
    payload = {c: it["choice"] for c, it in cfg["criteria"].items() if it["choice"]["treatment"] != "exclude"}
    return hashlib.md5(json.dumps([payload, active_weights(cfg), cfg["method"], cfg["mix"]], sort_keys=True, default=str).encode()).hexdigest()[:10]


def ensure_adverse():
    if state.get("adverse") is None and state.get("active_model") is not None:
        state["adverse"] = C.adverse_classes(domain, state["active_model"].classes_, bundle["y_train"])
    return state.get("adverse") or []


def effective_cfg_weights(cfg):
    return {c: 1 / len(active_criteria(cfg)) for c in active_criteria(cfg)} if cfg["method"] == "TOPSIS only" else active_weights(cfg)


def get_decision(scope, ml_weight=0.5, top_share=0.20, mid_share=0.50):
    cfg = get_cfg()
    sig = f"{cfg_signature(cfg)}|{scope}|{ml_weight}|{top_share}|{mid_share}|{state.get('active_version')}"
    cache = state.setdefault("decisions", {})
    if sig not in cache:
        sd = scope_data(scope)
        tbl, mc = N.decision_table(domain, state["active_model"], ensure_adverse(), sd["X"], sd["mrows"], sd["keys"], {**cfg, "weights": effective_cfg_weights(cfg)},
                                   cfg["method"], cfg["mix"], ml_weight, top_share, mid_share)
        cache.clear()
        cache[sig] = (tbl, mc, sd)
    return cache[sig]


def need_model(msg="Train a model in Model Studio first."):
    if not state.get("trained") or state.get("active_model") is None:
        T.note(msg, warn=True)
        return False
    return True


def tidy_table(d: pd.DataFrame, rounding=3):
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(rounding)
    return d


# ============================ Pages ============================
def page_overview():
    T.header("Executive overview", f"{domain} · {meta['objective']}")
    m = state.get("model_metrics") or {}
    k = st.columns(4)
    T.kpi(k[0], f"{meta['entities'].capitalize()} analysed", f"{len(df):,}", f"{bundle['X_all'].shape[1]} predictor fields")
    adv = ensure_adverse() if state.get("trained") else []
    pos_rate = float(bundle["y_all"].isin(adv).mean()) if adv else float(bundle["y_all"].value_counts(normalize=True).min())
    T.kpi(k[1], f"{meta['adverse_label'].capitalize()} rate", f"{pos_rate:.1%}", "share of records in the adverse class")
    T.kpi(k[2], "Active model", state.get("active_name") or "Not trained", f"version {state.get('active_version')}" if state.get("active_version") else "train in Model Studio")
    T.kpi(k[3], "Validation macro-F1", f"{m.get('f1_macro'):.3f}" if m.get("f1_macro") is not None else "n/a", f"ROC-AUC {m['roc_auc']:.3f}" if m.get("roc_auc") else "")
    st.write("")
    T.chips([("Data loaded", True), ("Model trained", bool(state.get("trained"))),
             ("Criteria configured", state.get("cfg") is not None), ("Decisions scored", bool(state.get("decisions"))),
             ("Reviews logged", len(get_feedback()) > 0)])
    st.write("")
    a, b = st.columns([1, 1])
    dist = bundle["y_all"].astype(str).value_counts().rename_axis("Outcome").reset_index(name="Records")
    fig = px.pie(dist, names="Outcome", values="Records", hole=0.55, title="Outcome mix")
    fig.update_traces(textinfo="percent", sort=False)
    a.plotly_chart(fig, width="stretch")
    part = pd.DataFrame({"Partition": ["Training", "Human review", "Validation", "Final test"],
                         "Records": [len(bundle["X_train"]), len(bundle["X_review"]), len(bundle["X_val"]), len(bundle["X_test"])]})
    fig = px.bar(part, x="Partition", y="Records", text="Records", title="Data partitions")
    fig.update_traces(texttemplate="%{text:,}", marker_color=T.BLUE)
    b.plotly_chart(fig, width="stretch")
    c, d = st.columns([1, 1])
    mt = state.get("metrics_table")
    if isinstance(mt, pd.DataFrame) and not mt.empty:
        ok = mt[mt["status"] == "OK"].melt(id_vars="model", value_vars=["f1_macro", "balanced_accuracy", "accuracy"], var_name="Metric", value_name="Score")
        fig = px.bar(ok, x="model", y="Score", color="Metric", barmode="group", title="Model comparison (validation)")
        fig.update_layout(xaxis_title=None, yaxis_range=[0, 1])
        c.plotly_chart(fig, width="stretch")
    else:
        miss = (df.drop(columns=["__row_key__"]).isna().mean() * 100).sort_values(ascending=False).head(8).rename_axis("Field").reset_index(name="Missing %")
        fig = px.bar(miss, x="Missing %", y="Field", orientation="h", title="Fields with most missing values")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        c.plotly_chart(fig, width="stretch")
    if state.get("decisions"):
        tbl = list(state["decisions"].values())[0][0]
        tc = tbl["recommendation"].value_counts().reindex(meta["tiers"]).fillna(0).reset_index()
        tc.columns = ["Recommendation", "Records"]
        fig = px.bar(tc, x="Recommendation", y="Records", color="Recommendation", title="Latest recommendations",
                     color_discrete_sequence=[T.RED, T.AMBER, T.TEAL])
        fig.update_layout(showlegend=False, xaxis_title=None)
        d.plotly_chart(fig, width="stretch")
    else:
        top_cat = bundle["X_all"].select_dtypes(exclude="number").nunique().sort_values(ascending=False).head(8).rename_axis("Field").reset_index(name="Distinct values")
        fig = px.bar(top_cat, x="Distinct values", y="Field", orientation="h", title="Category fields by distinct values")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        d.plotly_chart(fig, width="stretch")


def page_data():
    T.header("Data profile", "Quality checks applied before modelling and ranking.")
    raw_missing = float(df.drop(columns=["__row_key__"]).isna().mean().mean())
    k = st.columns(4)
    T.kpi(k[0], "Records", f"{len(df):,}")
    T.kpi(k[1], "Fields", f"{df.shape[1] - 2}", "excluding target and key")
    T.kpi(k[2], "Average missing", f"{raw_missing:.1%}")
    T.kpi(k[3], "Fields dropped (high cardinality)", f"{len(bundle['dropped_high_cardinality'])}")
    st.write("")
    miss = (df.drop(columns=["__row_key__"]).isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(15).rename_axis("Field").reset_index(name="Missing %")
    a, b = st.columns([1.1, 1])
    if len(miss):
        fig = px.bar(miss, x="Missing %", y="Field", orientation="h", title="Missing values by field")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        a.plotly_chart(fig, width="stretch")
    else:
        a.info("No missing values in this dataset.")
    prof = pd.DataFrame({"Type": df.drop(columns=["__row_key__"]).dtypes.astype(str), "Distinct": df.drop(columns=["__row_key__"]).nunique(dropna=True),
                         "Missing": df.drop(columns=["__row_key__"]).isna().sum()}).reset_index(names="Field")
    b.markdown("**Field profile**")
    b.dataframe(prof, hide_index=True, height=380, width="stretch")
    st.markdown("**Record preview**")
    st.dataframe(df.head(50), width="stretch", height=300)
    st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode(), file_name=f"{C.SLUG(domain)}_{key}_cleaned.csv", mime="text/csv")
    T.note("Cleaning applied: placeholder markers (“?” and blanks) converted to missing, records without a target removed, duplicates removed, identifier fields excluded from modelling. "
           + ("Impact, urgency and outcome-time fields are excluded from the ITSM prediction model because they define the recorded priority; they stay available for ranking." if domain == "ITSM Incident Priority" else ""))


def get_feedback(approved=None):
    q = "SELECT * FROM feedback WHERE domain=? AND dataset_key=?"
    args = [domain, key]
    if approved is not None:
        q += " AND approved=?"; args.append(int(approved))
    with C.db_conn() as con:
        return pd.read_sql_query(q + " ORDER BY feedback_id DESC", con, params=args)


def page_models():
    T.header("Model studio", "Train candidate models, compare them on validation data, and confirm the final result on the untouched test set.")
    options = list(C.candidate_models().keys())
    c1, c2, c3 = st.columns([2, 1, 1])
    chosen = c1.multiselect("Candidate models", options, default=[o for o in options if o != "SGD Logistic"], key=f"models_{key}")
    metric = c2.selectbox("Selection metric", ["f1_macro", "balanced_accuracy", "f1_weighted", "mcc", "roc_auc"], key=f"metric_{key}")
    big = len(bundle["X_train"]) > 20000
    cap = c3.selectbox("Training sample", ["Full training set", "30,000 records", "15,000 records"], index=1 if big else 0, key=f"cap_{key}")
    cap_n = {"Full training set": None, "30,000 records": 30000, "15,000 records": 15000}[cap]
    if st.button("Train and compare models", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Starting")
        try:
            fitted, table, best = C.train_models(bundle, chosen, progress=lambda f, t: bar.progress(f, text=t), max_train=cap_n)
            ok = table[table["status"] == "OK"].copy()
            if metric in ok.columns:
                ok[metric] = pd.to_numeric(ok[metric], errors="coerce")
                ok = ok.sort_values([metric, "balanced_accuracy"], ascending=False, na_position="last")
                if pd.notna(ok.iloc[0][metric]):
                    best = str(ok.iloc[0]["model"])
            model = fitted[best]
            version = "v1"
            C.save_candidate_models(domain, fitted, table, key, version)
            metrics = C.metrics_for(model, bundle["X_val"], bundle["y_val"])
            C.write_active_bundle(domain, bundle, best, model, metrics, key, version)
            state.update({"models": fitted, "metrics_table": table, "best_name": best, "trained": True, "active_model": model,
                          "active_name": best, "active_version": version, "model_metrics": metrics, "decisions": {}, "adverse": None})
            ensure_adverse()
            table.to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_model_comparison_validation.csv", index=False)
            bar.empty()
            st.success(f"Training complete. {best} is the active model.")
        except Exception as e:
            bar.empty(); st.error(f"Training failed: {e}")
    mt = state.get("metrics_table")
    if isinstance(mt, pd.DataFrame):
        st.subheader("Validation comparison")
        ok = mt[mt["status"] == "OK"]
        fig = px.bar(ok.melt(id_vars="model", value_vars=["f1_macro", "balanced_accuracy", "accuracy"], var_name="Metric", value_name="Score"),
                     x="model", y="Score", color="Metric", barmode="group")
        fig.update_layout(xaxis_title=None, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tidy_table(mt.drop(columns=[c for c in ["error"] if c in mt and mt[c].eq("").all()])), hide_index=True, width="stretch")
    if state.get("trained") and state.get("active_model") is not None:
        model = state["active_model"]
        st.subheader("Final test evaluation")
        T.note("The test set has not been used for training or model selection. Run this once the model is final.")
        if st.button("Evaluate on final test set"):
            Xt, yt = bundle["X_test"], bundle["y_test"]
            tm = C.metrics_for(model, Xt, yt)
            state["test_metrics"] = tm
            pd.DataFrame([tm]).to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_final_test_metrics.csv", index=False)
            pred = model.predict(Xt)
            k = st.columns(4)
            T.kpi(k[0], "Accuracy", f"{tm['accuracy']:.3f}"); T.kpi(k[1], "Balanced accuracy", f"{tm['balanced_accuracy']:.3f}")
            T.kpi(k[2], "Macro-F1", f"{tm['f1_macro']:.3f}"); T.kpi(k[3], "ROC-AUC", f"{tm['roc_auc']:.3f}" if tm.get("roc_auc") else "n/a")
            a, b = st.columns(2)
            labels = [str(x) for x in model.classes_]
            cm = confusion_matrix(yt, pred, labels=model.classes_)
            fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix", labels=dict(x="Predicted", y="Actual"))
            fig.update_coloraxes(showscale=False)
            a.plotly_chart(fig, width="stretch")
            if len(labels) == 2 and hasattr(model, "predict_proba"):
                fpr, tpr, _ = roc_curve(yt, model.predict_proba(Xt)[:, 1], pos_label=model.classes_[1])
                fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=T.BLUE, width=3)),
                                 go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=T.GREY, dash="dash"))])
                fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
                b.plotly_chart(fig, width="stretch")
            elif len(labels) > 2 and hasattr(model, "predict_proba"):
                proba, ya = model.predict_proba(Xt), np.asarray(yt).astype(str)
                fig = go.Figure()
                per = C.ovr_auc_table(model, Xt, yt)
                for i, cls in enumerate(labels):
                    pos = ya == cls
                    if 0 < pos.sum() < len(ya):
                        fpr, tpr, _ = roc_curve(pos.astype(int), proba[:, i])
                        auc_c = float(per.loc[per["class"] == cls, "auc"].iloc[0])
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cls} (AUC {auc_c:.3f})", line=dict(width=2.5)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=T.GREY, dash="dash")))
                fig.update_layout(title="ROC curves (one class vs the rest)", xaxis_title="False positive rate", yaxis_title="True positive rate")
                b.plotly_chart(fig, width="stretch")
                st.markdown("**One-vs-rest ROC-AUC by class**")
                st.dataframe(tidy_table(per.rename(columns={"class": "Class", "records": "Records in test set", "auc": "ROC-AUC"})), hide_index=True, width="stretch")
                st.caption("Overall ROC-AUC is the macro average of the classes above. Classes absent from the test set cannot be scored and are left out.")
            rep = pd.DataFrame(classification_report(yt, pred, output_dict=True, zero_division=0)).transpose()
            rep.to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_final_test_report.csv")
            st.dataframe(tidy_table(rep), width="stretch")


# ---------------- Prioritisation ----------------
SAATY = ["1/9", "1/7", "1/5", "1/3", "1", "3", "5", "7", "9"]
SAATY_V = {"1/9": 1 / 9, "1/7": 1 / 7, "1/5": 1 / 5, "1/3": 1 / 3, "1": 1.0, "3": 3.0, "5": 5.0, "7": 7.0, "9": 9.0}


def criteria_tab(cfg):
    T.note("Each field is given a scoring rule that fits its meaning. A rule is only offered when it makes sense for the field; "
           "fields without a defensible direction (such as age or credit limit) are excluded from the ranking by default but remain in the prediction model and the explanations. "
           + SELECTION_NOTE)
    items = list(cfg["criteria"].items())
    primary = [(c, it) for c, it in items if it["spec"].get("default", "exclude") != "exclude" or it["spec"].get("derived") or c in K.SPEC.get(domain, {})]
    secondary = [(c, it) for c, it in items if (c, it) not in primary]

    def render(col, it):
        spec, ch = it["spec"], it["choice"]
        allowed = spec["allowed"]
        cur = ch["treatment"] if ch["treatment"] in allowed else allowed[-1]
        with st.container(border=True):
            l, r = st.columns([1.5, 1])
            l.markdown(f"**{spec['label']}** &nbsp; `{col}`")
            l.caption(spec["meaning"])
            l.caption("Why: " + spec["why"])
            t = r.selectbox("Scoring rule", allowed, index=allowed.index(cur), format_func=lambda x: K.TREATMENTS[x], key=f"tr_{key}_{col}", label_visibility="collapsed")
            ch["treatment"] = t
            if t == "band":
                vals = K.recode_numeric(state["mcdm_X"][col], spec.get("recode")).dropna()
                lo0, hi0 = ch.get("band") or [float(vals.quantile(.5)), float(vals.quantile(.9))]
                lo = r.number_input("Range from", value=float(lo0), key=f"lo_{key}_{col}")
                hi = r.number_input("Range to", value=float(hi0), key=f"hi_{key}_{col}")
                ch["band"] = [lo, hi]
                r.caption("Records inside the range score 1; the score falls steadily with distance from the range.")
            elif t == "map":
                s = state["mcdm_X"][col]
                cats = (K.recode_numeric(s, "bool").dropna().astype(int).astype(str) if spec.get("recode") == "bool" else s.dropna().astype(str)).value_counts().head(12).index.tolist()
                base = pd.DataFrame({"category": cats, "score": [float(ch.get("map", {}).get(c, 0.5)) for c in cats]})
                ed = st.data_editor(base, hide_index=True, disabled=["category"], key=f"map_{key}_{col}", width="stretch",
                                    column_config={"score": st.column_config.NumberColumn("Priority score (0-1)", min_value=0.0, max_value=1.0, step=0.05)})
                ch["map"] = {str(a): float(b) for a, b in zip(ed["category"], ed["score"])}

    st.markdown("#### Recommended criteria")
    for c, it in primary: render(c, it)
    if secondary:
        with st.expander(f"Other fields available ({len(secondary)}) — excluded unless you assign a rule"):
            for c, it in secondary: render(c, it)
    act = active_criteria(cfg)
    st.markdown(f"**{len(act)} criteria active:** " + (", ".join(K.label_of(domain, c) for c in act) if act else "none"))
    if len(act) < 2: T.note("Select at least two criteria to build a ranking.", warn=True)
    if len(act) > 8: T.note("More than eight criteria makes pairwise weighting hard to keep consistent. Consider excluding overlapping fields.", warn=True)


def weights_tab(cfg):
    act = active_criteria(cfg)
    if len(act) < 2:
        T.note("Choose at least two criteria in the previous tab.", warn=True); return
    cfg["weight_method"] = st.radio("Weighting method", ["Equal weights", "AHP pairwise comparison"], horizontal=True, key=f"wm_{key}")
    if cfg["weight_method"] == "Equal weights":
        cfg["weights"] = {c: 1 / len(act) for c in act}; cfg["cr"] = None
    else:
        st.caption("Compare each pair on Saaty's scale. A value above 1 means the first criterion is more important than the second; below 1 means the second is more important.")
        M = np.ones((len(act), len(act)))
        pairs = [(i, j) for i in range(len(act)) for j in range(i + 1, len(act))]
        cols = st.columns(2)
        for n, (i, j) in enumerate(pairs):
            v = cols[n % 2].select_slider(f"{K.label_of(domain, act[i])}  vs  {K.label_of(domain, act[j])}", SAATY, value="1", key=f"ahp_{key}_{act[i]}_{act[j]}")
            M[i, j] = SAATY_V[v]; M[j, i] = 1 / SAATY_V[v]
        w, cr = K.ahp(M)
        cfg["weights"] = dict(zip(act, w)); cfg["cr"] = cr
    w = active_weights(cfg)
    wd = pd.DataFrame({"Criterion": [K.label_of(domain, c) for c in w], "Weight": list(w.values())}).sort_values("Weight")
    fig = px.bar(wd, x="Weight", y="Criterion", orientation="h", title="Criterion weights", text=wd["Weight"].map(lambda v: f"{v:.1%}"))
    fig.update_traces(marker_color=T.BLUE)
    st.plotly_chart(fig, width="stretch")
    if cfg.get("cr") is not None:
        ok = cfg["cr"] < 0.10
        T.note(f"Consistency ratio {cfg['cr']:.3f}. " + ("The comparisons are internally consistent (below the 0.10 guideline)." if ok else
               "The comparisons contradict each other (0.10 or above). Revisit the pairs where one criterion is ranked far above another before relying on the ranking."), warn=not ok)


def results_tab(cfg):
    act = active_criteria(cfg)
    if len(act) < 2:
        T.note("Choose at least two criteria first.", warn=True); return
    c1, c2, c3 = st.columns([1, 1, 1])
    scope = c1.selectbox("Records to rank", SCOPES, key=f"rs_{key}")
    cfg["method"] = c2.selectbox("Ranking method", ["AHP + TOPSIS", "TOPSIS only", "Weighted sum (AHP)", "Hybrid blend"], key=f"rm_{key}")
    cfg["mix"] = c3.slider("Weighted-sum share (hybrid)", 0.0, 1.0, 0.5, 0.05, disabled=cfg["method"] != "Hybrid blend", key=f"mix_{key}")
    if st.button("Calculate ranking", type="primary"):
        sd = scope_data(scope)
        try:
            w = effective_cfg_weights(cfg)
            mc = K.run_mcdm(domain, sd["mrows"], {**cfg, "weights": w}, w, cfg["method"], cfg["mix"])
            res = mc["result"].copy()
            res.insert(0, "row_key", sd["keys"].to_numpy())
            for c in mc["scores"].columns: res[K.label_of(domain, c)] = sd["mrows"][c].to_numpy()
            res = res.sort_values("Final_score", ascending=False).reset_index(drop=True)
            res.insert(0, "rank", np.arange(1, len(res) + 1))
            state["rank_result"] = dict(table=res, mc=mc, scope=scope, method=cfg["method"], weights=w, n=len(res))
            wt = pd.DataFrame([{"criterion": c, "label": K.label_of(domain, c), "rule": K.TREATMENTS[cfg["criteria"][c]["choice"]["treatment"]], "weight": w[c]} for c in w])
            wt.to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_criterion_weights.csv", index=False)
            res.to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_priority_ranking.csv", index=False)
        except Exception as e:
            st.error(f"Ranking failed: {e}")
    rr = state.get("rank_result")
    if not rr:
        return
    res, mc = rr["table"], rr["mc"]
    k = st.columns(4)
    T.kpi(k[0], "Records ranked", f"{rr['n']:,}", rr["scope"]); T.kpi(k[1], "Method", rr["method"]); T.kpi(k[2], "Criteria used", f"{len(rr['weights'])}")
    T.kpi(k[3], "Median score", f"{res['Final_score'].median():.3f}", f"top score {res['Final_score'].max():.3f}")
    st.write("")
    a, b = st.columns(2)
    fig = px.histogram(res, x="Final_score", nbins=40, title="Distribution of priority scores")
    fig.update_traces(marker_color=T.BLUE); a.plotly_chart(fig, width="stretch")
    top = mc["contrib"].loc[res.index.map(lambda i: i)].head(0)  # placeholder for index alignment below
    order = mc["result"]["Final_score"].sort_values(ascending=False).head(15).index
    cb = mc["contrib"].loc[order].copy(); cb.columns = [K.label_of(domain, c) for c in cb.columns]
    cb["Record"] = [f"#{i + 1}" for i in range(len(cb))]
    fig = px.bar(cb.melt(id_vars="Record", var_name="Criterion", value_name="Contribution"), x="Contribution", y="Record", color="Criterion", orientation="h",
                 title="What drives the 15 highest-priority records")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    b.plotly_chart(fig, width="stretch")
    st.markdown("**Highest-priority records**")
    st.dataframe(tidy_table(res.head(100)), hide_index=True, width="stretch", height=360)
    # sensitivity
    S, w = mc["scores"], np.array([rr["weights"][c] for c in mc["scores"].columns])
    base = res.set_index("row_key")["Final_score"] if False else mc["result"]["Final_score"]
    eq = pd.Series(K.topsis(S.to_numpy(), np.ones(S.shape[1]) / S.shape[1]), index=S.index)
    topn = max(1, int(0.1 * len(base)))
    overlap = len(set(base.nlargest(topn).index) & set(eq.nlargest(topn).index)) / topn
    rows = []
    for j, c in enumerate(S.columns):
        keep = [i for i in range(S.shape[1]) if i != j]
        w2 = w[keep] / w[keep].sum()
        alt = pd.Series(K.topsis(S.to_numpy()[:, keep], w2), index=S.index)
        rows.append({"Criterion removed": K.label_of(domain, c), "Rank correlation with current ranking": round(float(base.rank().corr(alt.rank())), 3),
                     "Top-10% records unchanged": f"{len(set(base.nlargest(topn).index) & set(alt.nlargest(topn).index)) / topn:.0%}"})
    st.markdown("**Sensitivity of the ranking**")
    T.note(f"Replacing the chosen weights with equal weights keeps {overlap:.0%} of the top-10% records unchanged "
           f"(rank correlation {base.rank().corr(eq.rank()):.3f}). The table shows how much the ranking depends on each criterion.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("Download ranking (CSV)", res.to_csv(index=False).encode(), file_name=f"{C.SLUG(domain)}_priority_ranking.csv", mime="text/csv")
    wt = pd.DataFrame({"criterion": [K.label_of(domain, c) for c in rr["weights"]], "weight": list(rr["weights"].values())})
    d2.download_button("Download criterion weights (CSV)", wt.to_csv(index=False).encode(), file_name=f"{C.SLUG(domain)}_criterion_weights.csv", mime="text/csv")


def page_prioritisation():
    T.header("Prioritisation (AHP–TOPSIS)", "Rank records by the criteria that matter for this decision, with explicit scoring rules and weights.")
    cfg = get_cfg()
    t1, t2, t3 = st.tabs(["1 · Criteria and scoring rules", "2 · Weights", "3 · Ranking and sensitivity"])
    with t1: criteria_tab(cfg)
    with t2: weights_tab(cfg)
    with t3: results_tab(cfg)


# ---------------- Decision rationale ----------------
def page_rationale():
    T.header("Decision rationale", "A plain-language explanation of why each record receives its recommendation.")
    if not need_model(): return
    c1, c2, c3, c4 = st.columns(4)
    scope = c1.selectbox("Records", SCOPES, key=f"dr_scope_{key}")
    mlw = c2.slider("Weight on prediction model", 0.0, 1.0, 0.5, 0.05, key=f"dr_mlw_{key}", help="The remaining weight goes to the criteria ranking.")
    top = c3.slider("Top tier share", 0.05, 0.40, 0.20, 0.05, key=f"dr_top_{key}")
    mid = c4.slider("Top two tiers share", 0.30, 0.80, 0.50, 0.05, key=f"dr_mid_{key}")
    if mid <= top: mid = min(0.95, top + 0.1)
    cfg = get_cfg()
    if len(active_criteria(cfg)) < 2:
        T.note("Fewer than two ranking criteria are active. Set them in Prioritisation.", warn=True); return
    with st.spinner("Scoring records..."):
        tbl, mc, sd = get_decision(scope, mlw, top, mid)
    n = len(tbl)
    k = st.columns(4)
    T.kpi(k[0], f"{meta['entities'].capitalize()} scored", f"{n:,}", scope)
    T.kpi(k[1], meta["tiers"][0], f"{(tbl['tier_rank'] == 1).sum():,}", f"{(tbl['tier_rank'] == 1).mean():.0%} of records")
    T.kpi(k[2], "Views aligned", f"{(tbl['views_agree'] == 'Aligned').mean():.0%}", "prediction model and criteria agree")
    T.kpi(k[3], "Average model risk", f"{tbl['model_risk'].mean():.1%}", meta["adverse_label"])
    st.write("")
    a, b = st.columns([1, 1.3])
    tc = tbl["recommendation"].value_counts().reindex(meta["tiers"]).fillna(0).reset_index(); tc.columns = ["Recommendation", "Records"]
    fig = px.bar(tc, x="Recommendation", y="Records", color="Recommendation", color_discrete_sequence=[T.RED, T.AMBER, T.TEAL], title="Recommendations")
    fig.update_layout(showlegend=False, xaxis_title=None); a.plotly_chart(fig, width="stretch")
    samp = tbl.sample(min(3000, n), random_state=1)
    fig = px.scatter(samp, x="model_risk", y="Final_score", color="recommendation", color_discrete_sequence=[T.RED, T.AMBER, T.TEAL],
                     category_orders={"recommendation": meta["tiers"]}, opacity=0.6, title="Prediction model risk vs criteria score",
                     labels={"model_risk": "Model-estimated probability", "Final_score": "Criteria priority score"})
    b.plotly_chart(fig, width="stretch")

    st.subheader("Priority queue")
    f1, f2 = st.columns(2)
    tier_f = f1.multiselect("Recommendation", meta["tiers"], default=meta["tiers"][:1], key=f"dr_tf_{key}")
    agree_f = f2.multiselect("View alignment", sorted(tbl["views_agree"].unique()), default=[], key=f"dr_af_{key}")
    q = tbl[tbl["recommendation"].isin(tier_f or meta["tiers"])]
    if agree_f: q = q[q["views_agree"].isin(agree_f)]
    q = q.head(300)
    show = q[["rank", "row_key", "recommendation", "model_risk", "Final_score", "combined_index", "top_criterion", "views_agree"]].rename(columns={
        "model_risk": "Model risk", "Final_score": "Criteria score", "combined_index": "Combined index", "top_criterion": "Strongest criterion", "views_agree": "Views", "row_key": "Record"})
    sel = st.dataframe(tidy_table(show), hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row", key=f"dr_tbl_{key}")
    if q.empty:
        st.info("No records match these filters."); return
    picked = sel.selection.rows[0] if sel and sel.selection.rows else 0
    rec = q.iloc[picked]
    rec = tbl.loc[rec.name]
    row = sd["X"].loc[[rec.name]]
    drivers, p0 = C.local_drivers(state["active_model"], bundle["X_train"], row, ensure_adverse())
    R = N.rationale(domain, rec, tbl, mc, {**cfg, "weights": effective_cfg_weights(cfg)}, sd["mrows"], drivers, p0, meta["adverse_label"], top)

    colour = T.TIER_COLORS[int(rec["tier_rank"])]
    st.markdown(f'<div class="verdict" style="background:{colour}"><div class="t">Recommendation for record {rec["row_key"]}</div><div class="h">{R["headline"]}</div></div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l: T.card("What the prediction model sees", R["model"])
    with r: T.card("What the criteria ranking sees", R["ranking"])
    if R["lowered"]:
        T.card("Factors holding the priority down", R["lowered"])
    T.card("How the two views compare", [R["agreement"], R["next_step"]])
    g1, g2 = st.columns(2)
    if len(drivers):
        dd = drivers.head(8).copy()
        dd["Field"] = [K.label_of(domain, f) + " = " + N.fmt_value(v) for f, v in zip(dd["field"], dd["value"])]
        dd["Effect (pp)"] = dd["effect"] * 100
        fig = px.bar(dd.sort_values("Effect (pp)"), x="Effect (pp)", y="Field", orientation="h", title="Model: effect on estimated probability vs a typical record")
        fig.update_traces(marker_color=[T.RED if v > 0 else T.TEAL for v in dd.sort_values("Effect (pp)")["Effect (pp)"]])
        g1.plotly_chart(fig, width="stretch")
    it = pd.DataFrame(R["items"])
    fig = px.bar(it.sort_values("delta"), x="delta", y="label", orientation="h", title="Ranking: contribution vs median record",
                 labels={"delta": "Difference in weighted score", "label": ""})
    fig.update_traces(marker_color=[T.RED if v > 0 else T.TEAL for v in it.sort_values("delta")["delta"]])
    g2.plotly_chart(fig, width="stretch")
    with st.expander("Scoring detail for this record"):
        det = pd.DataFrame({"Criterion": it["label"], "Value": [N.fmt_value(v) for v in it["raw"]], "Score (0-1)": it["score"], "Weight": it["weight"],
                            "Contribution": it["contrib"], "Share of total": it["share"], "Why it matters": it["why"]})
        st.dataframe(tidy_table(det), hide_index=True, width="stretch")
    report = "\n".join([f"Record {rec['row_key']} — {domain}", R["headline"], "", "Prediction model:", *R["model"], "", "Criteria ranking:", *R["ranking"], "",
                        "Holding priority down:", *R["lowered"], "", R["agreement"], R["next_step"]])
    st.download_button("Download this explanation (TXT)", report.encode(), file_name=f"rationale_{rec['row_key']}.txt")
    st.download_button("Download all scored decisions (CSV)", tbl.reset_index(drop=True).to_csv(index=False).encode(), file_name=f"{C.SLUG(domain)}_decisions.csv", mime="text/csv")


# ---------------- Technical insights ----------------
def page_insights():
    T.header("Model insights (SHAP / LIME)", "Technical attribution for analysts. Business users should start with Decision rationale.")
    if not need_model(): return
    source = st.radio("Explain a record from", ["Human review pool", "Final test set"], horizontal=True)
    Xs = bundle["X_review"] if source == "Human review pool" else bundle["X_test"]
    r = st.number_input("Record position in this partition", 0, max(0, len(Xs) - 1), 0, 1)
    row = Xs.iloc[[int(r)]]
    model = state["active_model"]
    pred = model.predict(row)[0]
    k = st.columns(2); T.kpi(k[0], "Predicted outcome", str(pred))
    if hasattr(model, "predict_proba"):
        pr = pd.DataFrame({"Outcome": [str(c) for c in model.classes_], "Probability": model.predict_proba(row)[0]}).sort_values("Probability", ascending=False)
        fig = px.bar(pr, x="Probability", y="Outcome", orientation="h", title="Outcome probabilities"); fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    with st.expander("Record fields"):
        st.dataframe(C.feature_row_for_display(row), hide_index=True, width="stretch")
    if st.button("Generate SHAP and LIME attributions", type="primary"):
        with st.spinner("Computing attributions..."):
            exp = C.get_explanations(model, bundle["X_train"], row)
        if exp["shap"] is not None: st.subheader("SHAP"); st.pyplot(exp["shap"])
        if exp["lime"] is not None: st.subheader("LIME"); st.pyplot(exp["lime"])
        for e in exp["errors"]: st.warning(e)
    T.note("Feature names are shown after preprocessing, so categorical fields appear as individual one-hot columns.")


# ---------------- Human review ----------------
def page_review():
    T.header("Human review", "Record reviewer decisions. Corrections are held for approval before they can influence retraining.")
    if not need_model(): return
    Xr, yr = bundle["X_review"], bundle["y_review"]
    preds = state["active_model"].predict(Xr)
    tbl = None
    cfg = get_cfg()
    if len(active_criteria(cfg)) >= 2:
        tbl, mc, sd = get_decision("Human review pool")
    order = list(tbl.index[:500]) if tbl is not None else list(range(min(500, len(Xr))))
    def lab(i):
        extra = f" · {tbl.loc[i, 'recommendation']} (rank {int(tbl.loc[i, 'rank'])})" if tbl is not None else ""
        return f"Record {i} · model: {preds[i]}{extra}"
    chosen = st.selectbox("Record to review", order, format_func=lab)
    row_key = str(bundle["row_keys"].iloc[int(bundle["idx_review"][chosen])])
    a, b = st.columns([1, 1])
    with a:
        T.card("Model output", [f"Predicted outcome: <b>{preds[chosen]}</b>", f"Recorded label: <b>{yr.iloc[chosen]}</b>", f"Record key: {row_key}"])
        if tbl is not None:
            rec = tbl.loc[chosen]
            T.card("Decision support", [f"Recommendation: <b>{rec['recommendation']}</b>", f"Combined rank {int(rec['rank']):,} of {len(tbl):,}", f"Strongest criterion: {rec['top_criterion']}", f"View alignment: {rec['views_agree']}"])
    with b:
        st.dataframe(C.feature_row_for_display(Xr.iloc[[chosen]]), hide_index=True, width="stretch", height=300)
    labels = sorted(bundle["y_train"].astype(str).unique().tolist())
    reasons = {"Hospital Readmission": ["Clinical review", "Prediction appears incorrect", "Missing clinical context", "Data quality issue", "Fairness or subgroup concern", "Policy exception", "Other"],
               "ITSM Incident Priority": ["Business impact review", "Urgency changed", "Service criticality", "Team capacity", "SLA or escalation policy", "Data quality issue", "Other"],
               "Credit Risk Assessment": ["Income or affordability review", "Credit history correction", "Missing financial context", "Policy exception", "Fairness or bias concern", "Data quality issue", "Other"]}[domain]
    c1, c2, c3 = st.columns(3)
    corrected = c1.selectbox("Reviewed label", labels, index=labels.index(str(yr.iloc[chosen])) if str(yr.iloc[chosen]) in labels else 0)
    reason = c2.selectbox("Reason", reasons, key=f"rev_reason_{domain}_{key}")
    reviewer = c3.text_input("Reviewer (initials or role)")
    if st.button("Save review", type="primary"):
        with C.db_conn() as con:
            con.execute("INSERT INTO feedback(domain,dataset_key,row_key,original_prediction,corrected_label,reason,reviewer,approved,model_version,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (domain, key, row_key, str(preds[chosen]), str(corrected), reason, reviewer, 0, state["active_version"], datetime.now(timezone.utc).isoformat()))
        st.success("Review saved and awaiting approval.")
    fb = get_feedback()
    st.subheader("Review log")
    if len(fb):
        st.dataframe(fb, hide_index=True, width="stretch")
    else:
        st.info("No reviews recorded yet.")


def page_retrain():
    T.header("Retrain and promote", "Approve reviewed corrections, train a candidate model, and promote it only if validation performance holds.")
    if not need_model(): return
    pend = get_feedback(approved=False)
    appr = get_feedback(approved=True)
    k = st.columns(3); T.kpi(k[0], "Pending reviews", len(pend)); T.kpi(k[1], "Approved reviews", len(appr)); T.kpi(k[2], "Active version", state.get("active_version"))
    st.write("")
    if len(pend):
        st.dataframe(pend[["feedback_id", "row_key", "original_prediction", "corrected_label", "reason", "reviewer", "created_at"]], hide_index=True, width="stretch")
        ids = st.multiselect("Approve review IDs", pend["feedback_id"].astype(int).tolist(), key=f"appr_{domain}_{key}")
        if st.button("Approve selected", disabled=not ids):
            with C.db_conn() as con:
                for i in ids: con.execute("UPDATE feedback SET approved=1 WHERE feedback_id=? AND domain=? AND dataset_key=?", (int(i), domain, key))
            st.rerun()
    else:
        st.info("No reviews waiting for approval.")
    T.note("Retraining updates the prediction model only. Criteria scoring rules and weights are changed in Prioritisation.")
    if st.button("Train candidate and apply promotion gate", type="primary", disabled=len(appr) == 0):
        try:
            Xc, yc = bundle["X_train"].copy(), bundle["y_train"].astype(str).copy()
            pos = {str(bundle["row_keys"].iloc[int(o)]): j for j, o in enumerate(bundle["idx_review"])}
            addX, addy, used, seen = [], [], [], set()
            for _, rec in appr.iterrows():
                rk = str(rec["row_key"])
                if rk in pos and rk not in seen:
                    addX.append(bundle["X_review"].iloc[[pos[rk]]]); addy.append(str(rec["corrected_label"])); used.append(int(rec["feedback_id"])); seen.add(rk)
            if not addX:
                st.error("None of the approved reviews map to the current review pool."); return
            Xc = pd.concat([Xc] + addX, ignore_index=True); yc = pd.concat([yc, pd.Series(addy, dtype=str)], ignore_index=True)
            cand = C.Pipeline([("preprocess", C.build_preprocessor(Xc)), ("model", C.candidate_models()[state["active_name"]])])
            with st.spinner("Training candidate model..."):
                cand.fit(Xc, yc)
            bv, cv = C.metrics_for(state["active_model"], bundle["X_val"], bundle["y_val"]), C.metrics_for(cand, bundle["X_val"], bundle["y_val"])
            comp = pd.DataFrame([{"model": "Current", **bv}, {"model": "Candidate", **cv}])
            comp.to_csv(C.REPORT_DIR / f"{C.SLUG(domain)}_{key}_retraining_{datetime.now():%Y%m%d_%H%M%S}.csv", index=False)
            state["last_retrain"] = comp
            passed = cv["f1_macro"] >= bv["f1_macro"] - 0.01 and cv["balanced_accuracy"] >= bv["balanced_accuracy"] - 0.02
            if passed:
                try: num = int(str(state["active_version"]).lower().replace("v", ""))
                except ValueError: num = 1
                ver = f"v{num + 1}"
                C.write_active_bundle(domain, bundle, state["active_name"], cand, cv, key, ver)
                state.update({"active_model": cand, "active_version": ver, "model_metrics": cv, "decisions": {}})
                with C.db_conn() as con:
                    for i in used: con.execute("UPDATE feedback SET retrained_at=? WHERE feedback_id=? AND domain=? AND dataset_key=?", (datetime.now(timezone.utc).isoformat(), i, domain, key))
                st.success(f"Candidate promoted as {ver}.")
            else:
                st.warning("The candidate did not meet the promotion gate; the current model remains active.")
        except Exception as e:
            st.error(f"Retraining failed: {e}")
    if state.get("last_retrain") is not None:
        st.subheader("Latest validation comparison")
        st.dataframe(tidy_table(state["last_retrain"]), hide_index=True, width="stretch")


def page_artifacts():
    T.header("Saved artifacts", "Datasets, models, reports and audit records stored by this application.")
    rows = []
    for folder in [C.DATA_DIR, C.ART_DIR, C.REPORT_DIR]:
        for p in sorted(folder.glob("*")):
            if p.is_file(): rows.append({"Folder": folder.name, "File": p.name, "Size (KB)": round(p.stat().st_size / 1024, 1), "Modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.info("No artifacts have been saved yet.")
    with C.db_conn() as con:
        st.subheader("Model registry")
        reg = pd.read_sql_query("SELECT * FROM model_registry WHERE domain=? ORDER BY registry_id DESC", con, params=(domain,))
        st.dataframe(reg, hide_index=True, width="stretch") if len(reg) else st.info("No model versions registered yet.")


# ============================ Navigation ============================
if not has_project:
    T.header("Explainable decision support", "Prediction, prioritisation and plain-language reasoning in one workflow.")
    k = st.columns(3)
    for col, (name, m) in zip(k, C.DOMAIN_META.items()):
        T.kpi(col, name, m["adverse_label"].capitalize(), m["objective"])
    st.write("")
    T.note(f"Select <b>{domain}</b> data from the sidebar: load the local file, download from UCI, or upload a CSV.")
    st.stop()

pages = {
    "Insight": [st.Page(page_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
                st.Page(page_data, title="Data profile", icon=":material/table_chart:", url_path="data")],
    "Analytics": [st.Page(page_models, title="Model studio", icon=":material/model_training:", url_path="models"),
                  st.Page(page_prioritisation, title="Prioritisation", icon=":material/stacked_bar_chart:", url_path="prioritisation"),
                  st.Page(page_rationale, title="Decision rationale", icon=":material/fact_check:", url_path="rationale"),
                  st.Page(page_insights, title="Model insights", icon=":material/query_stats:", url_path="insights")],
    "Governance": [st.Page(page_review, title="Human review", icon=":material/rate_review:", url_path="review"),
                   st.Page(page_retrain, title="Retrain and promote", icon=":material/published_with_changes:", url_path="retrain"),
                   st.Page(page_artifacts, title="Saved artifacts", icon=":material/inventory_2:", url_path="artifacts")],
}
st.navigation(pages).run()
