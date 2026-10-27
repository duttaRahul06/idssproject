"""Criteria registry, scoring rules, AHP weighting and TOPSIS ranking.

Every MCDM criterion is converted to a 0-1 *priority score* (higher = reviewed sooner) using an explicit,
documented treatment. Columns without a defensible direction are excluded by default rather than being forced
into Benefit/Cost.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

TREATMENTS = {
    "higher": "Higher value raises priority",
    "lower": "Lower value raises priority",
    "band": "Contextual: priority peaks inside a defined range",
    "map": "Contextual: score each category",
    "exclude": "Exclude from ranking (kept for model and explanations)",
}

# recode: how raw values become numbers before scoring
#   lead_int  '1 - High' -> 1          pay_status  -2/-1/0 -> 0 (no delay), k>=1 -> k months delayed
#   age_band  '[70-80)' -> 75          bool        True/False -> 1/0
SPEC = {
 "Hospital Readmission": {
  "number_inpatient":  dict(label="Prior inpatient visits", allowed=["higher","band","exclude"], default="higher",
      meaning="Hospital admissions in the year before this encounter.",
      why="Repeat admissions are one of the strongest recorded markers of early readmission (a component of the HOSPITAL score)."),
  "number_emergency":  dict(label="Prior emergency visits", allowed=["higher","band","exclude"], default="higher",
      meaning="Emergency department visits in the year before this encounter.",
      why="Prior emergency use is a component of the LACE readmission index."),
  "number_outpatient": dict(label="Prior outpatient visits", allowed=["higher","band","exclude"], default="higher",
      meaning="Outpatient visits in the year before this encounter.",
      why="Frequent outpatient contact signals ongoing, unresolved care needs."),
  "time_in_hospital":  dict(label="Length of stay (days)", allowed=["higher","lower","band","exclude"], default="higher",
      meaning="Days between admission and discharge.",
      why="Longer stays indicate greater illness severity and are part of both LACE and HOSPITAL scores."),
  "number_diagnoses":  dict(label="Number of diagnoses", allowed=["higher","band","exclude"], default="higher",
      meaning="Diagnoses entered for the encounter.",
      why="A higher diagnosis count is a practical proxy for comorbidity burden."),
  "num_medications":   dict(label="Distinct medications", allowed=["higher","band","exclude"], default="exclude",
      meaning="Distinct medications administered.",
      why="Medication count has no fixed direction: it reflects both severity and good disease management, so it is excluded unless a range is defined."),
  "num_lab_procedures":dict(label="Lab procedures", allowed=["higher","band","exclude"], default="exclude",
      meaning="Laboratory tests performed.",
      why="Testing volume follows protocol as much as risk, so it has no default direction."),
  "num_procedures":    dict(label="Procedures", allowed=["higher","band","exclude"], default="exclude",
      meaning="Non-lab procedures performed.",
      why="Procedures can lower or raise risk depending on the condition; no default direction."),
  "age":               dict(label="Age band", allowed=["band","higher","lower","exclude"], default="exclude", recode="age_band",
      meaning="Ten-year age band of the patient.",
      why="Neither older nor younger is preferred by default. Define an age range only if your review policy targets one."),
 },
 "ITSM Incident Priority": {
  "impact":   dict(label="Business impact", allowed=["lower","higher","exclude"], default="lower", recode="lead_int",
      meaning="Coded impact: 1 - High, 2 - Medium, 3 - Low.",
      why="In ITIL coding, 1 is the most severe level, so lower codes mean greater business impact."),
  "urgency":  dict(label="Urgency", allowed=["lower","higher","exclude"], default="lower", recode="lead_int",
      meaning="Coded urgency: 1 - High, 2 - Medium, 3 - Low.",
      why="In ITIL coding, 1 is the most urgent level, so lower codes mean a faster response is needed."),
  "reassignment_count": dict(label="Reassignments", allowed=["higher","band","exclude"], default="higher",
      meaning="Times the incident was moved between assignment groups.",
      why="Repeated hand-offs show an incident is stuck and consuming service-desk effort."),
  "reopen_count": dict(label="Reopen count", allowed=["higher","band","exclude"], default="higher",
      meaning="Times the incident was reopened after resolution.",
      why="Reopened incidents were not fixed first time and carry a higher chance of repeat escalation."),
  "sys_mod_count": dict(label="Record updates", allowed=["higher","band","exclude"], default="exclude",
      meaning="Number of updates to the ticket.",
      why="Update volume reflects workflow habits as much as difficulty, so it has no default direction."),
  "made_sla": dict(label="SLA met", allowed=["map","exclude"], default="exclude", recode="bool",
      default_map={"0": 1.0, "1": 0.0},
      meaning="Whether the service-level target was met.",
      why="A missed SLA should raise priority. Excluded by default because this flag is set late in the ticket life-cycle."),
  "knowledge": dict(label="Knowledge article used", allowed=["map","exclude"], default="exclude", recode="bool",
      meaning="Whether a knowledge-base article was linked.",
      why="Having an article can shorten or lengthen handling; no default direction."),
 },
 "Credit Risk Assessment": {
  "PAY_0": dict(label="Repayment status, latest month", allowed=["higher","exclude"], default="higher", recode="pay_status",
      meaning="Months of payment delay in the latest month (-2/-1/0 = no delay, 1 = one month late, 9 = nine or more).",
      why="Recent delay is the strongest single indicator of next-month default in this data."),
  "PAY_2": dict(label="Repayment status, 2 months ago", allowed=["higher","exclude"], default="exclude", recode="pay_status",
      meaning="Months of payment delay two months ago.", why="Earlier delays add history but overlap with the maximum-delay measure."),
  "PAY_3": dict(label="Repayment status, 3 months ago", allowed=["higher","exclude"], default="exclude", recode="pay_status",
      meaning="Months of payment delay three months ago.", why="Earlier delays add history but overlap with the maximum-delay measure."),
  "PAY_4": dict(label="Repayment status, 4 months ago", allowed=["higher","exclude"], default="exclude", recode="pay_status",
      meaning="Months of payment delay four months ago.", why="Earlier delays add history but overlap with the maximum-delay measure."),
  "PAY_5": dict(label="Repayment status, 5 months ago", allowed=["higher","exclude"], default="exclude", recode="pay_status",
      meaning="Months of payment delay five months ago.", why="Earlier delays add history but overlap with the maximum-delay measure."),
  "PAY_6": dict(label="Repayment status, 6 months ago", allowed=["higher","exclude"], default="exclude", recode="pay_status",
      meaning="Months of payment delay six months ago.", why="Earlier delays add history but overlap with the maximum-delay measure."),
  "max_delay_months": dict(label="Longest payment delay (6 months)", allowed=["higher","exclude"], default="higher", derived=True,
      meaning="Longest delay, in months, across the six-month history.",
      why="Captures the worst recent repayment behaviour in one measure."),
  "months_delayed": dict(label="Months with a delay (of 6)", allowed=["higher","exclude"], default="exclude", derived=True,
      meaning="How many of the last six months had any delay.", why="Shows how persistent late payment has been."),
  "utilisation_ratio": dict(label="Credit utilisation", allowed=["higher","band","exclude"], default="higher", derived=True,
      meaning="Latest bill amount divided by the credit limit.",
      why="A balance close to the limit leaves little headroom and is associated with repayment stress."),
  "repayment_ratio": dict(label="Repayment ratio", allowed=["lower","exclude"], default="lower", derived=True,
      meaning="Latest payment divided by the latest bill (capped at 1).",
      why="Paying only a small share of the bill raises concern; lower ratios increase priority."),
  "LIMIT_BAL": dict(label="Credit limit", allowed=["band","higher","lower","exclude"], default="exclude",
      meaning="Credit limit granted to the account holder.",
      why="A limit alone does not indicate default risk, so it is excluded unless you define a range."),
  "AGE": dict(label="Age", allowed=["band","exclude"], default="exclude",
      meaning="Age of the account holder.",
      why="Neither younger nor older is preferred by default; age stays in the model but not in the ranking."),
  "BILL_AMT1": dict(label="Latest bill amount", allowed=["higher","band","exclude"], default="exclude",
      meaning="Bill statement amount, latest month.", why="Absolute bill size depends on limit; see Credit utilisation."),
  "PAY_AMT1": dict(label="Latest payment amount", allowed=["lower","band","exclude"], default="exclude",
      meaning="Payment made, latest month.", why="Needs the bill amount for context; see Repayment ratio."),
 },
}


# ---------- Derived criteria ----------
def _pay_status(s):
    v = pd.to_numeric(s, errors="coerce")
    return v.clip(lower=0)


def derive_columns(domain: str, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if domain == "Credit Risk Assessment":
        pay_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in X.columns]
        if pay_cols:
            d = pd.concat([_pay_status(X[c]) for c in pay_cols], axis=1)
            X["max_delay_months"] = d.max(axis=1)
            X["months_delayed"] = (d > 0).sum(axis=1)
        if {"BILL_AMT1", "LIMIT_BAL"} <= set(X.columns):
            lim = pd.to_numeric(X["LIMIT_BAL"], errors="coerce").replace(0, np.nan)
            X["utilisation_ratio"] = (pd.to_numeric(X["BILL_AMT1"], errors="coerce").clip(lower=0) / lim).clip(0, 1.5)
        if {"BILL_AMT1", "PAY_AMT1"} <= set(X.columns):
            bill = pd.to_numeric(X["BILL_AMT1"], errors="coerce")
            pay = pd.to_numeric(X["PAY_AMT1"], errors="coerce")
            X["repayment_ratio"] = np.where(bill > 0, (pay / bill.replace(0, np.nan)).clip(0, 1), 1.0)
    return X


# ---------- Recoding ----------
def recode_numeric(s: pd.Series, how: str | None) -> pd.Series:
    if how == "pay_status":
        return _pay_status(s)
    if how == "lead_int":
        return pd.to_numeric(s.astype(str).str.extract(r"^\s*(\d+)", expand=False), errors="coerce")
    if how == "age_band":
        return pd.to_numeric(s.astype(str).str.extract(r"\[(\d+)-(\d+)\)").astype(float).mean(axis=1), errors="coerce")
    if how == "bool":
        m = s.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0})
        return pd.to_numeric(m, errors="coerce")
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    ex = s.astype(str).str.extract(r"([-+]?(?:\d+\.?\d*|\.\d+))", expand=False)
    return pd.to_numeric(ex, errors="coerce")


# ---------- Candidate discovery ----------
def candidate_columns(domain: str, X: pd.DataFrame, target: str) -> dict:
    """Return {column: spec} for every column that can sensibly act as a criterion."""
    reg = SPEC.get(domain, {})
    out = {}
    tgt = str(target).strip().lower()
    for col in X.columns:
        low = str(col).strip().lower()
        if low == tgt or low in {"__row_key__", "id", "index"} or low.startswith("unnamed"):
            continue
        if col in reg:
            out[col] = dict(reg[col]); continue
        s = X[col]
        if re.search(r"(_id|_code|_nbr|^diag_)", low) or low.endswith("id"):
            continue  # coded identifiers have no numeric meaning
        rec = recode_numeric(s, None)
        non_missing = s.notna().sum()
        if non_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and rec.nunique() > 2:
            out[col] = dict(label=pretty(col), allowed=["higher", "lower", "band", "exclude"], default="exclude",
                            meaning="Numeric field with no documented decision meaning.",
                            why="No direction is assumed for this field. Choose a treatment only if your policy defines one.")
        elif s.nunique(dropna=True) <= 12:
            out[col] = dict(label=pretty(col), allowed=["map", "exclude"], default="exclude", categorical=True,
                            meaning="Category field with no documented ordering.",
                            why="Categories have no natural order. Score them explicitly or leave them out.")
    # registry columns first, in registry order, then others
    ordered = {c: out[c] for c in reg if c in out}
    ordered.update({c: v for c, v in out.items() if c not in ordered})
    return ordered


def pretty(col: str) -> str:
    name = re.sub(r"^u_", "", str(col))
    return re.sub(r"\s+", " ", name.replace("_", " ").replace(".", " ")).strip().capitalize()


def label_of(domain: str, col: str) -> str:
    return SPEC.get(domain, {}).get(col, {}).get("label") or pretty(col)


# ---------- Scoring ----------
def default_choice(spec: dict, values: pd.Series | None = None) -> dict:
    ch = {"treatment": spec.get("default", "exclude")}
    if values is not None and "band" in spec.get("allowed", []):
        q = values.dropna()
        if len(q):
            lo, hi = float(q.quantile(0.50)), float(q.quantile(0.90))
            if hi <= lo:
                hi = float(q.max())
            ch["band"] = [round(lo, 4), round(hi, 4)]
    if "map" in spec.get("allowed", []):
        ch["map"] = dict(spec.get("default_map", {}))
    return ch


def score_criterion(raw: pd.Series, spec: dict, choice: dict) -> pd.Series:
    """Convert raw values into a 0-1 priority score according to the chosen treatment."""
    t = choice.get("treatment", "exclude")
    if t == "map":
        key = raw.astype(str).str.strip()
        if spec.get("recode") == "bool":
            key = recode_numeric(raw, "bool").map(lambda v: None if pd.isna(v) else str(int(v)))
        mp = {str(k): float(v) for k, v in choice.get("map", {}).items()}
        s = key.map(mp)
        s = s.fillna(s.median() if s.notna().any() else 0.5)
        return s.astype(float).clip(0, 1)
    x = recode_numeric(raw, spec.get("recode"))
    x = x.fillna(x.median() if x.notna().any() else 0.0).astype(float)
    lo_c, hi_c = x.quantile(0.01), x.quantile(0.99)
    if t in ("higher", "lower"):
        if hi_c <= lo_c:
            lo_c, hi_c = x.min(), x.max()
        s = pd.Series(0.5, index=x.index) if hi_c <= lo_c else ((x.clip(lo_c, hi_c) - lo_c) / (hi_c - lo_c))
        return s if t == "higher" else 1.0 - s
    if t == "band":
        lo, hi = choice.get("band", [x.quantile(0.5), x.quantile(0.9)])
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        reach = max(lo - x.min(), x.max() - hi, 1e-9)
        dist = np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))
        return pd.Series(1.0 - dist / reach, index=x.index).clip(0, 1)
    raise ValueError(f"Treatment '{t}' cannot be scored.")


def build_score_matrix(domain: str, rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = {}
    for col, item in cfg["criteria"].items():
        if item["choice"]["treatment"] == "exclude" or col not in rows.columns:
            continue
        cols[col] = score_criterion(rows[col], item["spec"], item["choice"])
    return pd.DataFrame(cols, index=rows.index)


# ---------- AHP ----------
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(matrix: np.ndarray):
    n = matrix.shape[0]
    vals, vecs = np.linalg.eig(matrix)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = 0.0 if n <= 2 else max(0.0, (lam - n) / (n - 1))
    cr = 0.0 if n <= 2 else ci / RI.get(n, 1.49)
    return w, cr


# ---------- TOPSIS / weighted sum ----------
def topsis(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = np.sqrt((scores ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    v = scores / norm * weights
    best, worst = v.max(axis=0), v.min(axis=0)
    d_b = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_w = np.sqrt(((v - worst) ** 2).sum(axis=1))
    return d_w / (d_b + d_w + 1e-12)


def run_mcdm(domain: str, rows: pd.DataFrame, cfg: dict, weights: dict | None = None, method: str = "AHP + TOPSIS", mix: float = 0.5):
    """Score every row. Returns dict with score matrix, contributions and the result frame."""
    S = build_score_matrix(domain, rows, cfg)
    if S.shape[1] < 2:
        raise ValueError("Select at least two criteria for ranking.")
    w = weights or cfg.get("weights") or {c: 1 / S.shape[1] for c in S.columns}
    w = np.array([w.get(c, 0.0) for c in S.columns], dtype=float)
    w = w / w.sum()
    ws = S.to_numpy() @ w
    tp = topsis(S.to_numpy(), w)
    final = {"TOPSIS only": tp, "AHP + TOPSIS": tp, "Weighted sum (AHP)": ws}.get(method, mix * ws + (1 - mix) * tp)
    res = pd.DataFrame({"AHP_score": ws, "TOPSIS_score": tp, "Final_score": final}, index=S.index)
    contrib = pd.DataFrame(S.to_numpy() * w, columns=S.columns, index=S.index)
    return {"scores": S, "weights": dict(zip(S.columns, w)), "result": res, "contrib": contrib}


def percentile_rank(a) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=float)).rank(pct=True, method="average").to_numpy()
