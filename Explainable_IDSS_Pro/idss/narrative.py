"""Decision table and plain-language rationale generation (rule-based, deterministic)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .criteria import (label_of, pretty, recode_numeric, run_mcdm, percentile_rank, TREATMENTS)
from .core import risk_scores, DOMAIN_META


def fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not recorded"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 100 else (f"{v:.2f}".rstrip("0").rstrip("."))
    return str(v)


def pct_words(p: float) -> str:
    p = float(p)
    if p >= 0.9:  return f"in the highest {max(1, round((1 - p) * 100))}% of records"
    if p >= 0.6:  return f"above {p * 100:.0f}% of records"
    if p >= 0.4:  return "close to the typical record"
    if p >= 0.1:  return f"below {(1 - p) * 100:.0f}% of records"
    return f"in the lowest {max(1, round(p * 100))}% of records"


# ---------- Combined decision table ----------
def decision_table(domain, model, adverse, X_model, mcdm_rows, row_keys, cfg, method, mix,
                   ml_weight=0.5, top_share=0.20, mid_share=0.50):
    meta = DOMAIN_META[domain]
    mc = run_mcdm(domain, mcdm_rows, cfg, method=method, mix=mix)
    risk = risk_scores(model, X_model, adverse)
    pred = model.predict(X_model).astype(str)
    out = pd.DataFrame({"row_key": list(row_keys), "model_prediction": pred, "model_risk": risk}, index=mc["result"].index)
    out = pd.concat([out, mc["result"]], axis=1)
    out["risk_percentile"] = percentile_rank(out["model_risk"])
    out["priority_percentile"] = percentile_rank(out["Final_score"])
    out["combined_index"] = ml_weight * out["risk_percentile"] + (1 - ml_weight) * out["priority_percentile"]
    out["combined_percentile"] = percentile_rank(out["combined_index"])
    t1, t2 = 1 - top_share, 1 - mid_share
    tiers = meta["tiers"]
    out["tier_rank"] = np.where(out["combined_percentile"] >= t1, 1, np.where(out["combined_percentile"] >= t2, 2, 3))
    out["recommendation"] = out["tier_rank"].map({1: tiers[0], 2: tiers[1], 3: tiers[2]})
    hi_ml, hi_mc = out["risk_percentile"] >= t1, out["priority_percentile"] >= t1
    out["views_agree"] = np.where(hi_ml == hi_mc, "Aligned", np.where(hi_ml, "Model higher than ranking", "Ranking higher than model"))
    out = out.sort_values("combined_index", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    top_c = mc["contrib"].loc[out.index]
    out["top_criterion"] = [label_of(domain, c) for c in top_c.idxmax(axis=1)]
    return out, mc


# ---------- Rationale for one record ----------
def _criterion_sentence(domain, col, item, raw_val, raw_series_num, score, weight, contrib, delta, share):
    spec, ch = item["spec"], item["choice"]
    lab = label_of(domain, col)
    t = ch["treatment"]
    shown = fmt_value(raw_val)
    parts = []
    if t in ("higher", "lower"):
        num = recode_numeric(pd.Series([raw_val]), spec.get("recode")).iloc[0]
        pb = float((raw_series_num < num).mean()) if pd.notna(num) else 0.5
        pe = float((raw_series_num == num).mean()) if pd.notna(num) else 0.0
        med = float(raw_series_num.median())
        if t == "higher":
            pos = pb + pe / 2
            parts.append(f"{lab} is {shown}, {pct_words(pos)} (typical value {fmt_value(med)}). Higher values raise priority.")
        else:
            pos = 1 - (pb + pe / 2)
            rel = (f"more severe than {pos * 100:.0f}% of records" if pos >= 0.5 else f"less severe than {(1 - pos) * 100:.0f}% of records")
            typical_label = "code" if spec.get("recode") == "lead_int" else "value"
            parts.append(f"{lab} is {shown}, {rel} (typical {typical_label} {fmt_value(med)}). Lower values mean higher priority.")
    elif t == "band":
        lo, hi = ch["band"]
        num = recode_numeric(pd.Series([raw_val]), spec.get("recode")).iloc[0]
        where = "inside" if pd.notna(num) and lo <= num <= hi else "outside"
        parts.append(f"{lab} is {shown}, {where} the review range {fmt_value(lo)} to {fmt_value(hi)}; scored {score:.2f} out of 1.")
    else:
        parts.append(f"{lab} is {shown}, which the category scoring rule rates {score:.2f} out of 1.")
    parts.append(f"It adds {contrib:.3f} to the weighted score ({share:.0%} of the total), compared with {contrib - delta:.3f} for the median record.")
    return " ".join(parts)


def rationale(domain, rec: pd.Series, table: pd.DataFrame, mc: dict, cfg: dict, mcdm_rows: pd.DataFrame,
              drivers: pd.DataFrame | None, model_risk: float, adverse_label: str, top_share: float):
    """Build structured plain-language reasoning for one record. Returns dict of text blocks + tables."""
    meta = DOMAIN_META[domain]
    n = len(table)
    idx = rec.name
    contrib = mc["contrib"].loc[idx]
    total = float(contrib.sum()) or 1.0
    med = mc["contrib"].median()
    items = []
    for col in mc["scores"].columns:
        item = cfg["criteria"][col]
        raw_series = recode_numeric(mcdm_rows[col], item["spec"].get("recode")) if item["choice"]["treatment"] != "map" else None
        if raw_series is None:
            raw_series = pd.Series(dtype=float)
        items.append(dict(col=col, label=label_of(domain, col), raw=mcdm_rows.loc[idx, col], score=float(mc["scores"].loc[idx, col]),
                          weight=float(mc["weights"][col]), contrib=float(contrib[col]), delta=float(contrib[col] - med[col]),
                          share=float(contrib[col] / total), why=item["spec"].get("why", ""),
                          sentence=_criterion_sentence(domain, col, item, mcdm_rows.loc[idx, col], raw_series, float(mc["scores"].loc[idx, col]),
                                                       float(mc["weights"][col]), float(contrib[col]), float(contrib[col] - med[col]), float(contrib[col] / total))))
    items.sort(key=lambda d: d["delta"], reverse=True)
    raised = [d for d in items if d["delta"] > 0.005]
    lowered = [d for d in items if d["delta"] < -0.005]

    tier = rec["recommendation"]
    pos = int(rec["rank"])
    pct_top = pos / n
    top_txt = "the top 0.1%" if pct_top < 0.001 else f"the top {pct_top:.1%}".replace(".0%", "%")
    headline = (f"{tier}. This {meta['entity']} ranks {pos:,} of {n:,} ({top_txt}) "
                f"with a combined index of {rec['combined_index']:.2f}.")

    # model view
    ml_lines = [f"The prediction model estimates a {model_risk:.0%} probability of {adverse_label}, "
                f"{pct_words(rec['risk_percentile'])} in this group. The model's predicted outcome is '{rec['model_prediction']}'."]
    if drivers is not None and len(drivers):
        up = drivers[drivers["effect"] > 0.002].head(3)
        down = drivers[drivers["effect"] < -0.002].head(2)
        for _, d in up.iterrows():
            ml_lines.append(f"{label_of(domain, d['field'])} = {fmt_value(d['value'])} increases the estimated probability by {d['effect'] * 100:.1f} percentage points compared with a typical record (typical: {fmt_value(d['typical'])}).")
        for _, d in down.iterrows():
            ml_lines.append(f"{label_of(domain, d['field'])} = {fmt_value(d['value'])} lowers the estimated probability by {abs(d['effect']) * 100:.1f} percentage points compared with a typical record (typical: {fmt_value(d['typical'])}).")

    # ranking view
    mc_lines = [f"On the weighted criteria, this {meta['entity']} scores {rec['Final_score']:.2f} and sits {pct_words(rec['priority_percentile'])}."]
    for d in raised[:3]:
        mc_lines.append(d["sentence"])
    low_txt = [d["sentence"] for d in lowered[:2]]

    # agreement
    if rec["views_agree"] == "Aligned":
        both_high = rec["risk_percentile"] >= 1 - top_share
        agree = ("The prediction model and the criteria ranking both place this record in the highest-attention group, so the case for this recommendation is strong."
                 if both_high else
                 "The prediction model and the criteria ranking both place this record outside the highest-attention group, so the recommendation is consistent across both views.")
    elif rec["views_agree"] == "Model higher than ranking":
        agree = ("The prediction model rates this record higher than the criteria ranking does. The model is reacting to fields that are not in the ranking criteria, so a reviewer should check those fields before acting.")
    else:
        agree = ("The criteria ranking rates this record higher than the prediction model does. The record scores high on the chosen criteria but its wider profile looks typical to the model, so confirm the criteria apply to this case.")

    next_step = {
        1: f"Next step: route this {meta['entity']} to the \u201c{tier}\u201d track and record the reviewer's decision in Human Review.",
        2: f"Next step: handle through the \u201c{tier}\u201d track.",
        3: f"Next step: no extra attention is needed beyond the \u201c{tier}\u201d track.",
    }[int(rec["tier_rank"])]
    return dict(headline=headline, model=ml_lines, ranking=mc_lines, lowered=low_txt, agreement=agree, next_step=next_step, items=items)
