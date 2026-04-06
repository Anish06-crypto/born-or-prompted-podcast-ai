"""
Metrics aggregation layer.

Reads all per-condition JSON files from experiments/data/metrics/,
flattens them into per-row records, and produces:

  - A flat list of dicts (one per condition) — useful for ad-hoc analysis
  - Per-group summary DataFrames (mean ± std across topics × runs)

Two public functions:
    load_experiment_a()  →  DataFrame, one row per condition (Experiment A)
    load_experiment_b()  →  DataFrame, one row per condition (Experiment B)

Each row contains every scalar metric extracted from the JSON:
    condition_id, group (model slug or persona slug), topic_index, run_index,
    lyra_gap, cipher_gap, lyra_own, cipher_own,
    coherence_mean, topic_drift_mean, drift_slope,
    sentiment_mean, sentiment_slope, sentiment_volatility,
    lyra_semantic_div, cipher_semantic_div, global_ttr
"""

import json
import os
import re
from typing import Optional

import pandas as pd

_METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "metrics")


def _parse_condition_id(cid: str) -> tuple[str, str, int, int]:
    """
    Returns (experiment_prefix, group_slug, topic_index, run_index).
    e.g. "modiso__llama-70b__t03__r02"    →  ("modiso",   "llama-70b",    3, 2)
    e.g. "periso__lyra-persona__t03__r02" →  ("periso",   "lyra-persona", 3, 2)
    e.g. "crossiso__baseline__t03__r02"   →  ("crossiso", "baseline",     3, 2)
    """
    m = re.match(r"^(modiso|periso|crossiso)__(.+)__t(\d{2})__r(\d{2})$", cid)
    if not m:
        raise ValueError(f"Cannot parse condition_id: {cid!r}")
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def _extract_row(path: str) -> Optional[dict]:
    """Load one metrics JSON and return a flat dict of scalars."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    cid = d.get("condition_id", "")
    try:
        prefix, group, t_idx, r_idx = _parse_condition_id(cid)
    except ValueError:
        return None

    disc = d.get("persona", {}).get("discrimination", {})
    lyra_disc   = disc.get("Lyra",   {})
    cipher_disc = disc.get("Cipher", {})

    coh  = d.get("coherence",    {})
    drft = d.get("topic_drift",  {})
    sent = d.get("sentiment",    {})
    div  = d.get("diversity",    {})

    div_by_spk = div.get("by_speaker", {})
    lyra_div   = div_by_spk.get("Lyra",   {})
    cipher_div = div_by_spk.get("Cipher", {})

    return {
        "condition_id":       cid,
        "experiment":         prefix,
        "group":              group,
        "topic_index":        t_idx,
        "run_index":          r_idx,
        # Persona discrimination
        "lyra_gap":           lyra_disc.get("gap",       None),
        "cipher_gap":         cipher_disc.get("gap",     None),
        "lyra_own":           lyra_disc.get("own_mean",  None),
        "cipher_own":         cipher_disc.get("own_mean", None),
        # Coherence
        "coherence_mean":     coh.get("mean",            None),
        # Topic drift
        "topic_drift_mean":   drft.get("mean",           None),
        "drift_slope":        drft.get("drift_slope",    None),
        # Sentiment
        "sentiment_mean":     sent.get("mean",           None),
        "sentiment_slope":    sent.get("slope",          None),
        "sentiment_vol":      sent.get("volatility",     None),
        # Diversity
        "lyra_sem_div":       lyra_div.get("semantic_diversity",   None),
        "cipher_sem_div":     cipher_div.get("semantic_diversity", None),
        "lyra_ttr":           lyra_div.get("ttr",                  None),
        "cipher_ttr":         cipher_div.get("ttr",                None),
        "global_ttr":         div.get("global_ttr",                None),
    }


def _load_all() -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(_METRICS_DIR)):
        if not fname.endswith(".json"):
            continue
        row = _extract_row(os.path.join(_METRICS_DIR, fname))
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def load_experiment_a(exclude_gemini: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame of Experiment A conditions (modiso__ prefix).

    Parameters
    ----------
    exclude_gemini : bool
        Drop the gemini-flash-lite condition (only 1 transcript — incomplete run).
        Default True.
    """
    df = _load_all()
    df = df[df["experiment"] == "modiso"].copy()
    if exclude_gemini:
        df = df[df["group"] != "gemini-flash-lite"]
    return df.reset_index(drop=True)


def load_experiment_b() -> pd.DataFrame:
    """Returns a DataFrame of Experiment B conditions (periso__ prefix)."""
    df = _load_all()
    return df[df["experiment"] == "periso"].copy().reset_index(drop=True)


def load_experiment_c() -> pd.DataFrame:
    """Returns a DataFrame of Experiment C conditions (crossiso__ prefix)."""
    df = _load_all()
    return df[df["experiment"] == "crossiso"].copy().reset_index(drop=True)


def summarise(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    """
    Compute mean and std for every numeric metric, grouped by group_col.

    Returns a DataFrame with columns: group, metric_mean, metric_std, ...
    """
    numeric_cols = [c for c in df.columns if c not in (
        "condition_id", "experiment", "group", "topic_index", "run_index"
    )]
    agg: dict = {col: ["mean", "std"] for col in numeric_cols}
    summary = df.groupby(group_col).agg(agg)
    summary.columns = ["__".join(c).strip() for c in summary.columns]
    return summary.reset_index()


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    df_a = load_experiment_a()
    df_b = load_experiment_b()
    df_c = load_experiment_c()

    cols = ["lyra_gap", "cipher_gap", "topic_drift_mean", "coherence_mean"]

    print(f"\nExperiment A: {len(df_a)} conditions across {df_a['group'].nunique()} models")
    print(df_a.groupby("group")[cols].mean().round(3).to_string())

    print(f"\nExperiment B: {len(df_b)} conditions across {df_b['group'].nunique()} personas")
    print(df_b.groupby("group")[cols].mean().round(3).to_string())

    print(f"\nExperiment C: {len(df_c)} conditions across {df_c['group'].nunique()} personas")
    if len(df_c) > 0:
        print(df_c.groupby("group")[cols].mean().round(3).to_string())
    else:
        print("  (no data yet — run Experiment C first)")
