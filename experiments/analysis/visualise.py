"""
Visualisation layer for DeepDive experiment results.

Produces publication-quality charts and summary CSVs in experiments/data/results/.

Usage:
  python -m experiments.analysis.visualise            # all charts + CSVs
  python -m experiments.analysis.visualise --show     # also display interactively
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiments.analysis.aggregate import (
    load_experiment_a,
    load_experiment_b,
    load_experiment_c,
    summarise,
)

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")

# ── Colour palettes ───────────────────────────────────────────────────────────

# Experiment A: one colour per model — ordered by scale
_MODEL_ORDER = [
    "llama-8b",
    "llama4-scout",
    "gpt-oss-20b",
    "qwen-32b",
    "llama-70b",
    "gpt-oss-120b",
    "qwen-235b",
]
_MODEL_LABELS = {
    "llama-8b":     "LLaMA-8B",
    "llama4-scout": "LLaMA4-Scout",
    "gpt-oss-20b":  "GPT-OSS-20B",
    "qwen-32b":     "Qwen-32B",
    "llama-70b":    "LLaMA-70B",
    "gpt-oss-120b": "GPT-OSS-120B",
    "qwen-235b":    "Qwen-235B",
}
_MODEL_COLORS = [
    "#4C72B0", "#4C72B0",   # LLaMA family — blue tones
    "#C44E52",              # GPT-OSS-20B  — red
    "#55A868",              # Qwen-32B     — green
    "#4C72B0",              # LLaMA-70B    — blue
    "#C44E52",              # GPT-OSS-120B — red
    "#55A868",              # Qwen-235B    — green
]
# Shade LLaMA family progressively
_MODEL_COLORS[0] = "#9DB8D9"   # llama-8b   (lightest)
_MODEL_COLORS[1] = "#6B99C8"   # llama4-scout
_MODEL_COLORS[4] = "#2F5EA3"   # llama-70b  (darkest)

# Experiment B persona colours
_PERSONA_ORDER  = ["baseline", "lyra-persona", "cipher-persona"]
_PERSONA_LABELS = {
    "baseline":       "Baseline",
    "lyra-persona":   "Lyra Persona",
    "cipher-persona": "Cipher Persona",
}
_PERSONA_COLORS = ["#888888", "#4C72B0", "#C44E52"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_results_dir() -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str, show: bool) -> None:
    path = os.path.join(_RESULTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def _error_bars(summary: pd.DataFrame, col_mean: str, col_std: str) -> np.ndarray:
    """Return std array (NaN → 0) for errorbar plots."""
    stds = summary[col_std].fillna(0).to_numpy()
    return stds


def _reorder(df: pd.DataFrame, order: list[str], group_col: str = "group") -> pd.DataFrame:
    """Reorder summary DataFrame rows to match a given slug order."""
    existing = [g for g in order if g in df[group_col].values]
    return df.set_index(group_col).loc[existing].reset_index()


# ── Experiment A charts ───────────────────────────────────────────────────────

def chart_a_persona_discrimination(df_a: pd.DataFrame, show: bool = False) -> None:
    """
    Grouped bar chart: Lyra gap and Cipher gap by model.
    The discrimination gap is the key metric — positive = agent's own persona score
    exceeds its cross-persona score.
    """
    summary = summarise(df_a)
    s = _reorder(summary, _MODEL_ORDER)
    labels = [_MODEL_LABELS[g] for g in s["group"]]

    x  = np.arange(len(labels))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_lyra = ax.bar(
        x - w / 2,
        s["lyra_gap__mean"],
        width=w,
        yerr=_error_bars(s, "lyra_gap__mean", "lyra_gap__std"),
        capsize=3,
        color="#4C72B0",
        alpha=0.85,
        label="Lyra gap",
        error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
    )
    bars_cipher = ax.bar(
        x + w / 2,
        s["cipher_gap__mean"],
        width=w,
        yerr=_error_bars(s, "cipher_gap__mean", "cipher_gap__std"),
        capsize=3,
        color="#C44E52",
        alpha=0.85,
        label="Cipher gap",
        error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Discrimination gap (own − cross)")
    ax.set_title("Experiment A — Persona Discrimination Gap by Model")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Annotate the native-model bar
    lyra_70b_idx = list(s["group"]).index("llama-70b")
    ax.annotate(
        "native\nmodel",
        xy=(lyra_70b_idx - w / 2, s.iloc[lyra_70b_idx]["lyra_gap__mean"]),
        xytext=(lyra_70b_idx - w / 2 - 0.5, s.iloc[lyra_70b_idx]["lyra_gap__mean"] + 0.04),
        fontsize=8,
        color="#4C72B0",
        arrowprops={"arrowstyle": "->", "color": "#4C72B0", "lw": 0.8},
    )

    fig.tight_layout()
    _save(fig, "exp_a_persona_discrimination", show)


def chart_a_conversation_quality(df_a: pd.DataFrame, show: bool = False) -> None:
    """
    Side-by-side subplots: topic drift mean and coherence mean by model.
    Higher is better for both metrics.
    """
    summary = summarise(df_a)
    s = _reorder(summary, _MODEL_ORDER)
    labels = [_MODEL_LABELS[g] for g in s["group"]]
    colors = [_MODEL_COLORS[_MODEL_ORDER.index(g)] for g in s["group"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, metric, title, col_m, col_s in [
        (axes[0], "topic_drift_mean", "Topic Adherence (↑ better)",
         "topic_drift_mean__mean", "topic_drift_mean__std"),
        (axes[1], "coherence_mean",   "Turn Coherence (↑ better)",
         "coherence_mean__mean",   "coherence_mean__std"),
    ]:
        ax.bar(
            x,
            s[col_m],
            color=colors,
            alpha=0.85,
            yerr=_error_bars(s, col_m, col_s),
            capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"Experiment A — {title}")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(0, max(s[col_m].max() * 1.25, 0.75))

    fig.tight_layout()
    _save(fig, "exp_a_conversation_quality", show)


def chart_a_overview(df_a: pd.DataFrame, show: bool = False) -> None:
    """
    2 × 2 overview grid: all four primary metrics side by side.
    """
    summary = summarise(df_a)
    s = _reorder(summary, _MODEL_ORDER)
    labels = [_MODEL_LABELS[g] for g in s["group"]]
    colors = [_MODEL_COLORS[_MODEL_ORDER.index(g)] for g in s["group"]]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Top left: Persona discrimination (grouped) ---
    ax = axes[0, 0]
    ax.bar(x - w / 2, s["lyra_gap__mean"],   width=w, color="#4C72B0", alpha=0.85, label="Lyra gap")
    ax.bar(x + w / 2, s["cipher_gap__mean"], width=w, color="#C44E52", alpha=0.85, label="Cipher gap")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Persona Discrimination Gap")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # --- Top right: Topic adherence ---
    ax = axes[0, 1]
    ax.bar(x, s["topic_drift_mean__mean"], color=colors, alpha=0.85,
           yerr=_error_bars(s, "topic_drift_mean__mean", "topic_drift_mean__std"),
           capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "0.4"})
    ax.set_title("Topic Adherence (mean similarity to topic string)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # --- Bottom left: Coherence ---
    ax = axes[1, 0]
    ax.bar(x, s["coherence_mean__mean"], color=colors, alpha=0.85,
           yerr=_error_bars(s, "coherence_mean__mean", "coherence_mean__std"),
           capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "0.4"})
    ax.set_title("Turn Coherence (mean consecutive cosine similarity)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # --- Bottom right: Semantic diversity (Lyra) ---
    ax = axes[1, 1]
    ax.bar(x - w / 2, s["lyra_sem_div__mean"],   width=w, color="#4C72B0", alpha=0.85, label="Lyra")
    ax.bar(x + w / 2, s["cipher_sem_div__mean"], width=w, color="#C44E52", alpha=0.85, label="Cipher")
    ax.set_title("Semantic Diversity per Agent")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Experiment A — Model Isolation: Key Metrics Overview", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "exp_a_overview", show)


def chart_a_scale_scatter(df_a: pd.DataFrame, show: bool = False) -> None:
    """
    Scatter plot: approximate parameter count vs Lyra discrimination gap.
    Tests the 'bigger model = stronger persona' hypothesis.
    """
    _PARAM_SCALE = {
        "llama-8b":     8,
        "llama4-scout": 17,   # active params (17B × 16 experts, but ~17B active)
        "gpt-oss-20b":  20,
        "qwen-32b":     32,
        "llama-70b":    70,
        "gpt-oss-120b": 120,
        "qwen-235b":    235,
    }

    summary = summarise(df_a)
    s = _reorder(summary, _MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in s.iterrows():
        g   = row["group"]
        x_v = _PARAM_SCALE.get(g, 0)
        y_v = row["lyra_gap__mean"]
        err = row["lyra_gap__std"] if pd.notna(row["lyra_gap__std"]) else 0
        ax.errorbar(x_v, y_v, yerr=err, fmt="o", capsize=4,
                    color=_MODEL_COLORS[_MODEL_ORDER.index(g)],
                    markersize=8, linewidth=1.2)
        ax.annotate(
            _MODEL_LABELS[g],
            xy=(x_v, y_v),
            xytext=(x_v + 3, y_v + 0.005),
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Approximate active parameters (B)")
    ax.set_ylabel("Lyra discrimination gap")
    ax.set_title("Experiment A — Model Scale vs Persona Consistency")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "exp_a_scale_vs_persona", show)


# ── Experiment B charts ───────────────────────────────────────────────────────

def chart_b_persona_effect(df_b: pd.DataFrame, show: bool = False) -> None:
    """
    Grouped bar chart: Lyra gap and Cipher gap by persona condition.
    Core result for Experiment B.
    """
    summary = summarise(df_b)
    s = _reorder(summary, _PERSONA_ORDER)
    labels = [_PERSONA_LABELS[g] for g in s["group"]]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        x - w / 2,
        s["lyra_gap__mean"],
        width=w,
        yerr=_error_bars(s, "lyra_gap__mean", "lyra_gap__std"),
        capsize=4,
        color="#4C72B0",
        alpha=0.85,
        label="Lyra gap",
        error_kw={"elinewidth": 0.9, "ecolor": "0.4"},
    )
    ax.bar(
        x + w / 2,
        s["cipher_gap__mean"],
        width=w,
        yerr=_error_bars(s, "cipher_gap__mean", "cipher_gap__std"),
        capsize=4,
        color="#C44E52",
        alpha=0.85,
        label="Cipher gap",
        error_kw={"elinewidth": 0.9, "ecolor": "0.4"},
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Discrimination gap (own − cross)")
    ax.set_title("Experiment B — Persona Effect on Agent Voice Discrimination")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "exp_b_persona_discrimination", show)


def chart_b_quality_comparison(df_b: pd.DataFrame, show: bool = False) -> None:
    """
    Four-panel comparison of persona conditions on all quality metrics.
    """
    summary = summarise(df_b)
    s = _reorder(summary, _PERSONA_ORDER)
    labels = [_PERSONA_LABELS[g] for g in s["group"]]
    colors = _PERSONA_COLORS[: len(labels)]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    metrics = [
        ("topic_drift_mean__mean", "topic_drift_mean__std",   "Topic Adherence (↑ better)"),
        ("coherence_mean__mean",   "coherence_mean__std",     "Turn Coherence (↑ better)"),
        ("sentiment_vol__mean",    "sentiment_vol__std",      "Sentiment Volatility (↓ calmer)"),
    ]

    for ax, (col_m, col_s, title) in zip(axes, metrics):
        ax.bar(
            x,
            s[col_m],
            color=colors,
            alpha=0.85,
            yerr=_error_bars(s, col_m, col_s),
            capsize=4,
            error_kw={"elinewidth": 0.9, "ecolor": "0.4"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Experiment B — Persona Condition vs Conversation Quality", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "exp_b_quality_comparison", show)


def chart_b_per_topic(df_b: pd.DataFrame, show: bool = False) -> None:
    """
    Lyra gap per persona condition, broken down by topic index.
    Shows whether persona effects are consistent across topics or topic-specific.
    """
    from experiments.topics import TOPICS
    topic_labels = [f"T{i}: {t[:30]}..." for i, t in enumerate(TOPICS)]

    # Average over the 3 runs per (persona × topic) cell
    cell_means = df_b.groupby(["group", "topic_index"])["lyra_gap"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))

    n_topics = df_b["topic_index"].nunique()
    x = np.arange(n_topics)
    w = 0.25

    for i, (persona, color) in enumerate(zip(_PERSONA_ORDER, _PERSONA_COLORS)):
        sub = cell_means[cell_means["group"] == persona].sort_values("topic_index")
        if sub.empty:
            continue
        ax.bar(
            x + (i - 1) * w,
            sub["lyra_gap"].values,
            width=w,
            color=color,
            alpha=0.85,
            label=_PERSONA_LABELS[persona],
        )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_topics)], fontsize=9)
    ax.set_xlabel("Topic index")
    ax.set_ylabel("Lyra discrimination gap")
    ax.set_title("Experiment B — Lyra Gap by Persona Condition × Topic")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Topic labels as secondary x annotation
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:20] for t in TOPICS], rotation=30, ha="left", fontsize=7.5)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    _save(fig, "exp_b_lyra_gap_by_topic", show)


# ── Experiment C charts ───────────────────────────────────────────────────────

def chart_c_persona_effect(df_c: pd.DataFrame, show: bool = False) -> None:
    """
    Grouped bar chart: Lyra gap and Cipher gap by persona condition — Experiment C.
    Mirrors chart_b_persona_effect for direct visual comparison.
    """
    summary = summarise(df_c)
    s = _reorder(summary, _PERSONA_ORDER)
    labels = [_PERSONA_LABELS[g] for g in s["group"]]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        x - w / 2,
        s["lyra_gap__mean"],
        width=w,
        yerr=_error_bars(s, "lyra_gap__mean", "lyra_gap__std"),
        capsize=4,
        color="#4C72B0",
        alpha=0.85,
        label="Lyra gap",
        error_kw={"elinewidth": 0.9, "ecolor": "0.4"},
    )
    ax.bar(
        x + w / 2,
        s["cipher_gap__mean"],
        width=w,
        yerr=_error_bars(s, "cipher_gap__mean", "cipher_gap__std"),
        capsize=4,
        color="#C44E52",
        alpha=0.85,
        label="Cipher gap",
        error_kw={"elinewidth": 0.9, "ecolor": "0.4"},
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Discrimination gap (own − cross)")
    ax.set_title(
        "Experiment C — Cross-Model Persona Discrimination\n"
        "(Agent A: llama-3.3-70b | Agent B: llama4-scout)"
    )
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "exp_c_persona_discrimination", show)


def chart_bc_comparison(df_b: pd.DataFrame, df_c: pd.DataFrame, show: bool = False) -> None:
    """
    Side-by-side comparison of Experiment B (intra-model) vs C (cross-model).
    Two panels: Lyra gap and Cipher gap.
    Each panel has grouped bars: B-baseline / B-lyra / B-cipher vs C-baseline / C-lyra / C-cipher.
    """
    sum_b = summarise(df_b)
    sum_c = summarise(df_c)
    sb = _reorder(sum_b, _PERSONA_ORDER)
    sc = _reorder(sum_c, _PERSONA_ORDER)

    labels = [_PERSONA_LABELS[g] for g in sb["group"]]
    x  = np.arange(len(labels))
    w  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_m, metric_s, title in [
        (axes[0], "lyra_gap__mean",   "lyra_gap__std",   "Lyra Discrimination Gap"),
        (axes[1], "cipher_gap__mean", "cipher_gap__std", "Cipher Discrimination Gap"),
    ]:
        # Experiment B bars
        ax.bar(
            x - w / 2,
            sb[metric_m],
            width=w,
            yerr=_error_bars(sb, metric_m, metric_s),
            capsize=3,
            color="#8CA9D0",
            alpha=0.85,
            label="Exp B (intra-model: scout+scout)",
            error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
        )
        # Experiment C bars
        ax.bar(
            x + w / 2,
            sc[metric_m],
            width=w,
            yerr=_error_bars(sc, metric_m, metric_s),
            capsize=3,
            color="#2F5EA3",
            alpha=0.85,
            label="Exp C (cross-model: llama70b+scout)",
            error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
        )
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Experiments B vs C — Intra-Model vs Cross-Model Persona Design",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "exp_bc_discrimination_comparison", show)


def chart_bc_quality_comparison(df_b: pd.DataFrame, df_c: pd.DataFrame, show: bool = False) -> None:
    """
    Side-by-side comparison of B vs C on topic drift and coherence.
    Tests whether cross-model design also affects conversation quality metrics.
    """
    sum_b = summarise(df_b)
    sum_c = summarise(df_c)
    sb = _reorder(sum_b, _PERSONA_ORDER)
    sc = _reorder(sum_c, _PERSONA_ORDER)

    labels = [_PERSONA_LABELS[g] for g in sb["group"]]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_m, metric_s, title in [
        (axes[0], "topic_drift_mean__mean", "topic_drift_mean__std", "Topic Adherence (↑ better)"),
        (axes[1], "coherence_mean__mean",   "coherence_mean__std",   "Turn Coherence (↑ better)"),
    ]:
        ax.bar(
            x - w / 2, sb[metric_m], width=w,
            yerr=_error_bars(sb, metric_m, metric_s), capsize=3,
            color="#E8A090", alpha=0.85, label="Exp B (intra-model)",
            error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
        )
        ax.bar(
            x + w / 2, sc[metric_m], width=w,
            yerr=_error_bars(sc, metric_m, metric_s), capsize=3,
            color="#C44E52", alpha=0.85, label="Exp C (cross-model)",
            error_kw={"elinewidth": 0.8, "ecolor": "0.4"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Experiments B vs C — Conversation Quality by Persona Condition",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "exp_bc_quality_comparison", show)


# ── CSV exports ───────────────────────────────────────────────────────────────

def export_csvs(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame) -> None:
    """Write per-condition and per-group summary CSVs to results/."""
    _ensure_results_dir()

    # Per-condition (raw)
    df_a.to_csv(os.path.join(_RESULTS_DIR, "exp_a_conditions.csv"),    index=False)
    df_b.to_csv(os.path.join(_RESULTS_DIR, "exp_b_conditions.csv"),    index=False)
    if len(df_c) > 0:
        df_c.to_csv(os.path.join(_RESULTS_DIR, "exp_c_conditions.csv"), index=False)

    # Per-group summary
    summarise(df_a).to_csv(os.path.join(_RESULTS_DIR, "exp_a_model_summary.csv"),    index=False)
    summarise(df_b).to_csv(os.path.join(_RESULTS_DIR, "exp_b_persona_summary.csv"),  index=False)
    if len(df_c) > 0:
        summarise(df_c).to_csv(os.path.join(_RESULTS_DIR, "exp_c_persona_summary.csv"), index=False)

    print(f"  CSVs written to {_RESULTS_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDive — Results Visualiser")
    parser.add_argument("--show", action="store_true", help="Display charts interactively after saving")
    args = parser.parse_args()

    _ensure_results_dir()

    print("\nLoading metrics...")
    df_a = load_experiment_a()
    df_b = load_experiment_b()
    df_c = load_experiment_c()
    print(f"  Experiment A: {len(df_a)} conditions / {df_a['group'].nunique()} models")
    print(f"  Experiment B: {len(df_b)} conditions / {df_b['group'].nunique()} persona conditions")
    print(f"  Experiment C: {len(df_c)} conditions" + (
        f" / {df_c['group'].nunique()} persona conditions" if len(df_c) > 0 else " (pending)"
    ))

    print("\nExporting CSVs...")
    export_csvs(df_a, df_b, df_c)

    print("\nGenerating Experiment A charts...")
    chart_a_persona_discrimination(df_a, args.show)
    chart_a_conversation_quality(df_a, args.show)
    chart_a_overview(df_a, args.show)
    chart_a_scale_scatter(df_a, args.show)

    print("\nGenerating Experiment B charts...")
    chart_b_persona_effect(df_b, args.show)
    chart_b_quality_comparison(df_b, args.show)
    chart_b_per_topic(df_b, args.show)

    if len(df_c) > 0:
        print("\nGenerating Experiment C charts...")
        chart_c_persona_effect(df_c, args.show)
        chart_bc_comparison(df_b, df_c, args.show)
        chart_bc_quality_comparison(df_b, df_c, args.show)
    else:
        print("\nExperiment C: no data yet — skipping C charts.")

    print(f"\nDone. All outputs in {_RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
