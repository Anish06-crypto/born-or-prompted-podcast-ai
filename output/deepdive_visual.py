import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = "#0D0D0D"
CARD       = "#141414"
LYRA_COL   = "#C084FC"
CIPHER_COL = "#22D3EE"
STAT_COL   = "#F59E0B"     # amber — system-level stats (neither agent)
TEXT_PRI   = "#F0F0F0"
TEXT_SEC   = "#555555"
DIVIDER    = "#252525"

fig = plt.figure(figsize=(10.8, 10.8), facecolor=BG)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.955, "DeepDive — Episode Behaviour Metrics",
         ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT_PRI)

# model labels — coloured dots inline
fig.text(0.5, 0.893,
         "          Lyra     ·     Meta Llama 3.3 70B                    Cipher     ·     Alibaba Qwen 3 32B",
         ha="center", va="top", fontsize=9.5, color=TEXT_SEC)
fig.text(0.330, 0.893, "●", ha="center", va="top", fontsize=9.5, color=LYRA_COL)
fig.text(0.610, 0.893, "●", ha="center", va="top", fontsize=9.5, color=CIPHER_COL)

fig.add_artist(plt.Line2D([0.04, 0.96], [0.853, 0.853],
               transform=fig.transFigure, color=DIVIDER, lw=0.8))

# ── Card grid — 2 rows ────────────────────────────────────────────────────────
lm = 0.04; rm = 0.04; gap_x = 0.02; gap_y = 0.05
cb_bot = 0.115
available_h = 0.823 - cb_bot
ch = (available_h - gap_y) / 2
cb_top = cb_bot + ch + gap_y

cw = (1 - lm - rm - 2 * gap_x) / 3

def make_ax(i):
    if i < 3:
        # Top row (3 cards)
        return fig.add_axes([lm + i * (cw + gap_x), cb_top, cw, ch], facecolor=CARD)
    else:
        # Bottom row (2 cards), centered
        start_x = (1 - (2 * cw + gap_x)) / 2
        j = i - 3
        return fig.add_axes([start_x + j * (cw + gap_x), cb_bot, cw, ch], facecolor=CARD)

axes = [make_ax(i) for i in range(5)]

# ── Comparison bar card (cards 1–3) ───────────────────────────────────────────
def draw_bar_card(ax, title, subtitle, rows, note=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=TEXT_PRI)
    ax.text(0.5, 0.845, subtitle, ha="center", va="top",
            fontsize=7.5, color=TEXT_SEC, linespacing=1.6)

    n_rows   = len(rows)
    zone_top = 0.70
    zone_bot = 0.22
    slot_h   = (zone_top - zone_bot) / n_rows
    bar_h    = 0.095
    bx       = 0.08
    bw       = 0.84

    max_val = max(r[1] for r in rows) * 1.25
    max_val = max(max_val, 0.01)

    for idx, (label, value, col, display) in enumerate(rows):
        cy = zone_top - (idx + 0.5) * slot_h
        by = cy - bar_h / 2

        ax.text(bx, by + bar_h + 0.035, label,
                ha="left", va="bottom",
                fontsize=8.5, fontweight="bold", color=col)

        # track
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bar_h,
            boxstyle="round,pad=0.004",
            facecolor="#1C1C1C", edgecolor="none",
            transform=ax.transAxes, clip_on=True))

        # fill
        fill = bw * min(value / max_val, 1.0)
        ax.add_patch(FancyBboxPatch(
            (bx, by), max(fill, 0.018), bar_h,
            boxstyle="round,pad=0.004",
            facecolor=col, edgecolor="none", alpha=0.82,
            transform=ax.transAxes, clip_on=True))

        # value label — inside if bar is wide, outside if narrow
        if fill / bw >= 0.22:
            ax.text(bx + fill - 0.025, cy, display,
                    ha="right", va="center",
                    fontsize=10, fontweight="bold", color="#0D0D0D", clip_on=True)
        else:
            ax.text(bx + fill + 0.025, cy, display,
                    ha="left", va="center",
                    fontsize=10, fontweight="bold", color=col, clip_on=False)

    if note:
        ax.text(0.5, 0.085, note, ha="center", va="bottom",
                fontsize=7.5, color=TEXT_SEC, style="italic", linespacing=1.5)

# ── Stat card (cards 4–5) — single big number ─────────────────────────────────
def draw_stat_card(ax, title, subtitle, big_number, big_unit, detail, note=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=TEXT_PRI)
    ax.text(0.5, 0.845, subtitle, ha="center", va="top",
            fontsize=7.5, color=TEXT_SEC, linespacing=1.6)

    # thin accent divider under subtitle
    ax.add_patch(FancyBboxPatch(
        (0.35, 0.755), 0.30, 0.006,
        boxstyle="round,pad=0.001",
        facecolor=STAT_COL, edgecolor="none", alpha=0.6,
        transform=ax.transAxes))

    # big number centred
    ax.text(0.5, 0.62, big_number,
            ha="center", va="center",
            fontsize=42, fontweight="bold", color=STAT_COL,
            fontfamily="DejaVu Sans")

    # unit label just below the number
    ax.text(0.5, 0.42, big_unit,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=STAT_COL, alpha=0.75)

    # detail line
    ax.text(0.5, 0.30, detail,
            ha="center", va="center",
            fontsize=8, color=TEXT_SEC, linespacing=1.5)

    if note:
        ax.text(0.5, 0.085, note, ha="center", va="bottom",
                fontsize=7.5, color=TEXT_SEC, style="italic", linespacing=1.5)

# ── Render all five cards ─────────────────────────────────────────────────────
draw_bar_card(
    axes[0], "Words per Turn", "avg across all episodes",
    [
        ("Lyra",   68, LYRA_COL,   "68 w"),
        ("Cipher", 50, CIPHER_COL, "50 w"),
    ],
    note="Lyra builds context first.\nCipher leads with the point."
)

draw_bar_card(
    axes[1], "Hedging Rate", "qualifications per 100 words\ngeopolitical episode",
    [
        ("Lyra",   2.0, LYRA_COL,   "2.0"),
        ("Cipher", 0.2, CIPHER_COL, "0.2"),
    ],
    note="Cipher's assertiveness sharpens\non charged topics."
)

draw_bar_card(
    axes[2], "Turn Split", "share of conversation\nper agent",
    [
        ("Lyra",   52, LYRA_COL,   "52%"),
        ("Cipher", 48, CIPHER_COL, "48%"),
    ],
    note="Balance emerges naturally.\nNot enforced by the system."
)

draw_stat_card(
    axes[3],
    "Episode Length",
    "avg audio per episode",
    "~7", "minutes",
    "21 turns · ~1,275 words\nfully voiced end to end",
    note="No script. No editing.\nGenerated and exported live."
)

draw_stat_card(
    axes[4],
    "Generation Speed",
    "time to produce a full episode",
    "24", "seconds",
    "to generate 21 turns of debate\nbefore audio rendering begins",
    note="Memory depth now at 4.\nBoth agents remember past episodes."
)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.04, 0.96], [0.108, 0.108],
               transform=fig.transFigure, color=DIVIDER, lw=0.8))
fig.text(0.5, 0.065,
         "DeepDive  ·  v3.5  ·  small dataset — patterns worth watching",
         ha="center", va="top", fontsize=8, color=TEXT_SEC)

out = "/Users/krat6s/Documents/podcast_ai/output/deepdive_metrics.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
