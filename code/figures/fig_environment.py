"""Figure: measurement environment (three panels, one shared legend row).

(a) Gemini citation drop and Perplexity interface gap; (b) Perplexity cited
sources per answer, interface vs API; (c) Perplexity corporate share of
citations, with ChatGPT as reference.

Input : out/paper/daily_engine_series.csv (engine x channel x day).
Output: figures/fig_environment.pdf and .png (300 dpi), 7.0 in x 2.5 in.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pathlib import Path
PAPER = Path(__file__).resolve().parents[1] / "out" / "paper"
FIGOUT = Path(__file__).resolve().parents[1] / "out" / "figures"
FIGOUT.mkdir(parents=True, exist_ok=True)
SRC = PAPER / "daily_engine_series.csv"
OUT = str(FIGOUT)

BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#E69F00", "#009E73", "#882255"
GREY_TXT = "0.4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

d = pd.read_csv(SRC, parse_dates=["date"])
start, end = d["date"].min(), d["date"].max()
full = pd.date_range(start, end, freq="D")


def series(mask, col):
    """Daily series on the full calendar; missing days stay NaN (gaps)."""
    return d.loc[mask].set_index("date")[col].reindex(full)


ui = d["channel"] == "ui"
api = d["channel"] == "api"
chat = d["model_id"] == "chatgpt-ui"
gem = d["model_id"] == "gemini-ui"
ppx_ui = (d["model_id"] == "perplexity-ui") & ui
ppx_api = (d["model_id"] == "perplexity-ui") & api

# Verified numbers (printed for the report).
g = d[gem]
pre = g[g["date"] < "2026-09-05"]["any_citation"].mean()
post = g[g["date"] > "2026-09-09"]["any_citation"].mean()
pu = d[ppx_ui]["citation_n"].mean()
pa = d[ppx_api]["citation_n"].mean()
cu = 100 * d[ppx_ui]["citation_mass_CORPORATE"].mean()
ca = 100 * d[ppx_api]["citation_mass_CORPORATE"].mean()
cu_pre = 100 * d[ppx_ui & (d["date"] < "2026-09-10")]["citation_mass_CORPORATE"].mean()
cu_post = 100 * d[ppx_ui & (d["date"] > "2026-09-15")]["citation_mass_CORPORATE"].mean()
ui_pre = d[ppx_ui & (d["date"] < "2026-09-10")]["citation_n"].mean()
ui_post = d[ppx_ui & (d["date"] > "2026-09-15")]["citation_n"].mean()
print(f"Gemini any_citation pre-Sep5 {pre:.3f} post-Sep9 {post:.3f}")
print(f"Perplexity citation_n ui {pu:.2f} api {pa:.2f}")
print(f"Perplexity corporate % ui {cu:.2f} api {ca:.2f}")
print(f"Perplexity UI citation_n pre-Sep10 {ui_pre:.2f} post-Sep15 {ui_post:.2f}")

fig = plt.figure(figsize=(6.95, 2.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.4,
                      left=0.065, right=0.935, top=0.88, bottom=0.25)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

band_b0, band_b1 = pd.Timestamp("2026-09-09 12:00"), pd.Timestamp("2026-09-15 12:00")
band_a0, band_a1 = pd.Timestamp("2026-09-04 12:00"), pd.Timestamp("2026-09-09 12:00")

# ---- (a) share of answers with >=1 citation ----
ax_a.axvspan(band_a0, band_a1, color="#E69F00", alpha=0.15, lw=0, zorder=0)
ax_a.plot(full, 100 * series(chat, "any_citation"), "-o", ms=2.2, lw=1.0,
          color=BLUE, zorder=3)
ax_a.plot(full, 100 * series(gem, "any_citation"), "-s", ms=2.0, lw=1.0,
          color=ORANGE, zorder=3)
ax_a.hlines(100 * pre, pd.Timestamp("2026-08-26"), band_a0, linestyles="--",
            colors=ORANGE, lw=0.9, zorder=2)
ax_a.hlines(100 * post, pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-28"),
            linestyles="--", colors=ORANGE, lw=0.9, zorder=2)
ax_a.text(pd.Timestamp("2026-09-11"), 70,
          f"Gemini: ≈{100 * pre:.0f}% to ≈{100 * post:.0f}%", fontsize=7.0,
          color=ORANGE, va="bottom", ha="left", zorder=4)
ax_a.set_ylim(0, 100)
ax_a.set_ylabel("Share of answers (%)")
ax_a.set_title("(a) Answers citing any source", loc="left", pad=3,
               fontweight="bold")

# ---- (b) Perplexity cited sources per answer ----
ax_b.axvspan(band_b0, band_b1, facecolor="#EEEEEE", lw=0, zorder=0)
ax_b.plot(full, series(ppx_ui, "citation_n"), "-^", ms=2.6, lw=1.0,
          color=GREEN, zorder=3)
ax_b.plot(full, series(ppx_api, "citation_n"), linestyle="none", marker="D",
          ms=3.0, color=PURPLE, zorder=4)
for m, x0, x1, tx in ((ui_pre, "2026-08-26", "2026-09-09", "2026-08-27"),
                      (ui_post, "2026-09-16", "2026-09-28", "2026-09-16")):
    ax_b.hlines(m, pd.Timestamp(x0), pd.Timestamp(x1), linestyles="--",
                colors=GREEN, lw=0.9, zorder=2)
    ax_b.text(pd.Timestamp(tx), m - 0.45, f"≈{m:.1f}", fontsize=7.0,
              color=GREEN, va="top", ha="left")
ax_b.text(pd.Timestamp("2026-09-18"), 8.3, f"API ≈ {pa:.1f}", fontsize=7.0,
          color=PURPLE, va="center", ha="left")
ax_b.set_ylim(0, 10)
ax_b.set_ylabel("Cited sources per answer")
ax_b.set_title("(b) Perplexity: cited sources", loc="left",
               pad=3, fontweight="bold")

# ---- (c) corporate share of citations ----
ax_c.axvspan(band_b0, band_b1, facecolor="#EEEEEE", lw=0, zorder=0)
ax_c.hlines(cu_pre, pd.Timestamp("2026-08-26"), band_b0, linestyles="--",
            colors=GREEN, lw=0.9, zorder=2)
ax_c.hlines(cu_post, band_b1, full[-1], linestyles="--", colors=GREEN, lw=0.9,
            zorder=2)
ax_c.text(pd.Timestamp("2026-08-27"), 13.5, f"≈{cu_pre:.1f}%",
          fontsize=7.0, color=GREEN, va="center", ha="left")
ax_c.text(full[-1], 13.5, f"≈{cu_post:.1f}%",
          fontsize=7.0, color=GREEN, va="center", ha="right")
ax_c.plot(full, 100 * series(ppx_ui, "citation_mass_CORPORATE"), "-^", ms=2.6,
          lw=1.0, color=GREEN, zorder=3)
ax_c.plot(full, 100 * series(ppx_api, "citation_mass_CORPORATE"),
          linestyle="none", marker="D", ms=4.5, mfc=PURPLE, mec="white",
          mew=0.8, zorder=5)
ax_c.text(pd.Timestamp("2026-09-16"), 24, f"API ≈ {ca:.1f}%",
          fontsize=7.0, color=PURPLE, va="center", ha="left")
ax_c.set_ylim(0, 30)
ax_c.set_ylabel("Corporate share, all answers (%)")
ax_c.set_title("(c) Perplexity: corporate sources", loc="left", pad=3,
               fontweight="bold")

ticks = pd.date_range(start, end, freq="14D")
for ax in (ax_a, ax_b, ax_c):
    ax.set_xlim(start - pd.Timedelta(days=0.5), end + pd.Timedelta(days=0.5))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.tick_params(length=2.5, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)

# ---- one shared legend row below the panels ----
handles = [
    Line2D([], [], color=BLUE, marker="o", ms=3.5, lw=1.0, label="ChatGPT"),
    Line2D([], [], color=ORANGE, marker="s", ms=3.2, lw=1.0, label="Gemini"),
    Line2D([], [], color=GREEN, marker="^", ms=3.6, lw=1.0,
           label="Perplexity interface"),
    Line2D([], [], color=PURPLE, marker="D", ms=3.2, lw=0,
           label="Perplexity API"),
    Patch(facecolor="#E69F00", alpha=0.15, edgecolor="none",
          label="Gemini change, Sep 5–9"),
    Patch(facecolor="#EEEEEE", edgecolor="none",
          label="Perplexity interface not collected"),
]
# place legend ~0.05 in below the lowest x-tick label (measured, not guessed)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_ymin = min(t.get_window_extent(_r).y0
            for _ax in (ax_a, ax_b, ax_c)
            for t in _ax.get_xticklabels() if t.get_text())
_y_top = fig.transFigure.inverted().transform((0, _ymin - 0.05 * fig.dpi))[1]
_leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, _y_top), handlelength=2.0,
                  columnspacing=1.6)
# correct for the legend's internal padding so its box top sits 0.05 in below the ticks
fig.canvas.draw()
_shift = (_leg.get_window_extent(_r).y1 - (_ymin - 0.05 * fig.dpi))
_y_px = fig.transFigure.transform((0, _y_top))[1] - _shift
_y_top = fig.transFigure.inverted().transform((0, _y_px))[1]
_leg.set_bbox_to_anchor((0.5, _y_top), transform=fig.transFigure)
fig.canvas.draw()
print("gap_in", round((_ymin - _leg.get_window_extent(_r).y1) / fig.dpi, 3), file=sys.stderr)

fig.savefig(f"{OUT}/fig_environment.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(f"{OUT}/fig_environment.png", dpi=300, bbox_inches="tight",
            pad_inches=0.08)
print("saved", file=sys.stderr)
