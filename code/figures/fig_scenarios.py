"""Forest plot of feature-disclosure effects across 25 shopping needs (Fig. 2)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
PAPER = Path(__file__).resolve().parents[1] / "out" / "paper"
FIGOUT = Path(__file__).resolve().parents[1] / "out" / "figures"
FIGOUT.mkdir(parents=True, exist_ok=True)
DATA = str(PAPER)
OUT = str(FIGOUT)

# Plain-language row labels (<= 26 characters), keyed by scenario code.
LABELS = {
    "BP01": "Multi-day backpacking",
    "BP02": "School backpack",
    "BP03": "Carry-on travel pack",
    "BP04": "Daypack for trips",
    "BP05": "IT commuter backpack",
    "CH01": "Comfortable office chair",
    "CH02": "Ergonomic desk chair",
    "CH03": "Breathable chair, long use",
    "CH04": "All-day desk sitting",
    "CH05": "Big and tall desk chair",
    "GL01": "Winter gloves (−30°C)",
    "GL02": "Heavy manual work gloves",
    "GL03": "Yardwork gloves",
    "GL04": "Cold-weather gardening",
    "GL05": "Commute gloves (−15°C)",
    "OW01": "Leather jacket, rain",
    "OW02": "Warm women's winter coat",
    "OW03": "Winter commute coat",
    "OW04": "Jacket, variable weather",
    "OW05": "Fall layering jacket",
    "WB01": "Purse carry, no leaks",
    "WB02": "Small-opening bottle",
    "WB03": "Child's school bottle",
    "WB04": "Steel, no plastic contact",
    "WB05": "Steel bottle, about 1 L",
}

TOPIC_ORDER = ["Water Bottles", "Backpacks", "Outerwear", "Gloves", "Office Chairs"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.0,
})

df = pd.read_csv(os.path.join(DATA, "fig4_fea_by_scenario.csv"))
meta = pd.read_csv(os.path.join(DATA, "fig4_meta_summary.csv")).iloc[0]
assert len(df) == 25
assert df.loc[df.ame_pp.idxmax(), "scenario"] == "OW05"
assert df.loc[df.ame_pp.idxmin(), "scenario"] == "OW01"
df["label"] = df.scenario.map(LABELS)
assert df.label.notna().all() and df.label.str.len().max() <= 26

# Build rows top-down: topic header, then scenarios sorted by effect (desc).
rows = []  # (y, kind, payload)
y = 0
for topic in TOPIC_ORDER:
    rows.append((y, "header", topic))
    y += 1
    sub = df[df.topic == topic].sort_values("ame_pp", ascending=False)
    for _, r in sub.iterrows():
        rows.append((y, "point", r))
        y += 1
    y += 0.25  # gap between groups
ymax = y

W, H = 3.33, 3.6
DISPLAY = {"Water Bottles": "Water bottles", "Backpacks": "Backpacks",
           "Outerwear": "Outerwear", "Gloves": "Gloves",
           "Office Chairs": "Office chairs"}
# Precision-weighted (random-effects) topic means, as reported in the text.
# Source: out/fmnl/rq2/jev_stage2.csv, columns mean_<topic>.
TOPIC_MEAN = {"Water Bottles": 14.4, "Backpacks": 10.2, "Outerwear": 8.2,
              "Gloves": 7.6, "Office Chairs": 2.3}
# Topic means computed from the scenario CSV (unweighted mean of ame_pp).
fig, ax = plt.subplots(figsize=(W, H))

lo, hi = meta.pi_low, meta.pi_high
# Prediction-interval band (behind everything).
ax.axvspan(lo, hi, color="#E4EAF2", zorder=0, lw=0)
# Zero line and mean line.
ax.axvline(0, color="black", lw=0.6, zorder=0.5)
ax.axvline(meta.mu, color="#444444", lw=0.7, ls=(0, (3, 2)), zorder=0.6)

ticks, ticklabels = [], []
for yy, kind, p in rows:
    if kind == "header":
        ticks.append(yy)
        ticklabels.append(DISPLAY[p])
        ax.plot(TOPIC_MEAN[p], yy, marker="D", ms=3.0, mfc="white",
                mec="black", mew=0.6, ls="none", zorder=5)
        continue
    r = p
    ticks.append(yy)
    ticklabels.append(r.label)
    ax.plot([r.ci_low, r.ci_high], [yy, yy], color="#555555", lw=0.7,
            solid_capstyle="butt", zorder=3)
    ax.plot(r.ame_pp, yy, marker="o", ms=3.0, mfc="#222222",
            mec="white", mew=0.4, zorder=4, ls="none")

ax.set_yticks([])
ax.set_ylim(ymax - 0.2, -0.8)  # top-down order
# Labels drawn manually: topic headers flush left, scenario rows indented.
from matplotlib.transforms import blended_transform_factory
tr = blended_transform_factory(fig.transFigure, ax.transData)
kinds = [k for _, k, _ in rows]
for yy, kind, text in zip(ticks, kinds, ticklabels):
    if kind == "header":
        ax.text(0.0, yy, text, transform=tr, ha="left", va="center",
                fontweight="bold", clip_on=False)
    else:
        ax.text(0.035, yy, text, transform=tr, ha="left", va="center",
                clip_on=False)

assert df.ci_high.max() < 45
ax.set_xlim(-16, 45)
ax.set_xticks([-10, 0, 10, 20, 30, 40])
ax.set_xlabel("Effect on corporate share of cited sources (pp)", labelpad=2)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="x", length=2, pad=1.5)
ax.grid(False)

# Direct labels replace the legend (caption explains band, dashed line, diamonds).
_bottom_y = ticks[-1]
ax.text(meta.mu + 1.0, _bottom_y, "mean 8.5", transform=ax.transData,
        ha="left", va="center", fontsize=6, color="#333333", clip_on=False)
# Key for topic means, bottom-right inside axes (empty region right of x=27).
ax.plot(28.0, ticks[-1], marker="D", ms=3.0, mfc="white", mec="black", mew=0.6,
        ls="none", zorder=5)
ax.text(29.8, ticks[-1], "topic mean", transform=ax.transData,
        ha="left", va="center", fontsize=6, color="#333333", clip_on=False)

fig.subplots_adjust(left=0.385, right=0.975, top=0.975, bottom=0.085)
fig.savefig(os.path.join(OUT, "fig_scenarios.pdf"))
fig.savefig(os.path.join(OUT, "fig_scenarios.png"), dpi=300)
print("saved", OUT, "rows plotted:", int(df.shape[0]))
