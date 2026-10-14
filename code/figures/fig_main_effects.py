"""Main effects figure: AME of prompt factors on corporate / editorial source shares.

Reads pooled and per-engine average marginal effects (pp) and writes
fig_main_effects.pdf and fig_main_effects.png (300 dpi) into the figures folder.
A point is "confirmed" (filled) only if gate_pass is True AND p_within < 0.05.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from pathlib import Path
PAPER = Path(__file__).resolve().parents[1] / "out" / "paper"
FIGOUT = Path(__file__).resolve().parents[1] / "out" / "figures"
FIGOUT.mkdir(parents=True, exist_ok=True)
IN_CSV = PAPER / "fig3_ame_by_engine.csv"
OUT_DIR = FIGOUT
OUT_STEM = "fig_main_effects"

# Okabe-Ito colours; "All engines" is black.
ENGINES = [  # key, label, colour, marker, size, dodge offset (y units)
    ("all", "All engines", "black", "D", 4.2, 0.27),
    ("chatgpt-ui", "ChatGPT", "#0072B2", "o", 3.4, 0.09),
    ("gemini-ui", "Gemini", "#E69F00", "s", 3.2, -0.09),
    ("perplexity-ui", "Perplexity", "#009E73", "^", 3.6, -0.27),
]
FACTORS = [  # key, label, y position (top = largest)
    ("fea", "Feature disclosure", 2.0),
    ("adv", "Advice framing", 1.0),
    ("prc", "Budget disclosure", 0.0),
]
PANELS = [("CORPORATE", "(a) Corporate sources"), ("EDITORIAL", "(b) Editorial sources")]
XLIM = (-25, 24)
MINUS = "−"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(IN_CSV)
df["gate_pass"] = df["gate_pass"].astype(str).str.lower().eq("true")
# Confirmed = gate_pass AND p_within < 0.05 (NaN p_within counts as not confirmed).
df["confirmed"] = df["gate_pass"] & (df["p_within"] < 0.05)

# Check against the value quoted in the text.
row = df[(df.engine == "all") & (df.type == "CORPORATE") & (df.factor == "fea")].iloc[0]
assert round(row.ame_pp, 2) == 8.12, row.ame_pp
assert round(row.ci_low, 2) == 3.92 and round(row.ci_high, 2) == 12.31


def get(engine, typ, factor):
    r = df[(df.engine == engine) & (df.type == typ) & (df.factor == factor)]
    assert len(r) == 1, (engine, typ, factor)
    return r.iloc[0]


eng_labels = []
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharex=True, sharey=True)
fig.subplots_adjust(left=0.165, right=0.99, top=0.86, bottom=0.20, wspace=0.06)

for ax_i, (ax, (typ, title)) in enumerate(zip(axes, PANELS)):
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    ax.axvline(0, color="0.35", lw=0.7, zorder=1)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Light separators between the three factor groups.
    for y_sep in (0.5, 1.5):
        ax.axhline(y_sep, color="0.88", lw=0.5, zorder=0)
    for key, _, colour, marker, size, dodge in ENGINES:
        for factor, _, y in FACTORS:
            r = get(key, typ, factor)
            yy = y + dodge
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=colour, lw=0.9, zorder=2,
                    solid_capstyle="butt")
            face = colour if r.confirmed else "white"
            ax.plot([r.ame_pp], [yy], linestyle="none", marker=marker,
                    markersize=size if r.confirmed else size * 1.4,
                    markerfacecolor=face, markeredgecolor=colour,
                    markeredgewidth=1.2 if r.confirmed else 1.0,
                    zorder=3)
            if key != "all" and factor == "fea":
                txt = f"{r.ame_pp:+.1f}".replace("-", MINUS)
                if r.ame_pp > 0:  # right of the CI end, away from zero
                    x_lab, ha_ = r.ci_high + 0.6, "left"
                else:  # left of the CI end, away from zero
                    x_lab, ha_ = r.ci_low - 0.6, "right"
                t_ = ax.text(x_lab, yy, txt, va="center", ha=ha_, fontsize=6.5,
                             color=colour, zorder=4)
                if typ == "CORPORATE":
                    eng_labels.append((ax, t_, r.ci_low, yy, txt, colour))
            if key == "all" and factor == "adv":
                txt = f"{r.ame_pp:+.1f}".replace("-", MINUS)
                if r.ame_pp > 0:
                    ax.text(r.ci_high + 0.6, yy, txt, va="center", ha="left", fontsize=6.5,
                            color="black", zorder=4)
                else:
                    ax.text(r.ci_low - 0.6, yy, txt, va="center", ha="right", fontsize=6.5,
                            color="black", zorder=4)
            if key == "gemini-ui" and factor == "adv":
                txt = f"{r.ame_pp:+.1f}".replace("-", MINUS)
                if r.ame_pp > 0:
                    x_lab, ha_ = r.ci_high + 0.6, "left"
                else:
                    x_lab, ha_ = r.ci_low - 0.6, "right"
                ax.text(x_lab, yy, txt, va="center", ha=ha_, fontsize=6.5,
                        color=colour, zorder=4)
            if key == "chatgpt-ui" and factor == "prc" and typ == "CORPORATE":
                txt = f"{r.ame_pp:+.1f}".replace("-", MINUS)
                ax.text(r.ci_high + 0.6, yy, txt, va="center", ha="left", fontsize=6.5,
                        color=colour, zorder=4)
            if key == "all" and factor == "fea":
                txt = f"{r.ame_pp:+.1f}".replace("-", MINUS)
                if r.ame_pp > 0:  # right of the bar end, away from zero
                    ax.text(r.ci_high + 0.8, yy, txt, va="center", ha="left", fontsize=6.5,
                            color="black", zorder=4)
                else:  # left of the bar end, away from zero
                    ax.text(r.ci_low - 0.6, yy, txt, va="center", ha="right", fontsize=6.5,
                            color="black", zorder=4)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xticklabels([f"{t:+d}".replace("-", MINUS) if t else "0" for t in [-20, -10, 0, 10, 20]])
    ax.set_yticks([f[2] for f in FACTORS])
    ax.set_ylim(-0.5, 2.5)
    ax.tick_params(axis="both", length=2, width=0.5, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    if ax_i == 0:
        ax.set_yticklabels([f[1] for f in FACTORS])
        ax.tick_params(axis="y", length=0)
    else:
        ax.tick_params(axis="y", length=0, labelleft=False)

engine_handles = [
    Line2D([], [], linestyle="none", marker=m, markersize=sz, markerfacecolor=c,
           markeredgecolor=c, label=lbl)
    for k, lbl, c, m, sz, _ in ENGINES
]
key_handles = [
    Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="black",
           markeredgecolor="black", markeredgewidth=1.2, label="confirmed"),
    Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="white",
           markeredgecolor="black", markeredgewidth=1.2, label="not confirmed"),
]
gap_handle = [Line2D([], [], linestyle="none", label=" ")]
fig.legend(handles=engine_handles + gap_handle + key_handles, loc="upper center", ncol=7,
           frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.25,
           columnspacing=1.0, borderaxespad=0.0)

fig.supxlabel("Change in share of cited sources (pp)", y=0.02, fontsize=7)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
for ax_, t_, lo_, yy_, txt_, col_ in eng_labels:
    if t_.get_window_extent(_r).x1 > ax_.get_window_extent(_r).x1:
        t_.remove()
        ax_.text(lo_ - 0.8, yy_, txt_, va="center", ha="right", fontsize=6.5, color=col_,
                 zorder=4)

_fig_r = fig.canvas.get_renderer()
for _ax in axes:
    _axb = _ax.get_window_extent(_fig_r)
    for _t in _ax.texts:
        _b = _t.get_window_extent(_fig_r)
        if _b.x0 < _axb.x0 or _b.x1 > _axb.x1:
            print("OUTSIDE AXES:", repr(_t.get_text()), round(_b.x0 - _axb.x0), round(_axb.x1 - _b.x1))
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf")
fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=300)

# Print which points are filled, for checking.
filled = df[df.confirmed][["engine", "type", "factor", "ame_pp", "p_within"]]
print(filled.round(4).to_string(index=False))
