"""Model-version and collection-channel precision for the Study 2 window.

ChatGPT is the only engine whose consumer UI streams a model identifier
(`the reported model field of the collection platform`); Gemini and Perplexity expose none, so their
versions are unobservable and only behaviour can be monitored. Outputs:

  chatgpt_model_by_day.csv   slug composition per day (gpt-5-6 / mini / blank / other)
  chatgpt_slug_outcomes.csv  outcome means by slug group (blank = truncated stream?)
  engine_day_series.csv      per engine x day: outcome mean net of prompt fixed effects (+SE)
  engine_breaks.csv          sup-|t| persistent-shift test per engine x outcome (permutation p),
                             MAD-z single-day flags
  channel_levels.csv         Perplexity UI vs API (sonar) level shift with same-day
                             ChatGPT / Gemini placebos
  channel_stability.csv      factor effects on UI-only vs API-only Perplexity days (TOST margin)
  fig/engine_daily_series.png

Break test: every split with >= MIN_SIDE days per side, Welch t on the day means;
p from 999 permutations of the day labels (null: exchangeable days, i.e. no
persistent shift; day shocks are noise). Weekday adjustment skipped at 22 days.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from design import DATA
from inference import FACTORS, contrast_weights, infer_mean
from revised_analysis import ENGINES, OUT, SEED, as_cube, build_runs, cell_means

FIG = Path(__file__).resolve().parent / "out" / "fig"
OUTCOMES = ["citation_mass_CORPORATE", "citation_mass_EDITORIAL", "citation_share_CORPORATE",
            "any_source", "any_citation", "citation_n"]
MIN_SIDE, PERMS, FLAG_Z = 3, 999, 4.0
API_DAYS = pd.to_datetime(["2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15"])


def slug_group(s):
    if pd.isna(s):
        return "blank"
    return "gpt-5-6" if s == "gpt-5-6" else ("mini" if "mini" in s else s)


def chatgpt_slugs(runs):
    slugs = pd.read_parquet(DATA / "chatgpt_model_by_chat.parquet")[["chat_id", "model_slug"]]
    d = runs[runs.model_id == "chatgpt-ui"].merge(slugs, on="chat_id", how="left")
    d["slug_group"] = d.model_slug.map(slug_group)
    by_day = d.groupby([d.date.dt.strftime("%Y-%m-%d"), "slug_group"]).size().unstack(fill_value=0)
    by_day["n_chats"] = by_day.sum(axis=1)
    known = by_day.drop(columns=["blank", "n_chats"]).sum(axis=1)
    by_day["share_gpt56_of_known"] = by_day.get("gpt-5-6", 0) / known
    by_day.index.name = "date"
    outcomes = d.groupby("slug_group").agg(n=("chat_id", "size"), **{o: (o, "mean") for o in OUTCOMES})
    return by_day.reset_index(), outcomes.reset_index()


def daily_series(runs, col):
    """Engine x day mean of the prompt-demeaned outcome (prompt FE), SE over prompts."""
    d = runs[runs.role == "factorial"].dropna(subset=[col])
    pe = d.groupby(["prompt_id", "model_id", "date"])[col].mean().reset_index()
    pe["resid"] = pe[col] - pe.groupby(["prompt_id", "model_id"])[col].transform("mean")
    out = pe.groupby(["model_id", "date"]).agg(mean=("resid", "mean"), se=("resid", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                                                n_prompts=("prompt_id", "nunique"), level=(col, "mean")).reset_index()
    out["outcome"] = col
    return out


def sup_t_break(days, values, rng):
    """Best single split by Welch t; permutation p over day labels."""
    v = np.asarray(values, float)
    if np.ptp(v) == 0:  # constant series (e.g. any_source == 1 on every Perplexity UI day)
        return dict(n_days=len(v), break_date="", shift=0., sup_t=np.nan, p_perm=np.nan, max_abs_rz=0., rz_flag_dates="")
    def best(v):
        ts = [(stats.ttest_ind(v[k:], v[:k], equal_var=False).statistic, k) for k in range(MIN_SIDE, len(v) - MIN_SIDE + 1)]
        t, k = max(ts, key=lambda x: abs(x[0]))
        return t, k
    t, k = best(v)
    perm = np.array([abs(best(rng.permutation(v))[0]) for _ in range(PERMS)])
    z = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12)
    return dict(n_days=len(v), break_date=str(days[k])[:10], shift=v[k:].mean() - v[:k].mean(),
                sup_t=t, p_perm=(perm >= abs(t)).mean(), max_abs_rz=np.abs(z).max(),
                rz_flag_dates=";".join(str(dd)[:10] for dd, zz in zip(days, z) if abs(zz) >= FLAG_Z))


def channel_exhibit(runs_all, signs):
    """Perplexity UI vs API days: level shift with placebos, and factor-effect stability."""
    core = runs_all[runs_all.role == "factorial"].copy()
    core["api_day"] = core.date.isin(API_DAYS)
    levels = []
    for e in ENGINES:
        d = core[core.model_id == e]
        for o in OUTCOMES:
            a, u = d.loc[d.api_day, o], d.loc[~d.api_day, o]
            levels.append(dict(engine=e, outcome=o, ui_days_mean=u.mean(), api_days_mean=a.mean(),
                               shift_pp=100 * (a.mean() - u.mean()), n_ui=u.notna().sum(), n_api=a.notna().sum(),
                               channel="sonar API" if e == "perplexity-ui" else "same-day placebo"))
    stability = []
    px = core[core.model_id == "perplexity-ui"]
    scenarios = sorted(px.scenario.unique())
    for metric, k in (("citation_share", "CORPORATE"), ("citation_share", "EDITORIAL"),
                      ("citation_mass", "CORPORATE"), ("citation_mass", "EDITORIAL")):
        col = f"{metric}_{k}"
        cubes = {}
        for flag in (False, True):
            sub = px[px.api_day == flag]
            cells = cell_means(sub, [col])
            cubes[flag] = as_cube(cells, col, "perplexity-ui", scenarios)
        for f in FACTORS:
            w = contrast_weights(f)
            ui, api = cubes[False] @ w, cubes[True] @ w
            r = infer_mean(api - ui, signs)
            m_min = abs(r["estimate"]) + stats.t.ppf(.95, len(scenarios) - 1) * r["se"]
            stability.append(dict(metric=metric, type=k, factor=f, effect_ui_days=ui.mean(), effect_api_days=api.mean(),
                                  diff=r["estimate"], se=r["se"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                                  p_wild=r["p_wild"], min_equivalence_margin_pp=100 * m_min, n_scenarios=len(scenarios)))
    return pd.DataFrame(levels), pd.DataFrame(stability)


def figure(series, breaks, by_day):
    engines = ENGINES
    outs = ["citation_mass_CORPORATE", "any_source"]
    fig, axes = plt.subplots(len(outs), len(engines), figsize=(12, 5.6), sharex=True)
    for j, e in enumerate(engines):
        for i, o in enumerate(outs):
            ax = axes[i, j]
            s = series.query("model_id == @e and outcome == @o").sort_values("date")
            # show collection gaps as gaps, not as a line across them
            s = s.set_index("date").reindex(pd.date_range(series.date.min(), series.date.max())).rename_axis("date").reset_index()
            ax.axhline(0, color="#898781", lw=1)
            ax.fill_between(s.date, 100 * (s["mean"] - 1.96 * s.se), 100 * (s["mean"] + 1.96 * s.se), color="#2a78d6", alpha=.12, lw=0)
            ax.plot(s.date, 100 * s["mean"], color="#2a78d6", lw=2, marker="o", ms=4, mec="#fcfcfb", mew=1)
            b = breaks.query("model_id == @e and outcome == @o").iloc[0]
            if b.break_date:
                ax.axvline(pd.Timestamp(b.break_date), color="#52514e", lw=1)
                sub = f"best split {b.break_date}: {100 * b['shift']:+.1f} pp, permutation p = {b.p_perm:.2f}"
            else:
                sub = "constant series"
            ax.set_title(f"{e.replace('-ui','')} · {o.replace('_', ' ')}\n{sub}", loc="left", fontsize=8.5)
            if e == "perplexity-ui":
                ax.axvspan(pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-15"), color="#f0efec", zorder=0)
            ax.grid(color="#e1e0d9", lw=.6); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].set_ylabel("pp, net of prompt FE"); axes[1, 0].set_ylabel("pp, net of prompt FE")
    if by_day is not None:
        g = by_day.set_index(pd.to_datetime(by_day.date)).share_gpt56_of_known
        axes[0, 0].text(.02, .04, f"gpt-5-6 share of identified chats: {g.min():.2f}–{g.max():.2f}", transform=axes[0, 0].transAxes, fontsize=7.5, color="#52514e")
    fig.suptitle("Daily engine behaviour over the window (prompt fixed effects removed; grey band = Perplexity UI outage / API fallback)",
                 x=.02, ha="left", fontsize=11, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(FIG / "engine_daily_series.png", dpi=200, bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    runs, *_ = build_runs("ui")
    by_day, slug_out = chatgpt_slugs(runs)
    by_day.to_csv(OUT / "chatgpt_model_by_day.csv", index=False)
    slug_out.to_csv(OUT / "chatgpt_slug_outcomes.csv", index=False)
    series = pd.concat([daily_series(runs, o) for o in OUTCOMES], ignore_index=True)
    series.to_csv(OUT / "engine_day_series.csv", index=False)
    breaks = []
    for (e, o), s in series.groupby(["model_id", "outcome"]):
        s = s.sort_values("date")
        breaks.append(dict(model_id=e, outcome=o, **sup_t_break(s.date.to_numpy(), s["mean"].to_numpy(), rng),
                           min_daily_prompts=int(s.n_prompts.min())))
    breaks = pd.DataFrame(breaks)
    # shares and rates in pp; citation_n stays a count
    breaks["shift_pp"] = np.where(breaks.outcome == "citation_n", np.nan, 100 * breaks["shift"])
    breaks.to_csv(OUT / "engine_breaks.csv", index=False)
    runs_all, *_ = build_runs("all")
    g = runs_all[runs_all.role == "factorial"].scenario.nunique()
    signs = rng.choice([-1., 1.], size=(9999, g))
    levels, stability = channel_exhibit(runs_all, signs)
    levels.to_csv(OUT / "channel_levels.csv", index=False)
    stability.to_csv(OUT / "channel_stability.csv", index=False)
    figure(series, breaks, by_day)
    print(slug_out.round(3).to_string(index=False))
    print(breaks[["model_id", "outcome", "break_date", "shift", "sup_t", "p_perm", "max_abs_rz", "rz_flag_dates"]].round(3).to_string(index=False))
    print(levels.query("outcome in ['citation_mass_CORPORATE','any_citation','citation_n']")[["engine", "outcome", "ui_days_mean", "api_days_mean", "shift_pp", "channel"]].round(3).to_string(index=False))
    print(stability[["metric", "type", "factor", "effect_ui_days", "effect_api_days", "diff", "p_wild", "min_equivalence_margin_pp"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
