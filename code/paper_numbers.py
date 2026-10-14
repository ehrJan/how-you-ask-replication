"""Numbers quoted in the text that no other script writes, plus the data behind Figures 3-5.

Outputs (out/paper/):
  daily_engine_series.csv   engine x channel x day means over factorial answers (Figure 5)
  environment_summary.csv   Gemini period means, Perplexity interface/API levels (Section 5.5, Figure 5)
  sample_table.csv          Table 2: responses, >=1 retrieved, >=1 cited, no-citation rate with 95% CI
  fig3_ame_by_engine.csv    pooled and per-engine AMEs on the cited composition (Figure 3)
  fig4_fea_by_scenario.csv  per-scenario feature effects on the corporate cited share (Figure 4)
  fig4_meta_summary.csv     precision-weighted mean and 95% prediction interval (Figure 4)
Run after fmnl_run.py and fmnl_hetero.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from revised_analysis import ENGINES, build_runs

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "paper"
COLS = ["any_source", "any_citation", "citation_n", "citation_mass_CORPORATE"]


def daily(runs_all):
    core = runs_all[runs_all.role == "factorial"]
    return core.groupby(["model_id", "channel", "date"])[COLS].mean().reset_index()


def environment(d):
    gem = d[(d.model_id == "gemini-ui")]
    ui = d[(d.model_id == "perplexity-ui") & (d.channel == "ui")]
    api = d[(d.model_id == "perplexity-ui") & (d.channel == "api")]
    pre, post = ui.date < "2026-09-10", ui.date > "2026-09-15"
    rows = {
        "gemini_any_citation_before_sep5_pct": 100 * gem[gem.date < "2026-09-05"].any_citation.mean(),
        "gemini_any_citation_after_sep9_pct": 100 * gem[gem.date > "2026-09-09"].any_citation.mean(),
        "perplexity_ui_cited_per_answer_before": ui[pre].citation_n.mean(),
        "perplexity_ui_cited_per_answer_after": ui[post].citation_n.mean(),
        "perplexity_api_cited_per_answer": api.citation_n.mean(),
        "perplexity_ui_any_citation_before_pct": 100 * ui[pre].any_citation.mean(),
        "perplexity_ui_any_citation_after_pct": 100 * ui[post].any_citation.mean(),
        "perplexity_ui_corporate_all_answers_pct": 100 * ui.citation_mass_CORPORATE.mean(),
        "perplexity_ui_corporate_before_pct": 100 * ui[pre].citation_mass_CORPORATE.mean(),
        "perplexity_ui_corporate_after_pct": 100 * ui[post].citation_mass_CORPORATE.mean(),
        "perplexity_api_corporate_all_answers_pct": 100 * api.citation_mass_CORPORATE.mean(),
        "perplexity_api_days": api.date.nunique(),
    }
    return pd.Series(rows).rename("value").rename_axis("quantity").reset_index()


def sample_table(runs):
    """Table 2. No-citation rate: scenario means per engine, CI from t(24) over scenarios;
    Total = equal-engine average of the scenario means."""
    core = runs[runs.role == "factorial"].assign(no_citation=lambda d: 1 - d.any_citation)
    by = core.groupby("model_id").agg(responses=("chat_id", "size"), retrieved=("any_source", "sum"),
                                      cited=("any_citation", "sum"))
    sc = core.groupby(["scenario", "model_id"]).no_citation.mean().unstack()[ENGINES]
    sc["total"] = sc[ENGINES].mean(axis=1)
    g = len(sc)
    q = stats.t.ppf(0.975, g - 1)
    ci = pd.DataFrame({"no_citation_pct": 100 * sc.mean(), "se": 100 * sc.std(ddof=1) / np.sqrt(g)})
    ci["ci_low"], ci["ci_high"] = ci.no_citation_pct - q * ci.se, ci.no_citation_pct + q * ci.se
    by.loc["total"] = by.sum()
    return by.join(ci.drop(columns="se")).rename_axis("engine").reset_index()


def fig3():
    return pd.read_csv(HERE / "out" / "fmnl" / "jev_citation" / "ame.csv")


def fig4(runs):
    rq2 = HERE / "out" / "fmnl" / "rq2"
    s1, s2 = pd.read_csv(rq2 / "jev_stage1.csv"), pd.read_csv(rq2 / "jev_stage2.csv")
    pick = lambda t: t[(t.engine == "all") & (t.type == "CORPORATE") & (t.factor == "fea")]
    topics = runs[["scenario", "topic"]].drop_duplicates()
    per = pick(s1).drop(columns="topic").merge(topics, on="scenario", validate="one_to_one")
    q = stats.t.ppf(0.975, per.n_days - 1)  # same reference as the stage-1 p-values
    per["ci_low"], per["ci_high"] = per.ame_pp - q * per.se_jack, per.ame_pp + q * per.se_jack
    return per, pick(s2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = build_runs("ui")[0]
    d = daily(build_runs("all")[0])
    d.to_csv(OUT / "daily_engine_series.csv", index=False)
    env = environment(d)
    env.to_csv(OUT / "environment_summary.csv", index=False)
    tab = sample_table(runs)
    tab.to_csv(OUT / "sample_table.csv", index=False)
    fig3().to_csv(OUT / "fig3_ame_by_engine.csv", index=False)
    per, meta = fig4(runs)
    per.to_csv(OUT / "fig4_fea_by_scenario.csv", index=False)
    meta.to_csv(OUT / "fig4_meta_summary.csv", index=False)
    print(tab.round(1).to_string(index=False))
    print(env.round(2).to_string(index=False))
    print(meta.T.to_string())


if __name__ == "__main__":
    main()
