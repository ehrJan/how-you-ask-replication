"""Descriptives with scenario-jackknife inference.

Convention of the revised layer: statistics are formed within scenario x engine,
averaged equally over scenarios, then engines ("all" = engine-equal mean); inference is
the leave-one-scenario-out jackknife with t(G-1). Answer-weighted pooled numbers are
kept only for comparison (`*_answer_weighted` stats). `unit` column:
shares and rates in pp, domain counts per answer in counts, conversion / Jaccard /
seen share as fractions.

Outputs (out/revised/, rows keyed by `labels`; one label set per process, the other
label set's rows already on disk are kept):
  descriptives_domain_counts.csv       FEA+ vs FEA- domains per answer + domain identity
  descriptives_zero_citation.csv       P(no citation), P(no source) per engine (selection)
  descriptives_perplexity_channel.csv  Perplexity API vs UI days, same-day ChatGPT/Gemini
                                       placebos and DiD; needs the channel='all' runs
  scenario_n_table.csv                 runs / sourced / cited answers and corporate cited share
                                       (answer-level SE, t CI) per scenario x engine x variant

The FEA-effect difference across Perplexity channels is not redone here: see
out/revised/channel_stability.csv (+11.43 pp [3.77, 19.09], p_wild .0061). The channel
contrast is confounded with its four calendar days; scenario jackknife intervals do not
cover day shocks, which is what the placebo rows are for.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "revised"
ARM_COLS = ["domains_retrieved", "corp_domains_retrieved", "corp_domains_cited",
            "corp_cited_given_retrieved"]


def load_runs(labels, channel="ui", pickle_path=None):
    """(runs, sources) from a frozen pickle, or build_runs(channel) under the label set.

    jev puts this directory first on sys.path (fmnl_run.py convention).
    """
    if pickle_path:
        d = pd.read_pickle(pickle_path)
        return d["runs"], d["sources"]
    if labels == "jev":
        sys.path.insert(0, str(HERE))
    import revised_analysis
    if labels != "jev":
        raise RuntimeError("revised_analysis already imported under the other label set")
    runs, sources, _, _ = revised_analysis.build_runs(channel)
    return runs, sources


def upsert(df, path):
    """Write this label set's rows, keeping the other label set's rows already on disk."""
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["labels"].isin(set(df["labels"]))], df], ignore_index=True)
    df.to_csv(path, index=False)


def scenario_jackknife(stat_fn, runs, *args):
    """Leave-one-scenario-out jackknife of a vector of named statistics.

    stat_fn(runs_subset, *args) -> pd.Series. SE = sqrt((G-1)/G * sum_g (theta_(g) -
    mean)^2), CI = theta +/- t_{.975,G-1} SE, p two-sided t(G-1) against 0. For a mean
    over scenario values this is exactly the usual t interval (inference.infer_mean).
    """
    scenarios = sorted(runs.scenario.unique())
    g = len(scenarios)
    theta = stat_fn(runs, *args)
    reps = pd.DataFrame([stat_fn(runs[runs.scenario != s], *args) for s in scenarios])
    th = theta.to_numpy(float)
    se = np.sqrt((g - 1) / g * ((reps - reps.mean()) ** 2).sum(min_count=1)).reindex(theta.index).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, th / se, np.where(th == 0, 0., np.copysign(np.inf, th)))
    t[np.isnan(se) | np.isnan(th)] = np.nan
    q = stats.t.ppf(.975, g - 1)
    return pd.DataFrame({"estimate": th, "se_jack": se, "ci_low": th - q * se, "ci_high": th + q * se,
                         "p_t": 2 * stats.t.sf(np.abs(t), g - 1), "n_scenarios": g}, index=theta.index)


def answer_domains(runs, sources):
    """Distinct domains per answer: all retrieved, corporate retrieved, corporate cited."""
    s = sources[sources.chat_id.isin(runs.chat_id)]
    corp = s[s.type == "CORPORATE"]
    per = pd.DataFrame({"domains_retrieved": s.groupby("chat_id").domain.nunique(),
                        "corp_domains_retrieved": corp.groupby("chat_id").domain.nunique(),
                        "corp_domains_cited": corp[corp.cited].groupby("chat_id").domain.nunique()})
    d = runs.merge(per, left_on="chat_id", right_index=True, how="left", validate="one_to_one")
    d[ARM_COLS[:3]] = d[ARM_COLS[:3]].fillna(0)
    # Definition: mean per-answer ratio over answers with >= 1 corporate domain
    d["corp_cited_given_retrieved"] = d.corp_domains_cited / d.corp_domains_retrieved.replace(0, np.nan)
    return d


def domain_stats(d, sources):
    """FEA+ vs FEA-: answer means per scenario x engine arm, within-scenario
    difference, scenario-equal then engine-equal; distinct corporate domains per
    scenario x engine arm (FEA+ set A, FEA- set B), Jaccard, |A \\ B|, and the share of
    FEA+ corporate retrievals (normalized URLs) that go to domains seen under FEA-.
    Conversion is undefined for an arm without corporate retrievals; such scenario x
    engine cells drop out of that engine's mean."""
    arm = d.groupby(["model_id", "scenario", "fea"])[ARM_COLS].mean()
    level = arm.groupby(["model_id", "fea"]).mean().unstack("fea")
    level.loc["all"] = level.mean()
    diff = (arm.xs(1., level="fea") - arm.xs(0., level="fea")).groupby("model_id").mean()
    diff.loc["all"] = diff.mean()
    aw = d.groupby(["model_id", "fea"])[ARM_COLS].mean().unstack("fea")
    aw.loc["all"] = d.groupby("fea")[ARM_COLS].mean().unstack()
    s = sources.merge(d[["chat_id", "model_id", "scenario", "fea"]], on="chat_id")
    s = s[s.type == "CORPORATE"]
    ident = {}
    for key, g in s.groupby(["model_id", "scenario"]):
        a, b = set(g.domain[g.fea == 1]), set(g.domain[g.fea == 0])
        ident[key] = {"distinct_corp_domains_fea0": len(b), "distinct_corp_domains_fea1": len(a),
                      "distinct_corp_domains_diff": len(a) - len(b),
                      "jaccard_corp_domains": len(a & b) / len(a | b),
                      "new_corp_domains_fea1": len(a - b),
                      "seen_share_fea1_corp_retrievals": g.domain[g.fea == 1].isin(b).mean()}
    cells = pd.MultiIndex.from_frame(d[["model_id", "scenario"]].drop_duplicates())
    ident = pd.DataFrame.from_dict(ident, orient="index").reindex(cells)
    count_cols = [c for c in ident if c.startswith(("distinct", "new"))]
    ident[count_cols] = ident[count_cols].fillna(0)  # no corporate source at all in the cell
    ident = ident.groupby(level=0).mean()
    ident.loc["all"] = ident.mean()
    out = {}
    for e in level.index:
        for m in ARM_COLS:
            out[(e, m, "fea0")], out[(e, m, "fea1")] = level.loc[e, (m, 0.)], level.loc[e, (m, 1.)]
            out[(e, m, "diff")] = diff.loc[e, m]
            out[(e, m, "fea0_answer_weighted")] = aw.loc[e, (m, 0.)]
            out[(e, m, "fea1_answer_weighted")] = aw.loc[e, (m, 1.)]
        for m, v in ident.loc[e].items():
            out[(e, m, "mean")] = v
    return pd.Series(out).rename_axis(["engine", "metric", "stat"])


def zero_citation_stats(d):
    """P(no citation), P(no source) in pp: scenario-equal within engine, engine-equal
    pooled; answer-weighted rates kept for the '30% of factorial answers' line."""
    x = d.assign(no_citation=100 * (1 - d.any_citation), no_source=100 * (1 - d.any_source))
    cols = ["no_citation", "no_source"]
    sc = x.groupby(["model_id", "scenario"])[cols].mean().groupby("model_id").mean()
    sc.loc["all"] = sc.mean()
    aw = x.groupby("model_id")[cols].mean()
    aw.loc["all"] = x[cols].mean()
    out = pd.concat({"rate": sc.stack(), "rate_answer_weighted": aw.stack()})
    return out.reorder_levels([1, 2, 0]).rename_axis(["engine", "metric", "stat"])


CHANNEL_OUTCOMES = {"citation_mass_CORPORATE": 100, "citation_n": 1, "any_citation": 100}


def channel_stats(d, api_days):
    """Per engine: API calendar days vs the engine's other days, means within
    scenario, scenario-equal. For Perplexity the API days are the sonar channel; for
    ChatGPT/Gemini they are same-day placebos. DiD = Perplexity minus placebo, per scenario."""
    x = d.assign(period=np.where(d.date.isin(api_days), "api_days", "other_days"))
    for c, scale in CHANNEL_OUTCOMES.items():
        x[c] = scale * x[c].astype(float)
    m = x.groupby(["model_id", "scenario", "period"])[list(CHANNEL_OUTCOMES)].mean().unstack("period")
    out = {}
    for c in CHANNEL_OUTCOMES:
        api, other = m[(c, "api_days")], m[(c, "other_days")]
        delta = (api - other).unstack("model_id")
        for e in delta:
            out[(e, c, "other_days")] = other.xs(e).mean()
            out[(e, c, "api_days")] = api.xs(e).mean()
            out[(e, c, "diff")] = delta[e].mean()
        for e in delta.columns.drop("perplexity-ui"):
            out[(f"perplexity-ui minus {e}", c, "did")] = (delta["perplexity-ui"] - delta[e]).mean()
    return pd.Series(out).rename_axis(["engine", "metric", "stat"])


def scenario_n_table(runs):
    """Runs, answers with >= 1 source / citation and corporate cited share (pp, mean over
    cited answers, answer-level SE, t CI) per scenario x engine x variant. 'all' =
    equal mean of the three engine means (NaN unless all three are defined), SE =
    sqrt(sum SE_e^2)/3, Welch-Satterthwaite df. CI truncated to [0, 100]."""
    r = runs.assign(variant=runs.cell + np.where(runs.is_reword, "-rw", ""),
                    y=100 * runs.citation_share_CORPORATE.astype(float))
    t = r.groupby(["scenario", "topic", "variant", "model_id"]).agg(
        runs=("chat_id", "size"), answers_with_source=("any_source", "sum"),
        answers_with_citation=("any_citation", "sum"), corp_cited_share_pp=("y", "mean"),
        sd=("y", "std"), n=("y", "count"))
    t[["answers_with_source", "answers_with_citation"]] = t[["answers_with_source", "answers_with_citation"]].astype(int)
    t["se_pp"] = t.sd / np.sqrt(t.n)
    t["df"] = t.n - 1
    g = t.groupby(level=["scenario", "topic", "variant"])
    a = g[["runs", "answers_with_source", "answers_with_citation"]].sum()
    a["corp_cited_share_pp"] = g.corp_cited_share_pp.mean().where(g.corp_cited_share_pp.count() == 3)
    v = t.se_pp ** 2
    a["se_pp"] = np.sqrt(v.groupby(level=["scenario", "topic", "variant"]).sum(min_count=3)) / 3
    a["df"] = (v.groupby(level=["scenario", "topic", "variant"]).sum(min_count=3) ** 2
               / (v ** 2 / t.df).groupby(level=["scenario", "topic", "variant"]).sum(min_count=3))
    a["model_id"] = "all"
    t = pd.concat([t.drop(columns=["sd", "n"]).reset_index(), a.reset_index()], ignore_index=True)
    q = stats.t.ppf(.975, t.df.where(t.df > 0))
    t["ci_low_pp"] = (t.corp_cited_share_pp - q * t.se_pp).clip(0, 100)
    t["ci_high_pp"] = (t.corp_cited_share_pp + q * t.se_pp).clip(0, 100)
    return t.rename(columns={"model_id": "engine"})


def _table(jk, labels, units):
    t = jk.reset_index()
    t.insert(0, "labels", labels)
    t.insert(4, "unit", t.metric.map(units))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--runs-pickle", default=None, help="frozen build_runs('ui') pickle")
    ap.add_argument("--runs-pickle-all", default=None, help="frozen build_runs('all') pickle")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, sources = load_runs(args.labels, "ui", args.runs_pickle)
    core = runs[runs.role == "factorial"]

    d = answer_domains(core, sources)
    dom = _table(scenario_jackknife(domain_stats, d, sources), args.labels,
                 lambda m: "fraction" if m.startswith(("jaccard", "seen", "corp_cited")) else "count")
    upsert(dom, OUT / "descriptives_domain_counts.csv")

    zero = _table(scenario_jackknife(zero_citation_stats, core), args.labels, lambda m: "pp")
    n = core.groupby("model_id").size()
    zero["runs"] = zero.engine.map({**n.to_dict(), "all": n.sum()})
    upsert(zero, OUT / "descriptives_zero_citation.csv")

    runs_all, _ = load_runs(args.labels, "all", args.runs_pickle_all)
    core_all = runs_all[runs_all.role == "factorial"]
    api_days = core_all.loc[core_all.channel.eq("api"), "date"].unique()
    ch = _table(scenario_jackknife(channel_stats, core_all, api_days), args.labels,
                lambda m: "count" if m == "citation_n" else "pp")
    ch["api_days"] = ";".join(sorted(pd.Series(api_days).dt.strftime("%Y-%m-%d")))
    upsert(ch, OUT / "descriptives_perplexity_channel.csv")

    nt = scenario_n_table(runs)
    nt.insert(0, "labels", args.labels)
    upsert(nt, OUT / "scenario_n_table.csv")

    show = dom[dom.engine.eq("all") | dom.stat.eq("diff")]
    print(show.round(3).to_string(index=False))
    print(zero.round(2).to_string(index=False))
    print(ch.round(2).to_string(index=False))
    neu = nt[(nt.variant == "NEU") & (nt.engine == "all")]
    print("NEU corporate cited share, pooled engines, over scenarios:",
          neu.corp_cited_share_pp.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
