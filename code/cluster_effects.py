"""Per-scenario (cluster) and per-topic factorial effects, all-variant baselines,
between-scenario heterogeneity. Granularity layer under revised_analysis.

The pooled estimand in revised_analysis is the equal-weight mean of 25 scenario
contrasts. This script reports those 25 contrasts (and the 5 topic means) with
their own uncertainty, tests whether scenarios differ beyond noise, and tabulates
the type shares of all 12 variants per scenario against the NEU baseline.

Per-scenario uncertainty: within a scenario the only replication is the repeated
run, so runs are the replication unit FOR THE PER-SCENARIO NUMBER ONLY. Runs of
one day share day shocks, so these intervals are somewhat narrow; they are for
reading heterogeneity and ranking, not for headline inference. Headline inference
stays at the scenario-cluster level (revised_analysis / inference.infer_mean).
Heterogeneity: DerSimonian-Laird random-effects on the 25 scenario contrasts
(Cochran Q, I^2, tau^2) with the per-scenario run-level variances as within
variances; topic moderation as Q_total - sum of within-topic Q (df = 4).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from inference import CELL_NAMES, FACTORS, TERMS, contrast_weights
from revised_analysis import ENGINES, OUT, TYPES, as_cube, build_runs, cell_means

FIG = Path(__file__).resolve().parent / "out" / "fig"
METRICS = {"citation_share": TYPES, "citation_mass": TYPES + ["NONE"], "source_share": TYPES}
VARIANTS = ["NEU", "NEU-rw", "NEU+FEA", "NEU+PRC", "NEU+FEA+PRC",
            "ADV", "ADV-rw", "ADV+FEA", "ADV+PRC", "ADV+FEA+PRC", "ADV+FEA+PRC-rw", "SEED"]


def scenario_contrasts(runs, metric, k):
    """25 scenario contrasts per engine (incl. 'all') and term, with run-level SE.

    Estimate: same code path as revised_analysis.analyze (cube @ contrast weights),
    so the scenario values average exactly to the pooled estimate. SE: pooled
    within-scenario x engine MSE over the eight cells, se^2 = MSE * sum_c w_c^2 / n_c
    with n_c = non-missing runs; 'all' = mean of three engine contrasts,
    se^2 = sum_e se_e^2 / 9.
    """
    col = f"{metric}_{k}"
    core = runs[runs.role == "factorial"]
    scenarios = sorted(core.scenario.unique())
    topics = core.groupby("scenario").topic.first()
    cells = cell_means(runs, [col])
    g = core.groupby(["scenario", "model_id", "cell"])[col].agg(["var", "count"])
    n = g["count"].unstack("cell").reindex(columns=CELL_NAMES)
    dof = (g["count"] - 1).clip(lower=0)
    mse = ((g["var"].fillna(0) * dof).groupby(level=[0, 1]).sum()
           / dof.groupby(level=[0, 1]).sum())
    rows, per_engine = [], {}
    for engine in ENGINES:
        cube = as_cube(cells, col, engine, scenarios)
        n_e = n.xs(engine, level="model_id").reindex(scenarios).to_numpy(dtype=float)
        mse_e = mse.xs(engine, level="model_id").reindex(scenarios).to_numpy()
        for term in TERMS:
            w = contrast_weights(term)
            est = cube @ w
            se = np.sqrt(mse_e * (w**2 / n_e).sum(axis=1))
            per_engine[(engine, term)] = (est, se)
            rows += [dict(scenario=s, topic=topics[s], engine=engine, metric=metric, type=k,
                          term=term, estimate=e, se=v, n_runs_used=int(nn), mse_run=m)
                     for s, e, v, nn, m in zip(scenarios, est, se, n_e.sum(axis=1), mse_e)]
    for term in TERMS:
        est = np.mean([per_engine[(e, term)][0] for e in ENGINES], axis=0)
        se = np.sqrt(sum(per_engine[(e, term)][1]**2 for e in ENGINES) / 9)
        n_all = n.groupby(level="scenario").sum().reindex(scenarios).sum(axis=1).to_numpy()
        rows += [dict(scenario=s, topic=topics[s], engine="all", metric=metric, type=k, term=term,
                      estimate=e, se=v, n_runs_used=int(nn), mse_run=np.nan)
                 for s, e, v, nn in zip(scenarios, est, se, n_all)]
    out = pd.DataFrame(rows)
    out["ci_low"] = out.estimate - 1.96 * out.se
    out["ci_high"] = out.estimate + 1.96 * out.se
    return out


def bh_q(p):
    """Benjamini-Hochberg q-values (monotone), NaN-safe."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]; n = len(pv); order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    q[np.flatnonzero(ok)[order]] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def add_p_q(scen):
    """Two-sided normal p from the run-level SE; BH q across the 25 scenarios of
    each engine x metric x type x term family. Both are for reading the per-scenario
    table, not for headline inference (which is scenario-cluster level)."""
    scen = scen.copy()
    z = scen.estimate / scen.se.replace(0, np.nan)
    scen["p"] = 2 * stats.norm.sf(np.abs(z))
    scen["q_bh"] = np.nan
    for _, idx in scen.groupby(["engine", "metric", "type", "term"]).groups.items():
        scen.loc[idx, "q_bh"] = bh_q(scen.loc[idx, "p"])
    return scen


def dl_heterogeneity(y, v):
    """DerSimonian-Laird: Q, df, p_Q, I2, tau2 for effects y with within variances v."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    if not (v > 0).any():  # outcome constant in every scenario (e.g. a type never cited)
        return dict(Q=np.nan, df=len(y) - 1, p_Q=np.nan, I2=np.nan, tau2=np.nan, tau_pp=np.nan, fe_mean=y.mean())
    v = np.where(v > 0, v, v[v > 0].min())  # a zero run variance gets the smallest observed one
    w = 1 / v
    ybar = (w * y).sum() / w.sum()
    Q = (w * (y - ybar)**2).sum()
    df = len(y) - 1
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0., (Q - df) / C) if C > 0 else 0.
    return dict(Q=Q, df=df, p_Q=stats.chi2.sf(Q, df), I2=max(0., (Q - df) / Q) if Q > 0 else 0.,
                tau2=tau2, tau_pp=100 * np.sqrt(tau2), fe_mean=ybar)


def heterogeneity(scen):
    """Per engine x metric x type x term: DL over 25 scenarios + topic moderation."""
    rows = []
    for key, d in scen.groupby(["engine", "metric", "type", "term"]):
        h = dl_heterogeneity(d.estimate, d.se**2)
        q_within = sum(dl_heterogeneity(t.estimate, t.se**2)["Q"] for _, t in d.groupby("topic"))
        q_topics = h["Q"] - q_within
        pooled = d.estimate.mean()
        rows.append(dict(zip(["engine", "metric", "type", "term"], key), n_scenarios=len(d),
                         pooled_estimate=pooled, sd_scenarios=d.estimate.std(ddof=1),
                         share_same_sign=(np.sign(d.estimate) == np.sign(pooled)).mean(),
                         share_ci_excludes_0=((d.ci_low > 0) | (d.ci_high < 0)).mean(),
                         **{k: h[k] for k in ("Q", "df", "p_Q", "I2", "tau2", "tau_pp")},
                         mean_within_var=(d.se**2).mean(),
                         Q_between_topics=q_topics, df_topics=d.topic.nunique() - 1,
                         p_between_topics=stats.chi2.sf(q_topics, d.topic.nunique() - 1)))
    return pd.DataFrame(rows)


def topic_effects(scen):
    """Topic means of scenario contrasts; t interval on the 5 scenarios (df = 4)."""
    rows = []
    for key, d in scen.groupby(["topic", "engine", "metric", "type", "term"]):
        h = dl_heterogeneity(d.estimate, d.se**2)
        m, se = d.estimate.mean(), d.estimate.std(ddof=1) / np.sqrt(len(d))
        tcrit = stats.t.ppf(.975, len(d) - 1)
        rows.append(dict(zip(["topic", "engine", "metric", "type", "term"], key), estimate=m, se=se,
                         ci_low=m - tcrit * se, ci_high=m + tcrit * se, n_scenarios=len(d),
                         p=2 * stats.t.sf(abs(m / se), len(d) - 1) if se > 0 else np.nan,
                         Q_within=h["Q"], I2_within=h["I2"]))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    for _, idx in out.groupby(["engine", "metric", "type", "term"]).groups.items():
        out.loc[idx, "q_bh"] = bh_q(out.loc[idx, "p"])
    return out


def variant_means(runs, columns):
    """Type shares of all 12 variants per scenario (incl. SEED and rewordings) vs NEU.

    Prompt-level mean, standard error of the mean over runs and run count per
    engine; 'all' = equal-weight mean of the three engine means. delta_vs_neu is
    against the scenario's original NEU prompt in the same engine and outcome.
    """
    r = runs.copy()
    r["variant"] = r.cell + np.where(r.is_reword, "-rw", "")
    keys = ["scenario", "topic", "prompt_id", "variant", "cell", "role", "is_reword", "template_id"]
    long = r.melt(id_vars=keys + ["model_id"], value_vars=columns, var_name="outcome", value_name="y")
    long["metric"] = long.outcome.str.rsplit("_", n=1).str[0]
    long["type"] = long.outcome.str.rsplit("_", n=1).str[1]
    per = long.groupby(keys + ["model_id", "metric", "type"], observed=True).y.agg(
        mean="mean", n="count", sd="std").reset_index()
    per["se"] = per.sd / np.sqrt(per.n)
    allrow = per.groupby(keys + ["metric", "type"], observed=True).agg(
        mean=("mean", "mean"), n=("n", "sum"), se=("se", lambda s: np.sqrt((s**2).sum()) / len(s))
    ).reset_index().assign(model_id="all")
    out = pd.concat([per.drop(columns="sd"), allrow], ignore_index=True).rename(columns={"model_id": "engine"})
    neu = out[(out.variant == "NEU")][["scenario", "engine", "metric", "type", "mean"]].rename(columns={"mean": "neu_mean"})
    out = out.merge(neu, on=["scenario", "engine", "metric", "type"], how="left")
    out["delta_vs_neu"] = out["mean"] - out["neu_mean"]
    return out


def variant_effects(var):
    """Mean delta vs NEU across the 25 scenarios per variant (t over scenarios, df 24);
    BH q across the 11 non-NEU variants within engine x metric x type."""
    d = var[var.variant != "NEU"]
    g = d.groupby(["engine", "metric", "type", "variant"]).delta_vs_neu.agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["t"] = g["mean"] / g.se.replace(0, np.nan)
    g["p"] = 2 * stats.t.sf(g.t.abs(), g["count"] - 1)
    g["q_bh"] = np.nan
    for _, idx in g.groupby(["engine", "metric", "type"]).groups.items():
        g.loc[idx, "q_bh"] = bh_q(g.loc[idx, "p"])
    return g.rename(columns={"mean": "mean_delta_vs_neu", "count": "n_scenarios"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, *_ = build_runs()
    pooled = pd.read_csv(OUT / "factor_effects.csv")[["engine", "metric", "type", "term", "estimate"]]
    pooled = pooled.rename(columns={"estimate": "pooled_estimate"})
    scen = pd.concat([scenario_contrasts(runs, m, k) for m, types in METRICS.items() for k in types],
                     ignore_index=True)
    scen = scen.merge(pooled, on=["engine", "metric", "type", "term"], how="left")
    # the 25 scenario contrasts must average to the published pooled estimate
    chk = scen.groupby(["engine", "metric", "type", "term"]).agg(m=("estimate", "mean"), p=("pooled_estimate", "first"))
    assert np.allclose(chk.m, chk.p, atol=1e-9), "scenario contrasts do not average to pooled estimates"
    scen["sign_conflict"] = ((scen.ci_low > 0) & (scen.pooled_estimate < 0)) | ((scen.ci_high < 0) & (scen.pooled_estimate > 0))
    scen = add_p_q(scen)
    scen.to_csv(OUT / "scenario_effects.csv", index=False)
    topic_effects(scen).to_csv(OUT / "topic_effects.csv", index=False)
    het = heterogeneity(scen)
    het.to_csv(OUT / "heterogeneity.csv", index=False)
    columns = [f"{m}_{k}" for m, types in METRICS.items() for k in types]
    var = variant_means(runs, columns)
    var["z_vs_neu"] = var.delta_vs_neu / np.sqrt(var.se**2 + var.merge(
        var[var.variant == "NEU"][["scenario", "engine", "metric", "type", "se"]].rename(columns={"se": "se_neu"}),
        on=["scenario", "engine", "metric", "type"], how="left").se_neu**2).replace(0, np.nan)
    var.to_csv(OUT / "scenario_variant_means.csv", index=False)
    variant_effects(var).to_csv(OUT / "variant_effects.csv", index=False)
    show = het.query("engine == 'all' and metric == 'citation_share' and type in ['CORPORATE','EDITORIAL'] and term in @FACTORS")
    print(show[["type", "term", "pooled_estimate", "sd_scenarios", "share_same_sign", "share_ci_excludes_0",
                "I2", "tau_pp", "p_Q", "p_between_topics"]].round(3).to_string(index=False))
    print(f"Wrote scenario_effects / topic_effects / heterogeneity / scenario_variant_means -> {OUT}")


if __name__ == "__main__":
    main()
