"""Exploratory Lee sensitivity bounds on balanced original factorial prompts.

Bounds target the always-selected principal stratum under individual-level
monotonicity of selection. The observed sign of the selection-rate difference
does not prove monotonicity. Report these as assumption-dependent sensitivity
analyses, never as a correction for all selection bias. Recompute bounds across
scenario bootstrap draws and provide an outer percentile uncertainty interval.
This interval is exploratory (especially near zero selection difference).

Rows: every engine x factor x tested type (CORPORATE, EDITORIAL, UGC), for the cited
composition (selection = any citation) and the retrieved composition (selection = any
source), one label set per process (`--labels`, rows upserted by `labels`). The
monotonicity sign check records the direction of the factor's effect on P(selected),
pooled and per scenario (same / opposite sign, each also significant at 5%, z test).
The eight original cells have equal replication, so the distributions of the
other factors and scenarios agree in the two marginal treatment arms.

Output -> out/revised/lee_bounds.csv (supersedes lee_sensitivity.csv).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from descriptives import OUT, load_runs, upsert
from inference import FACTORS

TESTED = ["CORPORATE", "EDITORIAL", "UGC"]
SELECTION = {"citation_share": "any_citation", "source_share": "any_source"}


def tail_means(values, weights, mass, upper=False):
    """Exact fractional trimming of a discrete weighted distribution.

    Unlike dropping floor(q*n) observations, this handles ties and a fractional
    observation at the cutoff without changing the intended trimming mass.
    """
    y, w = np.asarray(values), np.atleast_2d(weights).astype(float)
    if upper:
        y, w = y[::-1], w[:, ::-1]
    prior = np.cumsum(w, axis=1) - w
    included = np.minimum(w, np.maximum(0., np.asarray(mass)[:, None] - prior))
    return np.divide(included @ y, mass, out=np.full(len(w), np.nan), where=mass > 0)


def bounds_from_hist(y, h0, h1, n0, n1):
    """Vectorized bounds; lower/upper orientation remains correct for p1<p0."""
    h0, h1 = np.atleast_2d(h0), np.atleast_2d(h1)
    n0, n1 = np.atleast_1d(n0), np.atleast_1d(n1)
    selected0, selected1 = h0.sum(axis=1), h1.sum(axis=1)
    p0, p1 = selected0 / n0, selected1 / n1
    rising = p1 >= p0
    high = np.where(rising[:, None], h1, h0)
    low = np.where(rising[:, None], h0, h1)
    low_mean = np.divide(low @ y, low.sum(axis=1), out=np.full(len(low), np.nan),
                         where=low.sum(axis=1) > 0)
    p_hi, p_lo = np.maximum(p0, p1), np.minimum(p0, p1)
    retained = np.divide(p_lo, p_hi, out=np.zeros(len(p_hi)), where=p_hi > 0)
    mass = retained * high.sum(axis=1)
    lower_hi = tail_means(y, high, mass)
    upper_hi = tail_means(y, high, mass, upper=True)
    lower = np.where(rising, lower_hi - low_mean, low_mean - upper_hi)
    upper = np.where(rising, upper_hi - low_mean, low_mean - lower_hi)
    return lower, upper, p0, p1


def one_bounds(d, column, factor, scenarios, bootstrap_counts):
    y = np.sort(d[column].dropna().unique())
    if not len(y):
        raise ValueError("No selected outcomes for bounds")
    hist, totals = [], []
    for arm in (0, 1):
        sub = d[d[factor] == arm]
        h = pd.crosstab(sub.scenario, sub[column]).reindex(
            index=scenarios, columns=y, fill_value=0).to_numpy(dtype=float)
        n = sub.groupby("scenario").size().reindex(scenarios).to_numpy(dtype=float)
        hist.append(h)
        totals.append(n)
    h0, h1 = hist[0].sum(axis=0), hist[1].sum(axis=0)
    lo, hi, p0, p1 = bounds_from_hist(y, h0, h1, totals[0].sum(), totals[1].sum())
    blo, bhi, _, _ = bounds_from_hist(y, bootstrap_counts @ hist[0], bootstrap_counts @ hist[1],
                                     bootstrap_counts @ totals[0], bootstrap_counts @ totals[1])
    if not np.isfinite(blo).all() or not np.isfinite(bhi).all():
        raise ValueError("Bootstrap has an empty selected arm; report unidentified bounds")
    return dict(selection_0=p0.item(), selection_1=p1.item(),
        assumed_selection_direction="increasing" if p1.item() >= p0.item() else "decreasing",
        naive_diff=(h1 @ y) / h1.sum() - (h0 @ y) / h0.sum(),
        lee_lower=lo.item(), lee_upper=hi.item(),
        outer_ci_low=np.quantile(blo, .025), outer_ci_high=np.quantile(bhi, .975),
        n_scenarios=len(scenarios), inference="scenario_percentile_outer_exploratory",
        identification="requires_individual_monotonicity_not_verified")


def selection_sign_check(d, factor, selected="any_citation"):
    """Direction of the factor's effect on P(selected): pooled (the direction the bounds
    assume), scenario-mean with t(G-1) p, and per-scenario agreement. Agreement is
    necessary for, not proof of, individual-level monotonicity."""
    p = d.groupby(["scenario", factor])[selected].agg(["mean", "size"]).unstack(factor)
    m0, m1, n0, n1 = p[("mean", 0.)], p[("mean", 1.)], p[("size", 0.)], p[("size", 1.)]
    diff = m1 - m0
    z = diff / np.sqrt(m1 * (1 - m1) / n1 + m0 * (1 - m0) / n0)
    pooled = d.loc[d[factor] == 1, selected].mean() - d.loc[d[factor] == 0, selected].mean()
    agree = np.sign(diff) * np.sign(pooled)  # 0 where either difference is exactly 0
    same, opposite = agree > 0, agree < 0
    with np.errstate(invalid="ignore", divide="ignore"):  # no selection variation at all
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    sig = z.abs() > stats.norm.ppf(.975)
    return dict(selection_diff_pp=100 * pooled, selection_diff_scenario_mean_pp=100 * diff.mean(),
                selection_diff_p_t=2 * stats.t.sf(abs(t), len(diff) - 1),
                n_scen_same_sign=int(same.sum()), n_scen_opposite_sign=int(opposite.sum()),
                n_scen_same_sig05=int((same & sig).sum()), n_scen_opposite_sig05=int((opposite & sig).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1999)
    parser.add_argument("--labels", choices=["jev"], required=True)
    parser.add_argument("--runs-pickle", default=None, help="frozen build_runs('ui') pickle")
    args = parser.parse_args()
    runs, _ = load_runs(args.labels, "ui", args.runs_pickle)
    from revised_analysis import ENGINES, SEED
    d = runs[runs.role == "factorial"]
    scenarios = sorted(d.scenario.unique())
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(scenarios), np.full(len(scenarios), 1 / len(scenarios)), args.reps)
    rows = []
    for eng in ENGINES:
        frame = d[d.model_id == eng]
        for metric, selected in SELECTION.items():
            for factor in FACTORS:
                check = selection_sign_check(frame, factor, selected)
                for k in TESTED:
                    rows.append(dict(labels=args.labels, engine=eng, metric=metric, type=k, factor=factor,
                        **one_bounds(frame, f"{metric}_{k}", factor, scenarios, counts), **check))
    res = pd.DataFrame(rows)
    pp = ["naive_diff", "lee_lower", "lee_upper", "outer_ci_low", "outer_ci_high"]
    res[pp] *= 100
    res = res.rename(columns={c: f"{c}_pp" for c in pp})
    OUT.mkdir(parents=True, exist_ok=True)
    upsert(res, OUT / "lee_bounds.csv")
    print(res.query("metric == 'citation_share' and engine != 'chatgpt-ui'")
          [["engine", "type", "factor", "selection_diff_pp", "n_scen_same_sig05", "n_scen_opposite_sig05", "naive_diff_pp",
            "lee_lower_pp", "lee_upper_pp", "outer_ci_low_pp", "outer_ci_high_pp"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
