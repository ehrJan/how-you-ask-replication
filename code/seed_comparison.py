"""Natural seed prompt vs the matched full template (ADV+FEA+PRC, factorial role).

Answers with >= 1 citation. Per scenario x engine: mean cited share of the seed minus mean
cited share of the full template (all collection days pooled, no day matching); engines
averaged equally within scenario, scenarios equally; leave-one-scenario-out jackknife
with t(G-1). Estimates in pp; full_pct / seed_pct are the two cited-share levels under
the same weighting.

Output -> out/revised/seed_comparison.csv (rows keyed by `labels`)
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from descriptives import OUT, load_runs, upsert

TYPES = ["CORPORATE", "EDITORIAL", "UGC"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], default="jev")
    ap.add_argument("--runs-pickle", default=None, help="frozen build_runs('ui') pickle")
    args = ap.parse_args()
    runs, _ = load_runs(args.labels, "ui", args.runs_pickle)
    a = runs[(runs.role == "seed-natural") | ((runs.role == "factorial") & (runs.cell == "ADV+FEA+PRC"))]
    a = a[a.citation_n > 0].assign(seed=lambda d: (d.role == "seed-natural").astype(int))
    rows = []
    for k in TYPES:
        m = a.groupby(["scenario", "model_id", "seed"])[f"citation_share_{k}"].mean().unstack("seed")
        m = m.dropna()
        level = m.groupby("scenario").mean().mean()  # same weighting: engines, then scenarios
        d = (m[1] - m[0]).groupby("scenario").mean()
        G = len(d)
        jk = np.array([d.drop(s).mean() for s in d.index])
        se = np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())
        q = stats.t.ppf(.975, G - 1)
        rows.append(dict(labels=args.labels, type=k, full_pct=100 * level[0], seed_pct=100 * level[1],
                         estimate_pp=100 * d.mean(),
                         ci_low=100 * (d.mean() - q * se), ci_high=100 * (d.mean() + q * se),
                         p=2 * stats.t.sf(abs(d.mean() / se), G - 1), n_scenarios=G,
                         n_seed=int(a.seed.sum()), n_full=int((1 - a.seed).sum())))
    R = pd.DataFrame(rows)
    upsert(R, OUT / "seed_comparison.csv")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
