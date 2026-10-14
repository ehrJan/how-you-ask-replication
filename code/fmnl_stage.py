"""RQ3: where does a factor effect enter, retrieval or citation? (FMNL, same answers)

Frame A = retrieved composition (`source_*` shares) of the factorial answers with at
least one citation; frame B = the cited composition of the same answers. The cited minus
retrieved AME is the filtering-stage effect. Its jackknife covariance pairs the two
leave-one-scenario-out replicate matrices by left-out scenario: both frames hold the same
answers, so this equals one loop refitting both. Gate per unit (pre-specified): rank-adjusted
F on the difference over the non-reference types -> Holm over the three factors -> Holm
over the tested types inside a passing factor.

Companions: the any-source retrieval fit ("what enters": every answer with >= 1 source),
and P(cited | retrieved) by type and factor arm. That last table is DESCRIPTIVE only:
conditioning on retrieval conditions on a post-treatment variable (a collider), so an arm
difference there is no causal effect of the factor on citation filtering.

Outputs out/fmnl/rq3/<labels>_{ame,joint,cite_rate}.csv.
Usage: fmnl_stage.py --labels jev|peec [--runs-pickle PATH] [--out DIR]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fmnl
from inference import FACTORS

HERE = Path(__file__).resolve().parent
COLLIDER = "descriptive: conditions on retrieval (post-treatment, collider); no causal reading"


def series(raw):
    s = pd.Series(raw, dtype=float)
    s.index.names = ["unit", "type", "factor"]
    return s


def fit_jack(frame):
    """Point AMEs of the full-sample fit and leave-one-scenario-out replicates, with the
    primary pipeline's guards (convergence; RuntimeError if a tested-type AME moves along
    a null direction). Flags on non-tested types stay in the fmnl layer: reps.attrs is
    cleared because pandas cannot carry a DataFrame attr through concat."""
    _, point, sens = fmnl._refit(frame, False, "full sample")
    fmnl._unidentified(fmnl._flag("full", sens))
    reps = fmnl.jackknife(frame)[1]
    reps.attrs.clear()
    return series(point), reps


def infer(point, reps):
    """Per (unit, type, factor): estimate, jackknife SE, t(G-1) 95% CI, two-sided p."""
    g = len(reps)
    r = reps[point.index]
    se = np.sqrt((g - 1) / g * ((r - r.mean()) ** 2).sum()).to_numpy()
    est = point.to_numpy()
    t = np.divide(est, se, out=np.zeros_like(est), where=se > 0)
    q = stats.t.ppf(.975, g - 1)
    return pd.DataFrame({"estimate": est, "se_jack": se, "ci_low": est - q * se,
                         "ci_high": est + q * se, "p_t": 2 * stats.t.sf(np.abs(t), g - 1),
                         "n_scenarios": g}, index=point.index).reset_index()


def joint_F(point, reps, groups):
    """groups {(unit, factor): [column keys]} -> rank-adjusted Hotelling F per group."""
    return pd.DataFrame([dict(unit=u, factor=f, **fmnl.wald_F(point.loc[keys].to_numpy(),
                                                              reps[keys].to_numpy()))
                         for (u, f), keys in groups.items()])


def gated(point, reps, types, tested):
    """Inference + per unit x factor joint F over the non-reference types + gatekeeping."""
    units = point.index.get_level_values("unit").unique()
    joint = joint_F(point, reps, {(u, f): [(u, k, f) for k in types[:-1]]
                                  for u in units for f in FACTORS})
    return fmnl.gatekeep(infer(point, reps), joint, tested), joint


def stage(frame_a, frame_b):
    """Point AMEs and replicates of both stages; B − A is the filtering-stage effect."""
    if not frame_a.chat_id.equals(frame_b.chat_id):
        raise ValueError("Stage frames must hold the same answers in the same order")
    if frame_a.attrs["types"] != frame_b.attrs["types"]:
        raise ValueError("Stage frames must share the type vocabulary")
    pa, ra = fit_jack(frame_a)
    pb, rb = fit_jack(frame_b)
    if not ra.index.equals(rb.index):
        raise ValueError("Jackknife replicates are not paired by left-out scenario")
    return {"retrieved": (pa, ra), "cited": (pb, rb), "cited_minus_retrieved": (pb - pa, rb - ra)}


def cite_rate(runs, types):
    """P(cited | retrieved) = cited / retrieved sources, per engine x type x factor arm."""
    d = runs[runs.role.eq("factorial")]
    rows = []
    for f in FACTORS:
        g = d.groupby(["model_id", f])[[f"{p}_n_{k}" for p in ("source", "citation")
                                        for k in types]].sum()
        for k in types:
            n = g[f"source_n_{k}"].unstack(f)
            rate = (g[f"citation_n_{k}"] / g[f"source_n_{k}"].replace(0, np.nan)).unstack(f)
            rate.loc["all"], n.loc["all"] = rate.mean(), n.sum()     # all = engine-equal mean
            for u in rate.index:
                rows.append(dict(unit=u, type=k, factor=f, p_cited_off=rate.loc[u, 0],
                                 p_cited_on=rate.loc[u, 1],
                                 diff_pp=100 * (rate.loc[u, 1] - rate.loc[u, 0]),
                                 n_retrieved_off=int(n.loc[u, 0]),
                                 n_retrieved_on=int(n.loc[u, 1]), note=COLLIDER))
    return pd.DataFrame(rows)


def load_runs(labels, runs_pickle, channel="ui", **window):
    """Frozen pickle (dict with `runs`) or a fresh build_runs(channel, **window)."""
    if runs_pickle:
        if labels not in Path(runs_pickle).name:
            raise SystemExit(f"{runs_pickle} does not look like a {labels} runs pickle")
        with open(runs_pickle, "rb") as fh:
            return pickle.load(fh)["runs"]
    if labels == "jev":
        sys.path.insert(0, str(HERE))
    import revised_analysis
    for m in ("revised_analysis", "design"):
        if labels == "jev" and Path(sys.modules[m].__file__).parent != HERE:
            raise RuntimeError(f"{m} was imported from another directory; one label set per process")
    return revised_analysis.build_runs(channel, **window)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--runs-pickle")
    ap.add_argument("--out", default=str(HERE / "out" / "fmnl" / "rq3"))
    args = ap.parse_args(argv)
    runs = load_runs(args.labels, args.runs_pickle)
    frame_b = fmnl.long_frame(runs, "citation")
    frame_a = fmnl.long_frame(runs, "retrieval", sample="citation")
    frame_r = fmnl.long_frame(runs, "retrieval")
    types, tested = frame_b.attrs["types"], fmnl.tested_types("citation")
    parts = stage(frame_a, frame_b)
    parts["retrieved_any_source"] = fit_jack(frame_r)
    tables, joints = [], []
    for name, (point, reps) in parts.items():
        t, j = gated(point, reps, types, tested)
        n = len(frame_r if name == "retrieved_any_source" else frame_b)
        tables.append(t.assign(stage=name, n_answers=n))
        joints.append(j.assign(stage=name, n_answers=n))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True).assign(labels=args.labels)
    joint = pd.concat(joints, ignore_index=True).assign(labels=args.labels)
    table.to_csv(out / f"{args.labels}_ame.csv", index=False)
    joint.to_csv(out / f"{args.labels}_joint.csv", index=False)
    cite_rate(runs, types).assign(labels=args.labels).to_csv(
        out / f"{args.labels}_cite_rate.csv", index=False)
    show = table[table.unit.eq("all") & table.type.isin(tested)]
    print(show.drop(columns=["labels"]).round(3).to_string(index=False))
    print(joint[joint.unit.eq("all")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
