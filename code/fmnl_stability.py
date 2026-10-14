"""RQ4: stability of the FMNL factor effects, and the holdout check.

(a) engine: pairwise engine differences of the primary cited-composition AMEs, from the
    primary fit's saved scenario-jackknife replicates; joint F per factor over
    engine pairs x non-reference types.
(b) period: grp = engine|period, late − early; split 2026-09-06 (ChatGPT, Perplexity) and
    2026-09-05 for Gemini (its engine-side shift); per engine + engine-equal pool.
(c) channel: Perplexity API days − UI days, build_runs("all"), grp = engine|channel.
(d) holdout_check: pre-stated success rule on the holdout-only fit, plus holdout − discovery
    (grp = engine|window) with TOST. `--holdout` refuses to run without holdout dates.

Every contrast: scenario jackknife t(24) CI; gate = joint rank-adjusted F per factor ->
Holm over the three factors -> Holm over the tested types inside a passing factor;
TOST ±5 pp (equivalent iff the 90% t(24) CI lies inside ±5 pp).
Outputs out/fmnl/rq4/<labels>_{engine,period,channel}.csv, out/fmnl/holdout/holdout_check.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fmnl
from fmnl_stage import fit_jack, gated, infer, joint_F, load_runs, series
from inference import FACTORS, holm

HERE = Path(__file__).resolve().parent
ENGINES = fmnl.ENGINES
MARGIN = 5.0
SPLIT = {"chatgpt-ui": "2026-09-06", "perplexity-ui": "2026-09-06", "gemini-ui": "2026-09-05"}
PAIRS = [("chatgpt-ui", "gemini-ui"), ("chatgpt-ui", "perplexity-ui"),
         ("gemini-ui", "perplexity-ui")]
RULE = ("holdout, JEV, pooled: FEA joint F p < .05 and FEA -> CORPORATE AME > 0 "
        "with the 95% t(24) CI excluding 0")


def combine(point, reps, weights):
    """Linear unit contrasts {name: {unit: weight}} -> (point, reps) with unit = name."""
    p = pd.concat({n: sum(w * point[u] for u, w in ws.items()) for n, ws in weights.items()},
                  names=["unit"])
    r = pd.concat({n: sum(w * reps[u] for u, w in ws.items()) for n, ws in weights.items()},
                  axis=1)
    return p, r


def level_diff(hi, lo, engines):
    """engine|hi − engine|lo per engine, plus the engine-equal pool when > 1 engine."""
    w = {e: {f"{e}|{hi}": 1.0, f"{e}|{lo}": -1.0} for e in engines}
    if len(engines) > 1:
        w["all"] = {u: v / len(engines) for e in engines for u, v in w[e].items()}
    return w


def tost(table, margin=MARGIN):
    """TOST at alpha .05: equivalent iff the 90% t(G-1) CI lies inside ±margin."""
    t = table.copy()
    df = t.n_scenarios - 1
    q = stats.t.ppf(.95, df)
    t["ci90_low"], t["ci90_high"] = t.estimate - q * t.se_jack, t.estimate + q * t.se_jack
    se = t.se_jack.to_numpy()
    nl, nu = (t.estimate + margin).to_numpy(), (t.estimate - margin).to_numpy()
    zl = np.divide(nl, se, out=np.copysign(np.inf, nl), where=se > 0)
    zu = np.divide(nu, se, out=np.copysign(np.inf, nu), where=se > 0)
    t["p_tost"] = np.maximum(stats.t.sf(zl, df), stats.t.cdf(zu, df))
    t["equivalent"] = (t.ci90_low > -margin) & (t.ci90_high < margin)
    t["m_min"] = t.estimate.abs() + q * t.se_jack
    t["margin"] = margin
    return t


def contrast_table(point, reps, weights, types, tested):
    """Differences + gate (per unit) + TOST, with both sides' levels and the joint F."""
    p, r = combine(point, reps, weights)
    table, joint = gated(p, r, types, tested)
    for side, keep in (("effect_hi", 1), ("effect_lo", -1)):
        s = combine(point, reps, {n: {u: keep * w for u, w in ws.items() if w * keep > 0}
                                  for n, ws in weights.items()})[0]
        table[side] = s.reindex(pd.MultiIndex.from_frame(table[["unit", "type", "factor"]])).to_numpy()
    return tost(table.merge(joint[["unit", "factor", "F", "df1", "df2", "p_F"]],
                            on=["unit", "factor"], validate="many_to_one"))


def engine_equality(point, reps, types, tested):
    """Pairwise engine differences; one joint F per factor over pairs x non-reference types."""
    w = {f"{a} - {b}": {a: 1.0, b: -1.0} for a, b in PAIRS}
    p, r = combine(point, reps, w)
    table = infer(p, r)
    joint = joint_F(p, r, {("engines", f): [(n, k, f) for n in w for k in types[:-1]]
                           for f in FACTORS})
    joint["gate_p"] = holm(joint.p_F)
    table = table.merge(joint[["factor", "F", "df1", "df2", "p_F", "gate_p"]], on="factor")
    table["gate_pass"] = table.gate_p < .05
    table["tested"] = table.type.isin(tested)
    table["p_within"] = np.nan
    for f, part in table[table.tested & table.gate_pass].groupby("factor"):
        table.loc[part.index, "p_within"] = holm(part.p_t)   # family: tested types x pairs
    return tost(table)


def add_period(runs):
    split = pd.to_datetime(runs.model_id.map(SPLIT))
    if split.isna().any():
        raise ValueError("Engine without a period split")
    return runs.assign(period=np.where(runs.date >= split, "late", "early"))


def primary(labels, runs):
    """Primary citation fit: fmnl_run's saved jackknife replicates when its saved point AMEs
    equal this data's to 1e-9 pp (same data, same code); a fresh jackknife otherwise."""
    frame = fmnl.long_frame(runs, "citation")
    d = HERE / "out" / "fmnl" / f"{labels}_citation"
    if (d / "jack_reps.csv.gz").exists():
        point = series(fmnl.ame_raw(fmnl.fit_fmnl(frame), frame))
        saved = pd.read_csv(d / "ame.csv").set_index(["engine", "type", "factor"]).ame_pp
        if np.allclose(saved.reindex(point.index), point, rtol=0, atol=1e-9):
            reps = pd.read_csv(d / "jack_reps.csv.gz", header=[0, 1, 2], index_col=0)
            return frame, point, reps, f"saved {d.name}/jack_reps.csv.gz"
    return (frame, *fit_jack(frame), "recomputed")


def success_rule(table, joint, labels):
    j = joint.set_index(["unit", "factor"]).p_F[("all", "fea")]
    a = table.set_index(["unit", "type", "factor"]).loc[("all", "CORPORATE", "fea")]
    ok = bool(j < .05 and a.estimate > 0 and a.ci_low > 0)
    return dict(section="success_rule", unit="all", type="CORPORATE", factor="fea",
                estimate=a.estimate, ci_low=a.ci_low, ci_high=a.ci_high, p_F=j,
                verdict="PASS" if ok else "FAIL", confirmatory=labels == "jev", rule=RULE)


def holdout_eval(frame_h, frame_w, labels):
    """(i) success rule + gate on the holdout-only fit; (ii) holdout − discovery TOST."""
    types, tested = frame_h.attrs["types"], fmnl.tested_types("citation")
    point, reps = fit_jack(frame_h)
    table, joint = gated(point, reps, types, tested)
    table = table.merge(joint[["unit", "factor", "F", "df1", "df2", "p_F"]], on=["unit", "factor"])
    pw, rw = fit_jack(frame_w)
    diff = contrast_table(pw, rw, level_diff("holdout", "discovery", ENGINES), types, tested)
    d = diff.set_index(["unit", "type", "factor"]).loc[("all", "CORPORATE", "fea")]
    verdicts = [success_rule(table, joint, labels),
                dict(section="stability_tost", unit="all", type="CORPORATE", factor="fea",
                     estimate=d.estimate, ci90_low=d.ci90_low, ci90_high=d.ci90_high,
                     p_tost=d.p_tost, confirmatory=False,
                     verdict="EQUIVALENT" if d.equivalent else "NOT_EQUIVALENT",
                     rule=f"holdout − discovery FEA -> CORPORATE, 90% CI inside ±{MARGIN:g} pp")]
    out = pd.concat([pd.DataFrame(verdicts), table.assign(section="holdout_gate"),
                     diff.assign(section="holdout_minus_discovery")], ignore_index=True)
    return out.assign(labels=labels, n_holdout_answers=len(frame_h),
                      holdout_days=frame_h.day.nunique())


def holdout_check(runs_full, labels="jev"):
    """runs with `window` in {discovery, holdout} -> holdout_check table (see holdout_eval)."""
    frame_h = fmnl.long_frame(runs_full[runs_full.window.eq("holdout")], "citation")
    frame_w = fmnl.long_frame(runs_full, "citation", grp="window")
    return holdout_eval(frame_h, frame_w, labels)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--runs-pickle", help="frozen build_runs('ui') pickle")
    ap.add_argument("--runs-all-pickle", help="frozen build_runs('all') pickle (channel)")
    ap.add_argument("--holdout", action="store_true", help="holdout check only")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.holdout:
        runs = load_runs(args.labels, args.runs_pickle, window="full")
        from revised_analysis import DISCOVERY_END
        if "window" not in runs or not ((runs.date > pd.Timestamp(DISCOVERY_END)).any()
                                         and runs.window.eq("holdout").any()):
            raise SystemExit(f"Refusing: no holdout dates (> {DISCOVERY_END}) in the runs")
        out = Path(args.out or HERE / "out" / "fmnl" / "holdout")
        out.mkdir(parents=True, exist_ok=True)
        res = holdout_check(runs, args.labels)
        name = "holdout_check.csv" if args.labels == "jev" else f"holdout_check_{args.labels}.csv"
        res.to_csv(out / name, index=False)
        print(res[res.section.isin(["success_rule", "stability_tost"])].to_string(index=False))
        return
    out = Path(args.out or HERE / "out" / "fmnl" / "rq4")
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.labels, args.runs_pickle)
    tested = fmnl.tested_types("citation")
    frame, point, reps, source = primary(args.labels, runs)
    types = frame.attrs["types"]
    res = {"engine": engine_equality(point, reps, types, tested).assign(reps_source=source)}
    fp = fmnl.long_frame(add_period(runs), "citation", grp="period")
    res["period"] = contrast_table(*fit_jack(fp), level_diff("late", "early", ENGINES),
                                   types, tested).assign(split=lambda t: t.unit.map(SPLIT))
    runs_all = load_runs(args.labels, args.runs_all_pickle, channel="all")
    fc = fmnl.long_frame(runs_all, "citation", grp="channel")
    res["channel"] = contrast_table(*fit_jack(fc), level_diff("api", "ui", ["perplexity-ui"]),
                                    types, tested)
    for name, t in res.items():
        t.assign(labels=args.labels, contrast=name).to_csv(out / f"{args.labels}_{name}.csv",
                                                           index=False)
        show = t[t.type.isin(tested)]
        print(f"\n== {name}\n" + show[["unit", "type", "factor", "estimate", "ci_low", "ci_high",
                                       "p_F", "gate_p", "gate_pass", "p_within", "ci90_low",
                                       "ci90_high", "equivalent"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
