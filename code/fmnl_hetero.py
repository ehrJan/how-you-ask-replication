"""RQ2: does product category moderate the factor effects? Two-stage random coefficients on the FMNL.

Stage 1: FMNL with scenario-specific cell effects (`long_frame(..., grp="scenario")`, units
engine|scenario) -> per-scenario AMEs (pp) per engine and tested type; `all` = engine-equal
mean. Sampling variance: leave-one-day-out jackknife, each engine over its own collection days
(a left-out day without its answers leaves its estimate unchanged); `all` over all days.
Stage 2, per unit x type x factor: REML random-effects meta-regression
AME_s = topic_s + u_s + e_s, e_s ~ N(0, v_s) from stage 1, u_s ~ N(0, tau2), Knapp-Hartung vcov
and F(T-1, G-T) for topic moderation; intercept-only model: mu, tau, generalised I2 and the 95%
prediction interval mu +- t_{G-1} sqrt(tau2 + SE_KH^2) for a new scenario. DerSimonian-Laird
(`cluster_effects.dl_heterogeneity`) as comparison.
Diagnostic: SD over scenarios of the per-scenario AMEs that the common-beta primary fit implies
(one logit effect on baselines 5-87% corporate) vs tau -> share of pp heterogeneity that is
baseline mechanics, plus the stage-2 meta-regression on those mechanical AMEs.
Multiplicity (own family, gate, per unit; `all` confirmatory): factor gate
p = min(1, T min_k p_topic) (Bonferroni: stage 2 has no cross-type covariance), Holm over the
factors, Holm over the tested types inside a passing factor; BH q on the per-scenario AMEs.

Output: out/fmnl/rq2/<labels>_{stage1,stage2,diagnostic}.csv + <labels>_meta.json.
"""
import multiprocessing as mp
import os
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import fmnl
from inference import holm

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "fmnl" / "rq2"
OUTCOME = "citation"
ALPHA = .05
KEY = ["engine", "type", "factor"]


def meta_reml(y, v, X=None):
    """REML random-effects meta-regression y = X b + u + e, u ~ N(0, tau2), e ~ N(0, v), v known.

    tau2 = root of the REML score y'PPy - tr(P) (brentq on a bracket; 0 when the score at 0 is
    <= 0). Fisher scoring, metafor's default, 2-cycles at the boundary. Knapp-Hartung vcov
    s2 (X'WX)^-1 with s2 = sum w e^2 / (G - p), untruncated (metafor test="knha").
    I2 = tau2 / (tau2 + v_typ), v_typ = (G - p) / tr(P0) (Higgins-Thompson, generalised).
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = np.ones((len(y), 1)) if X is None else np.asarray(X, float)
    g, p = X.shape

    def proj(t2):
        wx = X / (v + t2)[:, None]
        a = np.linalg.inv(X.T @ wx)
        return np.diag(1 / (v + t2)) - wx @ a @ wx.T, a, wx

    def score(t2):
        P = proj(t2)[0]
        py = P @ y
        return py @ py - np.trace(P)

    t2 = 0.0
    if score(0.0) > 0:
        hi = 2 * np.var(y)
        while score(hi) > 0:
            hi *= 2
        t2 = optimize.brentq(score, 0.0, hi)
    p0 = proj(0.0)[0]
    q_e = float(y @ p0 @ y)
    _, a, wx = proj(t2)
    beta = a @ wx.T @ y
    s2 = float(((y - X @ beta) ** 2 / (v + t2)).sum()) / (g - p)
    return dict(beta=beta, vcov=s2 * a, s2=s2, tau2=t2, df=g - p, Q=q_e, p_Q=float(stats.chi2.sf(q_e, g - p)),
                I2=t2 / (t2 + (g - p) / np.trace(p0)))


def kh_f(fit, L):
    """Knapp-Hartung F test of L b = 0 -> (F, rank L, G - p, p)."""
    L = np.atleast_2d(L)
    b = L @ fit["beta"]
    F = float(b @ np.linalg.solve(L @ fit["vcov"] @ L.T, b)) / len(L)
    return F, len(L), fit["df"], float(stats.f.sf(F, len(L), fit["df"]))


def _floor(v):
    """A separated scenario (type never cited: AME 0 in every replicate) has zero variance;
    it gets the smallest observed one, as in cluster_effects.dl_heterogeneity."""
    v = np.asarray(v, float)
    tiny = v <= 1e-10
    return np.where(tiny, v[~tiny].min(), v), int(tiny.sum())


def meta_summary(y, v, topic):
    """Stage 2 for one unit x type x factor: y = per-scenario AMEs (pp), v = their variances."""
    from cluster_effects import dl_heterogeneity      # lazy: imports the label set's revised_analysis
    y, topic = np.asarray(y, float), np.asarray(topic)
    v, n_floor = _floor(v)
    levels = sorted(set(topic))
    m0 = meta_reml(y, v)
    m1 = meta_reml(y, v, (topic[:, None] == np.array(levels)[None, :]).astype(float))
    F, df1, df2, p_topic = kh_f(m1, np.hstack([-np.ones((len(levels) - 1, 1)), np.eye(len(levels) - 1)]))
    mu, se = float(m0["beta"][0]), float(np.sqrt(m0["vcov"][0, 0]))
    t = stats.t.ppf(.975, m0["df"])
    half = t * np.sqrt(m0["tau2"] + se ** 2)
    dl = dl_heterogeneity(y, v)
    q_within = sum(dl_heterogeneity(y[topic == k], v[topic == k])["Q"] for k in levels)
    return dict(n_scenarios=len(y), n_v_floored=n_floor, mean_equal=y.mean(), mu=mu, se_kh=se,
                ci_low=mu - t * se, ci_high=mu + t * se, p_mu=float(2 * stats.t.sf(abs(mu / se), m0["df"])),
                tau2=m0["tau2"], tau=np.sqrt(m0["tau2"]), I2=m0["I2"], Q=m0["Q"], p_Q=m0["p_Q"],
                pi_low=mu - half, pi_high=mu + half, tau2_topic=m1["tau2"],
                R2_topic=max(0., 1 - m1["tau2"] / m0["tau2"]) if m0["tau2"] > 0 else np.nan,
                F_topic=F, df1=df1, df2=df2, p_topic=p_topic,
                dl_tau2=dl["tau2"], dl_tau=np.sqrt(dl["tau2"]), dl_I2=dl["I2"],
                dl_p_topic=float(stats.chi2.sf(dl["Q"] - q_within, len(levels) - 1)),
                **{f"mean_{k}": b for k, b in zip(levels, m1["beta"])},
                **{f"se_{k}": s for k, s in zip(levels, np.sqrt(np.diag(m1["vcov"])))})


def _split(unit):
    a, b = unit.split("|")
    return (a, b) if a in fmnl.ENGINES else (b, a)


def _estimate(d, label, nan_unidentified=False):
    """Fit + AMEs with fmnl._refit's guards (gradient, null-direction sensitivity). A grp with a
    cell left empty (a day deleted) is not estimable: dropped here, where fmnl._worlds fails.
    nan_unidentified (day-jackknife refits only): an
    engine|scenario unit whose tested-type AME moves along a null direction is not estimable
    without that day either -> all its AMEs NaN in this replicate instead of a RuntimeError
    (perplexity-ui|CH04, |OW04 without 2026-09-18). The full-sample fit keeps the hard guard."""
    fit = fmnl.fit_fmnl(d)
    if not fit.max_gradient < fmnl.GRAD_GUARD:
        raise RuntimeError(f"{label} did not converge: max|gradient| {fit.max_gradient:.2e}")
    keep = np.nonzero(d.grp.map(d.groupby("grp").cell.nunique().eq(len(fmnl.CELL_NAMES))).to_numpy())[0]
    fit = replace(fit, fe=fit.fe[keep], active=fit.active[keep])
    d = d.iloc[keep].reset_index(drop=True)
    flagged = fmnl._flag(label, fmnl._sensitivity(fit, d))
    est = fmnl.ame_raw(fit, d)
    if not nan_unidentified:
        fmnl._unidentified(flagged)
        return fit, d, est
    df = pd.DataFrame(flagged, columns=fmnl.UNID_COLS)
    bad = df[df.type.isin(fmnl.TESTED + ["NONE"]) & (df.sensitivity > fmnl.SENS_TOL)]
    units = set(bad.unit[bad.unit.str.contains("|", regex=False)])
    # engine / `all` aggregates of the scenario fit are not used by stage 1; their flags follow
    # from the NaN'd scenario units. Anything else still fails hard.
    fmnl._unidentified([r for r in flagged if r["unit"] not in units and ("|" in r["unit"] or not units)])
    for u in sorted(units):
        print(f"  {label}: {u} not identified -> NaN in this replicate", flush=True)
    return fit, d, {k: (np.nan if k[0] in units else v) for k, v in est.items()}


def _day_refit(frame, g):
    fit, _, est = _estimate(frame[frame.day != g].reset_index(drop=True), f"refit without day {g}",
                            nan_unidentified=True)
    return est, fit.iterations


def day_jackknife(frame):
    """Leave-one-day-out replicates of the AMEs (fmnl.jackknife(by="day") through _estimate):
    a grp not estimable without that day is NaN there (perplexity-ui|OW01 NEU+FEA+PRC has one
    citing answer, 2026-09-08). Index = left-out day, columns (unit, type, factor)."""
    days = sorted(set(frame.day))
    workers = int(os.environ.get("STUDY2_WORKERS", "1"))   # refits are independent; 1 = sequential
    if workers > 1:
        ctx = mp.get_context("spawn")   # fork deadlocks after BLAS threads start
        with ctx.Pool(workers) as pool:
            out = pool.starmap(_day_refit, [(frame, g) for g in days])
    else:
        out = [_day_refit(frame, g) for g in days]
    rows = []
    for g, (est, its) in zip(days, out):
        rows.append(est)
        print(f"day jackknife: without {g} ({len(rows)}/{len(days)}), {its} Newton its", flush=True)
    reps = pd.DataFrame(rows, index=pd.Index(days, name="left_out"))
    reps.columns = pd.MultiIndex.from_tuples(reps.columns, names=["unit", "type", "factor"])
    return reps


def stage1(point, reps, frame, topics, mech):
    """Per-scenario AMEs with leave-one-day-out jackknife variances, engine units and `all`.

    point = fmnl.ame_raw of the grp = engine|scenario fit, reps = day_jackknife(frame);
    an engine's jackknife runs over its own collection days, `all` (engine-equal mean of the
    replicates) over all days; a NaN replicate (unit not estimable without that day) is left
    out of that unit's jackknife. topics = scenario -> topic, mech = fmnl.ame_by_scenario of
    the common-beta fit.
    """
    from cluster_effects import bh_q
    keys = [k for k in point if "|" in k[0]]
    idx = pd.MultiIndex.from_tuples([(*_split(u), k, f) for u, k, f in keys],
                                    names=["engine", "scenario", "type", "factor"])
    full = pd.Series([point[k] for k in keys], idx, name="full")
    tab = pd.concat([full, reps.loc[:, keys].T.set_axis(idx)], axis=1)
    level = ["scenario", "type", "factor"]
    if not tab.groupby(level=level).size().eq(len(fmnl.ENGINES)).all():
        raise ValueError("A scenario is missing an engine: `all` would not be engine-equal")
    units = {e: (tab.xs(e, level="engine"), sorted(set(frame.day[frame.engine == e]))) for e in fmnl.ENGINES}
    pooled = tab.groupby(level=level).sum(min_count=len(fmnl.ENGINES)) / len(fmnl.ENGINES)
    units["all"] = (pooled, sorted(set(frame.day)))
    parts = []
    for e, (t, days) in units.items():
        x = t[days].to_numpy()
        n = np.isfinite(x).sum(axis=1)
        var = (n - 1) / n * np.nansum((x - np.nanmean(x, axis=1, keepdims=True)) ** 2, axis=1)
        parts.append(t[["full"]].rename(columns={"full": "ame_pp"}).reset_index().assign(
            engine=e, var_jack=var, n_days=n))
    st1 = pd.concat(parts, ignore_index=True)
    st1["topic"] = st1.scenario.map(topics)
    st1["se_jack"] = np.sqrt(st1.var_jack)
    st1["p"] = 2 * stats.t.sf(np.abs(st1.ame_pp / st1.se_jack), st1.n_days - 1)
    st1["q_bh"] = st1.groupby(KEY).p.transform(bh_q)
    m = mech.rename(columns={"ame_pp": "ame_mech_pp"})
    m = pd.concat([m, m.groupby(level, as_index=False).ame_mech_pp.mean().assign(engine="all")],
                  ignore_index=True)
    st1 = st1.merge(m, on=["engine", "scenario", "type", "factor"], how="left", validate="one_to_one")
    return st1[["engine", "scenario", "topic", "type", "factor", "ame_pp", "se_jack", "var_jack",
                "n_days", "p", "q_bh", "ame_mech_pp"]]


def stage2(st1):
    return pd.DataFrame([dict(zip(KEY, key), **meta_summary(d.ame_pp, d.var_jack, d.topic))
                         for key, d in st1.groupby(KEY, sort=False)])


def gate(st2, alpha=ALPHA):
    """Gate for the RQ2 family, separately per unit (`all` confirmatory).

    p_holm_all (Holm over all factor x type tests of the unit) is the strict per-hypothesis
    reference: the gate controls FWER for factor-level claims, not across all type claims.
    """
    out = st2.copy()
    by_f = out.groupby(["engine", "factor"]).p_topic
    out["gate_p"] = by_f.transform(lambda p: min(1.0, len(p) * p.min()))
    fam = out.drop_duplicates(["engine", "factor"])
    adj = {(e, f): q for _, d in fam.groupby("engine") for e, f, q in zip(d.engine, d.factor, holm(d.gate_p))}
    out["gate_p_holm"] = [adj[k] for k in zip(out.engine, out.factor)]
    out["type_p_holm"] = by_f.transform(holm)
    out["p_holm_all"] = out.groupby("engine").p_topic.transform(holm)
    out["factor_pass"] = out.gate_p_holm <= alpha
    out["type_pass"] = out.factor_pass & (out.type_p_holm <= alpha)
    return out


def diagnostic(st1, st2):
    """Baseline mechanics: pp spread one common logit effect implies vs the stage-2 tau."""
    tau2 = st2.set_index(KEY).tau2
    rows = []
    for key, d in st1.groupby(KEY, sort=False):
        m, t2 = d.ame_mech_pp.to_numpy(), tau2[key]
        reg = meta_reml(d.ame_pp, _floor(d.var_jack)[0], np.column_stack([np.ones(len(m)), m]))
        rows.append(dict(zip(KEY, key), mean_mech=m.mean(), sd_mech=m.std(ddof=1), min_mech=m.min(),
                         max_mech=m.max(), sd_scenario=d.ame_pp.std(ddof=1), tau=np.sqrt(t2),
                         share_mech_var=m.var(ddof=1) / t2 if t2 > 0 else np.nan,
                         corr=np.corrcoef(d.ame_pp, m)[0, 1], slope_mech=reg["beta"][1],
                         se_slope=np.sqrt(reg["vcov"][1, 1]), tau_resid_mech=np.sqrt(reg["tau2"]),
                         R2_mech=max(0., 1 - reg["tau2"] / t2) if t2 > 0 else np.nan))
    return pd.DataFrame(rows)


def run(frame_c, frame, topics):
    """Common-beta fit (mechanics), scenario-specific fit, day jackknife, the three tables."""
    tested = fmnl.tested_types(OUTCOME)
    t0 = time.perf_counter()
    fits, ests = {}, {}
    for name, d in (("common", frame_c), ("scenario", frame)):
        fits[name], d_used, ests[name] = _estimate(d, f"{name} fit")
        if len(d_used) != len(d):
            raise ValueError(f"{name} fit: a grp x cell has no answers in the full sample")
    mech = fmnl.ame_by_scenario(fits["common"], frame_c)
    point = {k: a for k, a in ests["scenario"].items() if k[1] in tested}
    t1 = time.perf_counter()
    reps = day_jackknife(frame)
    st1 = stage1(point, reps, frame, topics, mech[mech.type.isin(tested)])
    st2 = gate(stage2(st1))
    meta = dict(outcome=OUTCOME, tested_types=list(tested), n_answers=len(frame),
                n_days={e: int(frame.day[frame.engine == e].nunique()) for e in fmnl.ENGINES},
                fits={name: dict(converged=bool(f.converged), iterations=int(f.iterations),
                                 max_gradient=float(f.max_gradient), n_params=int(f.support.sum()),
                                 separation_removed=int(f.separation_removed))
                      for name, f in fits.items()},
                nonestimable=[f"{u} without {g}" for (g, u), v in
                              reps.isna().T.groupby(level="unit").any().T.stack().items() if v],
                fit_seconds=t1 - t0, jackknife_seconds=time.perf_counter() - t1)
    return st1, st2, diagnostic(st1, st2), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--runs-pickle", default=None, help="pickled dict(runs=...) from build_runs('ui')")
    args = ap.parse_args()
    if args.labels == "jev":
        sys.path.insert(0, str(HERE))
    if args.runs_pickle:
        if args.labels not in Path(args.runs_pickle).name:
            raise ValueError(f"--runs-pickle {args.runs_pickle} does not look like a {args.labels} run")
        runs = pd.read_pickle(args.runs_pickle)["runs"]
    else:
        from revised_analysis import build_runs
        runs, *_ = build_runs("ui")
    topics = runs[runs.role.eq("factorial")].groupby("scenario").topic.first()
    st1, st2, diag, meta = run(fmnl.long_frame(runs, OUTCOME), fmnl.long_frame(runs, OUTCOME, grp="scenario"),
                               topics)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, tab in (("stage1", st1), ("stage2", st2), ("diagnostic", diag)):
        tab.insert(0, "labels", args.labels)
        tab.insert(1, "outcome", OUTCOME)
        tab.to_csv(OUT / f"{args.labels}_{name}.csv", index=False)
    (OUT / f"{args.labels}_meta.json").write_text(json.dumps(dict(labels=args.labels, **meta), indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    show = ["engine", "type", "factor", "mu", "tau", "dl_tau", "I2", "pi_low", "pi_high", "p_topic",
            "gate_p_holm", "type_pass"]
    print(st2[show].round(3).to_string(index=False))
    print(diag[["engine", "type", "factor", "sd_mech", "tau", "share_mech_var", "R2_mech"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
