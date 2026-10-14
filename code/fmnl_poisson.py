"""FMNL via the Poisson equivalence (second implementation), pyfixest `fepois`.

Why the Poisson slopes ARE the multinomial betas
------------------------------------------------
Rows are answer x category. Put an answer fixed effect phi_i in the linear index,
mu_ik = exp(phi_i + eta_ik), and maximise the weighted Poisson quasi-likelihood
    sum_i sum_k w_i (n_ik log mu_ik - mu_ik).
The score for phi_i is sum_k w_i (n_ik - mu_ik) = 0, so the fitted means obey
sum_k mu_ik = n_i, hence mu_ik = n_i * softmax_k(eta_ik): the answer FE absorbs
the multinomial normaliser exactly, at every candidate eta. Profiling phi_i out
leaves w_i * sum_k n_ik log p_ik + const, and with w_i = 1/n_i that is the
answer-equal multinomial quasi-likelihood the FMNL objective. So the Poisson
coefficient on (cell c, engine e, type k) equals beta_{c,e,k} of the FMNL and the
fitted mu give p_ik = mu_ik / n_i directly.
(Baker 1994, Appl. Statist. 43(4); Guimaraes, Figueiredo & Woodward 2003, J. Reg. Sci. 43(4).)

CORPORATE is the reference category (eta = 0): it carries no regressor and no
non-chat FE, so its rows sit in one shared FE level, which the chat FE and the
non-reference levels already span - a reparameterised intercept, nothing more.

Counterfactual probabilities need eta for
(answer, category) pairs that are NOT in the estimation frame, because a category
separated by engine x cell in the answer's observed cell becomes live again in a
counterfactual cell where it is not separated. `fit.fixef()` supplies the two
type-specific FE, with the shared REF level as their reference, so for any cell c
    eta_ik(c) - eta_i,REF(c) = alpha_{s,e,k} + lambda_{e,t,k} + beta[c, e, k],
defined whenever both FE levels survived separation - which is exactly the
condition, and unlike a table keyed on (scenario, engine, day) it stays defined on
days where every cell that keeps the category happens not to have run.

Guards: a cell dummy collinear with the fixed effects is dropped by pyfixest; its beta is
set to 0, a normalisation, not an estimate - the fit is unchanged. Typically the (engine,
type) block lost its NEU rows to separation and the drop is the block's level. That is
harmless for AMEs when (engine, NEU, type) is a separated pair: NEU is masked in every
counterfactual and the level cancels (`reason = level`). When the NEU rows went through
the scenario or day family instead, NEU counterfactuals read the level, and the engine's
AMEs hinge on the normalisation (`reason = collinear`, also any second drop in a block).
Every drop, in the full fit and in every jackknife refit, goes to `unidentified.csv`, the
coefficient is NaN in `coefs.csv`, and per-refit convergence is in `fit_meta.json`.

CLI: fmnl_poisson.py --labels jev
                     --outcome citation|retrieval|citation_mass|source_mass [--boot B]
                     [--runs-pickle PATH] [--out DIR]
"""
import argparse
import json
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TYPES = ["CORPORATE", "EDITORIAL", "UGC", "REFERENCE", "INSTITUTIONAL", "OTHER"]
REF = "CORPORATE"
ENGINES = ["chatgpt-ui", "gemini-ui", "perplexity-ui"]  # engines in the analysis
FACTORS = ("adv", "fea", "prc")
PREFIX = {"citation": "citation", "retrieval": "source",
          "citation_mass": "citation", "source_mass": "source"}
# The three separation group families, keyed by the columns that identify a group.
FAMILIES = {"se": ("scen", "eng"), "ed": ("eng", "day"), "ce": ("eng", "cell")}
# ponytail: module-level CATS instead of threading a config object - one outcome
# per process anyway (the label set already forces that).
CATS = list(TYPES)
# ponytail: tolerances fixed here, not CLI flags - they are part of the contract.
# MAP demeaning does not converge on this 3-way FE (chat x scen-eng-type x eng-day-type)
# within 10k sweeps; LSMR with the additive-Schwarz preconditioner does, in ~2 min.
IWLS_TOL, IWLS_MAXITER, FIXEF_TOL, FIXEF_MAXITER = 1e-10, 200, 1e-11, 100_000
# fit.fixef() recovers the FE by lsqr at atol = btol = 1e-6 unless told otherwise, which
# left them 0.006-0.026 off in the linear index on the real panels; 1e-14 costs ~0.1 s
# and brings `fixef_vs_exact_max_abs` to ~1e-10 (jev citation).
FIXEF_RECOVER_TOL = 1e-14


def set_outcome(outcome):
    """citation/retrieval = six conditional shares; *_mass = seven unconditional."""
    global CATS
    CATS = TYPES + ["NONE"] if outcome.endswith("_mass") else list(TYPES)
    return CATS


def label_path(labels):
    """One label set per process: `design` caches the classifier on import."""
    sys.path.insert(0, str(HERE))
    return HERE


# ---------------------------------------------------------------- data


def long_frame(runs, outcome):
    """Answer x category rows.

    citation/retrieval: answers with a non-empty composition, y = counts,
    total = n sources, weight 1/n. *_mass: every factorial run, y = the mass
    shares (which already sum to one over the seven categories), total = 1.
    """
    pre = PREFIX[outcome]
    mass = outcome.endswith("_mass")
    d = runs[runs.role.eq("factorial")]
    if not mass:
        d = d[d[f"any_{pre}"].eq(1)]
    cats, n = CATS, len(d)
    field = f"{pre}_mass_" if mass else f"{pre}_n_"
    rep = lambda s: np.repeat(np.asarray(s), len(cats))  # noqa: E731
    long = pd.DataFrame({
        "chat": rep(d.chat_id.astype(str)),
        "type": np.tile(cats, n),
        "y": d[[field + c for c in cats]].to_numpy().ravel().astype(float),
        "tot": np.ones(n * len(cats)) if mass else rep(d[f"{pre}_n"].astype(float)),
        "scen": rep(d.scenario.astype(str)),
        "eng": rep(d.model_id.astype(str)),
        "day": rep(d.date.dt.strftime("%Y-%m-%d")),
        "cell": rep(d.cell.astype(str)),
    })
    long["w"] = 1.0 / long["tot"]
    return long


def drop_separated(long):
    """Spec separation rule (same convention as fmnl._separation since 2026-09-26).

    Every (group, category) pair whose total weighted count is zero on the full frame is
    recorded, in all three families, and its rows are dropped (p_ik = 0). Dropped rows carry
    zero count, so one pass on the raw frame is the fixed point and the surviving rows equal
    the iterated rule. Recording also the pairs whose rows another family already removed is
    a convention for a cell that is not identified: an answer moved into such a cell gets
    p = 0 for that category rather than a never-fitted beta = 0 (worth up to 0.12 pp on thin
    Perplexity types under JEV labels; tested types <= 0.04 pp pooled).
    CORPORATE is the reference (eta = 0) and carries no parameter, so the rule never applies to it.
    """
    counts, sets, drop = {}, {}, np.zeros(len(long), bool)
    for fam, cols in FAMILIES.items():
        keys = [*cols, "type"]
        zero = long.groupby(keys, observed=True)["y"].transform("sum").to_numpy() == 0
        zero &= long["type"].ne(REF).to_numpy()
        pairs = long.loc[zero, keys].drop_duplicates()
        counts[fam] = len(pairs)
        sets[fam] = {(a, b, k) for a, b, k in pairs.itertuples(index=False)}
        drop |= zero
    return long[~drop].reset_index(drop=True), counts, sets


def design(long, cells, engines):
    """Saturated cell x engine x category dummies plus the two FE key columns."""
    ref = long["type"].eq(REF).to_numpy()
    long = long.assign(
        f_se=np.where(ref, "BASE", long.scen + "|" + long.eng + "|" + long.type),
        f_ed=np.where(ref, "BASE", long.eng + "|" + long.day + "|" + long.type))
    cell, eng, typ = long.cell.to_numpy(), long.eng.to_numpy(), long.type.to_numpy()
    x, keys = {}, []
    for ei, e in enumerate(engines):
        for ki, k in enumerate(CATS):
            if k == REF:
                continue
            base = (eng == e) & (typ == k)
            for ci, c in enumerate(cells):
                col = base & (cell == c)
                if c == "NEU" or not col.any():  # reference cell / separated away
                    continue
                x[f"x{ci}_{ei}_{ki}"] = col.astype(float)
                keys.append((c, e, k))
    return pd.concat([long, pd.DataFrame(x, index=long.index)], axis=1), list(x), keys


def fit_poisson(frame, cols):
    import pyfixest as pf
    from pyfixest.demeaners import LsmrDemeaner
    fml = "y ~ " + " + ".join(cols) + " | chat + f_se + f_ed"
    # separation_check=[] disables pyfixest's own check: `None` means "run it".
    # drop_separated() already did the separation rule (and on the citation sample
    # removes the same 11,904 rows pyfixest's "fe" check finds).
    return pf.fepois(fml, data=frame, weights="w", vcov="iid", fixef_rm="none",
                     separation_check=[], iwls_tol=IWLS_TOL, iwls_maxiter=IWLS_MAXITER,
                     demeaner=LsmrDemeaner(fixef_atol=FIXEF_TOL, fixef_btol=FIXEF_TOL,
                                           fixef_maxiter=FIXEF_MAXITER))


# ---------------------------------------------------------------- AME


def cell_bits(cells):
    return np.array([[int("ADV" in c.split("+")), int("FEA" in c.split("+")),
                      int("PRC" in c.split("+"))] for c in cells])


class Sample:
    """One answer per row: keys, the eta table net of the cell term, separation masks.

    eta0[i, k] = eta_ik - eta_i,REF with the cell term removed, from three sources,
    in order of precision:
    1. the answer's own fitted row, log p_ik - log p_i,REF - beta[c_i, e, k], which
       reproduces the fit exactly;
    2. the same quantity from another answer in the (scenario, engine, day) - it equals
       alpha_{s,e,k} + lambda_{e,t,k}, so it does not depend on the answer - for
       categories separation kept out of this answer's cell but not out of that one;
    3. `fit.fixef()`, for a category no surviving row of the (scenario, engine, day)
       covers - the only way to price a counterfactual cell that brings the category
       back. It covers 0-3 of ~55,000 entries on the real panels; its accuracy is
       `fixef_vs_exact_max_abs` (~1e-10 at FIXEF_RECOVER_TOL; the lsqr default 1e-6 gave
       0.006-0.026, i.e. up to ~2e-4 pp of AME per entry).
    `g_spread` is the within-(scenario, engine, day, category) spread of source 2 and
    must be ~0 unless a cell dummy was dropped for collinearity.
    """

    def __init__(self, long, cells, engines, sets, fe, p, beta):
        se, ed = fe
        base = se.get("BASE", 0.0) + ed.get("BASE", 0.0)
        chat, chats = pd.factorize(long.chat.to_numpy())
        a = long.iloc[np.unique(chat, return_index=True)[1]]
        self.n, self.nk = len(chats), len(CATS)
        self.scenarios = np.unique(a.scen.to_numpy())
        self.si = np.searchsorted(self.scenarios, a.scen.to_numpy())
        self.ei = np.array([engines.index(e) for e in a.eng])
        self.ci = np.array([cells.index(c) for c in a.cell])
        self.ns, self.ne = len(self.scenarios), len(engines)
        self.denom = np.bincount(self.si * self.ne + self.ei,
                                 minlength=self.ns * self.ne).astype(float)
        # engine x cell x category separation: the only family a counterfactual moves
        self.ce = np.zeros((len(engines), len(cells), self.nk), bool)
        for e, c, k in sets["ce"]:
            self.ce[engines.index(e), cells.index(c), CATS.index(k)] = True
        # exact source
        pref = np.zeros(self.n)
        ref_rows = long.type.eq(REF).to_numpy()
        pref[chat[ref_rows]] = p[ref_rows]
        g = np.log(p) - np.log(pref[chat]) - beta[
            np.array([cells.index(c) for c in long.cell]),
            np.array([engines.index(e) for e in long.eng]),
            np.array([CATS.index(t) for t in long.type])]
        table = pd.Series(g).groupby(
            (long.scen + "|" + long.eng + "|" + long.day + "|" + long.type).to_numpy())
        self.g_spread = float(table.max().sub(table.min()).max())
        own = np.full((self.n, self.nk), np.nan)
        own[chat, np.array([CATS.index(t) for t in long.type])] = g
        gmap = table.mean().to_dict()
        gkey = (a.scen + "|" + a.eng + "|" + a.day + "|").to_numpy()
        se_key = (a.scen + "|" + a.eng + "|").to_numpy()
        ed_key = (a.eng + "|" + a.day + "|").to_numpy()
        shared, approx = [], []
        for k in CATS:
            shared.append(np.array([gmap.get(x + k, np.nan) for x in gkey]))
            approx.append(np.zeros(self.n) if k == REF else
                          np.array([se.get(x + k, np.nan) for x in se_key])
                          + np.array([ed.get(x + k, np.nan) for x in ed_key]) - base)
        shared, approx = np.column_stack(shared), np.column_stack(approx)
        both = ~np.isnan(shared) & ~np.isnan(approx)
        self.fe_consistency = float(np.abs(shared[both] - approx[both]).max())
        self.n_fallback = int((np.isnan(shared) & ~np.isnan(approx)).sum())
        eta0 = np.where(np.isnan(own), np.where(np.isnan(shared), approx, shared), own)
        self.eta0 = np.where(np.isnan(eta0), -np.inf, eta0)

    def probs(self, beta, cells_assigned):
        """p_ik under an arbitrary per-answer cell assignment."""
        eta = self.eta0 + beta[cells_assigned, self.ei, :]
        eta = np.where(self.ce[self.ei, cells_assigned, :], -np.inf, eta)
        eta -= eta.max(axis=1, keepdims=True)
        w = np.exp(eta)
        return w / w.sum(axis=1, keepdims=True)


def ame_cube(s, beta, cells):
    """(factor, engine, category) AMEs in pp, averaged equally over the scenarios present.

    The separation mask is rebuilt for the counterfactual cell (Sample.probs), so a
    category separated in the observed cell can be live in the counterfactual one and
    vice versa - the fix for the route difference on thin types.
    """
    bits = cell_bits(cells)
    to_cell = {tuple(b): i for i, b in enumerate(bits)}
    out = np.zeros((len(FACTORS), s.ne, s.nk))
    for fi in range(len(FACTORS)):
        pref = []
        for value in (1, 0):
            nb = bits[s.ci].copy()
            nb[:, fi] = value
            pref.append(s.probs(beta, np.array([to_cell[tuple(b)] for b in nb])))
        diff = pref[0] - pref[1]
        code = s.si * s.ne + s.ei
        cube = np.stack([np.bincount(code, weights=diff[:, k],
                                     minlength=s.ns * s.ne).reshape(s.ns, s.ne)
                         for k in range(s.nk)], axis=2)
        out[fi] = 100 * (cube / s.denom.reshape(s.ns, s.ne, 1)).mean(axis=0)
    return out


def with_all(cube):
    """Prepend engine = all (equal mean of the engine AMEs)."""
    return np.concatenate([cube.mean(axis=1, keepdims=True), cube], axis=1)


# ---------------------------------------------------------------- estimation


def estimate(long, cells, engines, sets):
    """Fit once. Returns beta, the Sample, the identified (cell, engine, type) keys and
    info = converged, collinear keys (beta 0 by normalisation), the AME-relevant subset
    of them (module docstring), eta_err, loglik.

    loglik is the multinomial quasi-loglik sum_i w_i sum_k y_ik log p_ik (= the scipy
    route's sum of share x log p), not pyfixest's Poisson one, so the routes compare.
    """
    frame, cols, keys = design(long, cells, engines)
    fit = fit_poisson(frame, cols)
    coef = fit.coef()  # pyfixest leaves collinear dummies out
    collinear = [key for col, key in zip(cols, keys) if col not in coef.index]
    drops = Counter((e, k) for _, e, k in collinear)
    relevant = [(c, e, k) for c, e, k in collinear
                if drops[(e, k)] > 1 or (e, "NEU", k) not in sets["ce"]]
    beta = np.zeros((len(cells), len(engines), len(CATS)))
    for col, (c, e, k) in zip(cols, keys):
        beta[cells.index(c), engines.index(e), CATS.index(k)] = coef.get(col, 0.0)
    p = fit.predict(type="response") / frame["tot"].to_numpy()
    raw = fit.fixef(atol=FIXEF_RECOVER_TOL, btol=FIXEF_RECOVER_TOL)
    fe = tuple(next(v for k, v in raw.items() if name in k) for name in ("f_se", "f_ed"))
    s = Sample(long, cells, engines, sets, fe, p, beta)
    # The reconstruction must reproduce the fit at the observed cells, to solver
    # precision, on exactly the rows that survived separation.
    chat = pd.factorize(long.chat.to_numpy())[0]
    ki = np.array([CATS.index(t) for t in long.type])
    check = s.probs(beta, s.ci)[chat, ki]
    info = dict(converged=bool(fit.convergence), collinear=collinear, ame_relevant=relevant,
                eta_err=float(np.abs(check - p).max()),
                loglik=float((long.w * long.y * np.log(p)).sum()))
    return beta, s, set(keys) - set(collinear), info


def jackknife(raw, cells, engines):
    """Leave-one-scenario-out refits; returns the (G, factor, engine, category) cube.

    Separation is re-derived from the RAW frame, not from the cleaned one: applied to
    an already-cleaned frame the rule reports only the *additional* removals, the
    counterfactual mask then misses the full-sample ones, and every refit's
    probabilities are wrong (0.5 on p, and the jackknife SEs with them).
    Also returns one record per refit (refit = left-out scenario, converged, collinear
    and AME-relevant keys, eta_err): a refit with an AME-relevant drop has AMEs in that
    engine that hinge on the normalisation, not data.
    """
    scenarios = sorted(raw.scen.unique())
    reps, refits = [], []
    for g in scenarios:
        sub, _, sets = drop_separated(raw[raw.scen.ne(g)].copy())
        beta, s, _, info = estimate(sub, cells, engines, sets)
        assert info["eta_err"] < 1e-6, f"eta reconstruction off by {info['eta_err']} in refit {g}"
        refits.append(dict(refit=g, **{k: info[k] for k in
                                       ("converged", "collinear", "ame_relevant", "eta_err")}))
        reps.append(with_all(ame_cube(s, beta, cells)))
        flag = "" if info["converged"] else " NOT CONVERGED"
        flag += f" collinear {info['collinear']}" if info["collinear"] else ""
        flag += f" AME-RELEVANT {info['ame_relevant']}" if info["ame_relevant"] else ""
        print(f"  jackknife {g}: done ({len(reps)}/{len(scenarios)}){flag}", flush=True)
    reps = np.array(reps)
    n = len(scenarios)
    return reps, np.sqrt((n - 1) / n * ((reps - reps.mean(0)) ** 2).sum(0)), n, refits


def bootstrap(raw, cells, engines, b, seed=20260925):
    """Pairs cluster bootstrap over scenarios; copies get distinct scenario ids."""
    rng = np.random.default_rng(seed)
    scenarios = sorted(raw.scen.unique())
    by_scen = {g: raw[raw.scen.eq(g)] for g in scenarios}
    out = []
    for r in range(b):
        parts = []
        for j, g in enumerate(rng.choice(scenarios, size=len(scenarios), replace=True)):
            part = by_scen[g].copy()
            part["scen"], part["chat"] = f"{g}#{j}", part["chat"] + f"#{j}"
            parts.append(part)
        sub, _, sets = drop_separated(pd.concat(parts, ignore_index=True))
        beta, s, _, _ = estimate(sub, cells, engines, sets)
        out.append(with_all(ame_cube(s, beta, cells)))
        print(f"  boot {r + 1}/{b}", flush=True)
    return np.array(out)


def hotelling(theta, reps):
    """Rank-adjusted jackknife Hotelling test of theta = 0 -> (W, q, p_chi2, p_F).

    cov = (G-1)/G sum_g dev dev', W = theta' pinv(cov) theta, q = numerical rank(cov),
    F = W (G-q) / (q (G-1)) ~ F(q, G-q); chi2(q) kept as the (over-rejecting) reference.
    A type with no jackknife variation (separated everywhere) costs a df, not a blow-up.
    """
    g = len(reps)
    dev = reps - reps.mean(0)
    cov = (g - 1) / g * dev.T @ dev
    # rtol=None = matrix_rank's cutoff (max dim x eps x largest eigenvalue): the
    # directions W inverts are exactly the q it counts.
    w = float(theta @ np.linalg.pinv(cov, hermitian=True, rtol=None) @ theta)
    q = int(np.linalg.matrix_rank(cov, hermitian=True))
    if q == 0:
        return w, 0, np.nan, np.nan
    return (w, q, float(stats.chi2.sf(w, q)),
            float(stats.f.sf(w * (g - q) / (q * (g - 1)), q, g - q)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    adj[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adj


# ---------------------------------------------------------------- output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], default="jev")
    ap.add_argument("--outcome", choices=list(PREFIX), default="citation")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--runs-pickle", help="frozen dict(runs=...) instead of build_runs('ui')")
    ap.add_argument("--out", help="output dir (default out/fmnl_poisson/<labels>_<outcome>)")
    args = ap.parse_args()
    label_path(args.labels)
    set_outcome(args.outcome)
    from inference import CELL_NAMES  # noqa: E402

    cells, engines = list(CELL_NAMES), list(ENGINES)
    t0 = time.time()
    if args.runs_pickle:
        runs = pd.read_pickle(args.runs_pickle)["runs"]
    else:
        from revised_analysis import build_runs  # noqa: E402
        runs = build_runs("ui")[0]
    raw = long_frame(runs, args.outcome)
    long, pairs, sets = drop_separated(raw)
    beta, s, present, info = estimate(long, cells, engines, sets)
    # A collinear dummy is recorded, not fatal - one thin cell must not kill a 25-fit
    # jackknife; its coefficient is NaN below and the term is in unidentified.csv.
    if info["collinear"]:
        print(f"WARNING collinear, beta normalised to 0: {info['collinear']}"
              f" (AME-relevant: {info['ame_relevant']})")
    assert info["eta_err"] < 1e-6, f"eta reconstruction off by {info['eta_err']}"
    cube = with_all(ame_cube(s, beta, cells))
    assert np.abs(cube.sum(axis=2)).max() < 1e-8, "adding-up violated"
    reps, se, g, refits = jackknife(raw, cells, engines)

    out = (Path(args.out) if args.out else
           HERE / "out" / "fmnl_poisson" / f"{args.labels}_{args.outcome}")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(labels=args.labels, outcome=args.outcome)
    names = ["all"] + engines
    n_ans = {e: int(long.loc[long.eng.eq(e), "chat"].nunique()) for e in engines}
    n_ans["all"] = int(long.chat.nunique())

    # Separation-removed coefficients are NaN (-inf limits), and so are collinear ones.
    pd.DataFrame([dict(**meta, engine=e, type=k, term=c,
                       coef=beta[ci, ei, ki] if (c, e, k) in present else np.nan)
                  for ci, c in enumerate(cells) if c != "NEU"
                  for ei, e in enumerate(engines)
                  for ki, k in enumerate(CATS) if k != REF]
                 ).to_csv(out / "coefs.csv", index=False)
    pd.DataFrame([dict(**meta, refit=r["refit"], engine=e, type=k, term=c,
                       reason="collinear" if (c, e, k) in r["ame_relevant"] else "level")
                  for r in [dict(refit="full", **info)] + refits for c, e, k in r["collinear"]],
                 columns=[*meta, "refit", "engine", "type", "term", "reason"]
                 ).to_csv(out / "unidentified.csv", index=False)

    q = stats.t.ppf(0.975, g - 1)
    rows = []
    for ei, e in enumerate(names):
        block = []
        for fi, f in enumerate(FACTORS):
            for ki, k in enumerate(CATS):
                est, sd = cube[fi, ei, ki], se[fi, ei, ki]
                t = est / sd if sd > 0 else 0.0
                block.append(dict(**meta, engine=e, type=k, factor=f, ame_pp=est,
                                  se_jack=sd, ci_low=est - q * sd, ci_high=est + q * sd,
                                  p_t=2 * stats.t.sf(abs(t), g - 1), p_holm=np.nan,
                                  n_answers=n_ans[e], n_scenarios=g,
                                  estimand="equal_scenario_engine", method="fmnl_poisson"))
        for r, a in zip(block, holm([r["p_t"] for r in block])):
            r["p_holm"] = a
        rows += block
    pd.DataFrame(rows).to_csv(out / "ame.csv", index=False)
    # the replicates themselves: joint tests re-derive without refits, and the
    # differential can pin an SE gap to one refit
    pd.DataFrame([dict(**meta, refit=r["refit"], engine=e, type=k, factor=f,
                       ame_pp=reps[gi, fi, ei, ki])
                  for gi, r in enumerate(refits) for ei, e in enumerate(names)
                  for fi, f in enumerate(FACTORS) for ki, k in enumerate(CATS)]
                 ).to_csv(out / "jack_reps.csv", index=False)

    # Joint test on the whole composition: the non-reference AMEs (the reference one is
    # minus their sum). df = their number, rank = q, the df both references use.
    keep = [ki for ki, k in enumerate(CATS) if k != REF]
    joint = []
    for ei, e in enumerate(names):
        for fi, f in enumerate(FACTORS):
            w, rank, p_chi2, p_f = hotelling(cube[fi, ei, keep], reps[:, fi, ei, :][:, keep])
            joint.append(dict(**meta, engine=e, factor=f, wald=w, df=len(keep), rank=rank,
                              p_chi2=p_chi2, p_F=p_f, n_scenarios=g, method="fmnl_poisson"))
    pd.DataFrame(joint).to_csv(out / "joint.csv", index=False)

    if args.boot:
        boot = bootstrap(raw, cells, engines, args.boot)
        pd.DataFrame([dict(**meta, engine=e, type=k, factor=f,
                           ci_low=float(np.quantile(boot[:, fi, ei, ki], .025)),
                           ci_high=float(np.quantile(boot[:, fi, ei, ki], .975)),
                           b=len(boot), method="fmnl_poisson")
                      for fi, f in enumerate(FACTORS) for ei, e in enumerate(names)
                      for ki, k in enumerate(CATS)]).to_csv(out / "ame_boot.csv", index=False)

    term = lambda key: "|".join(key)  # noqa: E731 - cell|engine|type
    (out / "fit_meta.json").write_text(json.dumps(dict(
        **meta, reference_type=REF, categories=CATS, n_answers=n_ans["all"],
        n_rows=int(len(long)), loglik=info["loglik"], converged=info["converged"],
        iterations=None,  # pyfixest Fepois does not expose the IWLS iteration count
        separation_pairs_removed=int(sum(pairs.values())), separation_pairs_by_group=pairs,
        collinear_dropped=[term(k) for k in info["collinear"]],
        iwls_tol=IWLS_TOL, fixef_tol=FIXEF_TOL, fixef_recover_tol=FIXEF_RECOVER_TOL,
        eta_reconstruction_max_err=info["eta_err"],
        eta_reconstruction_max_err_jackknife=max(r["eta_err"] for r in refits),
        jackknife_not_converged=[r["refit"] for r in refits if not r["converged"]],
        jackknife_collinear={r["refit"]: [term(k) for k in r["collinear"]]
                             for r in refits if r["collinear"]},
        ame_relevant_dropped=[term(k) for k in info["ame_relevant"]],
        jackknife_ame_relevant={r["refit"]: [term(k) for k in r["ame_relevant"]]
                                for r in refits if r["ame_relevant"]},
        fixef_fallback_entries=s.n_fallback, fixef_vs_exact_max_abs=s.fe_consistency,
        g_spread=s.g_spread, runs_pickle=args.runs_pickle,
        runtime_seconds=round(time.time() - t0, 1)), indent=2) + "\n")
    print(pd.DataFrame(rows).query("engine == 'all'")
          [["factor", "type", "ame_pp", "se_jack", "p_holm"]].to_string(index=False))
    print(f"Wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
