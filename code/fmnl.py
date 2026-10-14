"""Fractional multinomial logit on the type composition (direct Newton fit).

Answer-level multinomial quasi-likelihood: answer i contributes sum_k y_ik log p_ik
with y_ik the observed share (sums to one), weight one per answer. The type vocabulary
travels with the frame (`frame.attrs["types"]`, reference last: CORPORATE, or REST for a
`binary_frame`). Linear index for k != reference: scenario x engine x type FE + engine x
day x type FE + seven non-NEU cell dummies per grp per type, NEU the reference cell.
grp = engine by default; a finer engine-nested grp (engine|scenario, engine|window, ...)
gives every group its own cell effects.

Every parameter is engine-specific, so the likelihood separates exactly into engine
blocks, each fitted by its own Newton loop. Fixed effects are left un-normalised: the
scenario and day blocks share one intercept per engine x type, so the information matrix
is singular by construction. Newton steps are taken in its range (min-norm,
eigen-truncated), which leaves the fitted probabilities, the cell coefficients and every
AME identified while keeping the separation rule expressible on the natural FE groups.
At the solution each block's null space is checked against that alias (one direction per
active type); extra directions are kept in `Fit.extra_null`, and the jackknife / bootstrap
record how far each AME moves along them (RuntimeError when a tested type moves).
"""
import time
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import linalg, stats

from inference import CELL_NAMES, FACTORS, holm

TYPES = ["EDITORIAL", "UGC", "REFERENCE", "INSTITUTIONAL", "OTHER", "CORPORATE"]  # default
ENGINES = ["chatgpt-ui", "gemini-ui", "perplexity-ui"]
TERMS = CELL_NAMES[1:]
INTERACTIONS = ["adv:fea", "adv:prc", "fea:prc", "adv:fea:prc"]
PREFIX = {"citation": "citation", "retrieval": "source",
          "citation_mass": "citation", "source_mass": "source"}
TESTED = ["CORPORATE", "EDITORIAL", "UGC"]
GRAD_GUARD = 1e-7                    # every refit must reach max|gradient| below this
SENS_FLOOR, SENS_TOL = 1e-9, 1e-6    # |dAME/dv| (pp per unit step): record / fail if tested
UNID_COLS = ["left_out", "unit", "type", "factor", "sensitivity"]
_CELLS = np.array(CELL_NAMES, dtype=object)          # index 4 adv + 2 fea + prc


def long_frame(runs, outcome, grp=None, sample=None):
    """Answer-level frame of the factorial runs; attrs["types"] = vocabulary, reference last.

    Conditional outcomes keep answers with any_<prefix> == 1, or any_<sample> == 1 when
    `sample` is given (sample="citation" with outcome "retrieval" = retrieved composition
    of the answers that cite). `*_mass` outcomes keep every factorial run and add NONE (no
    source / no citation) before the CORPORATE reference, so selection is an outcome of the
    same model. grp=None -> grp = engine, else engine + "|" + runs[grp].
    """
    p = PREFIX[outcome]
    mass = outcome.endswith("_mass")
    if sample not in (None, "citation", "source") or (mass and sample):
        raise ValueError(f"sample={sample!r} is not valid for outcome {outcome!r}")
    types = TYPES[:-1] + ["NONE", TYPES[-1]] if mass else list(TYPES)
    d = runs[runs.role.eq("factorial")]
    if mass:
        d = d.assign(**{f"{p}_n_NONE": (1 - d[f"any_{p}"]).astype(int),
                        f"{p}_share_NONE": d[f"{p}_mass_NONE"]},
                     **{f"{p}_share_{k}": d[f"{p}_mass_{k}"] for k in TYPES})
    else:
        d = d[d[f"any_{sample or p}"].eq(1)]
    engine = d.model_id.astype(str).to_numpy().astype(object)
    out = pd.DataFrame({
        "chat_id": d.chat_id.astype(str).to_numpy(),
        "scenario": d.scenario.astype(str).to_numpy(),
        "engine": engine,
        "day": d.date.dt.strftime("%Y-%m-%d").to_numpy(),
        "cell": d.cell.astype(str).to_numpy(),
        "grp": engine if grp is None else engine + "|" + d[grp].astype(str).to_numpy().astype(object),
        "n": d[f"{p}_n"].to_numpy(),
        **{f: d[f].astype(int).to_numpy() for f in FACTORS},
        **{f"count_{k}": d[f"{p}_n_{k}"].to_numpy() for k in types},
        **{f"share_{k}": d[f"{p}_share_{k}"].to_numpy(float) for k in types}})
    if not set(out.engine) <= set(ENGINES) or not set(out.cell) <= set(CELL_NAMES):
        raise ValueError("Unexpected engine or cell in the factorial sample")
    if not np.allclose(out[[f"share_{k}" for k in types]].sum(axis=1), 1):
        raise ValueError("Shares do not add to one")
    out.attrs["types"] = types
    return out


def binary_frame(frame, k):
    """Type k against all other types pooled (REST, the reference): per-type fractional logit."""
    types = frame.attrs["types"]
    rest = [t for t in types if t != k]
    out = frame.drop(columns=[f"{c}_{t}" for c in ("count", "share") for t in types]).assign(
        **{f"count_{k}": frame[f"count_{k}"],
           "count_REST": frame[[f"count_{t}" for t in rest]].sum(axis=1),
           f"share_{k}": frame[f"share_{k}"],
           "share_REST": frame[[f"share_{t}" for t in rest]].sum(axis=1)})
    out.attrs = {**frame.attrs, "types": [k, "REST"]}
    return out


def _dummies(values, vocab):
    """n x len(vocab) sparse indicator; values outside the vocabulary are all-zero rows."""
    code = pd.Series(values).map({v: j for j, v in enumerate(vocab)}).to_numpy()
    keep = np.nonzero(~pd.isna(code))[0]
    return sp.csr_matrix((np.ones(len(keep)), (keep, code[keep].astype(int))),
                         shape=(len(code), len(vocab)))


def _cells(frame, factor=None, value=None):
    """grp|cell per answer, with `factor` set to `value` for a counterfactual."""
    bits = {f: frame[f].to_numpy() for f in FACTORS}
    if factor is not None:
        bits[factor] = np.full(len(frame), value)
    return (frame.grp.to_numpy().astype(object) + "|"
            + _CELLS[4 * bits["adv"] + 2 * bits["fea"] + bits["prc"]])


def _design(frame):
    """FE design (n x F), parameter names, cell vocabulary (grp|cell), engine per column."""
    se = sorted(set(frame.scenario + "|" + frame.engine))
    ed = sorted(set(frame.engine + "|" + frame.day))
    ce = [f"{g}|{c}" for g in sorted(set(frame.grp)) for c in TERMS]
    fe = sp.hstack([_dummies(frame.scenario + "|" + frame.engine, se),
                    _dummies(frame.engine + "|" + frame.day, ed)]).tocsr()
    names = [f"se:{v}" for v in se] + [f"ed:{v}" for v in ed] + [f"ce:{v}" for v in ce]
    engine = [v.rsplit("|", 1)[1] for v in se] + [v.split("|", 1)[0] for v in ed + ce]
    return fe, names, ce, np.array(engine, dtype=object)


def _X(fe, cells, ce):
    return sp.hstack([fe, _dummies(cells, ce)]).tocsr()


def _probs(X, params, active):
    """n x (K+1) probabilities, reference last; inactive (answer, type) pairs are exactly zero."""
    eta = np.where(active, np.asarray(X @ params.T), -np.inf)
    m = np.maximum(0, np.max(eta, axis=1))
    e = np.where(active, np.exp(eta - m[:, None]), 0.0)
    denom = np.exp(-m) + e.sum(axis=1)
    return np.column_stack([e, np.exp(-m)]) / denom[:, None]


def _keys(frame):
    return {"se": (frame.scenario + "|" + frame.engine).to_numpy(),
            "ed": (frame.engine + "|" + frame.day).to_numpy()}


def _separation(frame, counts):
    """Drop (group, type) pairs with zero total count (families: scenario x engine,
    engine x day, grp x cell). A dropped answer has zero count of that type, so dropping
    never changes another group's total: one pass is the fixed point. Every zero-total
    group is listed, also one whose answers all left through another family -- the
    counterfactual mask needs it (skipping those would leave an answer
    moved into such a cell was scored with a never-fitted beta = 0).
    Returns the answer x type mask, the number of pairs, and the pairs per family.
    """
    active = np.ones(counts.shape, bool)
    sets = {}
    for fam, v in {**_keys(frame), "ce": _cells(frame)}.items():
        g, uniq = pd.factorize(v)
        zero = np.column_stack([np.bincount(g, weights=counts[:, j], minlength=len(uniq))
                                for j in range(counts.shape[1])]) == 0
        active &= ~zero[g]
        sets[fam] = {(uniq[b], j) for b, j in zip(*np.nonzero(zero))}
    return active, sum(len(v) for v in sets.values()), sets


def _active_mask(frame, cells, sets):
    """Separation mask for an arbitrary grp|cell assignment (counterfactual AMEs)."""
    keys = {**_keys(frame), "ce": np.asarray(cells)}
    mask = np.ones((len(frame), len(frame.attrs["types"]) - 1), bool)
    for fam, pairs in sets.items():
        for j in {j for _, j in pairs}:
            mask[np.isin(keys[fam], [key for key, jj in pairs if jj == j]), j] = False
    return mask


@dataclass
class Fit:
    params: np.ndarray                      # (K, P), K = len(types) - 1
    names: list
    active: np.ndarray
    loglik: float
    iterations: int                         # max over engine blocks
    converged: bool
    max_gradient: float
    separation_removed: int
    seconds: float
    fe: object = field(repr=False, default=None)
    support: np.ndarray = field(repr=False, default=None)
    removed_sets: dict = field(repr=False, default=None)
    types: list = None
    ce: list = field(repr=False, default=None)
    null_dims: dict = None                  # engine -> (null dim at solution, se/ed aliases)
    extra_null: list = field(repr=False, default_factory=list)   # (K, P) unit directions


def _newton(X, y, active, support, fam, tol, maxiter):
    """One engine block: Newton in the range of the information (min-norm solution)."""
    k, p_dim = support.shape
    free = np.nonzero(support.ravel())[0]
    params = np.zeros((k, p_dim))

    def loglik(par):
        p = _probs(X, par, active)
        lp = np.log(np.where(p > 0, p, 1))
        return float((np.where(y[:, :k] > 0, y[:, :k] * lp[:, :k], 0)).sum()
                     + np.where(y[:, k] > 0, y[:, k] * lp[:, -1], 0).sum())

    ll, it, converged, gmax = loglik(params), 0, not len(free), 0.0
    out = dict(params=params, loglik=ll, iterations=0, converged=converged, max_gradient=0.0,
               null=0, alias=0, extra=[])
    if not len(free):
        return out
    for it in range(1, maxiter + 1):
        p = _probs(X, params, active)[:, :k]
        grad = np.asarray((X.T @ (y[:, :k] - p)).T).ravel()
        gmax = float(np.abs(grad[free]).max())
        info = np.zeros((k * p_dim, k * p_dim))
        for a in range(k):
            for b in range(a, k):
                w = p[:, a] * ((a == b) - p[:, b])
                block = (X.T @ sp.diags(w) @ X).toarray()
                info[a * p_dim:(a + 1) * p_dim, b * p_dim:(b + 1) * p_dim] = block
                if a != b:
                    info[b * p_dim:(b + 1) * p_dim, a * p_dim:(a + 1) * p_dim] = block.T
        vals, vecs = linalg.eigh(info[np.ix_(free, free)])
        keep = vals > max(vals.max(), 1e-300) * 1e-12
        if gmax < tol:
            converged = True
            break
        step = vecs[:, keep] @ ((vecs[:, keep].T @ grad[free]) / vals[keep])
        scale = 1.0
        while scale > 1e-8:
            trial = params.copy()
            trial.ravel()[free] += scale * step
            new = loglik(trial)
            # rounding slack: near the optimum a full step gains less than the loglik's
            # float noise; without it one block stalls at |grad| ~ 1e-8 (seen on simulate(7))
            if new >= ll - 1e-12 * abs(ll):
                params, ll = trial, new
                break
            scale /= 2
        else:                       # no uphill step left: stop, report max|gradient|
            break
    # identification at the solution: null space vs the se/ed alias of each active type
    null = vecs[:, ~keep]
    t_of, c_of = divmod(free, p_dim)
    alias = []
    for j in range(k):
        s, d = (t_of == j) & (fam[c_of] == "se"), (t_of == j) & (fam[c_of] == "ed")
        if s.any() and d.any():
            a = s.astype(float) - d
            alias.append(a / np.linalg.norm(a))
    extra = []
    if null.shape[1] > len(alias):
        A = np.array(alias).reshape(len(alias), len(free)).T
        u, sv, _ = np.linalg.svd(null - A @ (A.T @ null), full_matrices=False)
        for v in u[:, sv > .5].T:
            full = np.zeros(k * p_dim)
            full[free] = v
            extra.append(full.reshape(k, p_dim))
    return dict(params=params, loglik=ll, iterations=it, converged=converged,
                max_gradient=gmax, null=int(null.shape[1]), alias=len(alias), extra=extra)


def fit_fmnl(frame, tol=1e-9, maxiter=100):
    """Quasi-ML fit, one Newton problem per engine block, stitched into one Fit."""
    t0 = time.perf_counter()
    types = list(frame.attrs["types"])
    counts = frame[[f"count_{k}" for k in types[:-1]]].to_numpy(float)
    y = frame[[f"share_{k}" for k in types]].to_numpy(float)
    active, removed, sets = _separation(frame, counts)
    fe, names, ce, col_engine = _design(frame)
    X = _X(fe, _cells(frame), ce)
    support = np.asarray((X.T @ active.astype(float)).T) > 0      # (K, P)
    fam = np.array([n[:2] for n in names])
    params = np.zeros((len(types) - 1, X.shape[1]))
    eng = frame.engine.to_numpy()
    blocks, extra = {}, []
    for e in sorted(set(eng)):
        rows, cols = np.nonzero(eng == e)[0], np.nonzero(col_engine == e)[0]
        b = blocks[e] = _newton(X[rows][:, cols], y[rows], active[rows], support[:, cols],
                                fam[cols], tol, maxiter)
        params[:, cols] = b["params"]
        for v in b["extra"]:
            full = np.zeros_like(params)
            full[:, cols] = v
            extra.append(full)
    b = blocks.values()
    return Fit(params, names, active, sum(x["loglik"] for x in b),
               max(x["iterations"] for x in b), all(x["converged"] for x in b),
               max(x["max_gradient"] for x in b), removed, time.perf_counter() - t0,
               fe=fe, support=support, removed_sets=sets, types=types, ce=ce,
               null_dims={e: (x["null"], x["alias"]) for e, x in blocks.items()},
               extra_null=extra)


def _worlds(fit, frame):
    """(8, n, K+1): probabilities with every answer moved to each factorial cell."""
    grp = frame.grp.to_numpy().astype(object)
    unfit = (~fit.support[:, len(fit.names) - len(fit.ce):]).T.astype(float)   # (C, K)
    out = []
    for name in CELL_NAMES:
        cells = grp + "|" + name
        cc = _dummies(cells, fit.ce)
        mask = _active_mask(frame, cells, fit.removed_sets)
        if (mask & (cc @ unfit > 0)).any():           # would silently use a never-fitted beta
            raise ValueError(f"Counterfactual cell {name} has no fitted coefficient in some grp")
        out.append(_probs(sp.hstack([fit.fe, cc]).tocsr(), fit.params, mask))
    return np.stack(out)


def _contrast(worlds, bits, factor):
    """Answer-level contrast (n, K+1) for 'adv' or 'adv:fea' etc., other factors as observed:
    the signed sum over the 2^m settings of the named factors (difference of differences)."""
    idx = [FACTORS.index(f) for f in factor.split(":")]
    rows = np.arange(len(bits))
    out = 0.0
    for a in product((0, 1), repeat=len(idx)):
        b = bits.copy()
        b[:, idx] = a
        out = out + (-1) ** (len(idx) - sum(a)) * worlds[4 * b[:, 0] + 2 * b[:, 1] + b[:, 2], rows]
    return out


def _unit_means(diff, frame):
    """{unit: vector}: mean within (unit, scenario), then equal mean over the scenarios
    present; units = grp values, engines, and "all" (equal mean over engines)."""
    d, scen, out = pd.DataFrame(diff), frame.scenario.to_numpy(), {}
    for key in ("grp", "engine"):
        m = d.groupby([frame[key].to_numpy(), scen]).mean().groupby(level=0).mean()
        out.update(zip(m.index, m.to_numpy()))
    out["all"] = np.mean([out[e] for e in sorted(set(frame.engine))], axis=0)
    return out


def ame_raw(fit, frame, interactions=False):
    """{(unit, type, factor): AME in pp}. Factors adv, fea, prc (+ the four interaction
    contrasts with interactions=True); adding-up over types is asserted per unit x factor."""
    worlds = _worlds(fit, frame)
    bits = frame[list(FACTORS)].to_numpy(int)
    factors = list(FACTORS) + (INTERACTIONS if interactions else [])
    width = len(fit.types)
    means = _unit_means(np.hstack([_contrast(worlds, bits, f) for f in factors]) * 100, frame)
    out = {}
    for i, f in enumerate(factors):
        for u, v in means.items():
            v = v[i * width:(i + 1) * width]
            if abs(v.sum()) > 1e-8:
                raise ValueError(f"AMEs do not add up: {u}, {f}, {v.sum()}")
            out.update({(u, k, f): float(a) for k, a in zip(fit.types, v)})
    return out


_ame_raw = ame_raw


def ame(fit, frame, interactions=False):
    n, s = {}, {}
    for key in ("grp", "engine"):
        n.update(frame.groupby(key).size().to_dict())
        s.update(frame.groupby(key).scenario.nunique().to_dict())
    n["all"], s["all"] = len(frame), frame.scenario.nunique()
    return pd.DataFrame([dict(engine=u, type=k, factor=f, ame_pp=v, n_answers=int(n[u]),
                              n_scenarios=int(s[u]), estimand="equal_scenario_engine",
                              method="fmnl_scipy")
                         for (u, k, f), v in ame_raw(fit, frame, interactions).items()])


def ame_by_scenario(fit, frame):
    """Within-(engine, scenario) AMEs of the three factors under the fit's own structure."""
    worlds, bits = _worlds(fit, frame), frame[list(FACTORS)].to_numpy(int)
    parts = []
    for f in FACTORS:
        m = pd.DataFrame(_contrast(worlds, bits, f) * 100, columns=fit.types).groupby(
            [frame.engine.to_numpy(), frame.scenario.to_numpy()]).mean()
        parts.append(m.rename_axis(["engine", "scenario"]).reset_index().melt(
            ["engine", "scenario"], var_name="type", value_name="ame_pp").assign(factor=f))
    return pd.concat(parts, ignore_index=True)[["engine", "scenario", "type", "factor", "ame_pp"]]


def _sensitivity(fit, frame, interactions=False, eps=1e-4):
    """{(unit, type, factor): |dAME/dv|} along the extra null directions (root-sum-square
    over their orthonormal basis); entries above SENS_FLOOR only."""
    acc = {}
    for v in fit.extra_null:
        hi = ame_raw(replace(fit, params=fit.params + eps * v), frame, interactions)
        lo = ame_raw(replace(fit, params=fit.params - eps * v), frame, interactions)
        for key, a in hi.items():
            acc[key] = float(np.hypot(acc.get(key, 0.0), (a - lo[key]) / (2 * eps)))
    return {key: s for key, s in acc.items() if s > SENS_FLOOR}


def _unidentified(rows):
    """Flagged AMEs as a frame; RuntimeError when a tested type moves by more than SENS_TOL."""
    df = pd.DataFrame(rows, columns=UNID_COLS)
    bad = df[df.type.isin(TESTED + ["NONE"]) & (df.sensitivity > SENS_TOL)]
    if len(bad):
        raise RuntimeError("Tested-type AME not identified (moves along a null direction):\n"
                           + bad.to_string(index=False))
    return df


def _refit(d, interactions, label):
    """Fit + convergence guard + AMEs + sensitivity along extra null directions."""
    fit = fit_fmnl(d)
    if not fit.max_gradient < GRAD_GUARD:
        raise RuntimeError(f"Refit {label} did not converge: max|gradient| {fit.max_gradient:.2e}")
    return fit, ame_raw(fit, d, interactions), _sensitivity(fit, d, interactions)


def _flag(left_out, sens):
    return [dict(left_out=left_out, unit=u, type=k, factor=f, sensitivity=s)
            for (u, k, f), s in sens.items()]


def jackknife(frame, by="scenario", interactions=False):
    """Leave-one-`by`-out refits -> (se_df[unit, type, factor, se_jack], replicates).

    reps: index = left-out id, columns (unit, type, factor); attrs: unidentified (flagged
    AMEs, RuntimeError if a tested type moves), max_gradient (worst refit), null_dims
    ({left-out id: Fit.null_dims}), by."""
    if by not in ("scenario", "day"):
        raise ValueError(f"by={by!r}")
    ids = sorted(set(frame[by]))
    rows, flagged, gmax, null = [], [], 0.0, {}
    for g in ids:
        d = frame[frame[by] != g].reset_index(drop=True)
        fit, est, sens = _refit(d, interactions, f"without {by} {g}")
        rows.append(est)
        flagged += _flag(g, sens)
        gmax, null[g] = max(gmax, fit.max_gradient), fit.null_dims
    unid = _unidentified(flagged)
    reps = pd.DataFrame(rows, index=pd.Index(ids, name="left_out"))
    reps.columns = pd.MultiIndex.from_tuples(reps.columns, names=["unit", "type", "factor"])
    n = len(ids)
    se = np.sqrt((n - 1) / n * ((reps - reps.mean()) ** 2).sum())
    reps.attrs.update(unidentified=unid, max_gradient=gmax, null_dims=null, by=by)
    return se.rename("se_jack").reset_index(), reps


def wald_F(theta, reps_matrix):
    """Rank-adjusted Hotelling F. cov = (G-1)/G sum dev dev', W = theta' pinv(cov) theta on
    the numerical range of cov (matrix_rank tolerance), q = rank, F = W (G-q)/(q (G-1)) ~
    F(q, G-q). q = 0 (no replicate variation): nothing estimable, p_F = 1."""
    theta, r = np.asarray(theta, float), np.asarray(reps_matrix, float)
    g = len(r)
    dev = r - r.mean(axis=0)
    vals, vecs = np.linalg.eigh((g - 1) / g * dev.T @ dev)
    keep = vals > np.abs(vals).max(initial=0) * len(vals) * np.finfo(float).eps
    q = int(keep.sum())
    if not q:
        return dict(wald=0.0, q=0, F=0.0, df1=0, df2=g, p_F=1.0, G=g)
    wald = float(np.sum((vecs[:, keep].T @ theta) ** 2 / vals[keep]))
    f = wald * (g - q) / (q * (g - 1))
    return dict(wald=wald, q=q, F=f, df1=q, df2=g - q, p_F=float(stats.f.sf(f, q, g - q)), G=g)


def joint_wald(table, reps, labels, outcome):
    """Per unit x factor: rank-adjusted joint F over the non-reference AMEs (jackknife cov).
    df = rank = q, the dimension actually tested; the reference type is the table's last."""
    point = table.set_index(["engine", "type", "factor"]).ame_pp
    types = list(dict.fromkeys(table.type))
    rows = []
    for u in table.engine.unique():
        for f in FACTORS:
            keys = [(u, k, f) for k in types[:-1]]
            r = wald_F(point.loc[keys].to_numpy(), reps[keys].to_numpy())
            rows.append(dict(labels=labels, outcome=outcome, engine=u, factor=f, wald=r["wald"],
                             df=r["q"], rank=r["q"],
                             p_chi2=float(stats.chi2.sf(r["wald"], r["q"])) if r["q"] else 1.0,
                             p_F=r["p_F"], n_scenarios=r["G"], method="fmnl_scipy_jackknife"))
    return pd.DataFrame(rows)


def tested_types(outcome):
    return TESTED + (["NONE"] if outcome.endswith("_mass") else [])


def gatekeep(table, joint, tested):
    """Per unit: gate_p = Holm over the three factors of the joint p_F, gate_pass = gate_p <
    .05; inside a passing factor p_within = Holm over the tested types' p_t (NaN otherwise).
    The unit column is `unit` when the table has one, else `engine`."""
    u = "unit" if "unit" in table else "engine"
    gate = joint[[u, "factor", "p_F"]].copy()
    gate["gate_p"] = np.nan
    for _, part in gate.groupby(u):
        gate.loc[part.index, "gate_p"] = holm(part.p_F)
    out = table.merge(gate[[u, "factor", "gate_p"]], on=[u, "factor"], how="left",
                      validate="many_to_one")
    out["tested"] = out.type.isin(tested)
    out["gate_pass"] = out.gate_p < .05
    out["p_within"] = np.nan
    for _, part in out[out.tested & out.gate_pass].groupby([u, "factor"]):
        out.loc[part.index, "p_within"] = holm(part.p_t)
    return out


def _boot_summary(boot, suffix=""):
    return pd.DataFrame({f"ame_pp_boot_mean{suffix}": boot.mean(), f"se_boot{suffix}": boot.std(ddof=1),
                         f"ci_low_boot{suffix}": boot.quantile(.025), f"ci_high_boot{suffix}": boot.quantile(.975),
                         f"p_boot{suffix}": np.minimum(1, 2 * np.minimum((boot <= 0).mean(),
                                                                         (boot >= 0).mean()))})


def pairs_bootstrap(frame, B, seed, draws_path=None, keep_unidentified=False):
    """Pairs cluster bootstrap over scenarios (resampled copies get distinct ids), same refit
    guards as the jackknife. p_boot = min(1, 2 min(P*(theta* <= 0), P*(theta* >= 0))).

    keep_unidentified: a draw whose tested-type AME moves along
    a null direction does not fail. The (unit, factor) is flagged in that draw (all types), the
    usual columns summarise ALL draws, the *_excl columns the draws without that flag
    (n_draws_excl), and the draws file carries a `flagged` column."""
    if keep_unidentified:
        return _pairs_bootstrap_flagged(frame, B, seed, draws_path)
    scenarios = np.array(sorted(set(frame.scenario)))
    rng = np.random.default_rng(seed)
    parts = {s: frame[frame.scenario == s] for s in scenarios}
    draws, flagged = [], []
    for b in range(B):
        pick = rng.choice(len(scenarios), len(scenarios), replace=True)
        d = pd.concat([parts[scenarios[j]].assign(scenario=f"{scenarios[j]}#{c}")
                       for c, j in enumerate(pick)], ignore_index=True)
        d.attrs = dict(frame.attrs)
        _, est, sens = _refit(d, False, f"bootstrap draw {b}")
        draws.append(est)
        flagged += _flag(b, sens)
    unid = _unidentified(flagged)
    boot = pd.DataFrame(draws)
    boot.columns = pd.MultiIndex.from_tuples(boot.columns)
    out = pd.DataFrame({"ame_pp_boot_mean": boot.mean(), "se_boot": boot.std(ddof=1),
                        "ci_low_boot": boot.quantile(.025), "ci_high_boot": boot.quantile(.975),
                        "p_boot": np.minimum(1, 2 * np.minimum((boot <= 0).mean(),
                                                               (boot >= 0).mean()))})
    out = out.rename_axis(["engine", "type", "factor"]).reset_index()
    out["B"] = B
    out["method"] = "fmnl_scipy_pairs_bootstrap"
    out.attrs["unidentified"] = unid
    if draws_path:
        pd.DataFrame([dict(draw=b, engine=u, type=k, factor=f, ame_pp=v)
                      for b, est in enumerate(draws) for (u, k, f), v in est.items()]
                     ).to_csv(draws_path, index=False)
    return out


def _pairs_bootstrap_flagged(frame, B, seed, draws_path=None):
    """pairs_bootstrap(keep_unidentified=True): same draws (same rng sequence), flags kept."""
    scenarios = np.array(sorted(set(frame.scenario)))
    rng = np.random.default_rng(seed)
    parts = {s: frame[frame.scenario == s] for s in scenarios}
    draws, flagged = [], []
    for b in range(B):
        pick = rng.choice(len(scenarios), len(scenarios), replace=True)
        d = pd.concat([parts[scenarios[j]].assign(scenario=f"{scenarios[j]}#{c}")
                       for c, j in enumerate(pick)], ignore_index=True)
        d.attrs = dict(frame.attrs)
        _, est, sens = _refit(d, False, f"bootstrap draw {b}")
        draws.append(est)
        flagged += _flag(b, sens)
    unid = pd.DataFrame(flagged, columns=UNID_COLS)
    bad = unid[unid.type.isin(TESTED + ["NONE"]) & (unid.sensitivity > SENS_TOL)]
    bad_uf = set(zip(bad.left_out, bad.unit, bad.factor))
    boot = pd.DataFrame(draws)
    boot.columns = pd.MultiIndex.from_tuples(boot.columns)
    mask = pd.DataFrame([[(b, u, f) in bad_uf for (u, k, f) in boot.columns] for b in range(B)],
                        columns=boot.columns)
    out = _boot_summary(boot).join(_boot_summary(boot.mask(mask), "_excl"))
    out["n_draws_excl"] = (~mask).sum()
    out["n_flagged"] = mask.sum()
    out = out.rename_axis(["engine", "type", "factor"]).reset_index()
    out["B"] = B
    out["method"] = "fmnl_scipy_pairs_bootstrap_flagged"
    out.attrs["unidentified"] = unid
    out.attrs["flagged_draws"] = int(bad.left_out.nunique())
    if draws_path:
        pd.DataFrame([dict(draw=b, engine=u, type=k, factor=f, ame_pp=v, flagged=(b, u, f) in bad_uf)
                      for b, est in enumerate(draws) for (u, k, f), v in est.items()]
                     ).to_csv(draws_path, index=False)
    return out


def assemble(fit, frame, labels, outcome, interactions=False):
    """(ame table: jackknife t(G-1) inference + gate columns, joint table, replicates).

    The full-sample fit gets the refits' identification check (left_out = "full");
    interactions=True also carries the interaction contrasts in the replicates."""
    full = _flag("full", _sensitivity(fit, frame, interactions))
    _unidentified(full)
    se, reps = jackknife(frame, interactions=interactions)
    reps.attrs["unidentified"] = pd.DataFrame(
        full + reps.attrs["unidentified"].to_dict("records"), columns=UNID_COLS)
    table = ame(fit, frame).merge(se.rename(columns={"unit": "engine"}), how="left",
                                  on=["engine", "type", "factor"], validate="one_to_one")
    g = len(reps)
    q = stats.t.ppf(.975, g - 1)
    table["ci_low"] = table.ame_pp - q * table.se_jack
    table["ci_high"] = table.ame_pp + q * table.se_jack
    t = np.divide(table.ame_pp.to_numpy(), table.se_jack.to_numpy(),
                  out=np.zeros(len(table)), where=table.se_jack.to_numpy() > 0)
    table["p_t"] = 2 * stats.t.sf(np.abs(t), g - 1)
    joint = joint_wald(table, reps, labels, outcome)
    table = gatekeep(table, joint, tested_types(outcome))
    table.insert(0, "labels", labels)
    table.insert(1, "outcome", outcome)
    cols = ["labels", "outcome", "engine", "type", "factor", "ame_pp", "se_jack", "ci_low",
            "ci_high", "p_t", "tested", "gate_p", "gate_pass", "p_within", "n_answers",
            "n_scenarios", "estimand", "method"]
    return table[cols], joint, reps


def coefs(fit, labels, outcome):
    """Cell coefficients per grp (column `engine`) x type x non-NEU cell; NaN if unsupported."""
    idx = {n: j for j, n in enumerate(fit.names)}
    grps = list(dict.fromkeys(v.rsplit("|", 1)[0] for v in fit.ce))
    return pd.DataFrame([
        dict(labels=labels, outcome=outcome, engine=g, type=k, term=c,
             coef=(fit.params[j, idx[f"ce:{g}|{c}"]]
                   if fit.support[j, idx[f"ce:{g}|{c}"]] else np.nan))
        for j, k in enumerate(fit.types[:-1]) for g in grps for c in TERMS])


def _demo():
    """Self-check: exact shares from known parameters are recovered by the fit."""
    rng = np.random.default_rng(0)
    frame = simulate(rng, noise=False)
    fit = fit_fmnl(frame)
    got = ame(fit, frame).set_index(["engine", "type", "factor"]).ame_pp
    want = frame.attrs["true_ame"]
    assert fit.converged, fit.max_gradient
    assert max(abs(got[key] - v) for key, v in want.items()) < 1e-6
    print("ok", fit.iterations, fit.loglik)


def simulate(rng, noise=True, n_scen=5, n_day=4, per_cell=20, draws=20, het_sd=0.0):
    """Small synthetic factorial with known cell effects (used by _demo and tests).

    Contract frame (chat_id, grp = engine, attrs["types"] = TYPES). het_sd > 0 adds
    scenario-specific cell effects ~ N(0, het_sd) on the logit scale (engine x scenario x
    cell x type): the truth is then the grp = engine|scenario model and attrs["true_ame"]
    also holds the engine|scenario units.
    """
    scen = [f"S{i}" for i in range(n_scen)]
    days = [f"2026-09-{i + 1:02d}" for i in range(n_day)]
    rows = [(s, e, d, c) for s in scen for e in ENGINES for d in days
            for c in CELL_NAMES for _ in range(per_cell)]
    frame = pd.DataFrame(rows, columns=["scenario", "engine", "day", "cell"])
    bits = frame.cell.map(lambda c: (int(c.startswith("ADV")), int("FEA" in c), int("PRC" in c)))
    for i, f in enumerate(FACTORS):
        frame[f] = [b[i] for b in bits]
    frame.insert(0, "chat_id", [f"c{i}" for i in range(len(frame))])
    frame["grp"] = frame.engine
    frame.attrs["types"] = list(TYPES)
    k = len(TYPES) - 1
    fe, names, ce, _ = _design(frame)
    n_se, n_ed = (sum(nm.startswith(p) for nm in names) for p in ("se:", "ed:"))
    true = np.zeros((k, len(names)))
    true[:, :n_se] = rng.normal(0, .3, (k, n_se))
    true[:, n_se:n_se + n_ed] = rng.normal(0, .15, (k, n_ed))
    true[:, n_se + n_ed:] = rng.normal(0, .4, (k, len(ce)))
    truth_frame = frame
    if het_sd:
        truth_frame = frame.assign(grp=frame.engine + "|" + frame.scenario)
        fe, names, ce_s, _ = _design(truth_frame)
        pooled = [ce.index(f"{v.split('|')[0]}|{v.rsplit('|', 1)[1]}") for v in ce_s]
        true = np.hstack([true[:, :n_se + n_ed], true[:, n_se + n_ed:][:, pooled]
                          + rng.normal(0, het_sd, (k, len(ce_s)))])
        ce = ce_s
    active = np.ones((len(frame), k), bool)
    p = _probs(_X(fe, _cells(truth_frame), ce), true, active)
    counts = rng.multinomial(draws, p) if noise else p * draws
    for j, t in enumerate(TYPES):
        frame[f"count_{t}"] = counts[:, j]
        frame[f"share_{t}"] = counts[:, j] / counts.sum(axis=1)
    frame["n"] = counts.sum(axis=1)
    truth = Fit(true, names, active, 0.0, 0, True, 0.0, 0, 0.0, fe=fe,
                support=np.ones_like(true, bool), removed_sets={"se": set(), "ed": set(), "ce": set()},
                types=list(TYPES), ce=ce)
    frame.attrs["true_ame"] = ame_raw(truth, truth_frame)
    return frame


if __name__ == "__main__":
    _demo()
