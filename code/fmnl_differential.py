"""Differential test of the two FMNL routes (agreement target).

For every <labels>_<outcome> dir with an ame.csv under both roots (scipy route
`out/fmnl`, Poisson route `out/fmnl_poisson`; both overridable, e.g. for test runs):

- AMEs: max |d ame_pp| (target < 1e-3 pp) and jackknife SEs: max |d se_jack|, each
  with where; the SE row also counts the refits (both routes) with an AME-relevant
  collinear drop in that engine (`reason != level`; no reason column = relevant) - such a
  refit's AMEs are a normalisation choice - and the AME row says whether its engine has
  one in the full fit.
- coefficients on identified directions (target |d| < 1e-5): both non-NaN (NaN =
  separated, a -inf limit) and the (engine, type) block free of collinear terms in the
  full fit of either route (`unidentified.csv`, refit == "full"; no refit column = full).
  A block that lost its reference cell has an unidentified level: pyfixest pins one dummy
  at 0, the scipy route takes the min-norm point, so the whole block moves by a constant
  at identical likelihood (jev/citation, chatgpt-ui/OTHER: 2.33). NaN on one side only,
  outside such blocks, is a separation disagreement and fails the check.
- jackknife replicates, when both dirs carry `jack_reps.csv`: max |d| per refit, the
  refits over the AME target, and which of those recorded a collinear term;
- joint tests: |d p_F| and both ranks per (engine, factor) row;
- the multinomial quasi-loglik of fit_meta.json (equal loglik = same optimum).

Writes differential.csv (one row per dir), differential_joint.csv (one row per joint
test) and differential.json (summary incl. pass flags and a note) to --out (default:
the Poisson root).

CLI: fmnl_differential.py [--scipy DIR] [--poisson DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COEF_TOL, AME_TOL = 1e-5, 1e-3  # agreement tolerances


def unidentified(d):
    """Collinear terms a route recorded: DataFrame with refit, engine, type, reason."""
    f = d / "unidentified.csv"
    u = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["engine", "type"])
    return u.assign(refit=u.refit.astype(str) if "refit" in u else "full",
                    reason=u.reason if "reason" in u else "collinear")


def top(frame, col, keys):
    """(max of col, 'k1/k2/..' where); (nan, '') when all NaN."""
    if not frame[col].notna().any():
        return np.nan, ""
    r = frame.loc[frame[col].idxmax()]
    return float(r[col]), "/".join(str(r[k]) for k in keys)


def loglik(d):
    f = d / "fit_meta.json"
    return json.loads(f.read_text()).get("loglik", np.nan) if f.exists() else np.nan


def compare(a, b):
    """One dir, a = scipy route, b = Poisson route -> (summary row, joint rows, |d AME|)."""
    ame = pd.read_csv(a / "ame.csv").merge(pd.read_csv(b / "ame.csv"), how="outer",
                                           on=["engine", "type", "factor"],
                                           suffixes=("_s", "_p"), validate="one_to_one")
    ame["d_ame"] = (ame.ame_pp_s - ame.ame_pp_p).abs()
    ame["d_se"] = (ame.se_jack_s - ame.se_jack_p).abs()

    u = pd.concat([unidentified(a).assign(route="s"), unidentified(b).assign(route="p")],
                  ignore_index=True)
    full, relevant = u.refit.eq("full"), u.reason.ne("level")
    blocked = set(zip(u.engine[full], u.type[full]))
    coef = pd.read_csv(a / "coefs.csv").merge(pd.read_csv(b / "coefs.csv"), how="outer",
                                              on=["engine", "type", "term"],
                                              suffixes=("_s", "_p"), validate="one_to_one")
    free = np.array([(e, k) not in blocked for e, k in zip(coef.engine, coef.type)], bool)
    both = coef.coef_s.notna() & coef.coef_p.notna() & free
    coef["d_coef"] = (coef.coef_s - coef.coef_p).abs().where(both)
    mismatch = int(((coef.coef_s.isna() != coef.coef_p.isna()) & free).sum())

    joint = pd.read_csv(a / "joint.csv").merge(pd.read_csv(b / "joint.csv"), how="outer",
                                               on=["engine", "factor"], suffixes=("_s", "_p"),
                                               validate="one_to_one")
    for col in ("wald_s", "wald_p", "rank_s", "rank_p", "p_F_s", "p_F_p"):
        if col not in joint:  # outputs from before the rank-adjusted F
            joint[col] = np.nan
    joint["d_p_F"] = (joint.p_F_s - joint.p_F_p).abs()
    joint.insert(0, "dir", a.name)

    over, max_rep, rep_where = [], np.nan, ""
    if (a / "jack_reps.csv").exists() and (b / "jack_reps.csv").exists():
        reps = pd.read_csv(a / "jack_reps.csv").merge(
            pd.read_csv(b / "jack_reps.csv"), on=["refit", "engine", "type", "factor"],
            suffixes=("_s", "_p"), validate="one_to_one")
        reps["d_rep"] = (reps.ame_pp_s - reps.ame_pp_p).abs()
        max_rep, rep_where = top(reps, "d_rep", ["refit", "engine", "type", "factor"])
        worst = reps.groupby(reps.refit.astype(str)).d_rep.max()
        over = sorted(worst.index[worst >= AME_TOL])

    max_ame, ame_where = top(ame, "d_ame", ["engine", "type", "factor"])
    max_se, se_where = top(ame, "d_se", ["engine", "type", "factor"])
    max_coef, coef_where = top(coef, "d_coef", ["engine", "type", "term"])
    max_pf, pf_where = top(joint, "d_p_F", ["engine", "factor"])
    in_engine = lambda where, e: where.split("/")[0] in ("all", e)  # noqa: E731
    refits = u[~full & relevant & u.engine.map(lambda e: in_engine(se_where, e)).astype(bool)]
    complete = bool(ame[["ame_pp_s", "ame_pp_p"]].notna().all(axis=None))
    row = dict(
        dir=a.name, n_ame=len(ame), max_ame_diff_pp=max_ame,
        median_ame_diff_pp=float(ame.d_ame.median()), ame_where=ame_where,
        ame_where_engine_blocked=any(in_engine(ame_where, e) for e in u.engine[full & relevant]),
        max_se_diff_pp=max_se, se_where=se_where,
        se_where_unidentified_refits=len(set(zip(refits.route, refits.refit))),
        max_rep_diff_pp=max_rep, rep_where=rep_where, refits_over_ame_tol=";".join(over),
        refits_over_tol_unidentified=";".join(sorted(set(over) & set(u.refit[~full & relevant]))),
        n_coef=int(both.sum()), n_coef_blocked=int((~free).sum()),
        blocked=";".join(sorted(f"{e}/{k}" for e, k in blocked)),
        coef_nan_mismatch=mismatch, max_coef_diff=max_coef, coef_where=coef_where,
        n_joint=len(joint), max_p_F_diff=max_pf, p_F_where=pf_where,
        rank_mismatch=int((joint.rank_s != joint.rank_p).sum()),
        loglik_diff=abs(loglik(a) - loglik(b)),
        ame_pass=complete and bool(max_ame < AME_TOL),
        coef_pass=mismatch == 0 and bool(both.any()) and bool(max_coef < COEF_TOL))
    return row, joint[["dir", "engine", "factor", "wald_s", "wald_p", "rank_s", "rank_p",
                       "p_F_s", "p_F_p", "d_p_F"]], ame.d_ame


def run(scipy_root, poisson_root, out):
    scipy_root, poisson_root = Path(scipy_root), Path(poisson_root)
    dirs = sorted(d.name for d in scipy_root.iterdir()
                  if (d / "ame.csv").exists() and (poisson_root / d.name / "ame.csv").exists())
    if not dirs:
        raise SystemExit(f"no <labels>_<outcome> dir in both {scipy_root} and {poisson_root}")
    parts = [compare(scipy_root / d, poisson_root / d) for d in dirs]
    rows = pd.DataFrame([p[0] for p in parts])
    joint = pd.concat([p[1] for p in parts], ignore_index=True)
    diffs = pd.concat([p[2] for p in parts])

    at = lambda col, where: (  # noqa: E731 - "dir where" of the largest col
        f"{rows.dir[rows[col].idxmax()]} {rows[where][rows[col].idxmax()]}"
        if rows[col].notna().any() else "n/a")
    blocked = "; ".join(f"{d}: {b}" for d, b in zip(rows.dir, rows.blocked) if b) or "none"
    over = "; ".join(f"{d}: {o} (collinear: {u or 'none'})" for d, o, u in zip(
        rows.dir, rows.refits_over_ame_tol, rows.refits_over_tol_unidentified) if o) or "none"
    summary = dict(
        n_dirs=len(rows), dirs=dirs, n_compared=int(rows.n_ame.sum()),
        max_ame_diff_pp=float(rows.max_ame_diff_pp.max()),
        median_ame_diff_pp=float(diffs.median()),
        n_coefs=int(rows.n_coef.sum()), max_coef_diff=float(rows.max_coef_diff.max()),
        max_se_diff_pp=float(rows.max_se_diff_pp.max()),
        max_p_F_diff=float(rows.max_p_F_diff.max()),
        max_rep_diff_pp=float(rows.max_rep_diff_pp.max()),
        coef_pass=bool(rows.coef_pass.all()), ame_pass=bool(rows.ame_pass.all()),
        targets=dict(coef=COEF_TOL, ame_pp=AME_TOL))
    summary["pass"] = summary["coef_pass"] and summary["ame_pass"]
    summary["note"] = (
        f"{len(rows)} dirs. AMEs: max {summary['max_ame_diff_pp']:.2g} pp at "
        f"{at('max_ame_diff_pp', 'ame_where')}, median {summary['median_ame_diff_pp']:.2g} "
        f"(target {AME_TOL:g}). Coefficients: max {summary['max_coef_diff']:.2g} on "
        f"{summary['n_coefs']} identified directions (target {COEF_TOL:g}); blocks with a "
        f"collinear term in the full fit, excluded: {blocked}. Jackknife SE: max "
        f"{summary['max_se_diff_pp']:.2g} pp at {at('max_se_diff_pp', 'se_where')}; "
        f"refits over the AME target (dirs with replicates on both sides): {over}. "
        f"Joint p_F: max |d| {summary['max_p_F_diff']:.2g} at "
        f"{at('max_p_F_diff', 'p_F_where')}.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "differential.csv", index=False)
    joint.to_csv(out / "differential_joint.csv", index=False)
    strict = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in summary.items()}
    (out / "differential.json").write_text(json.dumps(strict, indent=1, allow_nan=False) + "\n")
    return rows, joint, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scipy", default=str(HERE / "out" / "fmnl"))
    ap.add_argument("--poisson", default=str(HERE / "out" / "fmnl_poisson"))
    ap.add_argument("--out", default=None, help="default: the Poisson root")
    args = ap.parse_args()
    rows, _, summary = run(args.scipy, args.poisson, args.out or args.poisson)
    print(rows[["dir", "max_ame_diff_pp", "max_se_diff_pp", "se_where", "n_coef",
                "max_coef_diff", "max_p_F_diff", "ame_pass", "coef_pass"]].to_string(index=False))
    print(summary["note"])


if __name__ == "__main__":
    main()
