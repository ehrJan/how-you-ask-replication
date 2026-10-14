"""Run the FMNL (scipy route) for one label set and one outcome.

One label set per process: `--labels jev` applies the JEV source-type labels (the only label set
shipped). Output under
out/fmnl/[<window>/]<labels>_<outcome>/:

  coefs.csv, ame.csv (jackknife t(G-1) + gate columns: tested, gate_p, gate_pass, p_within),
  joint.csv (rank-adjusted F), interactions.csv (interaction contrasts + per unit x type
  additivity F over the four), jack_reps.csv.gz (wide: header rows unit/type/factor, index
  left-out scenario), unidentified.csv, fit_meta.json; --boot B adds ame_boot.csv (p_boot)
  and boot_draws.csv.gz.

--window discovery|holdout|full calls build_runs("ui", window=...) and writes a `window`
column; without it build_runs("ui") runs exactly as before. --runs-pickle PATH loads frozen
build_runs output (a dict with key "runs") instead of calling build_runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fmnl

HERE = Path(__file__).resolve().parent
SEED = 20260925


def interactions(fit, frame, reps, labels, outcome):
    """Interaction contrasts (jackknife SE, t(G-1) CI, p) + per unit x type additivity F."""
    t = fmnl.ame(fit, frame, interactions=True)
    t = t[t.factor.isin(fmnl.INTERACTIONS)].reset_index(drop=True)
    g = len(reps)
    r = reps[list(zip(t.engine, t.type, t.factor))]
    t["se_jack"] = np.sqrt((g - 1) / g * ((r - r.mean()) ** 2).sum()).to_numpy()
    q = stats.t.ppf(.975, g - 1)
    t["ci_low"], t["ci_high"] = t.ame_pp - q * t.se_jack, t.ame_pp + q * t.se_jack
    z = np.divide(t.ame_pp.to_numpy(), t.se_jack.to_numpy(), out=np.zeros(len(t)),
                  where=t.se_jack.to_numpy() > 0)
    t["p_t"] = 2 * stats.t.sf(np.abs(z), g - 1)
    add = []
    for (u, k), part in t.groupby(["engine", "type"], sort=False):
        w = fmnl.wald_F(part.ame_pp.to_numpy(), reps[list(zip(part.engine, part.type,
                                                               part.factor))].to_numpy())
        add.append(dict(engine=u, type=k, additivity_wald=w["wald"], additivity_q=w["q"],
                        additivity_F=w["F"], additivity_p_F=w["p_F"]))
    t = t.merge(pd.DataFrame(add), on=["engine", "type"], validate="many_to_one")
    t.insert(0, "labels", labels)
    t.insert(1, "outcome", outcome)
    return t


def run(runs, labels, outcome, boot=0, seed=SEED, draws_path=None, boot_keep_unidentified=False):
    """All tables of one label x outcome combination (pure: no files unless draws_path)."""
    t0 = time.perf_counter()
    frame = fmnl.long_frame(runs, outcome)
    fit = fmnl.fit_fmnl(frame)
    table, joint, reps = fmnl.assemble(fit, frame, labels, outcome, interactions=True)
    unid = reps.attrs["unidentified"]
    meta = dict(labels=labels, outcome=outcome, n_answers=len(frame),
                n_rows=len(frame) * len(fit.types), n_scenarios=int(frame.scenario.nunique()),
                n_params=int(fit.support.sum()), loglik=fit.loglik,
                converged=bool(fit.converged), iterations=fit.iterations,
                max_gradient=fit.max_gradient, separation_pairs_removed=fit.separation_removed,
                null_dims={e: list(v) for e, v in fit.null_dims.items()},
                n_refits_unidentified=int(unid.left_out[unid.left_out.ne("full")].nunique()),
                n_refits_extra_null=sum(any(a > b for a, b in nd.values())
                                        for nd in reps.attrs["null_dims"].values()),
                max_refit_gradient=reps.attrs["max_gradient"],
                tested_types=fmnl.tested_types(outcome), fit_seconds=fit.seconds)
    out = dict(coefs=fmnl.coefs(fit, labels, outcome), ame=table, joint=joint,
               interactions=interactions(fit, frame, reps, labels, outcome),
               unidentified=unid, reps=reps, meta=meta)
    if boot:
        b = fmnl.pairs_bootstrap(frame, boot, seed, draws_path, keep_unidentified=boot_keep_unidentified)
        b.insert(0, "labels", labels)
        b.insert(1, "outcome", outcome)
        out["boot"] = b
        meta["boot_reps"], meta["boot_seed"] = boot, seed
        if boot_keep_unidentified:
            meta["boot_keep_unidentified"] = True
            meta["boot_draws_flagged"] = b.attrs["flagged_draws"]
    meta["total_seconds"] = time.perf_counter() - t0
    return out


def load_runs(args):
    if args.runs_pickle:
        runs = pd.read_pickle(args.runs_pickle)["runs"]
        if args.window:
            if "window" not in runs:
                raise SystemExit(f"--window {args.window}: the pickle has no window column")
            if args.window != "full":
                runs = runs[runs.window.eq(args.window)]
        return runs
    from revised_analysis import build_runs
    return (build_runs("ui", window=args.window) if args.window else build_runs("ui"))[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--outcome", choices=list(fmnl.PREFIX), required=True)
    ap.add_argument("--window", choices=["discovery", "holdout", "full"], default=None)
    ap.add_argument("--runs-pickle", default=None)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--boot-keep-unidentified", action="store_true",
                    help="flag unidentified bootstrap draws instead of failing (all-draws + *_excl columns)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.labels == "jev":
        sys.path.insert(0, str(HERE))
    base = HERE / "out" / "fmnl" / (args.window or "")
    out = Path(args.out) if args.out else base / f"{args.labels}_{args.outcome}"
    out.mkdir(parents=True, exist_ok=True)
    res = run(load_runs(args), args.labels, args.outcome, args.boot, args.seed,
              out / "boot_draws.csv.gz" if args.boot else None, args.boot_keep_unidentified)
    res["meta"]["window"] = args.window
    for name in ("coefs", "ame", "joint", "interactions", "unidentified", "boot"):
        if name in res:
            t = res[name] if args.window is None else res[name].assign(window=args.window)
            t.to_csv(out / f"{'ame_boot' if name == 'boot' else name}.csv", index=False)
    res["reps"].to_csv(out / "jack_reps.csv.gz")
    (out / "fit_meta.json").write_text(json.dumps(res["meta"], indent=2) + "\n")
    print(json.dumps(res["meta"], indent=2))
    print(res["ame"][res["ame"].engine.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
