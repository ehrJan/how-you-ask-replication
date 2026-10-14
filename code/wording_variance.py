"""Wording variance from nested random effects, with a method-of-moments estimate as
comparison.

Per engine, statsmodels MixedLM (REML) on the answer-level share (pp) of CORPORATE and
EDITORIAL, roles factorial + template-replicate:
    y = C(cell) + C(day) + u_scenario + v_(scenario x cell) + w_prompt + e
groups = scenario (random intercept); variance components nested in scenario:
scenario x cell (0 + C(cell)) and prompt = wording (0 + C(prompt_id)); residual = run
noise. sigma_wording is identified by the three replicate cells (two templates per
scenario x cell); in the other cells prompt and scenario x cell coincide.
metric citation_share (answers with >= 1 citation) is primary; source_share (answers
with >= 1 source) is the like-for-like row for a moment estimate on all source rows. sigma_word_mom_same_sample applies the moment formula to
exactly this model's sample. engine 'all' = sqrt of the engine-mean variance.

Output -> out/revised/wording_variance.csv (rows keyed by `labels`)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from descriptives import OUT, load_runs, upsert

HERE = Path(__file__).resolve().parent
TYPES = ["CORPORATE", "EDITORIAL"]
COMPONENTS = ["sigma_scenario", "sigma_scenario_cell", "sigma_wording", "sigma_run"]


def fit_components(d, col):
    """Nested REML standard deviations (pp) for one engine and outcome column."""
    x = pd.DataFrame({"y": 100 * d[col].to_numpy(float), "scenario": d.scenario.to_numpy(object),
                      "cell": d.cell.to_numpy(object), "prompt_id": d.prompt_id.to_numpy(object),
                      "day": pd.to_datetime(d.date).dt.strftime("%m%d").to_numpy(object)})
    vcs = {"cell": "0 + C(cell)", "prompt": "0 + C(prompt_id)"}
    # Single optimizers stall off the REML optimum on some draws (synthetic checks: the
    # default bfgs->lbfgs chain 27 log-lik units short, sigma_scenario +56%; lbfgs alone
    # 18 units short, sigma_wording -65%), and near sigma_wording = 0 none converges
    # (Gemini, JEV labels: |grad| 16-23). Fit each optimizer plus the boundary model
    # without the wording component (same fixed effects, so REML log-liks compare) and
    # keep the highest REML log-lik; `converged` reports that fit.
    fits = {}
    for method, vc_formula in [("lbfgs", vcs), ("bfgs", vcs), ("cg", vcs),
                               ("lbfgs_wording0", {"cell": vcs["cell"]})]:
        model = smf.mixedlm("y ~ C(cell) + C(day)", x, groups="scenario", re_formula="1",
                            vc_formula=vc_formula)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(reml=True, method=[method.split("_")[0]])
        fits[method] = fit, caught, dict(zip(model.exog_vc.names, fit.vcomp))
    method = max(fits, key=lambda m: fits[m][0].llf)
    fit, caught, vc = fits[method]
    return dict(sigma_scenario=np.sqrt(fit.cov_re.iloc[0, 0]), sigma_scenario_cell=np.sqrt(vc["cell"]),
                sigma_wording=np.sqrt(vc.get("prompt", 0.)), sigma_run=np.sqrt(fit.scale), n=len(x),
                n_scenarios=x.scenario.nunique(), n_prompts=x.prompt_id.nunique(),
                converged=bool(fit.converged), optimizer=method, reml_loglik=fit.llf,
                fit_warnings="; ".join(sorted({str(w.message).split(".")[0][:70] for w in caught})))


def mom_pairs(d, col):
    """The replicate pairs on this sample (pp): scenario x cell with an
    original and a replicate prompt; d2 = squared difference of the prompt means,
    samp = var1/n1 + var2/n2 (run noise in that difference)."""
    g = (d.assign(y=100 * d[col].astype(float)).groupby(["scenario", "cell", "is_reword"]).y
         .agg(["mean", "var", "count"]).unstack("is_reword").dropna())
    return pd.DataFrame({"d2": (g[("mean", False)] - g[("mean", True)]) ** 2,
                         "samp": g[("var", False)] / g[("count", False)] + g[("var", True)] / g[("count", True)]})


def mom_sigma(pairs):
    """E[d2] = 2 sigma2_word + E[samp]  =>  sigma_word = sqrt(max(0, (mean d2 - mean samp) / 2))."""
    return np.sqrt(max(0., (pairs.d2.mean() - pairs.samp.mean()) / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], required=True)
    ap.add_argument("--runs-pickle", default=None, help="frozen build_runs('ui') pickle")
    args = ap.parse_args()
    runs, _ = load_runs(args.labels, "ui", args.runs_pickle)
    d = runs[runs.role.isin(["factorial", "template-replicate"])]
    mom_csv = pd.Series(dtype=float)  # cross-check column of the platform-label run; not used for JEV
    rows = []
    for metric in ("citation_share", "source_share"):
        for k in TYPES:
            col = f"{metric}_{k}"
            per, pairs = [], []
            for e, frame in d.groupby("model_id"):
                x = frame.dropna(subset=[col])
                pairs.append(mom_pairs(x, col))
                per.append(dict(engine=e, **fit_components(x, col),
                                sigma_word_mom_same_sample=mom_sigma(pairs[-1]), n_pairs=len(pairs[-1])))
            per.append(dict(engine="all", **{c: np.sqrt(np.mean([r[c] ** 2 for r in per])) for c in COMPONENTS},
                            n=sum(r["n"] for r in per), converged=all(r["converged"] for r in per),
                            sigma_word_mom_same_sample=mom_sigma(pd.concat(pairs)),
                            n_pairs=sum(len(p) for p in pairs)))
            for r in per:
                csv = mom_csv.get((r["engine"], k), np.nan) if args.labels == "peec" else np.nan
                rows.append(dict(labels=args.labels, metric=metric, type=k, **r,
                                 sigma_word_mom_template_variance_csv=csv))
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    upsert(res, OUT / "wording_variance.csv")
    print(res.drop(columns=["fit_warnings"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
