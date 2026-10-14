"""Inference on independent, equally weighted scenario contrasts.

The cluster is the scenario, not a response, prompt, source, or engine. Every
bootstrap draw keeps all outcomes and factorial cells of a scenario together.
Wild p-values impose the zero-mean null using Rademacher signs on scenario
contrasts; these are bootstrap tests, not assignment-based randomization tests.
Validity requires independent scenario clusters and a suitable approximation
to the distribution of cluster errors. The scenarios are not a probability
sample of commercial demand.
"""
from itertools import combinations

import numpy as np
from scipy import stats

FACTORS = ("adv", "fea", "prc")
TERMS = [":".join(c) for n in (1, 2, 3) for c in combinations(FACTORS, n)]
CELL_BITS = np.array([(a, f, p) for a in (0, 1) for f in (0, 1) for p in (0, 1)])
CELL_NAMES = ["+".join(["ADV" if a else "NEU"] + (["FEA"] if f else [])
                       + (["PRC"] if p else [])) for a, f, p in CELL_BITS]


def contrast_weights(term):
    """Main effects are high-minus-low means; interactions are differences
    in differences, averaged equally over factors outside the interaction.
    """
    columns = [FACTORS.index(f) for f in term.split(":")]
    signs = np.prod(2 * CELL_BITS[:, columns] - 1, axis=1)
    return signs / (2 ** (3 - len(columns)))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Holm family contains missing p-values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def _tstat(mean, variance, n):
    se = np.sqrt(np.maximum(variance, 0) / n)
    out = np.divide(mean, se, out=np.zeros_like(np.asarray(mean, dtype=float)),
                    where=se > 1e-15)
    return np.where((se <= 1e-15) & (np.abs(mean) > 1e-15),
                    np.copysign(np.inf, mean), out)


def infer_mean(values, signs):
    """Student-t CI and null-imposed, studentized wild bootstrap p-value."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError("Need at least three complete scenario contrasts")
    g = len(x)
    if signs.shape[1] != g:
        raise ValueError("Bootstrap cluster count mismatch")
    mean, var = x.mean(), x.var(ddof=1)
    se = np.sqrt(var / g)
    t = float(_tstat(mean, var, g))
    means = signs @ x / g
    variances = (np.square(x).sum() - g * means ** 2) / (g - 1)
    tb = _tstat(means, variances, g)
    p_wild = (1 + np.count_nonzero(np.abs(tb) >= abs(t) - 1e-12)) / (len(signs) + 1)
    q = stats.t.ppf(0.975, g - 1)
    return {"estimate": mean, "se": se, "ci_low": mean - q * se,
            "ci_high": mean + q * se, "p_t": 2 * stats.t.sf(abs(t), g - 1),
            "p_wild": p_wild, "n_scenarios": g, "df": g - 1}


def joint_test(values, signs):
    """Joint zero effect across outcomes, with the same scenario wild draws.

    Drop one composition category before calling. Rank deficiency is reported
    explicitly; the pseudoinverse tests the estimable outcome subspace.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 3:
        raise ValueError("Need a complete scenario-by-outcome matrix")
    g, k = x.shape
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise ValueError("No scenario variation for joint inference")
    mean = x.mean(axis=0)
    observed = float(g * mean @ np.linalg.pinv(covariance) @ mean)
    cross = x.T @ x
    exceed = 0
    for block in np.array_split(signs, max(1, len(signs) // 256)):
        means = block @ x / g
        covs = (cross[None, :, :] - g * means[:, :, None] * means[:, None, :]) / (g - 1)
        inv = np.linalg.pinv(covs, hermitian=True)
        boot = g * np.einsum("bi,bij,bj->b", means, inv, means)
        exceed += np.count_nonzero(boot >= observed - 1e-10)
    return {"wald": observed, "rank": rank, "n_outcomes": k,
            "n_scenarios": g, "p_wild": (1 + exceed) / (len(signs) + 1)}


def envelope(cube, draws):
    """Cube = scenario x eight cells. Scenario bootstrap includes extrema
    selection in each draw. Intervals are exploratory percentile intervals,
    not a guarantee of coverage at tied or nearly tied cell means.
    """
    x = np.asarray(cube, dtype=float)
    if x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError("Envelope requires eight complete factorial cells")
    theta = x.mean(axis=0)
    boot = x[draws].mean(axis=1)
    widths = np.ptp(boot, axis=1)
    return {"min": theta.min(), "max": theta.max(), "width": np.ptp(theta),
            "width_ci_low": np.quantile(widths, .025),
            "width_ci_high": np.quantile(widths, .975),
            "min_cell": CELL_NAMES[int(theta.argmin())],
            "max_cell": CELL_NAMES[int(theta.argmax())]}
