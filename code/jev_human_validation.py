"""Accuracy of the JEV and Peec source-type labels against a human labeller (Section 3.3).

Sample (data/jev_human_labels.csv): 150 domains and 150 URLs drawn from the sources of an
earlier panel of 250 shopping prompts, labelled by the same JEV classifier. Within each kind,
two strata: items on which Peec and JEV agree (60 drawn) and disagree (90 drawn). One author
labelled every item blind to both classifier outputs, using the classifiers' category
definitions verbatim.

Estimator. Strata are sampled at unequal rates, so accuracy is the stratified mean
sum_s (N_s/N) mean_s with Var = sum_s (N_s/N)^2 var_s / n_s; the JEV-minus-Peec
difference uses the same formula on the paired item-level difference. Items marked
`unreachable` (the page no longer loads; the human label is then OTHER) are excluded in
the primary sample and kept in the `all` sample.

Output -> out/revised/jev_human_validation.csv
"""
import math

import pandas as pd

from descriptives import OUT
from design import DATA


def stratified(df: pd.DataFrame, col: str) -> tuple[float, float]:
    N = df.groupby("stratum").N_stratum.first()
    W = N / N.sum()
    g = df.groupby("stratum")[col]
    est = (W * g.mean()).sum()
    se = math.sqrt((W ** 2 * g.var(ddof=1) / g.size()).sum())
    return est, se


def main():
    lab = pd.read_csv(DATA / "jev_human_labels.csv", keep_default_na=False)
    lab["jev_hit"] = (lab.jev_label == lab.human_label).astype(float)
    lab["peec_hit"] = (lab.peec_label == lab.human_label).astype(float)
    lab["diff"] = lab.jev_hit - lab.peec_hit
    rows = []
    for kind, block in lab.groupby("kind"):
        for sample, d in (("primary", block[~block.unreachable]), ("all", block)):
            dis = d[d.stratum == "disagree"]
            row = dict(kind=kind, sample=sample, n=len(d), n_agree=int((d.stratum == "agree").sum()),
                       n_disagree=len(dis))
            for col, name in (("jev_hit", "jev"), ("peec_hit", "peec"), ("diff", "jev_minus_peec")):
                est, se = stratified(d, col)
                row.update({name: est, f"{name}_lo": est - 1.96 * se, f"{name}_hi": est + 1.96 * se})
            row.update(h2h_jev=int(dis.jev_hit.sum()), h2h_peec=int(dis.peec_hit.sum()),
                       h2h_neither=int(len(dis) - dis.jev_hit.sum() - dis.peec_hit.sum()))
            rows.append(row)
    res = pd.DataFrame(rows)
    path = OUT / "jev_human_validation.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
