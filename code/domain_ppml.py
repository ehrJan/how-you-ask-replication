"""Domain-level stage model (Appendix F): stacked retrieval + citation PPML.

Cell = prompt x engine x day x domain on the build_runs('ui') run universe (factorial and
template-replicate prompts; seeds have no factor values). Retrieval and citation counts
are stacked with a `cited` indicator; zero cells are case-control sampled per topic
(ZERO_RATIO per positive cell, weight = zero universe / sampled), exposure log(runs).
The model never conditions on retrieval.

  delta_fk   factor effect on retrieval (log-rate) per domain type
  tau_fk     additional effect at the citation stage
  delta+tau  effect on citations

Fixed effects: domain x engine, domain x topic, engine x day. Standard errors clustered
by scenario (the paper's inferential unit). --engine fits one engine alone.

Output -> out/revised/domain_ppml[_<engine>].csv (all terms) and _summary.csv
"""
import argparse

import numpy as np
import pandas as pd
import pyfixest as pf

from descriptives import OUT, load_runs
from design import DATA
from jev_labels import swap

FACTORS = ["adv", "fea", "prc"]
SEED = 42
ZERO_RATIO = 4        # sampled zero cells per positive cell
MIN_DOMAIN_ROWS = 3   # drop ultra-rare domains (fixed-effect groups without signal)
KEYS = ["prompt_id", "model_id", "date"]


def domain_cells(runs):
    """Positive cells: retrieval rows and cited rows per prompt x engine x day x domain."""
    m = swap(pd.read_parquet(DATA / "sources.parquet"))
    m = m[m.chat_id.isin(runs.chat_id)].merge(
        runs[["chat_id", "scenario", "topic", "cell"]], on="chat_id", validate="many_to_one")
    m["cited"] = m.metric_citation_count >= 1
    n_runs = runs.groupby(KEYS).chat_id.nunique().rename("runs")
    cells = (m.groupby(KEYS + ["domain"], observed=True)
             .agg(n_retrieved=("chat_id", "size"), n_cited=("cited", "sum"),
                  domain_type=("domain_classification", lambda s: s.mode().iloc[0]),
                  scenario=("scenario", "first"), topic=("topic", "first"), cell=("cell", "first"))
             .reset_index().merge(n_runs, on=KEYS))
    return cells


def sample_zeros(cells, rng):
    """Uniform zero-cell sample per topic; weight = N_zeros_universe / n_sampled."""
    run_cells = cells[KEYS + ["runs", "cell", "topic", "scenario"]].drop_duplicates(KEYS)
    out = []
    for topic, grp in cells.groupby("topic"):
        domains = grp[["domain", "domain_type"]].drop_duplicates("domain")
        rc = run_cells[run_cells.topic == topic].reset_index(drop=True)
        n_zero_universe = len(rc) * len(domains) - len(grp)
        m = min(ZERO_RATIO * len(grp), n_zero_universe)
        if m <= 0:
            continue
        draw = int(m * 1.3)  # oversample, drop collisions with positives, trim to m
        z = pd.concat([rc.iloc[rng.integers(0, len(rc), draw)].reset_index(drop=True),
                       domains.iloc[rng.integers(0, len(domains), draw)].reset_index(drop=True)], axis=1)
        pos = set(map(tuple, grp[KEYS + ["domain"]].values))
        z = z[[tuple(r) not in pos for r in z[KEYS + ["domain"]].values]]
        z = z.drop_duplicates(KEYS + ["domain"]).head(m)
        z["n_retrieved"], z["n_cited"] = 0, 0
        z["weight"] = n_zero_universe / len(z)
        out.append(z)
    return pd.concat(out, ignore_index=True)


def fit(cells):
    keep = cells.groupby("domain").n_retrieved.transform("sum") >= MIN_DOMAIN_ROWS
    cells = cells[keep].assign(weight=1.0)
    zeros = sample_zeros(cells, np.random.default_rng(SEED))
    cols = KEYS + ["domain", "domain_type", "runs", "cell", "topic", "scenario",
                   "n_retrieved", "n_cited", "weight"]
    base = pd.concat([cells[cols], zeros[cols]], ignore_index=True)
    s = pd.concat([base.assign(n=base.n_retrieved, cited=0),
                   base.assign(n=base.n_cited, cited=1)], ignore_index=True)
    parts = s.cell.str.split("+")
    s["adv"] = (parts.str[0] == "ADV").astype(int)
    s["fea"] = parts.apply(lambda p: "FEA" in p).astype(int)
    s["prc"] = parts.apply(lambda p: "PRC" in p).astype(int)
    s["log_runs"] = np.log(s.runs.astype(float))
    s["n"], s["weight"] = s.n.astype(float), s.weight.astype(float)
    s["date"] = s.date.astype(str)
    s["dom_eng"] = s.domain + "_" + s.model_id
    s["eng_date"] = s.model_id + "_" + s.date
    lab = " + ".join(FACTORS)
    fml = (f"n ~ C(domain_type):({lab}) + cited + cited:C(domain_type) "
           f"+ cited:C(domain_type):({lab}) | dom_eng + domain^topic + eng_date")
    return pf.fepois(fml, data=s, offset="log_runs", weights="weight",
                     vcov={"CRV1": "scenario"}).tidy().reset_index().rename(columns={"Coefficient": "term"})


def summarize(t):
    ext = t.term.str.extract(r"C\(domain_type\)\[(?P<domain_type>\w+)\]:(?P<label>adv|fea|prc)")
    t = pd.concat([t, ext], axis=1).dropna(subset=["label"])
    t["block"] = np.where(t.term.str.startswith("cited:"), "tau", "delta")
    m = t.pivot_table(index=["label", "domain_type"], columns="block", values=["Estimate", "Std. Error"])
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    m = m.reset_index()
    m["citation"] = m.Estimate_delta + m.Estimate_tau
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", choices=["jev"], default="jev")
    ap.add_argument("--engine", default="all", choices=["all", "chatgpt-ui", "gemini-ui", "perplexity-ui"])
    ap.add_argument("--runs-pickle", default=None, help="frozen build_runs('ui') pickle")
    args = ap.parse_args()
    runs, _ = load_runs(args.labels, "ui", args.runs_pickle)
    runs = runs[runs.role.isin(["factorial", "template-replicate"])]
    if args.engine != "all":
        runs = runs[runs.model_id == args.engine]
    t = fit(domain_cells(runs))
    tag = "" if args.engine == "all" else f"_{args.engine}"
    t.to_csv(OUT / f"domain_ppml{tag}.csv", index=False)
    m = summarize(t)
    m.to_csv(OUT / f"domain_ppml{tag}_summary.csv", index=False)
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
