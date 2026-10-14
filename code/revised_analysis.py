"""Main analysis: standardized factorial contrasts on source-type shares.

Primary design: one original prompt per scenario x factorial cell (200 prompts).
Each cell has 16 collected runs per engine. First average runs within each
scenario/engine/cell; then average engines and scenarios equally. This estimates
the finite portfolio's standardized factorial contrasts without imposing common
factor effects across scenarios. Natural seeds and template replicates are not
part of the eight-cell main experiment.

The original primary outcome, citation share conditional on citations, remains
reported and labelled conditional. Companion outcomes give
zero-citation runs an explicit category: citation_mass_k = citation_share_k if
citations are recorded, zero otherwise. The six masses plus NO_CITATION sum to
one on every run. This is an operational exposure measure, not an imputation
of the undefined conditional share.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from design import DATA, DERIVED, complete_engine_days, exclusion_audit, source_chat_ids
from jev_labels import swap
from inference import (CELL_NAMES, FACTORS, TERMS, contrast_weights, envelope,
                       holm, infer_mean, joint_test)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "revised"
TYPES = ["CORPORATE", "EDITORIAL", "UGC", "REFERENCE", "INSTITUTIONAL", "OTHER"]
ENGINES = ["chatgpt-ui", "gemini-ui", "perplexity-ui"]
SEED = 20260908
MIN_RUNS_PER_CELL = 10
DISCOVERY_END = "2026-09-16"  # last discovery day; later days = holdout (not used for design decisions)
WINDOWS = ("full", "discovery", "holdout")


def build_runs(channel="ui", window="full", end=None):
    """Full run universe and normalized-URL counts, including empty answers.

    Duplicate normalized URLs count once per answer; citation flag is any
    positive record. Conflicting labels within a normalized source become OTHER
    and are exported for human review rather than resolved using outcomes.
    Engines are canonicalised and the mechanical exclusion rules applied
    (design.complete_engine_days); `channel='ui'` (primary) excludes the
    Perplexity API fallback days, `channel='all'` keeps them (`channel` column).
    Every run gets `window` = discovery (date <= DISCOVERY_END) / holdout;
    window='discovery' / 'holdout' keeps only that part. `end` (YYYY-MM-DD) drops
    runs dated after it; later loaded days serve only as rule-3 look-ahead (the last collection day is
    loaded but excluded here). Exclusions are
    evaluated on everything loaded first, so they never depend on `window` or `end`.
    `end=None` falls back to the STUDY2_END environment variable (set once by run_all.sh).
    It must be set: without it the look-ahead day would be analysed.
    """
    end = end or os.environ.get("STUDY2_END") or None
    if end is None:
        raise RuntimeError("STUDY2_END is not set (e.g. 2026-09-28, the last analysis day); "
                           "run via run_all.sh or export it")
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    u = pd.read_parquet(DATA / "chats_universe.parquet")
    design = pd.read_parquet(DERIVED / "prompt_design.parquet")
    if not u.chat_id.is_unique or not design.prompt_id.is_unique:
        raise ValueError("Run/design identifiers must be unique")
    u = complete_engine_days(u, channel, source_chat_ids(DATA / "sources.parquet"))
    u = u.merge(design, on="prompt_id", validate="many_to_one")
    u["date"] = pd.to_datetime(u["date"])
    u["window"] = np.where(u["date"] <= pd.Timestamp(DISCOVERY_END), "discovery", "holdout")
    if window != "full":
        u = u[u["window"] == window]
    if end is not None:
        u = u[u["date"] <= pd.Timestamp(end)]
    raw = swap(pd.read_parquet(DATA / "sources.parquet"))
    raw = raw[raw.chat_id.isin(u.chat_id)].copy()
    raw["cited"] = raw.metric_citation_count.ge(1)
    key = ["chat_id", "url_normalized"]
    if raw.url_normalized.isna().any() or raw.url_normalized.eq("").any():
        raise ValueError("Missing normalized URL")
    sources = raw.groupby(key, sort=False).agg(
        cited=("cited", "max"), type=("domain_classification", "first"),
        n_labels=("domain_classification", "nunique"), domain=("domain", "first"))
    conflicts = sources[sources.n_labels > 1].reset_index()
    sources.loc[sources.n_labels > 1, "type"] = "OTHER"
    if not sources.type.isin(TYPES).all():
        raise ValueError("Unexpected or missing source type")
    sources = sources.reset_index()
    for prefix, df in [("source", sources), ("citation", sources[sources.cited])]:
        count = df.groupby(["chat_id", "type"]).size().unstack(fill_value=0)
        count = count.reindex(columns=TYPES, fill_value=0)
        count.columns = [f"{prefix}_n_{k}" for k in TYPES]
        u = u.merge(count, left_on="chat_id", right_index=True, how="left", validate="one_to_one")
        cols = count.columns.tolist()
        u[cols] = u[cols].fillna(0).astype(int)
        u[f"{prefix}_n"] = u[cols].sum(axis=1)
        u[f"any_{prefix}"] = u[f"{prefix}_n"].gt(0).astype(float)
        for k in TYPES:
            share = u[f"{prefix}_n_{k}"] / u[f"{prefix}_n"].replace(0, np.nan)
            u[f"{prefix}_share_{k}"] = share
            u[f"{prefix}_mass_{k}"] = share.fillna(0)
            u[f"{prefix}_presence_{k}"] = u[f"{prefix}_n_{k}"].gt(0).astype(float)
        u[f"{prefix}_mass_NONE"] = 1 - u[f"any_{prefix}"]
        if not np.allclose(u[[f"{prefix}_mass_{k}" for k in TYPES + ["NONE"]]].sum(axis=1), 1):
            raise ValueError("Unconditional composition does not add to one")
    audit = {"raw_source_rows": len(raw), "unique_normalized_sources": len(sources),
             "duplicate_normalized_rows_removed": len(raw) - len(sources),
             "conflicting_source_labels": len(conflicts), "full_runs": len(u),
             "status_counts": u.status.value_counts().to_dict()}
    return u, sources, conflicts, audit


def cell_means(runs, columns, include_replicates=False):
    roles = ["factorial", "template-replicate"] if include_replicates else ["factorial"]
    d = runs[runs.role.isin(roles)]
    # Equal weight per prompt before combining template replicates; repeated
    # measurements are never independent experimental units.
    prompt = d.groupby(["scenario", "model_id", "cell", "prompt_id"], observed=True)[columns].mean()
    cells = prompt.groupby(["scenario", "model_id", "cell"], observed=True).mean()
    return cells


def as_cube(cells, column, engine, scenarios):
    if engine == "all":
        counts = cells[column].groupby(["scenario", "cell"]).count()
        if not counts.eq(len(ENGINES)).all():
            raise ValueError(f"Incomplete engines for {column}")
        values = cells[column].groupby(["scenario", "cell"]).mean()
    else:
        values = cells.xs(engine, level="model_id")[column]
    cube = values.unstack("cell").reindex(index=scenarios, columns=CELL_NAMES).to_numpy()
    if not np.isfinite(cube).all():
        raise ValueError(f"Missing cell mean: {column}, {engine}; do not silently drop scenarios")
    return cube


def analyze(runs, reps=9999, boot_reps=1999, out=None):
    OUT = out or globals()["OUT"]
    OUT.mkdir(parents=True, exist_ok=True)
    core = runs[runs.role == "factorial"]
    scenarios = sorted(core.scenario.unique())
    g = len(scenarios)
    counts = core.groupby(["scenario", "model_id", "cell"]).size()
    # Every scenario x engine x cell must be observed with enough runs. Run counts
    # per prompt may differ (dropped partial engine-days, provider capture failures):
    # cell means average within prompt first, so prompts keep equal weight and
    # engines enter the pooled estimand with equal weight regardless of run counts.
    if len(counts) != g * 3 * 8 or counts.min() < MIN_RUNS_PER_CELL:
        raise ValueError(f"Core design incomplete: {len(counts)} cells, min runs {counts.min()}")
    print("runs per scenario x engine x cell:",
          counts.groupby(level="model_id").agg(["min", "median", "max"]).to_dict("index"))
    metrics = {
        "citation_share": TYPES, "source_share": TYPES,
        "citation_mass": TYPES + ["NONE"], "source_mass": TYPES + ["NONE"],
        "citation_presence": TYPES, "citation_n": TYPES, "source_n": TYPES,
    }
    columns = [f"{metric}_{k}" for metric, types in metrics.items() for k in types]
    cells = cell_means(runs, columns)
    cells.reset_index().to_csv(OUT / "scenario_cell_means.csv", index=False)
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1., 1.], size=(reps, g))
    draws = rng.integers(0, g, size=(boot_reps, g))
    rows, joint, ranges, cell_table, sensitivity = [], [], [], [], []
    all_effects = {}
    for engine in ["all"] + ENGINES:
        for metric, types in metrics.items():
            matrices = {}
            for k in types:
                cube = as_cube(cells, f"{metric}_{k}", engine, scenarios)
                for cell, theta in zip(CELL_NAMES, cube.mean(axis=0)):
                    cell_table.append(dict(engine=engine, metric=metric, type=k, cell=cell, theta=theta))
                for term in TERMS:
                    contrast = cube @ contrast_weights(term)
                    all_effects[(engine, metric, k, term)] = contrast
                    rows.append(dict(engine=engine, metric=metric, type=k, term=term,
                                     estimand="equal_scenario_engine_cell", **infer_mean(contrast, signs)))
                matrices[k] = cube
                if metric in ("citation_share", "source_share", "citation_mass"):
                    ranges.append(dict(engine=engine, metric=metric, type=k, n_cells=8,
                                       **envelope(cube, draws)))
            if metric in ("citation_share", "source_share", "citation_mass", "source_mass"):
                # Six conditional shares add to one; seven unconditional masses
                # add to one. Always drop one category before joint inference.
                for factor in FACTORS:
                    x = np.column_stack([matrices[k] @ contrast_weights(factor) for k in types[:-1]])
                    joint.append(dict(engine=engine, metric=metric, factor=factor,
                                      excluded_type=types[-1], **joint_test(x, signs)))
    effects = pd.DataFrame(rows)
    effects["p_holm"] = np.nan
    effects["multiplicity_family"] = "interaction_exploratory_unadjusted"
    main = effects.term.isin(FACTORS)
    for key, frame in effects[main].groupby(["engine", "metric"]):
        effects.loc[frame.index, "p_holm"] = holm(frame.p_wild)
        effects.loc[frame.index, "multiplicity_family"] = f"3_factors_x_{len(metrics[key[1]])}_types"
    effects.to_csv(OUT / "factor_effects.csv", index=False)
    joint = pd.DataFrame(joint)
    for _, frame in joint.groupby(["engine", "metric"]):
        joint.loc[frame.index, "p_holm"] = holm(frame.p_wild)
    joint.to_csv(OUT / "joint_tests.csv", index=False)
    pd.DataFrame(ranges).to_csv(OUT / "composition_envelopes.csv", index=False)
    pd.DataFrame(cell_table).to_csv(OUT / "cell_means.csv", index=False)
    # Fixed, paired engine contrasts: use the same scenario in both engines.
    heterogeneity = []
    for metric in ("citation_share", "citation_mass", "source_share"):
        for k in ("CORPORATE", "EDITORIAL"):
            for factor in FACTORS:
                for engine in ENGINES[1:]:
                    diff = all_effects[(engine, metric, k, factor)] - all_effects[(ENGINES[0], metric, k, factor)]
                    result = infer_mean(diff, signs)
                    # TOST uses the scenario t critical value at alpha .05.
                    from scipy import stats
                    m_min = abs(result["estimate"]) + stats.t.ppf(.95, g-1) * result["se"]
                    heterogeneity.append(dict(metric=metric, type=k, factor=factor,
                        contrast=f"{engine}-chatgpt-ui", min_equivalence_margin=m_min, **result))
    pd.DataFrame(heterogeneity).to_csv(OUT / "engine_heterogeneity.csv", index=False)
    # Replicates are a sensitivity, with equal prompt weights within each cell.
    repcells = cell_means(runs, columns, include_replicates=True)
    topics = core.groupby("scenario").topic.first().reindex(scenarios)
    for metric in ("citation_share", "citation_mass", "source_share"):
        for k in ("CORPORATE", "EDITORIAL"):
            for factor in FACTORS:
                x = all_effects[("all", metric, k, factor)]
                rep = as_cube(repcells, f"{metric}_{k}", "all", scenarios) @ contrast_weights(factor)
                sensitivity.append(dict(metric=metric, type=k, factor=factor,
                    core_estimate=x.mean(), with_replicates=rep.mean(),
                    min_leave_one_scenario=np.min([(x.sum()-v)/(g-1) for v in x]),
                    max_leave_one_scenario=np.max([(x.sum()-v)/(g-1) for v in x]),
                    min_leave_one_topic=min(x[(topics != t).to_numpy()].mean() for t in topics.unique()),
                    max_leave_one_topic=max(x[(topics != t).to_numpy()].mean() for t in topics.unique())))
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    # Report selection unconditionally for EVERY engine, alongside source shares.
    selection = core.groupby("model_id").agg(n_runs=("chat_id", "size"),
            source_rate=("any_source", "mean"), citation_rate=("any_citation", "mean"))
    selection.to_csv(OUT / "selection_rates.csv")
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=9999)
    parser.add_argument("--boot-reps", type=int, default=1999)
    parser.add_argument("--channel", choices=["ui", "all"], default="ui",
                        help="ui = scraped consumer UI only (primary); all = include Perplexity API fallback days")
    args = parser.parse_args()
    out = OUT if args.channel == "ui" else OUT.with_name("revised_all")
    out.mkdir(parents=True, exist_ok=True)
    runs, sources, conflicts, audit = build_runs(args.channel)
    conflicts.to_csv(out / "classification_conflicts.csv", index=False)
    # label- and channel-independent: always the full-universe audit, in out/revised/
    exclusion_audit(pd.read_parquet(DATA / "chats_universe.parquet")).to_csv(OUT / "exclusions.csv", index=False)
    effects = analyze(runs, args.reps, args.boot_reps, out=out)
    paths = [DATA / "chats_universe.parquet", DATA / "sources.parquet",
             DERIVED / "prompt_design.parquet", Path(__file__), HERE / "inference.py"]
    manifest = dict(analysis_status="post_review_exploratory", seed=SEED, channel=args.channel,
        engine_days=runs.groupby("model_id").date.nunique().to_dict(),
        wild_reps=args.reps, scenario_bootstrap_reps=args.boot_reps,
        core_prompts=int(runs.loc[runs.role == "factorial", "prompt_id"].nunique()),
        core_runs=int(runs.role.eq("factorial").sum()), date_min=str(runs.date.min()),
        date_max=str(runs.date.max()), audit=audit,
        sha256={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(effects.query("engine=='all' and term in ['adv','fea','prc'] and "
        "metric in ['citation_share','citation_mass'] and type in ['CORPORATE','EDITORIAL']")
        [["metric", "type", "term", "estimate", "ci_low", "ci_high", "p_wild", "p_holm"]].to_string(index=False))
    print(f"Wrote revised results: {out}")


if __name__ == "__main__":
    main()
