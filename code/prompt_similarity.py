"""Semantic structure of the Study 2 prompt set: how far the 12 variants of a
scenario move from its NEU baseline, and how distinct the 25 scenarios are.

Two steps so the statistics venv never imports torch:
  --embed   sentence-transformers/all-mpnet-base-v2 (downloaded on first use;
            needs `pip install sentence-transformers`):
            python prompt_similarity.py --embed
  (default) numpy cosine tables from out/revised/prompt_embeddings.npy
  --tfidf   last-resort lexical fallback (sklearn) if the embedding env is unavailable.

Writes the per-prompt cosine to the scenario's NEU prompt and the minimum cosine to
other scenarios to out/revised/prompt_similarity.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from design import DERIVED

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "revised"
FIG = HERE / "out" / "fig"
MODEL = "sentence-transformers/all-mpnet-base-v2"
VARIANTS = ["NEU", "NEU-rw", "NEU+FEA", "NEU+PRC", "NEU+FEA+PRC",
            "ADV", "ADV-rw", "ADV+FEA", "ADV+PRC", "ADV+FEA+PRC", "ADV+FEA+PRC-rw", "SEED"]


def load_design():
    d = pd.read_parquet(DERIVED / "prompt_design.parquet")
    d = d.sort_values(["scenario", "cell", "is_reword"]).reset_index(drop=True)
    d["variant"] = d.cell + np.where(d.is_reword, "-rw", "")
    return d


def _sfx(tag):
    return "" if not tag else f"_{tag}"


def embed(d, model_name=MODEL, tag=""):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    e = model.encode(d.text.tolist(), normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    np.save(OUT / f"prompt_embeddings{_sfx(tag)}.npy", np.asarray(e, dtype=np.float32))
    d[["prompt_id"]].assign(embedding_model=model_name).to_csv(OUT / f"prompt_embeddings_ids{_sfx(tag)}.csv", index=False)
    print(f"embedded {len(d)} prompts with {model_name} -> {OUT / f'prompt_embeddings{_sfx(tag)}.npy'} {e.shape}")


def tfidf(d, tag="tfidf"):
    """Lexical baseline: sublinear TF-IDF over word uni+bigrams, L2-normalised (numpy, no sklearn)."""
    import re
    from collections import Counter
    docs = []
    for t in d.text:
        w = re.findall(r"[a-z0-9]+", t.lower())
        docs.append(Counter(w + [f"{a} {b}" for a, b in zip(w, w[1:])]))
    vocab = {k: i for i, k in enumerate(sorted({k for c in docs for k in c}))}
    n = len(docs)
    df = np.zeros(len(vocab))
    for c in docs:
        for k in c:
            df[vocab[k]] += 1
    idf = np.log((1 + n) / (1 + df)) + 1
    e = np.zeros((n, len(vocab)))
    for i, c in enumerate(docs):
        for k, v in c.items():
            e[i, vocab[k]] = (1 + np.log(v)) * idf[vocab[k]]
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    np.save(OUT / f"prompt_embeddings{_sfx(tag)}.npy", e.astype(np.float32))
    d[["prompt_id"]].assign(embedding_model="tfidf-1-2gram").to_csv(OUT / f"prompt_embeddings_ids{_sfx(tag)}.csv", index=False)


def similarity_tables(d, e):
    """Per-prompt, per-scenario and per-variant cosine tables from unit-norm embeddings."""
    s = np.clip(e @ e.T, -1, 1)
    scen = d.scenario.to_numpy()
    same = scen[:, None] == scen[None, :]
    eye = np.eye(len(d), dtype=bool)
    other = ~same
    neu_idx = {r.scenario: i for i, r in d.iterrows() if r.variant == "NEU"}
    orig_idx = {(r.scenario, r.cell): i for i, r in d.iterrows() if not r.is_reword}
    rows = []
    for i, r in d.iterrows():
        so = s[i, other[i]]
        j = np.flatnonzero(other[i])[so.argmax()]
        same_var = (d.variant.to_numpy() == r.variant) & other[i]
        rows.append({
            "prompt_id": r.prompt_id, "scenario": r.scenario, "topic": r.topic, "cell": r.cell,
            "variant": r.variant, "role": r.role, "is_reword": r.is_reword, "template_id": r.template_id,
            "text": r.text,
            "cosine_vs_scenario_NEU": s[i, neu_idx[r.scenario]],
            "min_cosine_other_scenarios": so.min(),
            "max_cosine_other_scenarios": so.max(), "mean_cosine_other_scenarios": so.mean(),
            "nearest_other_scenario": scen[j], "nearest_other_topic": d.topic.iloc[j],
            "mean_cosine_within_scenario": s[i, same[i] & ~eye[i]].mean(),
            "cosine_vs_same_cell_original": s[i, orig_idx[(r.scenario, r.cell)]] if r.is_reword else np.nan,
            "mean_cosine_same_variant_other_scenarios": s[i, same_var].mean(),
        })
    prompts = pd.DataFrame(rows)
    scenarios = sorted(d.scenario.unique())
    m = np.zeros((len(scenarios), len(scenarios)))
    for a, sa in enumerate(scenarios):
        ia = scen == sa
        for b, sb in enumerate(scenarios):
            ib = scen == sb
            block = s[np.ix_(ia, ib)]
            m[a, b] = block[~np.eye(ia.sum(), dtype=bool)].mean() if a == b else block.mean()
    scenario_sim = pd.DataFrame(m, index=scenarios, columns=scenarios)
    variant_sim = prompts.groupby("variant").agg(
        n=("prompt_id", "size"),
        cosine_vs_neu_mean=("cosine_vs_scenario_NEU", "mean"),
        cosine_vs_neu_min=("cosine_vs_scenario_NEU", "min"),
        cosine_vs_neu_max=("cosine_vs_scenario_NEU", "max"),
        same_variant_other_scenarios=("mean_cosine_same_variant_other_scenarios", "mean"),
        within_scenario=("mean_cosine_within_scenario", "mean"),
    ).reindex(VARIANTS).reset_index()
    # silhouette of the scenario labels under cosine distance (numpy, no sklearn)
    dist = 1 - s
    a = np.array([dist[i, same[i] & ~eye[i]].mean() for i in range(len(d))])
    b = np.array([min(dist[i, scen == t].mean() for t in scenarios if t != scen[i]) for i in range(len(d))])
    topic = d.topic.to_numpy()
    same_topic = topic[:, None] == topic[None, :]
    summary = pd.DataFrame([dict(
        embedding_model=d.attrs.get("embedding_model", "unknown"),
        within_scenario=s[same & ~eye].mean(),
        within_topic_between_scenario=s[same_topic & other].mean(),
        between_topic=s[~same_topic].mean(),
        silhouette_scenario=((b - a) / np.maximum(a, b)).mean(),
        share_nearest_other_same_topic=(prompts.nearest_other_topic == prompts.topic).mean(),
        share_within_gt_max_other=(prompts["mean_cosine_within_scenario"] > prompts["max_cosine_other_scenarios"]).mean(),
    )])
    return prompts, scenario_sim, variant_sim, summary


def figures(d, scenario_sim, prompts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    FIG.mkdir(parents=True, exist_ok=True)
    topics = ["Backpacks", "Office Chairs", "Gloves", "Outerwear", "Water Bottles"]
    topic_of = d.drop_duplicates("scenario").set_index("scenario").topic
    order = sorted(scenario_sim.index, key=lambda s: (topics.index(topic_of[s]), s))
    m = scenario_sim.loc[order, order]
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(m.to_numpy(), cmap=LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for t in topics[:-1]:
        last = max(i for i, s_ in enumerate(order) if topic_of[s_] == t)
        ax.axhline(last + .5, color="#fcfcfb", lw=2.5); ax.axvline(last + .5, color="#fcfcfb", lw=2.5)
    for t in topics:
        members = [i for i, s_ in enumerate(order) if topic_of[s_] == t]
        ax.text(-2.6, np.mean(members), t, ha="right", va="center", fontsize=8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.outline.set_visible(False)
    cb.set_label("mean pairwise cosine similarity (all-mpnet-base-v2)")
    ax.set_title("How similar are the 25 scenarios? (diagonal = within-scenario)", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "scenario_cosine_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    data = [prompts.loc[prompts.variant == v, "cosine_vs_scenario_NEU"].to_numpy() for v in VARIANTS[1:]]
    bp = ax.boxplot(data, positions=range(len(data)), widths=.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color="#0b0b0b", lw=1.4), flierprops=dict(marker="o", ms=3, mfc="#898781", mec="none"))
    for box in bp["boxes"]:
        box.set(facecolor="#cde2fb", edgecolor="#2a78d6", lw=1)
    for k in ("whiskers", "caps"):
        for l in bp[k]: l.set(color="#898781", lw=1)
    ax.axhline(prompts["max_cosine_other_scenarios"].mean(), color="#eb6834", lw=1.4)
    ax.text(len(data) - .5, prompts["max_cosine_other_scenarios"].mean() + .012, "mean nearest other-scenario prompt", ha="right", fontsize=8, color="#eb6834")
    ax.set_xticks(range(len(data))); ax.set_xticklabels(VARIANTS[1:], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("cosine to the scenario's NEU prompt"); ax.set_ylim(.4, 1.02)
    ax.grid(axis="y", color="#e1e0d9", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("How far does each variant move from its scenario's neutral prompt? (25 scenarios per box)", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "variant_cosine_vs_neu.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def effect_link(prompts, tag=""):
    """Does the text distance a switch introduces predict the size of the scenario's
    effect? Spearman over the 25 scenarios: mean (1 - cosine to NEU) of the variants
    carrying the switch vs |per-scenario effect| on corporate cited share."""
    from scipy import stats
    scen = pd.read_csv(OUT / "scenario_effects.csv").query(
        "engine == 'all' and metric == 'citation_share' and type == 'CORPORATE'")
    rows = []
    for factor, cells in (("fea", ["NEU+FEA", "ADV+FEA", "NEU+FEA+PRC", "ADV+FEA+PRC"]),
                          ("prc", ["NEU+PRC", "ADV+PRC", "NEU+FEA+PRC", "ADV+FEA+PRC"]),
                          ("adv", ["ADV", "ADV+FEA", "ADV+PRC", "ADV+FEA+PRC"])):
        dist = prompts[prompts.variant.isin(cells)].groupby("scenario")["cosine_vs_scenario_NEU"].mean().rsub(1)
        eff = scen[scen.term == factor].set_index("scenario").estimate.abs()
        x = pd.concat([dist.rename("text_distance"), eff.rename("abs_effect")], axis=1).dropna()
        r, p = stats.spearmanr(x.text_distance, x.abs_effect)
        rows.append(dict(embedding=tag or "mpnet", factor=factor, n_scenarios=len(x),
                         mean_text_distance=dist.mean(), spearman_rho=r, p=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--model", default=MODEL, help="sentence-transformers model id (downloaded on first use)")
    ap.add_argument("--tag", default="", help="output suffix; '' = primary (all-mpnet-base-v2)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_design()
    if a.embed:
        return embed(d, a.model, a.tag)
    if a.tfidf:
        tfidf(d, a.tag or "tfidf")
    tag = a.tag or ("tfidf" if a.tfidf else "")
    e = np.load(OUT / f"prompt_embeddings{_sfx(tag)}.npy")
    ids = pd.read_csv(OUT / f"prompt_embeddings_ids{_sfx(tag)}.csv")
    assert (ids.prompt_id.to_numpy() == d.prompt_id.to_numpy()).all(), "embedding order != design order"
    d.attrs["embedding_model"] = ids.embedding_model.iloc[0]
    prompts, scenario_sim, variant_sim, summary = similarity_tables(d, e)
    prompts.to_csv(OUT / f"prompt_similarity{_sfx(tag)}.csv", index=False)
    scenario_sim.to_csv(OUT / f"scenario_similarity{_sfx(tag)}.csv")
    variant_sim.to_csv(OUT / f"variant_similarity{_sfx(tag)}.csv", index=False)
    summary.to_csv(OUT / f"similarity_summary{_sfx(tag)}.csv", index=False)
    link = effect_link(prompts, tag)
    link.to_csv(OUT / f"similarity_vs_effect{_sfx(tag)}.csv", index=False)
    if not a.no_figures and not tag:
        figures(d, scenario_sim, prompts)
    print(summary.T.to_string()); print(variant_sim.round(3).to_string(index=False)); print(link.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
