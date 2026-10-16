# How You Ask Changes How AI Cites: replication materials

Data and code for

> Emilija Gjorgjevska, Jan Ehrlinspiel, and Malte Landwehr. 2027. *How You Ask Changes How AI Cites: An Empirical Study of Shopping Search.*

The study is a matched-scenario factorial experiment. It uses 25 shopping scenarios × 12 prompts (8 factorial conditions crossing advice framing, feature disclosure and budget disclosure, plus one natural seed and three rewordings). The 300 prompts were submitted daily to the logged-out consumer interfaces of ChatGPT, Gemini and Perplexity from 26 August to 28 September 2026. Every response records the sources the interface lists as retrieved and the sources it visibly cites.

## Contents

| Path | What it is |
|---|---|
| `data/prompt_design.parquet` | The 300 prompts with scenario, topic, condition (`cell`), factor indicators (`adv`, `fea`, `prc`), role (factorial / template replicate / seed) and template id |
| `data/chats_universe.parquet` | One row per submitted prompt run: response id, prompt id, engine (`model_id`; `sonar` = Perplexity API), collection provider (`P1`–`P8`), date, status |
| `data/sources.parquet` | One row per source record: response id, prompt id, engine, date, normalized URL, domain, source type (`domain_classification`), and `metric_citation_count` (0 = retrieved but not cited) |
| `data/jev_labels_domains.parquet` | The frozen source-type label of every domain in `sources.parquet` (TypeSafe JEV, `jev-latest`), with the classifier's confidence |
| `data/chats_structure.parquet` | Answer length (`message_length`, characters) and answer element types per response (Gemini change, Section 5.5) |
| `data/chatgpt_model_by_chat.parquet` | The model version that ChatGPT reported per response (Section 3.2) |
| `code/` | Estimation and figure code; `run_all.sh` runs it in order |

**De-identification.** Response and prompt ids are salted hashes. Collection-provider names are replaced by `P1`–`P8`. Answer texts, page titles, brand annotations and all identifiers of the collection platform are not included. The source-type labels are the frozen JEV labels used in the paper.

## Running

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh                  # about 4 hours (fmnl_hetero.py alone about 3); BOOT=0 skips the bootstrap, EMBED=1 adds Table 5
```

Environment variables read by `run_all.sh` and the code:

| Variable | Meaning | Default |
|---|---|---|
| `STUDY2_END` | Last analysis day (`YYYY-MM-DD`). Collection ran one day longer; that day is loaded only as look-ahead for the exclusion rules. `revised_analysis.py` raises an error if it is unset. | `2026-09-28` (set in `run_all.sh`) |
| `DATA_DIR` | Directory holding the files of `data/` | `data/` |
| `PY` | Python interpreter used by `run_all.sh` | `../.venv/bin/python` |
| `BOOT` | Bootstrap replications; `0` skips the bootstrap | `999` |
| `EMBED` | `1` adds Table 5 (prompt similarity). The first run downloads the sentence-transformers models `all-mpnet-base-v2` and `all-MiniLM-L6-v2` and needs `pip install sentence-transformers` (not in `requirements.txt`). | `0` |

Table 5 (`EMBED=1`) downloads the sentence-transformers models on first use (`all-mpnet-base-v2` as primary; `all-MiniLM-L6-v2` is selected with `prompt_similarity.py --embed --model sentence-transformers/all-MiniLM-L6-v2 --tag minilm`, then the same call without `--embed`) and needs `pip install sentence-transformers`.

All outputs are written to `code/out/`. Every model is estimated on the analysis window 26 August–28 September 2026 (`STUDY2_END` in `run_all.sh`). The exclusion rules of Section 3.2 are applied mechanically (`code/design.py`, `exclusions`).

## Where each result comes from

| Paper | Command (from `code/`) | Output |
|---|---|---|
| Section 3.2 (exclusion audit) | `revised_analysis.py --channel ui` | `out/revised/` |
| Table 2 (sample) | `paper_numbers.py` | `out/paper/sample_table.csv` |
| Table 3, Figure 3 (average marginal effects, cited composition) | `fmnl_run.py --labels jev --outcome citation` | `out/fmnl/jev_citation/ame.csv`, `joint.csv` |
| Section 5.1 bootstrap interval | `fmnl_run.py --labels jev --outcome citation --boot 999 --boot-keep-unidentified --out out/fmnl/jev_citation_boot` | `out/fmnl/jev_citation_boot/ame_boot.csv` |
| Section 5.2 (no-citation as seventh category) | `fmnl_run.py --labels jev --outcome citation_mass` | `out/fmnl/jev_citation_mass/ame.csv` |
| Section 5.2, Table 8 (Lee bounds) | `selection_bounds.py --labels jev` | `out/revised/lee_bounds.csv` |
| Section 5.3 (variance decomposition) | `wording_variance.py --labels jev` | `out/revised/wording_variance.csv` |
| Section 5.3 (natural seed vs full template) | `seed_comparison.py --labels jev` | `out/revised/seed_comparison.csv` |
| Appendix F, Section 5.4 (domain-level stage model, pooled and per engine) | `domain_ppml.py --labels jev --engine <all\|chatgpt-ui\|gemini-ui\|perplexity-ui>` | `out/revised/domain_ppml[_<engine>]_summary.csv` |
| Section 5.3, Figure 4 (scenario effects, meta-regression) | `fmnl_hetero.py --labels jev` | `out/fmnl/rq2/jev_stage1.csv`, `jev_stage2.csv` |
| Section 5.4 (retrieval vs citation) | `fmnl_stage.py --labels jev` | `out/fmnl/rq3/jev_ame.csv` |
| Section 5.5 (stability) | `fmnl_stability.py --labels jev` | `out/fmnl/rq4/` |
| Table 4 (domains per response), Perplexity API | `descriptives.py --labels jev` | `out/revised/descriptives_domain_counts.csv` |
| Section 5.5 (model share, Gemini break, Perplexity API) | `model_regime.py` | `out/revised/engine_breaks.csv`, `channel_levels.csv` |
| Section 5.5 (Gemini answer structure) | `gemini_structure.py` | `out/revised/gemini_structure_breaks.csv` |
| Remaining text numbers, Figure 5 data | `paper_numbers.py` | `out/paper/environment_summary.csv`, `daily_engine_series.csv` |
| Figures 3-5 | `figures.py` | `out/figures/` |
| Table 5 (prompt similarity; `EMBED=1`) | `cluster_effects.py`, `prompt_similarity.py --embed`, `prompt_similarity.py` | `out/revised/similarity_summary.csv` |
| Table 6 (second implementation) | `fmnl_poisson.py --labels jev --outcome <o>` for the four outcomes, then `fmnl_differential.py` | `out/fmnl_poisson/differential*.csv` |

(`<o>` = `citation`, `retrieval`, `citation_mass`, `source_mass`; `fmnl_run.py` is likewise run for all four outcomes.)

`fmnl.py` implements the fractional multinomial logit of Eq. (1) together with the scenario jackknife (`jackknife`), the Wald F test (`wald_F`) and the gatekeeping procedure (`gatekeep`). `fmnl_poisson.py` implements the equivalent weighted Poisson regression (second implementation, Table 6). `inference.py` holds the Holm correction and the helpers for the equal-weight contrasts of `revised_analysis.py`.

## Data dictionary

`chat_id` is the response id (one row of `chats_universe` = one submitted prompt run). All ids are salted hashes.

**`prompt_design.parquet`** (300 rows, one per prompt)

| Column | Meaning |
|---|---|
| `text` | Prompt text as submitted |
| `topic`, `scenario` | Product topic (5) and shopping scenario (25, e.g. `BP01`) |
| `cell` | Condition: `NEU`, `ADV`, `+FEA`, `+PRC` combinations (8 factorial cells) or `SEED` |
| `adv`, `fea`, `prc` | 0/1 indicators for advice framing, feature disclosure, budget disclosure; empty for the 25 seed prompts |
| `is_seed`, `is_reword` | Natural seed prompt; second (reworded) prompt of a cell |
| `role` | `factorial` (200), `template-replicate` (75), `seed-natural` (25) |
| `frame` | `neutral`, `advice` or `natural` |
| `template_id` | Wording template (`N1`-`N3`, `A1`-`A3`, `original` for seeds) |
| `clause_order` | Position of the added clause (`before`/`after`); empty for cells without one |
| `prc_type` | Budget disclosure as a number or as a value statement |
| `v4_text_changed` | The final prompt text differs from the earlier design draft |
| `seed_branded` | The prompt names a brand |
| `prompt_id` | Salted prompt hash; join key |

`prompt_design.parquet` will not contain `qa_flags` (internal QA notes).

**`chats_universe.parquet`** (one row per run, 33,006)

On the first collection day, 26 August 2026, the providers ran every prompt three times per engine; on all later days, once.

| Column | Meaning |
|---|---|
| `chat_id`, `prompt_id` | Response id, prompt id |
| `model_id` | `chatgpt-ui`, `gemini-ui`, `perplexity-ui`, or `sonar` (Perplexity API, used on days the interface collection was down) |
| `provider_id` | Collection provider `P1`-`P8`; empty for the Perplexity API (`sonar`) |
| `date` | Collection day |
| `status` | `SUCCEEDED` or `NOT_AVAILABLE` |

**`sources.parquet`** (one row per source record, 255,747)

| Column | Meaning |
|---|---|
| `date`, `chat_id`, `prompt_id`, `model_id` | As above |
| `url_normalized` | Normalized URL without scheme and `www.`. Query strings are replaced by a salted hash (`?q=<hash>`), so URLs with tracking parameters are not published |
| `domain`, `full_domain` | Registrable domain; full host name (the join key to the label file) |
| `domain_classification` | Source type of the domain: the frozen JEV label (`CORPORATE`, `EDITORIAL`, `UGC`, `REFERENCE`, `INSTITUTIONAL`, `OTHER`) |
| `metric_citation_count` | Times the answer cites the source; 0 = retrieved but not cited |

**`jev_labels_domains.parquet`**: `full_domain`; `jev_cls` (source type); `jev_conf` (classifier confidence, 0-1).

**`chats_structure.parquet`**: `chat_id`, `date`, `model_id`, `provider_id`; `message_length` (answer length in characters); `element_types` (list of the answer's element types, e.g. table, sources).

**`chatgpt_model_by_chat.parquet`**: `chat_id`, `date`, `model_slug` (model version ChatGPT reported for the response; empty when the stream was truncated).

## How to cite

Cite the paper (machine-readable metadata in `CITATION.cff`):

> Emilija Gjorgjevska, Jan Ehrlinspiel, and Malte Landwehr. 2027. *How You Ask Changes How AI Cites: An Empirical Study of Shopping Search.* In Proceedings of The Web Conference 2027 (WWW '27).

## Not included

- **Collection.** Prompts were submitted through the infrastructure of Peec AI. The collection code depends on that platform and is not part of this repository.
- **Labelling.** Domains were labelled once through the TypeSafe API. The question and the criteria passed to the classifier are given verbatim in the paper (Appendix D). The labels are frozen in `data/jev_labels_domains.parquet`.
- **Second label set.** Table 6 in the paper also covers fits with the platform's own source labels, which are not released. The code here reproduces the JEV half of that comparison.

## License

Code: MIT. Data: CC BY 4.0. See `LICENSE`.
