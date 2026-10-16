#!/usr/bin/env bash
# Reproduces every estimate in the paper from data/ (see README.md for the map to tables and figures).
#   ./run_all.sh            # everything, about 4 hours on a laptop (fmnl_hetero.py alone takes about 3)
#   BOOT=0 ./run_all.sh     # skip the pairs-cluster bootstrap (the slowest step)
set -euo pipefail
cd "$(dirname "$0")/code"
PY="${PY:-../.venv/bin/python}"
BOOT="${BOOT:-999}"
export STUDY2_END=2026-09-28            # last analysis day (collection ran to 2026-09-29, used as look-ahead only)
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 MKL_NUM_THREADS=3
LOG=out/logs; mkdir -p "$LOG"
step() { local name=$1; shift; echo "$(date +%H:%M:%S) $name"; "$PY" "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see code/$LOG/$name.log)"; exit 1; }; }

step revised       revised_analysis.py --channel ui              # exclusion audit (Section 3.2)
step human_labels  jev_human_validation.py                       # Section 3.3 label accuracy vs. human labels
for oc in citation retrieval citation_mass source_mass; do
  step fmnl_$oc    fmnl_run.py --labels jev --outcome $oc         # Table 3, Figure 3, Section 5.2
  step poisson_$oc fmnl_poisson.py --labels jev --outcome $oc     # second implementation (Table 6)
done
step differential  fmnl_differential.py                          # Table 6
step hetero        fmnl_hetero.py --labels jev                    # Section 5.3, Figure 4
step stage         fmnl_stage.py --labels jev                     # Section 5.4
step stability     fmnl_stability.py --labels jev                 # Section 5.5
step descriptives  descriptives.py --labels jev                   # Tables 2 and 4, Perplexity API
step wording_var   wording_variance.py --labels jev               # Section 5.3 variance decomposition
step seed          seed_comparison.py --labels jev                # Section 5.3 seed vs full template
for e in all chatgpt-ui gemini-ui perplexity-ui; do
  step ppml_$e    domain_ppml.py --labels jev --engine $e        # Appendix F, Section 5.4 (domain-level stage model)
done
step lee_bounds    selection_bounds.py --labels jev               # Section 5.2, Table 8
step gemini        gemini_structure.py                            # Section 5.5 (answer structure)
step regime        model_regime.py                                # Section 3.2 model share, Section 5.5 Gemini break + Perplexity API
step paper_numbers paper_numbers.py                               # remaining text numbers + figure data
step figures       figures.py                                     # Figures 3-5
if [ "${EMBED:-0}" = 1 ]; then                                     # Table 5 (needs sentence-transformers)
  step clusters    cluster_effects.py                       # scenario_effects.csv, read by prompt_similarity.py
  step embed       prompt_similarity.py --embed
  step similarity  prompt_similarity.py
fi
if [ "$BOOT" != 0 ]; then
  step boot        fmnl_run.py --labels jev --outcome citation --boot "$BOOT" --boot-keep-unidentified --out out/fmnl/jev_citation_boot
fi
echo "$(date +%H:%M:%S) done; outputs in code/out/"
