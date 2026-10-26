#!/usr/bin/env bash
# CCEC-G experiment grid: 16 variants x 5 seeds, then mean ± std tables with significance tests.
#   * resumable  - runs that already finished (report/metrics_summary.csv exists) are skipped,
#                  so your earlier `ccec_s*` and `fusion_baseline_s*` runs are reused
#   * multi-GPU  - one job per GPU listed in GPUS
#   * logged     - runs/logs/<run>.log
#
# Usage:
#   GPUS="1 2 3 4 5" bash run_ablations.sh IndicCrisis_clean.csv "Merged Images"
#   ONLY="full corr_cmcw ccec" GPUS="1 2" bash run_ablations.sh ...     # run only some variants
#   SEEDS="42 43 44" bash run_ablations.sh ...                           # fewer seeds
#   EXTRA="--epochs 20" bash run_ablations.sh ...                         # extra flags for every run
set -u
CSV="${1:?usage: bash run_ablations.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_ablations.sh <csv> <image_dir>}"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-runs}"
ONLY="${ONLY:-}"
VLM_VIEW="${VLM_VIEW:-vlm_views.npz}"   # made by run_vlm_view.sh
COMP_LR="${COMP_LR:-0.02}"              # learning rate of the composition tables (full_vlm_comp)
mkdir -p "$OUT/logs"

# random batch partners: kNN batches created false negatives and hurt (see analysis of the first grid)
FULL="--corr graph --nbr_sampler random --ordinal_cmc"
VARIANTS=(
  # proposed model
  "full|$FULL"
  "full_fine|$FULL --fine_align"
  # correlation-learning ladder: CCEC -> +contrastive -> +CMC weights -> +graph -> +ordinal (= full)
  "ccec|"
  "corr_clip|--corr clip"
  "corr_cmcw|--corr cmcw"
  "corr_graph|--corr graph --nbr_sampler random"
  "full_knn_nbrs|--corr graph --nbr_sampler knn --ordinal_cmc"
  # composition ablations on top of correlation learning
  "full_no_regimes|$FULL --no_regimes"
  "full_no_view_sup|$FULL --no_view_sup"
  "full_no_residual|$FULL --no_residual"
  "fusion_full|$FULL --fusion_baseline"
  "fusion_baseline|--fusion_baseline"
  # how much is each annotation worth?
  "full_cmc50|$FULL --cmc_frac 0.5"
  "full_cmc25|$FULL --cmc_frac 0.25"
  "full_no_cmc|$FULL --no_cmc_sup"
  "full_no_cmc_no_view|$FULL --no_cmc_sup --no_view_sup"
  # zero-shot VLM reasoning as a knowledge view for the pair-level components
  "full_vlm|$FULL --vlm_view $VLM_VIEW"
  "fusion_full_vlm|$FULL --fusion_baseline --vlm_view $VLM_VIEW"
  "ccec_vlm|--vlm_view $VLM_VIEW"
  # composition tables with their own learning rate (they barely move at the shared rate)
  "full_vlm_comp|$FULL --vlm_view $VLM_VIEW --comp_lr $COMP_LR"
)

JOBS=()
for seed in $SEEDS; do
  for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
    JOBS+=("$name|${v#*|}|$seed")
  done
done

run_job() {   # $1 = gpu, $2 = job
  local gpu="$1" name flags seed dir
  IFS="|" read -r name flags seed <<< "$2"
  dir="$OUT/${name}_s${seed}"
  if [ -f "$dir/report/metrics_summary.csv" ]; then
    echo "[skip]  ${name}_s${seed} (already done)"; return
  fi
  echo "[start] ${name}_s${seed} on GPU $gpu"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python train_ccec.py --csv "$CSV" --image_dir "$IMG" \
      --seed "$seed" --save_dir "$dir" $flags $EXTRA > "$OUT/logs/${name}_s${seed}.log" 2>&1
  if [ $? -eq 0 ]; then echo "[done]  ${name}_s${seed}"; else echo "[FAIL]  ${name}_s${seed} - see $OUT/logs/${name}_s${seed}.log"; fi
}

# fail fast if a VLM variant is requested but the knowledge view has not been built
for j in "${JOBS[@]}"; do
  if [[ "$j" == *"--vlm_view"* ]] && [ ! -f "$VLM_VIEW" ]; then
    echo "ERROR: $VLM_VIEW not found - run run_vlm_view.sh first (or set VLM_VIEW=...)"; exit 1
  fi
done

read -r -a GPU_LIST <<< "$GPUS"
G=${#GPU_LIST[@]}
echo "${#JOBS[@]} runs on ${G} GPU(s): ${GPUS}"
for ((w = 0; w < G; w++)); do
  (
    for ((j = w; j < ${#JOBS[@]}; j += G)); do
      run_job "${GPU_LIST[$w]}" "${JOBS[$j]}"
    done
  ) &
done
wait

echo "all runs finished - aggregating"
python aggregate_runs.py --runs "$OUT" --out "$OUT/summary" --ref full
