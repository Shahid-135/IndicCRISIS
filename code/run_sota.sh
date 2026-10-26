#!/usr/bin/env bash
# Re-implemented SOTA crisis models on the final split: 6 models x 3 seeds, one GPU, resumable.
#   GPU=7 bash run_sota.sh IndicCrisis_clean.csv "Merged Images"
#   MODELS="dmcc mann" SEEDS="42" GPU=7 bash run_sota.sh ...      # subset
# Results: $OUT/sota_<model>_s<seed>  (default OUT=runs_v2, next to the baselines)
set -u
CSV="${1:?usage: bash run_sota.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_sota.sh <csv> <image_dir>}"
GPU="${GPU:-0}"; SEEDS="${SEEDS:-42 43 44}"; OUT="${OUT:-runs_v2}"; EXTRA="${EXTRA:-}"
MODELS="${MODELS:-mann camm robertamft dmcc clip align}"
mkdir -p "$OUT/logs"
for seed in $SEEDS; do
  for m in $MODELS; do
    r="sota_${m}_s${seed}"; dir="$OUT/$r"
    if [ -f "$dir/report/metrics_summary.csv" ]; then echo "[skip]  $r"; continue; fi
    echo "[start] $r on GPU $GPU"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" python train_sota.py --model "$m" --csv "$CSV" --image_dir "$IMG" \
        --seed "$seed" --save_dir "$dir" $EXTRA > "$OUT/logs/$r.log" 2>&1 \
      && echo "[done]  $r" || echo "[FAIL]  $r - see $OUT/logs/$r.log"
  done
done
