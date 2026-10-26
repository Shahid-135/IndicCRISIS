#!/usr/bin/env bash
# Single-modality baselines on the final split: 8 models x 3 seeds, one GPU, resumable.
#   GPU=6 bash run_baselines.sh IndicCrisis_clean.csv "Merged Images"
#   MODELS="muril xlmr" SEEDS="42" GPU=6 bash run_baselines.sh ...      # subset
# Results go to $OUT/txt_<model>_s<seed> and $OUT/img_<model>_s<seed> (default OUT=runs_v2).
set -u
CSV="${1:?usage: bash run_baselines.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_baselines.sh <csv> <image_dir>}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${MODELS:-mbert indicbert muril xlmr vgg19 resnet50 vit_b16 convnext}"
OUT="${OUT:-runs_v2}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT/logs"
for seed in $SEEDS; do
  for m in $MODELS; do
    case "$m" in mbert|indicbert|muril|xlmr) pre=txt ;; *) pre=img ;; esac
    dir="$OUT/${pre}_${m}_s${seed}"
    if [ -f "$dir/report/metrics_summary.csv" ]; then echo "[skip]  ${pre}_${m}_s${seed}"; continue; fi
    echo "[start] ${pre}_${m}_s${seed} on GPU $GPU"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" python train_baselines.py --model "$m" --csv "$CSV" --image_dir "$IMG" \
        --seed "$seed" --save_dir "$dir" $EXTRA > "$OUT/logs/${pre}_${m}_s${seed}.log" 2>&1 \
      && echo "[done]  ${pre}_${m}_s${seed}" || echo "[FAIL]  ${pre}_${m}_s${seed} - see $OUT/logs/${pre}_${m}_s${seed}.log"
  done
done
