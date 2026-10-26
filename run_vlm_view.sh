#!/usr/bin/env bash
# Build the VLM knowledge view for CCEC-G (run once; both steps are resumable / re-runnable).
#   Step 1 (env vlmhf): zero-shot chain-of-thought for ALL posts. No labels are used, so it is
#                       leak-free for train, validation and test. ~3 h on one A100 (hf backend).
#   Step 2 (env base) : embed the reasoning with frozen SigLIP-2 -> vlm_views.npz (~5 min).
#
#   conda activate vlmhf;  STEP=1 GPU=6 bash run_vlm_view.sh IndicCrisis_clean.csv "Merged Images"
#   conda activate base;   STEP=2 GPU=6 bash run_vlm_view.sh IndicCrisis_clean.csv "Merged Images"
set -eu
CSV="${1:?usage: bash run_vlm_view.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_vlm_view.sh <csv> <image_dir>}"
STEP="${STEP:?set STEP=1 (generate) or STEP=2 (embed)}"
GPU="${GPU:-0}"
OUT="${OUT:-vlm_all}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
if [ "$STEP" = "1" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" python pilot_reasoning_graph.py --csv "$CSV" --image_dir "$IMG" \
      --model "$MODEL" --backend hf --modes cot --split all --n 0 --out "$OUT" 2>&1 | tee "$OUT.log"
else
  CUDA_VISIBLE_DEVICES="$GPU" python embed_vlm_views.py --pilot_out "$OUT" --out vlm_views.npz
fi
