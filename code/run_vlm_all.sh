#!/usr/bin/env bash
# Overnight VLM pipeline: retrieval index -> few-shot -> LoRA training -> fine-tuned inference -> tables.
# Every stage is resumable: if the night is interrupted, run the same command again.
#   GPU=0 bash run_vlm_all.sh IndicCrisis_clean.csv "Merged Images"
# Run it in the conda env that ran the zero-shot VLM (Qwen2.5-VL with Hugging Face transformers) + peft.
set -u
CSV="${1:?usage: bash run_vlm_all.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_vlm_all.sh <csv> <image_dir>}"
GPU="${GPU:-0}"; OUT="${OUT:-pilot_test}"; EXTRA="${EXTRA:-}"
mkdir -p "$OUT/logs"
run() {   # stage name, then arguments
  local st="$1"; shift
  echo "[$(date +%H:%M)] start $st"
  if CUDA_VISIBLE_DEVICES="$GPU" python vlm_adapt.py --stage "$st" --csv "$CSV" --image_dir "$IMG" --out "$OUT" "$@" $EXTRA \
       > "$OUT/logs/$st.log" 2>&1; then
    echo "[$(date +%H:%M)] done  $st | $(grep -E '^\[(fewshot|finetuned|train|index|check)\]' "$OUT/logs/$st.log" | tail -1)"
  else
    echo "[$(date +%H:%M)] FAIL  $st - see $OUT/logs/$st.log"; return 1
  fi
}
run check
run index
run fewshot   --k 4 --batch 4
run train     --epochs 2 --lora_r 16 --lr 1e-4 --grad_accum 8 && run finetuned --batch 8
echo "[$(date +%H:%M)] all stages finished - results in $OUT/{fewshot,finetuned}/metrics.json"
