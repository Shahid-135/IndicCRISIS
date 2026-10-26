#!/usr/bin/env bash
# Zero-shot (direct, CoT) -> few-shot -> LoRA fine-tuning -> fine-tuned inference, for several VLMs.
# Each model writes to pilot_test_<short>/ ; every stage is resumable.
#   GPU=0 MODELS="gemma3" nohup bash run_vlm_models.sh IndicCrisis_clean.csv "Merged Images" > vlm_gemma3.log 2>&1 &
#   GPU=1 MODELS="internvl3" nohup bash run_vlm_models.sh ... > vlm_internvl3.log 2>&1 &
# short names: gemma3 internvl3 llama32v qwen25vl   (Qwen is normally run by run_vlm_all.sh into pilot_test/)
set -u
CSV="${1:?usage: bash run_vlm_models.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_vlm_models.sh <csv> <image_dir>}"
GPU="${GPU:-0}"; MODELS="${MODELS:-gemma3 internvl3 llama32v}"; STAGES="${STAGES:-zeroshot fewshot train finetuned}"
declare -A HF=( [gemma3]="google/gemma-3-12b-it" [internvl3]="OpenGVLab/InternVL3-8B-hf"
                [llama32v]="meta-llama/Llama-3.2-11B-Vision-Instruct" [qwen25vl]="Qwen/Qwen2.5-VL-7B-Instruct" )
for m in $MODELS; do
  name="${HF[$m]:?unknown model $m}"; out="pilot_test_$m"; mkdir -p "$out/logs"
  echo "[$(date +%H:%M)] ===== $m ($name) -> $out"
  for st in $STAGES; do
    log="$out/logs/$st.log"
    if [ "$st" = zeroshot ]; then
      if [ -f "$out/direct/metrics.json" ] && [ -f "$out/cot/metrics.json" ]; then echo "  [skip] zeroshot"; continue; fi
      cmd=(python pilot_reasoning_graph.py --csv "$CSV" --image_dir "$IMG" --out "$out" --model "$name"
           --backend hf --hf_batch 4 --split test --n 0 --modes direct cot)
    else
      [ "$st" = fewshot ] && [ -f "$out/fewshot/metrics.json" ] && { echo "  [skip] fewshot"; continue; }
      [ "$st" = finetuned ] && [ -f "$out/finetuned/metrics.json" ] && { echo "  [skip] finetuned"; continue; }
      extra=(); [ "$st" = fewshot ] && extra=(--k 4 --batch 2); [ "$st" = finetuned ] && extra=(--batch 4)
      [ "$st" = train ] && extra=(--epochs 2 --lora_r 16 --lr 1e-4 --grad_accum 8)
      cmd=(python vlm_adapt.py --stage "$st" --csv "$CSV" --image_dir "$IMG" --out "$out" --model "$name"
           --index_from pilot_test/retrieval_index.npz "${extra[@]}")
    fi
    echo "  [$(date +%H:%M)] start $st"
    if CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" > "$log" 2>&1; then
      echo "  [$(date +%H:%M)] done  $st | $(grep -E '^\[(fewshot|finetuned|train)\]|combined.avg_macro_f1' "$log" | tail -1 | cut -c1-120)"
    else
      echo "  [$(date +%H:%M)] FAIL  $st - see $log"
      [ "$st" = train ] && break          # no adapter -> skip fine-tuned inference for this model
    fi
  done
done
echo "[$(date +%H:%M)] finished: $MODELS"
