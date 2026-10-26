#!/usr/bin/env bash
# Reasoning-graph pilot on the DGX (200 validation posts x 3 prompting modes).
# Needs:  pip install vllm        (in its own environment is safest)
#
# Option A - offline vLLM, one GPU (simplest):
#   GPU=4 bash run_pilot.sh IndicCrisis_clean.csv "Merged Images"
# On a GPU that is partly used by others, lower the reservation (0.5 = 20 GB of 40 GB):
#   GPU=3 GPU_MEM=0.5 bash run_pilot.sh IndicCrisis_clean.csv "Merged Images"
# Option B - no vLLM needed: Hugging Face transformers in the environment that trains CCEC
#   conda activate <your CCEC env>;  BACKEND=hf GPU=4 bash run_pilot.sh IndicCrisis_clean.csv "Merged Images"
# Option C - if vLLM installs but its offline API fails, start a server and use it:
#   CUDA_VISIBLE_DEVICES=6 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' &
#   BACKEND=openai bash run_pilot.sh IndicCrisis_clean.csv "Merged Images"
set -eu
CSV="${1:?usage: bash run_pilot.sh <csv> <image_dir>}"
IMG="${2:?usage: bash run_pilot.sh <csv> <image_dir>}"
GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
BACKEND="${BACKEND:-vllm}"
N="${N:-200}"            # 0 = whole split
SPLIT="${SPLIT:-val}"      # val while designing prompts, test for the final baseline
MODES="${MODES:-direct cot graph}"
OUT="${OUT:-pilot_out}"
GPU_MEM="${GPU_MEM:-0.85}"   # share of the GPU's TOTAL memory vLLM may reserve

CUDA_VISIBLE_DEVICES="$GPU" python pilot_reasoning_graph.py --csv "$CSV" --image_dir "$IMG" \
    --model "$MODEL" --backend "$BACKEND" --modes $MODES --n "$N" --split "$SPLIT" --out "$OUT" --gpu_mem "$GPU_MEM" \
    2>&1 | tee "$OUT.log"
