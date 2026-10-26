#!/usr/bin/env bash
# CrisisMMD (benchmark splits of Ofli et al., 2020): XLM-R text-only, Fusion and CCEC-G (latent CMC),
# informativeness and humanitarian, 3 seeds -> 18 runs.  The six re-implemented crisis models of the paper run
# with the same splits, seeds and model selection:  MODELS="mann camm robertamft dmcc clip align"
# (original English text encoders, --faithful; MANN's discriminator predicts the disaster event, --adv_col).  Resumable (finished runs are skipped), multi-GPU,
# logs in $OUT/logs/.  Settings = the IndicCrisis XLM-R runs; checkpoints are selected on dev weighted F1
# (the metric reported on CrisisMMD).
#
# Usage:
#   IMG=/path/to/CrisisMMD_v2.0 GPUS="0 1 2" bash run_crisismmd.sh
#   TSV=...  (default /home/nagendra/cafunet/crisismmd)   MODELS="ccecg"   SEEDS="42"   TASKS="informative"
#   MODELS="mann camm robertamft dmcc clip align" TASKS="informative humanitarian damage" GPUS="4 4 5 5"
#   EXTRA="--epochs 5"  (extra flags for every run)
# IMG is the folder that contains data_image/ (the paths in the TSVs are relative to it).
set -u
TSV="${TSV:-/home/nagendra/cafunet/crisismmd}"
IMG="${IMG:-}"
DATA="${DATA:-crisismmd_csv}"
OUT="${OUT:-runs_crisismmd}"
TASKS="${TASKS:-informative humanitarian}"
MODELS="${MODELS:-xlmr fusion ccecg}"
SEEDS="${SEEDS:-42 43 44}"
GPUS="${GPUS:-0}"
EXTRA="${EXTRA:-}"
SELECT="--select_metric avg_combined_weighted_f1"
FULL="--corr graph --nbr_sampler random --ordinal_cmc"          # CCEC-G as in the IndicCrisis runs
mkdir -p "$OUT/logs"

# 1) convert the TSVs (checks the image paths; stops if many images are missing)
# shellcheck disable=SC2086
python prepare_crisismmd.py --tsv_dir "$TSV" ${IMG:+--image_root "$IMG"} --out_dir "$DATA" --tasks $TASKS \
    || { echo "conversion failed"; exit 1; }
if [ -z "$IMG" ]; then
  IMG=$(python -c "import json,sys; print(json.load(open('$DATA/crisismmd_${TASKS%% *}_summary.json'))['image_root'])")
fi
echo "image root: $IMG"

# 2) job list
JOBS=()
for task in $TASKS; do
  case "$task" in informative) T=Informativeness ;; humanitarian) T=Humanitarian ;; damage) T=Damage ;; esac
  for seed in $SEEDS; do
    for m in $MODELS; do JOBS+=("$task|$T|$m|$seed"); done
  done
done

run_job() {   # $1 = gpu, $2 = job
  local gpu="$1" task T m seed dir csv common
  IFS="|" read -r task T m seed <<< "$2"
  dir="$OUT/$task/${m}_s${seed}"
  csv="$DATA/crisismmd_${task}.csv"
  if [ -f "$dir/results.json" ]; then echo "[skip]  $task/${m}_s${seed}"; return; fi
  echo "[start] $task/${m}_s${seed} on GPU $gpu"
  common="--csv $csv --tasks $T --split_col split --keep_source --seed $seed --save_dir $dir"
  log="$OUT/logs/${task}_${m}_s${seed}.log"
  # shellcheck disable=SC2086
  case "$m" in
    xlmr)   CUDA_VISIBLE_DEVICES="$gpu" python train_baselines.py --model xlmr --image_dir "$IMG" $common \
                $SELECT $EXTRA > "$log" 2>&1 ;;
    fusion) CUDA_VISIBLE_DEVICES="$gpu" python train_ccec.py --image_dir "$IMG" $common --text_encoder xlmr \
                --fusion_baseline --no_cmc_sup --delete_ckpt $SELECT $EXTRA > "$log" 2>&1 ;;
    ccecg)  CUDA_VISIBLE_DEVICES="$gpu" python train_ccec.py --image_dir "$IMG" $common --text_encoder xlmr \
                $FULL --no_cmc_sup --delete_ckpt $SELECT $EXTRA > "$log" 2>&1 ;;
    mann|camm|robertamft|dmcc|clip|align)
            CUDA_VISIBLE_DEVICES="$gpu" python train_sota.py --model "$m" --image_dir "$IMG" $common --faithful \
                --adv_col event_name $SELECT $EXTRA > "$log" 2>&1 ;;
    *) echo "unknown model $m"; return ;;
  esac
  if [ $? -eq 0 ]; then echo "[done]  $task/${m}_s${seed}"; else echo "[FAIL]  $task/${m}_s${seed} - see $log"; fi
}

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

echo "all runs finished - table"
python crisismmd_table.py --runs "$OUT" --tasks $TASKS
