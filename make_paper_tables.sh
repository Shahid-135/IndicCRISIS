#!/usr/bin/env bash
# Build every results table of the paper from the runs, into paper_tables_out/.
#   bash make_paper_tables.sh                      # defaults below
#   RUNS=runs_xlmr BASE=runs_v2 SIGLIP=runs_v2 VLM=pilot_test GPU=3 bash make_paper_tables.sh
# Tables produced (copy them into the paper's tables/ folder):
#   main_results.tex, language_table.tex, detail_{en,hi,all}.tex   (paper_tables.py)
#   ablation_table.tex, alignment_table.tex                         (aggregate_runs.py)
#   diagnostics_table.tex                                           (modality_reliance.py, seed 42 + VLM)
#   overlap_table.tex                                               (check_image_overlap.py, needs a GPU)
set -u
RUNS="${RUNS:-runs_xlmr}"; BASE="${BASE:-runs_v2}"; SIGLIP="${SIGLIP:-}"; VLM="${VLM:-pilot_test}"
CSV="${CSV:-IndicCrisis_clean.csv}"; IMG="${IMG:-Merged Images}"; GPU="${GPU:-0}"; TE="${TE:-xlmr}"
OUT="${OUT:-paper_tables_out}"
mkdir -p "$OUT"
VLMARG=""; [ -d "$VLM" ] && VLMARG="--vlm $VLM" || echo "note: VLM folder '$VLM' not found - VLM rows skipped"
SIGARG=""; [ -n "$SIGLIP" ] && SIGARG="--siglip_runs $SIGLIP"

echo "== 1/4 main, language and detail tables"
python paper_tables.py --runs "$RUNS" --baseline_runs "$BASE" $SIGARG $VLMARG --text_encoder "$TE" --out "$OUT/main" \
  && cp "$OUT"/main/*.tex "$OUT"/

echo "== 2/4 ablation and alignment tables"
python aggregate_runs.py --runs "$RUNS" --out "$OUT/ablation" --ref full_vlm \
  && cp "$OUT/ablation/ablation_table.tex" "$OUT/ablation/alignment_table.tex" "$OUT"/

echo "== 3/4 step-wise diagnostics (seed 42, same test posts as the VLM)"
python modality_reliance.py --runs "$RUNS" --variants ccec full full_vlm fusion_baseline fusion_full fusion_full_vlm \
  --seeds 42 $VLMARG --out "$OUT/diagnostics" && cp "$OUT/diagnostics/modality_reliance_table.tex" "$OUT/diagnostics_table.tex"

echo "== 4/4 image overlap between train and test (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" python check_image_overlap.py --csv "$CSV" --image_dir "$IMG" --runs "$RUNS" $VLMARG \
  --out "$OUT/overlap" && cp "$OUT/overlap/overlap_table.tex" "$OUT"/

echo; ls "$OUT"/*.tex
echo "done - send me the whole $OUT folder"
