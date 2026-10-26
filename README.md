# IndicCrisis

**IndicCrisis: A Benchmark and Explainable Correlation-Aware Framework for Indic-Multilingual Multimodal
Crisis Understanding** (under review, ACM Web Conference 2026)

> **Anonymous review.** This repository is anonymised for double-blind review. Author names, the dataset
> download link and the data licence will be added after the review.

IndicCrisis is a bilingual **English–Hindi** dataset of **6,862 disaster news articles** (headline + image).
Every article is annotated for five crisis tasks (crisis type, severity, informativeness, humanitarian category,
resource need) **three times**: from the text alone, from the image alone and for the article as a whole,
together with the **cross-modal correlation (CMC)** between text and image (*Low / Partial / Strong*). This
step-wise annotation makes it possible to evaluate *how* a model uses each modality, not only whether its final
label is right.

This repository contains the model of the paper, **CCEC-G**, all baselines, the vision–language-model (VLM)
experiments, the step-wise diagnostics, the CrisisMMD experiments and the scripts that build every table and
figure of the paper and its supplementary material.

<p align="center">
  <img src="assets/arch_final.png" width="100%" alt="CCEC-G architecture">
</p>

**CCEC-G** (Correlation-Conditioned Evidence Composition with Graph-contrastive correlation learning) follows
the annotation: a text-only and an image-only head judge every task from one modality, a CMC router estimates
how strongly text and image correspond, and composition experts (one table per correlation regime) combine the
two judgements. Every prediction can be traced back to the text judgement, the image judgement, the regime and
the table entry that produced it. A CMC-weighted graph contrastive loss and an ordinal loss teach the model to
measure correlation; an optional frozen VLM "knowledge view" feeds only the router and the joint-evidence expert.

## Contents

- [Main results](#main-results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Step-wise diagnostics for your own model](#step-wise-diagnostics-for-your-own-model)
- [Smoke test](#smoke-test)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation) · [Licence](#licence)

## Main results

**IndicCrisis test set** (macro-F1 %, mean ± std over seeds 42, 43, 44; Avg.₅ = mean over the five tasks of the
combined label):

| Model | Avg.₅ | CMC |
|---|---|---|
| ConvNeXt-B (image only) | 46.37 ± 1.58 | – |
| XLM-R (text only) | 63.04 ± 1.41 | – |
| MANN / CAMM (re-implemented) | 36.52 / 46.23 | – |
| ALIGN / CLIP (re-implemented) | 49.13 / 56.26 | – |
| DMCC / RoBERTaMFT (re-implemented) | 59.98 / 62.92 | – |
| Fusion (ours, XLM-R) | 64.38 ± 1.44 | 52.83 |
| **CCEC-G (ours, XLM-R)** | **63.86 ± 1.22** | **53.99** |

CCEC-G is ahead of all six re-implemented crisis classifiers in every seed and matches the Fusion classifier
while exposing its intermediate judgements.

**Vision–language models** (test split of seed 42, Avg.₅ macro-F1 %; CMC for the fine-tuned models):

| Model | Params | Zero-shot | CoT | Few-shot (k = 4) | LoRA | CMC (LoRA) |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 8.3B | 39.79 | 42.53 | 48.46 | 61.87 | 47.58 |
| InternVL3-8B | 8.0B | 42.14 | 44.03 | 50.16 | 61.31 | 44.58 |
| Gemma 3 12B | 12.3B | 42.54 | 43.41 | 50.29 | 62.48 | 45.64 |
| **CCEC-G (ours)** | **0.38B** | – | – | – | **62.55** | **54.17** |

LoRA-tuned VLMs 21–33 times larger reach the same Avg.₅ as CCEC-G (differences not significant) but a
significantly lower CMC (paired bootstrap, p ≤ 0.019). CCEC-G trains in about 18 minutes on one A100.

**Step-wise diagnostics.** Decisions in which the image changed the annotators' judgement are solved about half
as often as the others (≈35% vs. ≈73% accuracy for the strongest models, LoRA-tuned VLMs included), and no model
changes its judgement like the annotators. CrisisMMD results are in the paper.

## Repository structure

```
IndicCrisis/
├── indiccrisis/                    Python package
│   ├── train_ccec.py               CCEC-G, CCEC and Fusion: training, evaluation, alignment analysis
│   ├── ccec_report.py              full evaluation report of a run (confusion matrices, per-class metrics)
│   ├── baselines/
│   │   ├── train_baselines.py      fine-tuned text-only (mBERT, IndicBERT, MuRIL, XLM-R) and image-only models
│   │   └── train_sota.py           MANN, CAMM, RoBERTaMFT, DMCC, CLIP, ALIGN (re-implemented)
│   ├── vlm/
│   │   ├── pilot_reasoning_graph.py   zero-shot VLM prompting (direct, chain-of-thought), parsing, metrics
│   │   ├── vlm_adapt.py            few-shot (retrieved demonstrations) and LoRA fine-tuning of VLMs
│   │   └── embed_vlm_views.py      VLM chain-of-thought -> frozen "knowledge view" for CCEC-G
│   ├── crisismmd/
│   │   ├── prepare_crisismmd.py    CrisisMMD benchmark splits -> IndicCrisis CSV format (with checks)
│   │   └── crisismmd_table.py      mean accuracy / weighted F1 of the CrisisMMD runs
│   ├── analysis/
│   │   ├── modality_reliance.py    step-wise diagnostics (image-dependent decisions, change rate, ...)
│   │   ├── vlm_significance.py     paired bootstrap tests for the VLM comparison
│   │   ├── aggregate_runs.py       mean ± std over seeds, paired t-tests, ablation / alignment tables
│   │   ├── paper_tables.py         main results, language and per-language detail tables
│   │   ├── supplementary_tables.py tables of the supplementary material
│   │   ├── make_figures.py         all figures (vector PDF, ACM column widths)
│   │   ├── qualitative.py          qualitative analysis (image, text, annotated and predicted step-wise labels)
│   │   ├── export_prompts.py       the exact VLM prompts as text files
│   │   └── ...                     language metrics, near-duplicate check, examples
│   └── tools/                      data checks, checkpoint conversion to safetensors
├── scripts/                        resumable multi-GPU run scripts (one per experiment family)
├── tests/                          synthetic data + smoke test (every pipeline with tiny mock models)
├── data/README.md                  expected data format
├── docs/reproduce.md               which command produces which table / figure of the paper
├── docs/qualitative/               English glosses and notes of the qualitative examples
├── assets/                         architecture figure
└── legacy/                         superseded table scripts, kept for reference
```

## Installation

Python ≥ 3.9 and a CUDA GPU. All paper runs used one NVIDIA A100 (40 GB) per run.

```bash
git clone <anonymous repository URL> IndicCrisis && cd IndicCrisis
# install PyTorch for your CUDA version first: https://pytorch.org
pip install -r requirements.txt
pip install -e .                    # optional: makes `python -m indiccrisis...` work from anywhere
```

| Environment | Used for | Install |
|---|---|---|
| main | CCEC-G, baselines, re-implemented models, CrisisMMD, analysis | `pip install -r requirements.txt` |
| VLM (separate env recommended) | zero-shot / few-shot / LoRA VLMs, VLM knowledge view | `pip install -e ".[vlm]"` + a recent `transformers` (Qwen2.5-VL, InternVL3, Gemma 3); `vllm` optional |

The scripts in `scripts/` put the repository on `PYTHONPATH` themselves, so they work without
`pip install -e .` and from any working directory (typically the folder that holds the data).

## Data

See [`data/README.md`](data/README.md) for the exact format. Expected layout:

```
<data folder>/
├── IndicCrisis_clean.csv           one row per article
└── Merged Images/                  the images (file names as in the CSV)
```

The CSV contains the article text, the image file name and, for each task `T` in {Crisis, Severity,
Informativeness, Humanitarian, Resource}, the columns `T_Text`, `T_Image` and `T_Combined`, plus `CMC`. Check a
copy of the data with `python -m indiccrisis.tools.check_data --csv IndicCrisis_clean.csv --image_dir "Merged Images"`.
CrisisMMD is converted to the same format with `python -m indiccrisis.crisismmd.prepare_crisismmd`.

## Quick start

```bash
CSV=IndicCrisis_clean.csv; IMG="Merged Images"

# CCEC-G (XLM-R text encoder, CMC-weighted graph contrastive learning, ordinal CMC loss)
python -m indiccrisis.train_ccec --csv $CSV --image_dir "$IMG" --text_encoder xlmr \
    --corr graph --nbr_sampler random --ordinal_cmc --seed 42 --save_dir runs/ccecg_s42

# the same with the VLM knowledge view (build vlm_views.npz first, see docs/reproduce.md)
python -m indiccrisis.train_ccec --csv $CSV --image_dir "$IMG" --text_encoder xlmr \
    --corr graph --nbr_sampler random --ordinal_cmc --vlm_view vlm_views.npz --seed 42 --save_dir runs/ccecg_vlm_s42

# Fusion baseline (same encoders and objective, a classifier instead of composition)
python -m indiccrisis.train_ccec --csv $CSV --image_dir "$IMG" --text_encoder xlmr \
    --fusion_baseline --seed 42 --save_dir runs/fusion_s42

# a text-only baseline and a re-implemented crisis classifier
python -m indiccrisis.baselines.train_baselines --model xlmr --csv $CSV --image_dir "$IMG" --save_dir runs/txt_xlmr_s42
python -m indiccrisis.baselines.train_sota --model dmcc --csv $CSV --image_dir "$IMG" --save_dir runs/sota_dmcc_s42
```

Each run writes to its `--save_dir`:

| File | Content |
|---|---|
| `results.json` | arguments, test metrics, training history, efficiency (parameters, time, memory) |
| `report/predictions.csv` | per test post: gold label, prediction and confidence of the text-only, image-only and combined step of every task, and of the CMC |
| `report/` (other files) | per-class metrics, confusion matrices, results per CMC level, `metrics_summary.csv` |

The split is a random 80/10/10 split per seed, stratified by crisis type and language, in which identical
headlines never cross splits. `--split_col` uses fixed splits from a CSV column instead. All paper runs use 10
epochs and seeds 42, 43 and 44.

## Reproducing the paper

[`docs/reproduce.md`](docs/reproduce.md) lists the command that trains each group of models and the command
that builds each table and figure. In short:

| Step | Command |
|---|---|
| Text-only and image-only baselines | `bash scripts/run_baselines.sh $CSV "$IMG"` |
| Re-implemented crisis classifiers | `bash scripts/run_sota.sh $CSV "$IMG"` |
| CCEC-G, CCEC, Fusion and ablations | `bash scripts/run_ablations.sh $CSV "$IMG"` |
| VLMs: zero-shot, few-shot, LoRA | `bash scripts/run_pilot.sh`, `run_vlm_all.sh`, `run_vlm_models.sh` |
| VLM knowledge view | `STEP=1 bash scripts/run_vlm_view.sh $CSV "$IMG"`, then `STEP=2 ...` |
| CrisisMMD | `bash scripts/run_crisismmd.sh` |
| All paper tables | `bash scripts/make_paper_tables.sh` |
| VLM significance tests | `python -m indiccrisis.analysis.vlm_significance ...` |
| Figures | `python -m indiccrisis.analysis.make_figures ...` |
| Supplementary tables, prompts, qualitative examples | `python -m indiccrisis.analysis.{supplementary_tables,export_prompts,qualitative} ...` |

Every run script is **resumable** (finished runs are skipped, so a script can simply be restarted) and
**multi-GPU** (`GPUS="0 1 2"` runs one job per listed GPU); logs go to `<out>/logs/`.

## Step-wise diagnostics for your own model

Any model that outputs text-only, image-only and combined predictions can be evaluated with the step-wise
protocol of the paper. Write a `predictions.csv` in the format of `train_ccec.py` (one row per test post with
`image_name`, `language`, and for every task `T` and view `V` in {Text, Image, Combined} the columns
`T_V_true` and `T_V_pred`, plus `CMC_true` and `CMC_pred`), save it as
`<folder>/<name>_s<seed>/report/predictions.csv`, and run

```bash
python -m indiccrisis.analysis.modality_reliance --runs <folder> --variants <name> --out diag_<name>
```

It reports, per model, the macro-F1 of every step, the combined accuracy on **image-dependent** decisions (the
annotators' combined label differs from their text-only label) and on **text-sufficient** decisions, the
**change rate** (how often the combined prediction differs from the model's own text-only prediction; annotators:
7.4%) and the **change precision** (how often such a change is needed and correct). VLM outputs of
`pilot_reasoning_graph.py` / `vlm_adapt.py` are read with `--vlm <folder>`.

## Smoke test

```bash
bash tests/smoke_test.sh        # CPU, a few minutes: synthetic data + tiny random encoders, no downloads
```

It runs CCEC-G, Fusion, a text baseline, the six re-implemented classifiers, the CrisisMMD conversion and the
CrisisMMD run script end to end. It checks that the pipelines run, not the numbers.

## Troubleshooting

**PyTorch < 2.6 refuses `.bin` checkpoints.** Recent `transformers` no longer load `.bin` weights with older
PyTorch. Convert the affected models (MuRIL, IndicBERT, CLIP, ALIGN, ...) once to safetensors:

```bash
export INDICCRISIS_MODEL_DIR=/path/to/converted_models    # default: the current directory
python -m indiccrisis.tools.convert_to_safetensors        # writes <short>-safetensors/ there
python -m indiccrisis.tools.convert_indicbert
```

Every training script looks for `<short>-safetensors/` in `$INDICCRISIS_MODEL_DIR` before downloading.

**Out of GPU memory.** CCEC-G with XLM-R needs about 7 GB (batch size 16), with the SigLIP-2 text encoder about
3 GB. LoRA fine-tuning of the VLMs fits on one 40 GB GPU with the settings of the scripts.

**An interrupted run.** Start the same script again; finished runs are skipped. A run counts as finished when
its `report/metrics_summary.csv` (CrisisMMD: `results.json`) exists; incomplete run folders are run again.

