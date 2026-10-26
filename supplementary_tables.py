"""
supplementary_tables.py - LaTeX tables of the supplementary material, built from results_data/.

    python supplementary_tables.py --data results_data --out supplementary/tables

Writes:
    diag_seeds.tex        step-wise diagnostics of the fine-tuned models (mean over 3 seeds, both text encoders)
    diag_vlm.tex          step-wise diagnostics of every VLM setting and our models (test split of seed 42)
    vlm_full.tex          per-task results, step F1, CMC and per-language Avg5 of the three VLMs in four settings
    vlm_sig.tex           paired bootstrap tests (Avg5, CMC vs CCEC-G; CoT vs direct by language and decision type)
    cost.tex              parameters, training time, memory and inference time of our models and the VLMs
    clip_align.tex        raw vs. L2-normalised CLIP/ALIGN embeddings: validation and test, both datasets
    cmmd_seeds.tex        CrisisMMD: every model, every seed (accuracy and weighted F1)
    cmmd_sig.tex          CrisisMMD: CCEC-G vs. every model, paired t-test over seeds (weighted F1)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
VLMS = [("pilot_test", "Qwen2.5-VL-7B"), ("pilot_test_internvl3", "InternVL3-8B"), ("pilot_test_gemma3", "Gemma 3 12B")]
MODES = [("direct", "zero-shot, direct"), ("cot", "zero-shot, CoT"), ("fewshot", "few-shot ($k=4$)"),
         ("finetuned", "LoRA")]


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def write(path, body):
    with open(path, "w") as f:
        f.write(body)
    print("wrote", path)


def table(caption, label, colspec, header, rows, star=True, size="\\footnotesize", width="\\textwidth"):
    env = "table*" if star else "table"
    resize_open = f"\\begin{{adjustbox}}{{max width={width}}}\n" if width else ""      # shrink only if too wide
    resize_close = "\n\\end{adjustbox}" if width else ""
    return (f"\\begin{{{env}}}[!htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
            f"\\setlength{{\\tabcolsep}}{{3pt}}\n{resize_open}\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n"
            f"\\midrule\n" + "\n".join(rows) + f"\n\\bottomrule\n\\end{{tabular}}{resize_close}\n\\end{{{env}}}\n")


# ── diagnostics ─────────────────────────────────────────────────────────────────────────────────────────
DIAG_COLS = [("text_step_f1", "Text"), ("image_step_f1", "Image"), ("cmc_f1", "CMC"), ("combined_f1", "Comb."),
             ("change_rate", "Change"), ("change_precision", "Ch. prec."), ("image_use_recall", "Img-use rec."),
             ("acc_image_dependent", "Acc$_{\\text{img-dep}}$"), ("acc_text_sufficient", "Acc$_{\\text{text-suff}}$")]


def diag_row(name, r):
    cells = []
    for k, _ in DIAG_COLS:
        v = r.get(k, np.nan)
        cells.append("--" if v is None or (isinstance(v, float) and np.isnan(v)) else pct(v))
    return f"{name} & " + " & ".join(cells) + " \\\\"


def diag_tables(a):
    D = os.path.join(a.data, "diagnostics")
    rows = []
    for f, tag in (("xlmr_3seeds.csv", "XLM-R text encoder"), ("siglip_3seeds.csv", "SigLIP-2 text encoder")):
        d = pd.read_csv(os.path.join(D, f))
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{tag}}}}} \\\\")
        for _, r in d.iterrows():
            rows.append(diag_row(r["model"], r))
        rows.append("\\midrule")
    rows = rows[:-1]
    head = "Model & " + " & ".join(h for _, h in DIAG_COLS) + " \\\\"
    cap = ("Step-wise diagnostics of the fine-tuned models on \\dataset{} (\\%, mean over three seeds). Text / Image /"
           " CMC / Comb.: macro-F1 of the text-only, image-only, CMC and combined predictions against the corresponding"
           " annotations (averaged over the five tasks). Change: share of decisions in which the combined prediction"
           " differs from the model's own text-only prediction (annotators: 7.4\\%); Ch.\\ prec.: share of these"
           " changes that are necessary and correct; Img-use rec.: share of image-dependent decisions solved by such a"
           " change; Acc: combined accuracy on image-dependent and text-sufficient decisions.")
    write(os.path.join(a.out, "diag_seeds.tex"), table(cap, "tab:s_diag_seeds", "l" + "c" * 9, head, rows))

    d42 = pd.read_csv(os.path.join(D, "xlmr_seed42_with_3vlms.csv"))
    rows = ["\\multicolumn{10}{l}{\\textit{Vision--language models}} \\\\"]
    for model in d42.model:
        if " | " in model:
            r = d42[d42.model == model].iloc[0]
            v, s = model.split(" | ")
            rows.append(diag_row(f"{v}, {s}", r))
    rows += ["\\midrule", "\\multicolumn{10}{l}{\\textit{Ours, XLM-R text encoder}} \\\\"]
    for model in d42.model:
        if " | " not in model:
            rows.append(diag_row(model, d42[d42.model == model].iloc[0]))
    cap = ("Step-wise diagnostics on the test split of seed 42 (\\%) for every VLM setting and our fine-tuned"
           " models; columns as in Table~\\ref{tab:s_diag_seeds}. On this split 9.0\\% of the decisions are"
           " image-dependent.")
    write(os.path.join(a.out, "diag_vlm.tex"), table(cap, "tab:s_diag_vlm", "l" + "c" * 9, head, rows))


# ── VLMs ────────────────────────────────────────────────────────────────────────────────────────────────
def vlm_tables(a):
    root = os.path.join(a.data, "vlm_three_models")
    rows = []
    for folder, name in VLMS:
        rows.append(f"\\multicolumn{{13}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for mode, mname in MODES:
            p = os.path.join(root, folder, mode, "metrics.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            cells = [pct(j[f"combined.{t}.macro_f1"]) for t in TASKS]
            cells += [pct(j["combined.avg_macro_f1"], 2), pct(j["lang=en.combined.avg_macro_f1"]),
                      pct(j["lang=hi.combined.avg_macro_f1"]), pct(j["text_only.avg_macro_f1"]),
                      pct(j["image_only.avg_macro_f1"]), pct(j["cmc.macro_f1"]), pct(j["parse_rate"], 0)]
            rows.append(f"{mname} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    head = ("Setting & Crisis & Sev. & Inf. & Hum. & Res. & Avg.$_5$ & EN & HI & Text & Image & CMC & Parsed \\\\")
    cap = ("Vision--language models on the test split of seed 42 (691 posts): macro-F1 (\\%) of the combined prediction"
           " per task, its mean (Avg.$_5$) on all, English and Hindi posts, the mean macro-F1 of the text-only and"
           " image-only steps, CMC macro-F1 and the share of parsable answers. The main paper reports Avg.$_5$ and the"
           " CMC of the fine-tuned models.")
    write(os.path.join(a.out, "vlm_full.tex"), table(cap, "tab:s_vlm_full", "l" + "c" * 12, head, rows))

    S = os.path.join(a.data, "vlm_significance")
    av = pd.read_csv(os.path.join(S, "avg5_vs_ccecg.csv"))
    cm = pd.read_csv(os.path.join(S, "cmc_vs_ccecg.csv"))
    rows = []
    nice = dict((f.split("pilot_test")[0] or n, n) for f, n in VLMS)
    for _, r in av.iterrows():
        m = r["model"]
        if "|" in m:
            v, mode = m.split("|")
            label = f"{v.replace('Gemma-3-12B', 'Gemma 3 12B')}, {dict(MODES)[mode]}"
        else:
            label = m
        c = cm[cm.model == m]
        cmc = (f"{c.cmc_f1.iloc[0]:.2f} & {c.ccecg_minus_model.iloc[0]:+.2f} [{c.ci_lo.iloc[0]:+.2f}, "
               f"{c.ci_hi.iloc[0]:+.2f}] & {fp(c.p.iloc[0])}") if len(c) else "-- & -- & --"
        rows.append(f"{label} & {r.avg5:.2f} & {r.ccecg_minus_model:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {fp(r.p)}"
                    f" & {cmc} \\\\")
    head = ("Model & Avg.$_5$ & CCEC-G $-$ model [95\\% CI] & $p$ & CMC & CCEC-G $-$ model [95\\% CI] & $p$ \\\\")
    cap = ("Paired bootstrap tests (10{,}000 resamples of the 691 test posts of seed 42) of CCEC-G (XLM-R; Avg.$_5$"
           " 62.55, CMC 54.17) against every VLM setting and Fusion. CMC is tested for the fine-tuned models only.")
    t1 = table(cap, "tab:s_vlm_sig", "lcccccc", head, rows)

    lang = pd.read_csv(os.path.join(S, "cot_by_language.csv"))
    imd = pd.read_csv(os.path.join(S, "cot_imgdep.csv"))
    rows = []
    for _, r in lang.iterrows():
        rows.append(f"{r.vlm.replace('Gemma-3-12B', 'Gemma 3 12B')} & {r.lang.upper()} posts, Avg.$_5$ & "
                    f"{r.direct:.2f} & {r.cot:.2f} & {r.cot_minus_direct:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {fp(r.p)} \\\\")
    for _, r in imd.iterrows():
        rows.append(f"{r.vlm.replace('Gemma-3-12B', 'Gemma 3 12B')} & {r.decisions} decisions, accuracy & -- & -- & "
                    f"{r.cot_minus_direct:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {fp(r.p)} \\\\")
    head = "VLM & Measure & Direct & CoT & CoT $-$ direct [95\\% CI] & $p$ \\\\"
    cap = ("Effect of chain-of-thought prompting (zero-shot, test split of seed 42; paired bootstrap over posts).")
    t2 = table(cap, "tab:s_vlm_cot", "llcccc", head, rows)
    write(os.path.join(a.out, "vlm_sig.tex"), t1 + "\n" + t2)


# ── cost ────────────────────────────────────────────────────────────────────────────────────────────────
def cost_table(a):
    rows = []
    for root, tag in (("runs_xlmr_results", "XLM-R"), ("runs_siglip_results", "SigLIP-2")):
        for v, name in (("fusion_baseline", "Fusion"), ("full", "CCEC-G"), ("full_vlm", "CCEC-G + VLM view")):
            es = [json.load(open(f)).get("efficiency") for f in
                  sorted(glob.glob(os.path.join(a.data, root, f"{v}_s4*", "results.json")))]
            es = [e for e in es if e]
            if not es:
                continue
            k = lambda key: np.mean([e[key] for e in es])
            rows.append(f"{name} ({tag} text) & {k('total_parameters') / 1e6:.0f}M & {k('trainable_parameters') / 1e6:.1f}M"
                        f" & {k('training_time_min'):.0f} min & {k('peak_gpu_memory_GB'):.1f} GB &"
                        f" {k('inference_ms_per_sample'):.0f} ms \\\\")
    rows.append("\\midrule")
    for name, total, train, hours in (("Qwen2.5-VL-7B, LoRA", "8.33B", "40.4M", "2.4 h"),
                                      ("InternVL3-8B, LoRA", "7.98B", "40.4M", "3.2 h"),
                                      ("Gemma 3 12B, LoRA", "12.25B", "65.5M", "4.1 h")):
        rows.append(f"{name} & {total} & {train} & {hours} & -- & -- \\\\")
    head = "Model & Parameters & Trainable & Training & Peak memory & Inference / post \\\\"
    cap = ("Cost of our models (mean over three seeds; one NVIDIA A100-SXM4-40GB, batch size 16, 10 epochs) and of"
           " LoRA fine-tuning of the VLMs (rank 16, 2 epochs; parameter counts and times from the training logs).")
    write(os.path.join(a.out, "cost.tex"), table(cap, "tab:s_cost", "lccccc", head, rows))


# ── CLIP / ALIGN variants ───────────────────────────────────────────────────────────────────────────────
def clip_align_table(a):
    rows = []
    f = os.path.join(a.data, "crisismmd", "clip_align_variant_selection.txt")
    d = pd.read_csv(f, sep=" ", comment="#", names=["ds", "task", "model", "var", "seed", "val", "test"])
    for (ds, task), g in d.groupby(["ds", "task"], sort=False):
        cells = []
        for m in ("clip", "align"):
            for var in ("plain", "l2"):
                h = g[(g.model == m) & (g["var"] == var)]
                cells.append(f"{h.val.mean():.2f} / {h.test.mean():.2f}")
        name = "\\dataset{} (Avg.$_5$)" if ds == "indic" else f"CrisisMMD {task} (W-F1)"
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    head = ("& \\multicolumn{2}{c}{CLIP} & \\multicolumn{2}{c}{ALIGN} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Data & raw & L2-normalised & raw & L2-normalised \\\\")
    cap = ("CLIP and ALIGN with raw or L2-normalised embeddings: validation / test score (\\%, mean over three"
           " seeds). For each model and dataset the variant with the higher validation score is reported in the"
           " main paper: CLIP raw everywhere, ALIGN L2-normalised everywhere.")
    write(os.path.join(a.out, "clip_align.tex"), table(cap, "tab:s_clip_align", "lcccc", head, rows))


# ── CrisisMMD ───────────────────────────────────────────────────────────────────────────────────────────
CMMD_MODELS = [("mann", "MANN"), ("camm", "CAMM"), ("robertamft", "RoBERTaMFT"), ("dmcc", "DMCC"), ("clip", "CLIP"),
               ("align", "ALIGN"), ("xlmr", "XLM-R (text only)"), ("fusion", "Fusion"), ("ccecg", "CCEC-G (latent CMC)")]


def cmmd_frame(a):
    C = os.path.join(a.data, "crisismmd")
    d = pd.read_csv(os.path.join(C, "crisismmd_per_seed_l2.csv"))
    sel = pd.read_csv(os.path.join(C, "clip_align_variant_selection.txt"), sep=" ", comment="#",
                      names=["ds", "task", "model", "var", "seed", "val", "test"])
    raw = sel[(sel.ds == "cmmd") & (sel.model == "clip") & (sel["var"] == "plain")]
    d = d[d.model != "clip"].copy()          # CLIP: the raw-embedding runs (selected on validation)
    summ = pd.read_csv(os.path.join(C, "crisismmd_summary.csv"))
    clip_acc = {r.task: r.acc for r in summ[summ.model == "clip"].itertuples()}   # first round = raw CLIP
    add = [{"task": r.task, "run": f"clip_{r.seed}", "acc": np.nan, "wf1": r.test, "model": "clip",
            "seed": int(r.seed[1:])} for r in raw.itertuples()]
    d = pd.concat([d, pd.DataFrame(add)], ignore_index=True)
    return d, clip_acc


def cmmd_tables(a):
    d, clip_acc = cmmd_frame(a)
    rows = []
    for m, name in CMMD_MODELS:
        cells = []
        for t in ("informative", "humanitarian", "damage"):
            g = d[(d.model == m) & (d.task == t)].sort_values("seed")
            cells += [" / ".join(f"{x:.2f}" for x in g.wf1)]
            accs = g.acc.dropna()
            cells += [f"{accs.mean():.2f}" if len(accs) else f"{clip_acc[t]:.2f}"]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    head = ("& \\multicolumn{2}{c}{Informativeness} & \\multicolumn{2}{c}{Humanitarian} & \\multicolumn{2}{c}{Damage} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Model & W-F1, seeds 42 / 43 / 44 & Acc (mean) & W-F1, seeds 42 / 43 / 44 & Acc (mean) & "
            "W-F1, seeds 42 / 43 / 44 & Acc (mean) \\\\")
    cap = ("CrisisMMD: weighted F1 of every seed and mean accuracy (\\%). All models trained by us on the agreed"
           " benchmark splits, checkpoints selected on dev weighted F1; CLIP with raw, ALIGN with L2-normalised"
           " embeddings (Table~\\ref{tab:s_clip_align}).")
    t1 = table(cap, "tab:s_cmmd_seeds", "lcccccc", head, rows)

    rows = []
    for m, name in CMMD_MODELS:
        if m == "ccecg":
            continue
        cells = []
        for t in ("informative", "humanitarian", "damage"):
            c = d[(d.model == "ccecg") & (d.task == t)].sort_values("seed").wf1.values
            o = d[(d.model == m) & (d.task == t)].sort_values("seed").wf1.values
            p = stats.ttest_rel(c, o).pvalue
            cells.append(f"{np.mean(c - o):+.2f} & {int((c > o).sum())}/3 & {fp(p)}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    head = ("& \\multicolumn{3}{c}{Informativeness} & \\multicolumn{3}{c}{Humanitarian} & \\multicolumn{3}{c}{Damage} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "CCEC-G vs. & $\\Delta$ & ahead & $p$ & $\\Delta$ & ahead & $p$ & $\\Delta$ & ahead & $p$ \\\\")
    cap = ("CrisisMMD: CCEC-G minus every other model in weighted F1 (points), number of seeds in which CCEC-G is"
           " ahead, and paired $t$-test over the three seeds.")
    t2 = table(cap, "tab:s_cmmd_sig", "lccccccccc", head, rows)
    write(os.path.join(a.out, "cmmd.tex"), t1 + "\n" + t2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results_data")
    ap.add_argument("--out", default="supplementary/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    diag_tables(a)
    vlm_tables(a)
    cost_table(a)
    clip_align_table(a)
    cmmd_tables(a)


if __name__ == "__main__":
    main()
