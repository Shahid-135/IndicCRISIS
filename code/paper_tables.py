"""
paper_tables.py - build the main results table, the language table and the per-language detail tables
for the paper directly from the runs (exact mean ± std over seeds, also for task averages).

    python paper_tables.py --runs runs_xlmr --baseline_runs runs_v2 --vlm pilot_test \
        --siglip_runs runs_v2 --out paper_tables

    --runs           multimodal models (fusion_baseline, ccec, fusion_full, full, fusion_full_vlm, full_vlm)
    --baseline_runs  text-only / image-only baselines (txt_*, img_*)
    --siglip_runs    optional: the same multimodal models with the SigLIP-2 text encoder, shown as an
                     extra group (text-encoder comparison)
    --vlm            optional: zero-shot VLM predictions (direct/, cot/)
Outputs (in --out): main_results.tex, language_table.tex, detail_en.tex, detail_hi.tex, detail_all.tex,
                    all_metrics.csv
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
MULTI = [("fusion_baseline", "Fusion"), ("ccec", "CCEC"), ("fusion_full", "Fusion + corr. learning"),
         ("full", "CCEC-G"), ("fusion_full_vlm", "Fusion + corr. learning + VLM view"),
         ("full_vlm", "CCEC-G + VLM view")]
TEXT = [("mbert", "mBERT"), ("indicbert", "IndicBERT"), ("muril", "MuRIL"), ("xlmr", "XLM-R")]
IMAGE = [("vgg19", "VGG-19"), ("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16"), ("convnext", "ConvNeXt-B")]
SOTA = [("mann", "MANN~\\cite{gao2020mann}"), ("camm", "CAMM~\\cite{khattar2022camm}"),
        ("robertamft", "RoBERTaMFT~\\cite{yoon2023robertamft}"), ("dmcc", "DMCC~\\cite{rezk2023dmcc}"),
        ("clip", "CLIP~\\cite{mandal2024contrastive}"), ("align", "ALIGN~\\cite{mandal2024contrastive}")]
LANGS = [("en", "English"), ("hi", "Hindi"), ("all", "All")]


def task_metrics(y, p, labels):
    _, _, wf, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    return {"acc": accuracy_score(y, p), "wf1": wf, "mp": mp, "mr": mr, "mf1": mf}


def frame_from_run(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    f = pd.DataFrame({"lang": d["language"]})
    for t in TASKS:
        f[f"{t}_y"], f[f"{t}_p"] = d[f"{t}_Combined_true"], d[f"{t}_Combined_pred"]
    return f


def frame_from_vlm(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    f = pd.DataFrame({"lang": d["lang"]})
    for t in TASKS:
        f[f"{t}_y"], f[f"{t}_p"] = d[f"{t}_Combined"], d[f"{t}_combined"]
    return f


def collect(a):
    """[(group, key, display, seed, frame, cmc)]"""
    out = []

    def runs_of(root, prefix, key):
        res = []
        for p in sorted(glob.glob(os.path.join(root, f"{prefix}{key}_s*", "report", "predictions.csv"))):
            m = re.search(r"_s(\d+)$", os.path.dirname(os.path.dirname(p)))
            rj = os.path.join(os.path.dirname(os.path.dirname(p)), "results.json")
            cmc = json.load(open(rj))["test"].get("cmc", {}).get("macro_f1") if os.path.exists(rj) else None
            res.append((int(m.group(1)), frame_from_run(p), cmc))
        return res

    if a.vlm:
        for mode, disp in (("direct", "VLM, direct answer"), ("cot", "VLM, chain-of-thought"),
                          ("fewshot", "VLM, few-shot ($k=4$)"), ("finetuned", "VLM, fine-tuned (LoRA)")):
            p = os.path.join(a.vlm, mode, "predictions.csv")
            if os.path.exists(p):
                cmc = None
                mj = os.path.join(a.vlm, mode, "metrics.json")
                if os.path.exists(mj):
                    cmc = json.load(open(mj)).get("cmc.macro_f1")
                out.append(("Zero-shot vision--language model (Qwen2.5-VL-7B)", f"vlm_{mode}", disp, 42,
                            frame_from_vlm(p), cmc))
    for grp, prefix, items in (("Text-only, fine-tuned", "txt_", TEXT), ("Image-only, fine-tuned", "img_", IMAGE),
                               ("Multimodal crisis models (re-implemented)", "sota_", SOTA)):
        for key, disp in items:
            for seed, f, _ in runs_of(a.baseline_runs, prefix, key):
                out.append((grp, prefix + key, disp, seed, f, None))
    if a.siglip_runs:
        for key, disp in MULTI:
            for seed, f, cmc in runs_of(a.siglip_runs, "", key):
                out.append(("Multimodal, fine-tuned (SigLIP-2 text encoder)", f"sig_{key}", disp, seed, f, cmc))
    tname = {"xlmr": "XLM-R", "muril": "MuRIL", "mbert": "mBERT"}.get(a.text_encoder, a.text_encoder)
    for key, disp in MULTI:
        for seed, f, cmc in runs_of(a.runs, "", key):
            out.append((f"Multimodal, fine-tuned ({tname} text encoder)", key, disp, seed, f, cmc))
    return out


def fmt(v, s, bold, single):
    if pd.isna(v):
        return "--"
    txt = f"{v:.2f}" if (single or pd.isna(s)) else f"{v:.2f}$\\pm${s:.2f}"
    return r"\textbf{" + txt + "}" if bold else txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="multimodal runs (e.g. runs_xlmr)")
    ap.add_argument("--baseline_runs", default="runs_v2")
    ap.add_argument("--siglip_runs", default=None)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--text_encoder", default="xlmr", help="text encoder of --runs (for the group title)")
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = collect(a)
    if not runs:
        raise SystemExit("no runs found")
    labels = {t: sorted(set().union(*[set(f[f"{t}_y"]) for *_, f, _ in runs])) for t in TASKS}
    rows = []
    for grp, key, disp, seed, f, cmc in runs:
        for lg, _ in LANGS:
            g = f if lg == "all" else f[f["lang"] == lg]
            per = {t: task_metrics(g[f"{t}_y"], g[f"{t}_p"], labels[t]) for t in TASKS}
            r = {"group": grp, "key": key, "model": disp, "seed": seed, "lang": lg,
                 "cmc": (100 * cmc if cmc is not None and lg == "all" else np.nan)}
            for t in TASKS:
                r[t] = 100 * per[t]["mf1"]
            for k in ("acc", "wf1", "mp", "mr", "mf1"):
                r[k] = 100 * np.mean([per[t][k] for t in TASKS])          # task average for this seed
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "all_metrics.csv"), index=False)
    cols = TASKS + ["acc", "wf1", "mp", "mr", "mf1", "cmc"]
    M = D.groupby(["key", "lang"])[cols].mean()
    S = D.groupby(["key", "lang"])[cols].std()
    N = D.groupby(["key", "lang"]).size()
    order, seen = [], set()
    for grp, key, disp, *_ in runs:
        if key not in seen:
            seen.add(key)
            order.append((grp, key, disp))
    groups = []
    for grp, key, disp in order:
        if not groups or groups[-1][0] != grp:
            groups.append((grp, []))
        groups[-1][1].append((key, disp))
    single = {k: N.get((k, "all"), 1) < 2 for _, k, _ in order}
    keys = [k for _, k, _ in order]

    def table(lg, columns, header, caption, label, ncol_fmt):
        best = {c: M.xs(lg, level="lang").loc[[k for k in keys if (k, lg) in M.index], c].max() for c, _ in columns}
        L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{ncol_fmt}}}",
             r"\toprule", header + r" \\"]
        for grp, items in groups:
            L += [r"\midrule", rf"\multicolumn{{{len(columns) + 1}}}{{l}}{{\textit{{{grp}}}}} \\"]
            for k, disp in items:
                cells = []
                for c, use_sd in columns:
                    v, s = M[c].get((k, lg), np.nan), S[c].get((k, lg), np.nan)
                    cells.append(fmt(v, s if use_sd else np.nan, not pd.isna(v) and np.isclose(v, best[c]), single[k]))
                L.append(disp + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
        return "\n".join(L) + "\n"

    # main table (all test posts)
    main_cols = [(t, True) for t in TASKS] + [("mf1", True), ("acc", False), ("wf1", False), ("cmc", True)]
    open(os.path.join(a.out, "main_results.tex"), "w").write(table(
        "all", main_cols, "Model & Crisis & Sev. & Inf. & Hum. & Res. & Avg.$_5$ & Acc & W-F1 & CMC",
        r"Results on the \dataset{} test set (all posts). Crisis--Res.: macro-F1 per task on the combined labels;"
        r" Avg.$_5$: their mean; Acc and W-F1: accuracy and weighted F1 averaged over the five tasks; CMC: macro-F1"
        r" of cross-modal correlation prediction (models that predict it). Fine-tuned models: mean $\pm$ standard"
        r" deviation over three seeds (Acc, W-F1: mean); zero-shot VLM: single deterministic run on the test split"
        r" of seed 42. All models use the same bilingual training split. Bold: best per column.",
        "tab:main_results", "l ccccc c cc c"))

    # language table
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Results by language of the test post (macro-F1 Avg.$_5$, accuracy and weighted F1, averaged"
         r" over the five tasks on the combined labels). All models are trained on the bilingual training split;"
         r" only the evaluation is split by language. Fine-tuned models: mean over three seeds ($\pm$ standard"
         r" deviation for Avg.$_5$). Bold: best per column.}",
         r"\label{tab:language}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
         r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{English} & \multicolumn{3}{c}{Hindi} & \multicolumn{3}{c}{All} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r" & Avg.$_5$ & Acc & W-F1 & Avg.$_5$ & Acc & W-F1 & Avg.$_5$ & Acc & W-F1 \\"]
    met = [("mf1", True), ("acc", False), ("wf1", False)]
    best = {(lg, c): M.xs(lg, level="lang")[c].max() for lg, _ in LANGS for c, _ in met}
    for grp, items in groups:
        L += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\textit{{{grp}}}}} \\"]
        for k, disp in items:
            cells = [fmt(M[c].get((k, lg)), S[c].get((k, lg)) if sd else np.nan,
                         np.isclose(M[c].get((k, lg)), best[(lg, c)]), single[k]) for lg, _ in LANGS for c, sd in met]
            L.append(disp + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.out, "language_table.tex"), "w").write("\n".join(L) + "\n")

    # detail tables per language
    det_cols = [(t, True) for t in TASKS] + [("mf1", True), ("mp", False), ("mr", False), ("acc", False), ("wf1", False)]
    for lg, title in LANGS:
        where = "complete test set" if lg == "all" else f"{title} test posts"
        open(os.path.join(a.out, f"detail_{lg}.tex"), "w").write(table(
            lg, det_cols, "Model & Crisis & Sev. & Inf. & Hum. & Res. & Avg.$_5$ & M-P & M-R & Acc & W-F1",
            rf"Detailed results on the {where} (combined labels). Crisis--Res.: macro-F1 per task; Avg.$_5$: their"
            r" mean (mean $\pm$ standard deviation over three seeds); M-P, M-R, Acc, W-F1: macro precision, macro"
            r" recall, accuracy and weighted F1, averaged over the five tasks (mean over seeds). Bold: best per column.",
            f"tab:detail_{lg}", "l ccccc c cccc"))

    pd.set_option("display.width", 220)
    summ = M["mf1"].unstack("lang").reindex(columns=["en", "hi", "all"]).loc[keys].round(2)
    summ.insert(0, "seeds", [N.get((k, "all"), 0) for k in keys])
    summ.index = [f"{d}  [{k}]" for _, k, d in order]
    print("Avg_5 macro-F1 by language (mean over seeds):\n" + summ.to_string())
    print(f"\nwritten to {a.out}/")


if __name__ == "__main__":
    main()
