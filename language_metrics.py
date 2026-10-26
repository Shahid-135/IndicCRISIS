"""
language_metrics.py - full metric set per language (English / Hindi / All) for every model,
computed from the per-post predictions (predictions.csv) of each run.

Metrics per task (combined labels), then averaged over the five tasks:
    accuracy, macro precision / recall / F1, weighted precision / recall / F1, Cohen's kappa, MCC
Fine-tuned models: mean ± std over seeds; zero-shot VLM: test split of seed 42.

    python language_metrics.py --runs runs_v2 --vlm pilot_test --out language_metrics
Outputs: language_metrics/metrics_long.csv (per task), metrics_avg.csv (averaged over tasks),
         language_metrics_tables.tex (one table per language)
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
NAMES = {"mbert": "mBERT", "indicbert": "IndicBERT", "muril": "MuRIL", "xlmr": "XLM-R",
         "vgg19": "VGG-19", "resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext": "ConvNeXt-B",
         "fusion_baseline": "Fusion", "ccec": "CCEC", "fusion_full": "Fusion + corr. learning",
         "full": "CCEC-G", "fusion_full_vlm": "Fusion + corr. + VLM view", "full_vlm": "CCEC-G + VLM view",
         "full_vlm_comp": "CCEC-G + VLM view (trained comp.)",
         "vlm_direct": "VLM zero-shot (direct)", "vlm_cot": "VLM zero-shot (CoT)",
         "sota_mann": "MANN", "sota_camm": "CAMM", "sota_robertamft": "RoBERTaMFT", "sota_dmcc": "DMCC",
         "sota_clip": "CLIP (linear)", "sota_align": "ALIGN (linear)"}
GROUPS = [("Zero-shot VLM", ["vlm_direct", "vlm_cot"]),
          ("Text-only (fine-tuned)", ["mbert", "indicbert", "muril", "xlmr"]),
          ("Image-only (fine-tuned)", ["vgg19", "resnet50", "vit_b16", "convnext"]),
          ("Multimodal crisis models (re-implemented)", ["sota_mann", "sota_camm", "sota_robertamft", "sota_dmcc", "sota_clip", "sota_align"]),
          ("Multimodal (fine-tuned)", ["fusion_baseline", "ccec", "fusion_full", "full",
                                       "fusion_full_vlm", "full_vlm", "full_vlm_comp"])]
METRICS = [("accuracy", "Acc"), ("macro_precision", "M-P"), ("macro_recall", "M-R"), ("macro_f1", "M-F1"),
           ("weighted_precision", "W-P"), ("weighted_recall", "W-R"), ("weighted_f1", "W-F1"),
           ("kappa", r"$\kappa$"), ("mcc", "MCC")]
LANGS = [("en", "English"), ("hi", "Hindi"), ("all", "All (English + Hindi)")]


def metrics(y, p, labels):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
            "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf,
            "kappa": cohen_kappa_score(y, p), "mcc": matthews_corrcoef(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--vlm", default=None, help="pilot output folder with direct/ and cot/ predictions")
    ap.add_argument("--out", default="language_metrics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    preds = []                                  # (model, seed, frame with lang, true/pred per task)
    for p in sorted(glob.glob(os.path.join(a.runs, "*_s*", "report", "predictions.csv"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        m = re.match(r"(.+)_s(\d+)$", run)
        if not m:
            continue
        name = m.group(1)[4:] if m.group(1).startswith(("txt_", "img_")) else m.group(1)   # sota_* keeps its prefix
        if name not in NAMES:
            continue
        d = pd.read_csv(p, encoding="utf-8-sig")
        f = pd.DataFrame({"lang": d["language"]})
        for t in TASKS:
            f[f"{t}_true"], f[f"{t}_pred"] = d[f"{t}_Combined_true"], d[f"{t}_Combined_pred"]
        preds.append((name, int(m.group(2)), f))
    if a.vlm:
        for mode in ("direct", "cot"):
            p = os.path.join(a.vlm, mode, "predictions.csv")
            if os.path.exists(p):
                d = pd.read_csv(p, encoding="utf-8-sig")
                f = pd.DataFrame({"lang": d["lang"]})
                for t in TASKS:
                    f[f"{t}_true"], f[f"{t}_pred"] = d[f"{t}_Combined"], d[f"{t}_combined"]
                preds.append((f"vlm_{mode}", 42, f))
            else:
                print(f"note: {p} not found - skipping VLM {mode}")
    if not preds:
        raise SystemExit(f"no predictions found under {a.runs}/*/report/")

    # label space per task = every gold label seen in any run (same for all models)
    labels = {t: sorted(set().union(*[set(f[f"{t}_true"]) for _, _, f in preds])) for t in TASKS}
    rows = []
    for name, seed, f in preds:
        for lg, _ in LANGS:
            g = f if lg == "all" else f[f["lang"] == lg]
            for t in TASKS:
                rows.append({"model": name, "seed": seed, "lang": lg, "task": t,
                             **metrics(g[f"{t}_true"], g[f"{t}_pred"], labels[t])})
    long = pd.DataFrame(rows)
    mcols = [m for m, _ in METRICS]
    per_task = long.groupby(["model", "lang", "task"])[mcols].agg(["mean", "std"]) * 100
    per_task.round(2).to_csv(os.path.join(a.out, "metrics_long.csv"))
    avg = long.groupby(["model", "seed", "lang"])[mcols].mean().reset_index()      # average over tasks
    M = avg.groupby(["model", "lang"])[mcols].mean() * 100
    S = avg.groupby(["model", "lang"])[mcols].std().fillna(0) * 100
    out = M.round(2).astype(str) + " ± " + S.round(2).astype(str)
    out.insert(0, "n_seeds", avg.groupby(["model", "lang"]).size())
    out.to_csv(os.path.join(a.out, "metrics_avg.csv"))

    present = set(long["model"])
    order = [m for _, ms in GROUPS for m in ms if m in present]
    pd.set_option("display.width", 250)
    for lg, title in LANGS:
        t = M.xs(lg, level="lang").reindex(order)[mcols].round(2)
        t.index = [NAMES[m] for m in order]
        t.columns = [h.replace("$\\kappa$", "kappa") for _, h in METRICS]
        print(f"\n=== {title} (averaged over the five tasks) ===\n{t.to_string()}")

    L = []
    for lg, title in LANGS:
        sub = M.xs(lg, level="lang")
        best = {c: sub.loc[[m for m in order if m in sub.index], c].max() for c in mcols}
        L += [r"\begin{table*}[t]", r"\centering",
              rf"\caption{{Results on the {title} test posts (\%, averaged over the five tasks on the combined"
              r" labels; fine-tuned models: mean over three seeds). M-: macro-averaged; W-: weighted;"
              r" $\kappa$: Cohen's kappa; MCC: Matthews correlation coefficient. Bold: best per column.}",
              rf"\label{{tab:metrics_{lg}}}", r"\small", r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
              "Model & " + " & ".join(h for _, h in METRICS) + r" \\"]
        for gtitle, ms in GROUPS:
            ms = [m for m in ms if m in present]
            if not ms:
                continue
            L += [r"\midrule", rf"\multicolumn{{{len(METRICS) + 1}}}{{l}}{{\textit{{{gtitle}}}}} \\"]
            for m in ms:
                cells = []
                for c in mcols:
                    v, s = M[c].get((m, lg), np.nan), S[c].get((m, lg), 0)
                    txt = "--" if pd.isna(v) else (f"{v:.2f}" if m.startswith("vlm") else f"{v:.2f}$\\pm${s:.2f}")
                    cells.append(r"\textbf{" + txt + "}" if not pd.isna(v) and np.isclose(v, best[c]) else txt)
                L.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    open(os.path.join(a.out, "language_metrics_tables.tex"), "w").write("\n".join(L))
    print(f"\nwritten to {a.out}/ (metrics_avg.csv, metrics_long.csv, language_metrics_tables.tex)")


if __name__ == "__main__":
    main()
