"""
old_style_tables.py - the new results in the layout of the earlier result sheets:
one sheet per model, blocks for English / Hindi / All, one row per task, metric columns
(mean ± std over seeds). Computed from the per-post predictions of every run (combined labels).

    python old_style_tables.py --runs runs_v2 --vlm pilot_test --out Results_new_format.xlsx
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TASKS = ["Crisis", "Humanitarian", "Informativeness", "Resource", "Severity"]      # order of the old sheets
NAMES = {"mbert": "mBERT", "indicbert": "IndicBERT", "muril": "MuRIL", "xlmr": "XLM-R",
         "vgg19": "VGG-19", "resnet50": "ResNet-50", "vit_b16": "ViT-B16", "convnext": "ConvNeXt-B",
         "fusion_baseline": "Fusion", "ccec": "CCEC", "fusion_full": "Fusion+corr",
         "full": "CCEC-G", "fusion_full_vlm": "Fusion+corr+VLM", "full_vlm": "CCEC-G+VLM",
         "full_vlm_comp": "CCEC-G+VLM (comp)", "vlm_direct": "VLM direct", "vlm_cot": "VLM CoT",
         "sota_mann": "MANN", "sota_camm": "CAMM", "sota_robertamft": "RoBERTaMFT", "sota_dmcc": "DMCC",
         "sota_clip": "CLIP (linear)", "sota_align": "ALIGN (linear)"}
COLS = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
        "macro_precision", "macro_recall", "macro_f1"]
LANGS = [("en", "English"), ("hi", "Hindi"), ("all", "All (English + Hindi)")]


def scores(y, p, labels):
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    return dict(zip(COLS, [accuracy_score(y, p), wp, wr, wf, mp, mr, mf]))


def load(a):
    runs = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*_s*", "report", "predictions.csv"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(p))))
        if not m:
            continue
        name = m.group(1)[4:] if m.group(1).startswith(("txt_", "img_")) else m.group(1)   # sota_* keeps its prefix
        if name in NAMES:
            d = pd.read_csv(p, encoding="utf-8-sig")
            f = pd.DataFrame({"lang": d["language"]})
            for t in TASKS:
                f[f"{t}_y"], f[f"{t}_p"] = d[f"{t}_Combined_true"], d[f"{t}_Combined_pred"]
            runs.append((name, f))
    for mode in ("direct", "cot"):
        p = os.path.join(a.vlm or "", mode, "predictions.csv")
        if a.vlm and os.path.exists(p):
            d = pd.read_csv(p, encoding="utf-8-sig")
            f = pd.DataFrame({"lang": d["lang"]})
            for t in TASKS:
                f[f"{t}_y"], f[f"{t}_p"] = d[f"{t}_Combined"], d[f"{t}_combined"]
            runs.append((f"vlm_{mode}", f))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--out", default="Results_new_format.xlsx")
    a = ap.parse_args()
    runs = load(a)
    if not runs:
        raise SystemExit(f"no predictions found under {a.runs}/*/report/")
    labels = {t: sorted(set().union(*[set(f[f"{t}_y"]) for _, f in runs])) for t in TASKS}
    rows = []
    for name, f in runs:
        for lg, _ in LANGS:
            g = f if lg == "all" else f[f["lang"] == lg]
            for t in TASKS:
                rows.append({"model": name, "lang": lg, "task": t, **scores(g[f"{t}_y"], g[f"{t}_p"], labels[t])})
    d = pd.DataFrame(rows)
    M = d.groupby(["model", "lang", "task"])[COLS].mean()
    S = d.groupby(["model", "lang", "task"])[COLS].std()
    N = d.groupby(["model", "lang", "task"]).size()
    order = [m for m in NAMES if m in set(d["model"])]

    with pd.ExcelWriter(a.out, engine="openpyxl") as xw:
        for m in order:
            blocks, r0 = [], 0
            sheet = NAMES[m][:31]
            for lg, title in LANGS:
                n = int(N.get((m, lg, TASKS[0]), 0))
                head = pd.DataFrame([[f"{title}  (n_seeds = {n})"] + [""] * len(COLS)])
                tab = pd.DataFrame([[f"{t}_Combined"] + [
                    (f"{M.loc[(m, lg, t), c]:.4f}" if n < 2 else
                     f"{M.loc[(m, lg, t), c]:.4f} ± {S.loc[(m, lg, t), c]:.4f}") for c in COLS] for t in TASKS],
                    columns=["Task"] + COLS)
                avg = ["Average (5 tasks)"] + [f"{M.loc[(m, lg)][c].mean():.4f}" for c in COLS]
                head.to_excel(xw, sheet_name=sheet, startrow=r0, index=False, header=False)
                tab.to_excel(xw, sheet_name=sheet, startrow=r0 + 1, index=False)
                pd.DataFrame([avg]).to_excel(xw, sheet_name=sheet, startrow=r0 + 2 + len(TASKS),
                                             index=False, header=False)
                r0 += len(TASKS) + 5
        # one overview sheet: macro-F1 average per model and language
        ov = d.groupby(["model", "lang"])["macro_f1"].mean().unstack("lang").reindex(order)[["en", "hi", "all"]]
        ov.index = [NAMES[m] for m in order]
        ov.columns = ["English", "Hindi", "All"]
        (ov * 100).round(2).to_excel(xw, sheet_name="Overview (macro-F1)")
    print(f"written {a.out}: one sheet per model ({len(order)}) + overview")


if __name__ == "__main__":
    main()
