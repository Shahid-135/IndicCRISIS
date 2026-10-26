"""
baseline_table.py - collect the single-modality baselines into one table (mean ± std over seeds)
and verify that every baseline was evaluated on exactly the same test posts as CCEC-G.

    python baseline_table.py --runs runs_v2 --ref full_vlm --out baseline_summary
Outputs: baseline_summary/baselines.csv, baseline_summary/baselines_table.tex
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
NAMES = {"mbert": "mBERT", "indicbert": "IndicBERT", "muril": "MuRIL", "xlmr": "XLM-R",
         "vgg19": "VGG-19", "resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext": "ConvNeXt-B"}
ORDER = ["mbert", "indicbert", "muril", "xlmr", "vgg19", "resnet50", "vit_b16", "convnext"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--ref", default="full_vlm", help="CCEC run whose test posts must match")
    ap.add_argument("--out", default="baseline_summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*_s*", "results.json"))):
        run = os.path.basename(os.path.dirname(p))
        m = re.match(r"(txt|img)_(.+)_s(\d+)$", run)
        if not m:
            continue
        kind, model, seed = m.group(1), m.group(2), int(m.group(3))
        r = json.load(open(p))
        t = r["test"]
        view = r["own_view"]
        row = {"model": model, "modality": "text" if kind == "txt" else "image", "seed": seed,
               "avg5": 100 * t["avg_combined_macro_f1"],
               "own_view_avg": 100 * np.mean([t[f"{x}_{view}"]["macro_f1"] for x in TASKS]),
               **{x: 100 * t[f"{x}_Combined"]["macro_f1"] for x in TASKS}}
        for lg in r.get("test_lang", []):
            row[f"avg5_{lg['language']}"] = 100 * lg["avg_combined_macro_f1"]
        # same test posts as the CCEC run of this seed?
        ref = os.path.join(a.runs, f"{a.ref}_s{seed}", "report", "predictions.csv")
        mine = os.path.join(os.path.dirname(p), "report", "predictions.csv")
        if os.path.exists(ref) and os.path.exists(mine):
            A = set(pd.read_csv(ref, encoding="utf-8-sig")["image_name"])
            B = set(pd.read_csv(mine, encoding="utf-8-sig")["image_name"])
            row["same_test_posts"] = A == B
        else:
            row["same_test_posts"] = None
        rows.append(row)
    if not rows:
        raise SystemExit(f"no baseline runs (txt_*/img_*) found in {a.runs}/")
    d = pd.DataFrame(rows)
    bad = d[d.same_test_posts == False]                                   # noqa: E712
    if len(bad):
        print("WARNING - these runs used different test posts than the CCEC runs:")
        print(bad[["model", "seed"]].to_string(index=False))
    elif d.same_test_posts.isna().any():
        print("note: could not verify the test posts for some runs (reference predictions missing)")
    else:
        print("check passed: all baselines use exactly the same test posts as the CCEC runs")

    cols = TASKS + ["avg5", "own_view_avg"] + [c for c in ["avg5_en", "avg5_hi"] if c in d.columns]
    M = d.groupby("model")[cols].mean()
    S = d.groupby("model")[cols].std().fillna(0)
    N = d.groupby("model").size()
    order = [m for m in ORDER if m in M.index]
    out = M.loc[order].round(2).astype(str) + " ± " + S.loc[order].round(2).astype(str)
    out.insert(0, "n_seeds", N.loc[order])
    out.to_csv(os.path.join(a.out, "baselines.csv"))
    pd.set_option("display.width", 250)
    print(out.to_string())

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Single-modality baselines on the \dataset{} test set (macro-F1, \%, mean $\pm$ std over"
         r" three seeds; same split as Table~\ref{tab:main_results}). Each model is fine-tuned with one encoder"
         r" and two heads per task. Avg.$_5$: combined labels; Own view: the model's own modality labels"
         r" (text labels for text models, image labels for image models).}",
         r"\label{tab:baselines}", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l ccccc c c}", r"\toprule",
         r"Model & Crisis & Sev. & Inf. & Hum. & Res. & Avg.$_5$ & Own view \\", r"\midrule"]
    for mod, title in (("text", "Text-only"), ("image", "Image-only")):
        ms = [m for m in order if d[d.model == m].modality.iloc[0] == mod]
        if not ms:
            continue
        L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
        best = {c: M.loc[ms, c].max() for c in TASKS + ["avg5", "own_view_avg"]}
        for m in ms:
            cells = []
            for c in TASKS + ["avg5", "own_view_avg"]:
                txt = f"{M.loc[m, c]:.2f}$\\pm${S.loc[m, c]:.2f}"
                cells.append(r"\textbf{" + txt + "}" if np.isclose(M.loc[m, c], best[c]) else txt)
            L.append(NAMES.get(m, m) + " & " + " & ".join(cells) + r" \\")
        if mod == "text":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.out, "baselines_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
