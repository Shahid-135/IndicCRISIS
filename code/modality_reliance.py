"""
modality_reliance.py - how do models use the text and the image? (step-wise diagnostics)

IndicCrisis annotates every task three times (text only, image only, combined) plus the
cross-modal correlation (CMC). This lets us evaluate each reasoning step and, above all,
whether a model lets the IMAGE change its judgement when the annotators did.

Inputs (either or both):
    --runs  runs_v2                   fine-tuned runs: <run>/report/predictions.csv (train_ccec.py)
    --vlm   pilot_test                VLM outputs:     <mode>/predictions.csv (pilot_reasoning_graph.py)

Measures, per model (averaged over the five tasks; fine-tuned variants averaged over seeds):
    text_step_f1 / image_step_f1   macro-F1 of the text-only / image-only prediction vs the text / image labels
    cmc_f1                         macro-F1 of the CMC prediction
    combined_f1                    macro-F1 of the combined prediction
    change_rate                    share of decisions where the model's combined label differs from its
                                   own text-only label (annotators: gold_change_rate)
    acc_image_dependent            combined accuracy on decisions where the ANNOTATORS' combined label differs
                                   from their text label - the cases that need the image
    acc_text_sufficient            combined accuracy on all other decisions
    change_precision               share of the model's own changes that were needed (annotators changed) and correct
    image_use_recall               on image-dependent decisions: share where the model also departed from
                                   its text-only label AND reached the correct combined label
    copy_rate                      share of decisions where combined == text-only (1 - change_rate)
    change_rate_CMC=<level>        change rate per annotated CMC level (should grow with correlation)

Outputs to --out: modality_reliance.csv, modality_reliance_per_task.csv, modality_reliance_table.tex
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
NAMES = {"ccec": "CCEC", "full": "CCEC-G", "fusion_baseline": "Fusion baseline",
         "fusion_full": "Fusion + corr. learning", "corr_cmcw": "CCEC + CMC-weighted contr.",
         "full_vlm": "CCEC-G + VLM view", "fusion_full_vlm": "Fusion + corr. + VLM view",
         "ccec_vlm": "CCEC + VLM view", "full_vlm_comp": "CCEC-G + VLM view (trained comp.)",
         "vlm:direct": "VLM zero-shot (direct)", "vlm:cot": "VLM zero-shot (CoT)",
         "vlm:fewshot": "VLM few-shot", "vlm:finetuned": "VLM fine-tuned (LoRA)",
         "vlm:graph": "VLM zero-shot (graph)"}


def from_run(path):
    """train_ccec.py predictions -> common frame."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame({"image_name": d["image_name"], "lang": d["language"],
                        "gold_cmc": d["CMC_true"], "pred_cmc": d["CMC_pred"]})
    for t in TASKS:
        for v, short in (("Text", "text"), ("Image", "image"), ("Combined", "combined")):
            out[f"gold_{t}_{short}"] = d[f"{t}_{v}_true"]
            out[f"pred_{t}_{short}"] = d[f"{t}_{v}_pred"]
    return out


def from_vlm(path):
    """pilot_reasoning_graph.py predictions -> common frame."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame({"image_name": d["image_name"], "lang": d["lang"],
                        "gold_cmc": d["CMC"], "pred_cmc": d["cmc"]})
    for t in TASKS:
        for v, short, pv in (("Text", "text", "text_only"), ("Image", "image", "image_only"),
                             ("Combined", "combined", "combined")):
            out[f"gold_{t}_{short}"] = d[f"{t}_{v}"]
            out[f"pred_{t}_{short}"] = d[f"{t}_{pv}"]
    return out


def measures(df):
    rows, per_task = {}, []
    cmc_labels = sorted(df["gold_cmc"].unique())
    rows["cmc_f1"] = f1_score(df["gold_cmc"], df["pred_cmc"], labels=cmc_labels, average="macro", zero_division=0)
    for t in TASKS:
        labels = sorted(set(df[f"gold_{t}_text"]) | set(df[f"gold_{t}_image"]) | set(df[f"gold_{t}_combined"]))
        f = lambda g, p: f1_score(df[g], df[p], labels=labels, average="macro", zero_division=0)
        gold_change = df[f"gold_{t}_combined"] != df[f"gold_{t}_text"]
        pred_change = df[f"pred_{t}_combined"] != df[f"pred_{t}_text"]
        correct = df[f"pred_{t}_combined"] == df[f"gold_{t}_combined"]
        r = {"task": t,
             "text_step_f1": f(f"gold_{t}_text", f"pred_{t}_text"),
             "image_step_f1": f(f"gold_{t}_image", f"pred_{t}_image"),
             "combined_f1": f(f"gold_{t}_combined", f"pred_{t}_combined"),
             "gold_change_rate": gold_change.mean(),
             "change_rate": pred_change.mean(),
             "acc_image_dependent": correct[gold_change].mean() if gold_change.any() else np.nan,
             "acc_text_sufficient": correct[~gold_change].mean(),
             "image_use_recall": (pred_change & correct)[gold_change].mean() if gold_change.any() else np.nan,
             "change_precision": (gold_change & correct)[pred_change].mean() if pred_change.any() else np.nan,
             "n_image_dependent": int(gold_change.sum())}
        for z in cmc_labels:
            m = df["gold_cmc"] == z
            r[f"change_rate_CMC={z}"] = pred_change[m].mean()
            r[f"gold_change_rate_CMC={z}"] = gold_change[m].mean()
        per_task.append(r)
    pt = pd.DataFrame(per_task)
    for c in pt.columns:
        if c not in ("task", "n_image_dependent"):
            rows[c] = pt[c].mean()
    rows["n_image_dependent"] = int(pt["n_image_dependent"].sum())
    rows["copy_rate"] = 1 - rows["change_rate"]
    return rows, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="folder with fine-tuned runs (<variant>_s<seed>)")
    ap.add_argument("--variants", nargs="+", default=["ccec", "full", "fusion_baseline", "fusion_full"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="only these seeds of the fine-tuned runs (e.g. 42 = the split the VLM was tested on)")
    ap.add_argument("--vlm", nargs="*", default=[], help="pilot output folder(s)")
    ap.add_argument("--out", default="modality_reliance")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    models = {}                                    # name -> list of frames (one per seed)
    if a.runs:
        for p in sorted(glob.glob(os.path.join(a.runs, "*_s*", "report", "predictions.csv"))):
            m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(p))))
            if m and m.group(1) in a.variants and (a.seeds is None or int(m.group(2)) in a.seeds):
                models.setdefault(m.group(1), []).append(from_run(p))
    if a.runs and not models:
        print(f"warning: no runs of {a.variants} found in {a.runs}/ - check the folder name")
    for folder in a.vlm:
        for mode in ("direct", "cot", "graph", "fewshot", "finetuned"):
            p = os.path.join(folder, mode, "predictions.csv")
            if os.path.exists(p):
                models.setdefault(f"vlm:{mode}", []).append(from_vlm(p))
    if not models:
        raise SystemExit("nothing found - check --runs / --vlm")

    # are VLM and fine-tuned models evaluated on the same posts?
    ref = next((v[0] for k, v in models.items() if not k.startswith("vlm")), None)
    for k, frames in models.items():
        if k.startswith("vlm") and ref is not None:
            ov = len(set(frames[0]["image_name"]) & set(ref["image_name"])) / len(set(frames[0]["image_name"]))
            flag = "" if ov > 0.99 else "   <-- NOT the same posts: compare with care"
            print(f"{NAMES.get(k, k)}: {ov:.0%} of its posts are in the fine-tuned test set{flag}")

    summary, per_task = [], []
    for k, frames in models.items():
        res = [measures(f) for f in frames]
        row = pd.DataFrame([r for r, _ in res]).mean(numeric_only=True).to_dict()
        row.update({"model": NAMES.get(k, k), "n_seeds": len(frames), "n_posts": len(frames[0])})
        summary.append(row)
        pt = pd.concat([p for _, p in res]).groupby("task").mean(numeric_only=True).reset_index()
        per_task.append(pt.assign(model=NAMES.get(k, k)))
    s = pd.DataFrame(summary)
    first = ["model", "n_seeds", "n_posts", "text_step_f1", "image_step_f1", "cmc_f1", "combined_f1",
             "gold_change_rate", "change_rate", "acc_image_dependent", "acc_text_sufficient", "image_use_recall",
             "change_precision"]
    s = s[first + [c for c in s.columns if c not in first]]
    s.to_csv(os.path.join(a.out, "modality_reliance.csv"), index=False)
    pd.concat(per_task).to_csv(os.path.join(a.out, "modality_reliance_per_task.csv"), index=False)

    cols = [("text_step_f1", "Text"), ("image_step_f1", "Image"), ("cmc_f1", "CMC"),
            ("combined_f1", "Comb."), ("change_rate", "Change"), ("acc_image_dependent", "Acc$_{\\text{img-dep}}$"),
            ("acc_text_sufficient", "Acc$_{\\text{text-suff}}$"), ("image_use_recall", "Img-use rec."),
            ("change_precision", "Change prec.")]
    gold = s["gold_change_rate"].mean()
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Step-wise diagnostics (\%). Text / Image / CMC / Comb.: macro-F1 of the text-only, image-only,"
         r" CMC and combined predictions against the corresponding annotations. Change: share of decisions in which"
         rf" the model's combined label differs from its own text-only label (annotators: {100 * gold:.1f}\%)."
         r" Acc$_{\text{img-dep}}$: combined accuracy on decisions where the annotators' combined label differs from"
         r" their text label, i.e. where the image matters; Acc$_{\text{text-suff}}$: on all other decisions."
         r" Change prec.: share of the model's own changes that were needed and correct."
         r" Img-use rec.: share of image-dependent decisions where the model departs from its text judgement"
         r" and reaches the correct combined label.}",
         r"\label{tab:diagnostics}", r"\small", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Model & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for _, r in s.iterrows():
        L.append(r["model"] + " & " + " & ".join(
            "--" if pd.isna(r[c]) else f"{100 * r[c]:.1f}" for c, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.out, "modality_reliance_table.tex"), "w").write("\n".join(L) + "\n")

    pd.set_option("display.width", 250)
    show = s[["model", "n_seeds", "n_posts"] + [c for c, _ in cols]].copy()
    for c, _ in cols:
        show[c] = (100 * show[c]).round(1)
    print("\n" + show.to_string(index=False))
    print(f"\nannotators' change rate (image changes the combined judgement): {100 * gold:.1f}%")
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
