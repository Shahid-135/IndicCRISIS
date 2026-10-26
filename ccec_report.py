"""
ccec_report.py - full evaluation report for CCEC (imported by train_ccec.py).

Writes into <save_dir>/report/:
    metrics_summary.csv        every field (CMC + 5 tasks x text/image/combined): Acc, macro/weighted
                               P/R/F1, Cohen's kappa, MCC, ROC-AUC, PR-AUC
    per_class_metrics.csv      precision / recall / F1 / support for every class of every field
    classification_reports.txt sklearn classification report for every field
    per_language.csv           main metrics split by English / Hindi
    per_cmc.csv                combined-label metrics split by CMC level
    confusion_matrices/        <field>.csv (counts), <field>_norm.csv (row-normalised), <field>.png
    predictions.csv            one row per test sample: gold, prediction and confidence per field
    efficiency.json            parameters, training time, peak GPU memory, inference time
    results.xlsx               all of the above in one workbook
    main_table.tex             LaTeX table of the combined-label results (booktabs)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
VIEWS = ["Text", "Image", "Combined"]
MAIN = ["cmc"] + [f"{t}_Combined" for t in TASKS]


def field_classes(key, spaces):
    return spaces["CMC"] if key == "cmc" else spaces[key.rsplit("_", 1)[0]]


def field_name(key):
    return "CMC" if key == "cmc" else key


def _auc_scores(y, prob, K):
    """Macro one-vs-rest ROC-AUC and PR-AUC over the classes present in y."""
    present = [c for c in range(K) if 0 < (y == c).sum() < len(y)]
    if not present:
        return np.nan, np.nan
    if K == 2:
        return roc_auc_score(y, prob[:, 1]), average_precision_score(y == 1, prob[:, 1])
    onehot = np.eye(K)[y]
    roc = np.mean([roc_auc_score(onehot[:, c], prob[:, c]) for c in present])
    pr = np.mean([average_precision_score(onehot[:, c], prob[:, c]) for c in present])
    return roc, pr


def evaluate_field(y, p, prob, classes):
    K = len(classes)
    labels = list(range(K))
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    roc, pr = _auc_scores(y, prob, K) if prob is not None else (np.nan, np.nan)
    summary = {"n": len(y), "accuracy": accuracy_score(y, p),
               "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
               "weighted_precision": wp, "weighted_recall": wr, "weighted_f1": wf,
               "kappa": cohen_kappa_score(y, p), "mcc": matthews_corrcoef(y, p),
               "roc_auc": roc, "pr_auc": pr}
    P, R, F, S = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": P, "recall": R, "f1": F, "support": S})
    cm = confusion_matrix(y, p, labels=labels)
    text = classification_report(y, p, labels=labels, target_names=classes, digits=4, zero_division=0)
    return summary, per_class, cm, text


def _plot_cm(cm, classes, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(4.5, 0.9 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _latex_table(summary_df, path, main=MAIN):
    cols = [("accuracy", "Acc"), ("macro_precision", "M-P"), ("macro_recall", "M-R"),
            ("macro_f1", "M-F1"), ("weighted_f1", "W-F1"), ("kappa", r"$\kappa$"),
            ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    rows = summary_df.set_index("field").loc[[field_name(k) for k in main]]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{CCEC test results on the combined labels (\%). $\kappa$: Cohen's kappa.}",
             r"\label{tab:ccec_results}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Task & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for name, r in rows.iterrows():
        label = name.replace("_Combined", "")
        lines.append(label + " & " + " & ".join(
            "--" if pd.isna(r[c]) else f"{100 * r[c]:.2f}" for c, _ in cols) + r" \\")
    avg = rows[[c for c, _ in cols]].mean()
    lines += [r"\midrule", r"\textbf{Average} & " + " & ".join(
        f"\\textbf{{{100 * avg[c]:.2f}}}" for c, _ in cols) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_full_report(out_dir, preds, gold, probs, names, spaces, efficiency, extra=None, report_cmc=True):
    """report_cmc=False: the data has no CMC labels (placeholder levels) - CMC is left out of every table."""
    rep = os.path.join(out_dir, "report")
    cm_dir = os.path.join(rep, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    TASKS = [k[:-len("_Combined")] for k in preds if k.endswith("_Combined")]     # tasks of this run
    MAIN = (["cmc"] if report_cmc else []) + [f"{t}_Combined" for t in TASKS]
    keys = MAIN + [f"{t}_{v}" for t in TASKS for v in ("Text", "Image")]
    keys = [k for k in keys if k in preds]

    summaries, per_class_all, texts, cms = [], [], [], {}
    for k in keys:
        classes = field_classes(k, spaces)
        s, pc, cm, txt = evaluate_field(gold[k], preds[k], probs.get(k), classes)
        summaries.append({"field": field_name(k), **s})
        per_class_all.append(pc.assign(field=field_name(k)))
        texts.append(f"{'=' * 70}\n{field_name(k)}\n{'=' * 70}\n{txt}\n")
        cms[k] = cm
        base = os.path.join(cm_dir, field_name(k))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(base + ".csv")
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1), index=classes,
                     columns=classes).round(4).to_csv(base + "_norm.csv")
        _plot_cm(cm, classes, field_name(k), base + ".png")

    summary_df = pd.DataFrame(summaries)
    main_mask = summary_df["field"].isin([field_name(k) for k in MAIN])
    avg = summary_df[main_mask].drop(columns=["field", "n"]).mean()
    summary_df = pd.concat([summary_df, pd.DataFrame([{"field": "AVERAGE (CMC + combined)" if report_cmc
                                                        else "AVERAGE (combined)", **avg}])],
                           ignore_index=True)
    per_class_df = pd.concat(per_class_all, ignore_index=True)[["field", "class", "precision",
                                                                "recall", "f1", "support"]]
    summary_df.to_csv(os.path.join(rep, "metrics_summary.csv"), index=False)
    per_class_df.to_csv(os.path.join(rep, "per_class_metrics.csv"), index=False)
    with open(os.path.join(rep, "classification_reports.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(texts))

    # ── per language ──
    lang_rows = []
    for lang in np.unique(gold["_lang"]):
        m = gold["_lang"] == lang
        for k in MAIN:
            s, *_ = evaluate_field(gold[k][m], preds[k][m], probs[k][m], field_classes(k, spaces))
            lang_rows.append({"language": lang, "field": field_name(k), **s})
    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(os.path.join(rep, "per_language.csv"), index=False)

    # ── per CMC level (combined labels) ──
    cmc_rows = []
    for zi, zname in enumerate(spaces["CMC"] if report_cmc else []):
        m = gold["cmc"] == zi
        if m.sum() == 0:
            continue
        for t in TASKS:
            k = f"{t}_Combined"
            s, *_ = evaluate_field(gold[k][m], preds[k][m], probs[k][m], spaces[t])
            cmc_rows.append({"CMC": zname, "field": k, **s})
    cmc_df = pd.DataFrame(cmc_rows)
    cmc_df.to_csv(os.path.join(rep, "per_cmc.csv"), index=False)

    # ── per-sample predictions ──
    pred_df = pd.DataFrame({"image_name": names, "language": gold["_lang"]})
    for k in keys:
        classes = field_classes(k, spaces)
        pred_df[f"{field_name(k)}_true"] = [classes[i] for i in gold[k]]
        pred_df[f"{field_name(k)}_pred"] = [classes[i] for i in preds[k]]
        pred_df[f"{field_name(k)}_conf"] = probs[k].max(1).round(4)
        pred_df[f"{field_name(k)}_correct"] = gold[k] == preds[k]
    pred_df.to_csv(os.path.join(rep, "predictions.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(rep, "efficiency.json"), "w") as f:
        json.dump(efficiency, f, indent=2, default=float)
    _latex_table(summary_df, os.path.join(rep, "main_table.tex"), MAIN)

    # ── one workbook ──
    try:
        with pd.ExcelWriter(os.path.join(rep, "results.xlsx"), engine="openpyxl") as xw:
            summary_df.round(4).to_excel(xw, sheet_name="Summary", index=False)
            per_class_df.round(4).to_excel(xw, sheet_name="Per_class", index=False)
            lang_df.round(4).to_excel(xw, sheet_name="Per_language", index=False)
            cmc_df.round(4).to_excel(xw, sheet_name="Per_CMC", index=False)
            pd.DataFrame(list(efficiency.items()), columns=["metric", "value"]).to_excel(
                xw, sheet_name="Efficiency", index=False)
            if extra:
                pd.json_normalize(extra, sep=".").T.reset_index().rename(
                    columns={"index": "metric", 0: "value"}).to_excel(xw, sheet_name="Extra", index=False)
            for k in MAIN:
                classes = field_classes(k, spaces)
                pd.DataFrame(cms[k], index=classes, columns=classes).to_excel(
                    xw, sheet_name=("CM_" + field_name(k).replace("_Combined", ""))[:31])
    except ImportError:
        print("openpyxl not installed - skipped results.xlsx (all CSVs were written)")

    # ── console summary ──
    show = summary_df[main_mask | summary_df["field"].str.startswith("AVERAGE")]
    print("\n" + "=" * 96)
    print("TEST RESULTS (combined labels)")
    print("=" * 96)
    print(show[["field", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_f1", "kappa", "roc_auc", "pr_auc"]].round(4).to_string(index=False))
    print("\nPer language (macro-F1):")
    print(lang_df.pivot(index="field", columns="language", values="macro_f1").round(4).to_string())
    print("\nEfficiency:")
    for k, v in efficiency.items():
        print(f"  {k:32s} {v}")
    print(f"\nFull report written to {rep}/")
    return summary_df
