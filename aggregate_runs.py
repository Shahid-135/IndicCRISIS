"""
aggregate_runs.py - combine seeds and ablations into mean ± std tables with significance tests.

    python aggregate_runs.py --runs runs --out runs/summary --ref full

Expects run folders <variant>_s<seed> (created by run_ablations.sh). Folders without a seed
suffix (one-off test runs) are skipped. Writes to --out:
    macro_f1_mean_std.csv        variant x task, "mean ± std" (combined labels + CMC + averages)
    significance_vs_<ref>.csv    paired t-test of every variant against the reference, per column
    alignment_mean_std.csv       CMC-alignment metrics (Spearman, text->image R@k, cross-lingual P@10)
    efficiency.csv               parameters / time / memory per variant (mean over seeds)
    all_metrics_long.csv         every metric of every run
    ablation_table.tex           grouped LaTeX table: macro-F1 mean±std, bold = best,
                                 * = significantly different from the reference (p < 0.05)
    alignment_table.tex          LaTeX table of the alignment metrics
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel
except ImportError:                                   # significance is skipped without scipy
    ttest_rel = None

# (folder name, display name, group) - order = order in the tables
VARIANTS = [
    ("full",               "CCEC-G (full)",                              "Proposed"),
    ("full_vlm_comp",      "CCEC-G + VLM view, trained composition",     "Proposed"),
    ("full_vlm",           "CCEC-G + VLM knowledge view",                "Proposed"),
    ("full_fine",          "CCEC-G + token-patch alignment",             "Proposed"),
    ("ccec",               "CCEC (no correlation learning)",             "Correlation learning"),
    ("corr_clip",          "+ standard contrastive",                     "Correlation learning"),
    ("corr_cmcw",          "+ CMC-weighted contrastive",                 "Correlation learning"),
    ("corr_graph",         "+ CMC graph contrastive",                    "Correlation learning"),
    ("full_knn_nbrs",      "CCEC-G, nearest-neighbour batches",          "Correlation learning"),
    ("full_rand_nbrs",     "CCEC-G, random batch partners",              "Correlation learning"),
    ("full_no_regimes",    "CCEC-G w/o CMC regimes",                     "Composition"),
    ("full_no_view_sup",   "CCEC-G w/o view supervision",                "Composition"),
    ("full_no_residual",   "CCEC-G w/o joint-evidence expert",           "Composition"),
    ("fusion_full",        "Fusion + correlation learning",              "Composition"),
    ("fusion_baseline",    "Fusion baseline",                            "Composition"),
    ("fusion_full_vlm",    "Fusion + corr. learning + VLM view",         "Composition"),
    ("ccec_vlm",           "CCEC + VLM knowledge view",                  "Composition"),
    ("full_cmc50",         "CCEC-G, CMC on 50\\% of posts",                "Supervision"),
    ("full_cmc25",         "CCEC-G, CMC on 25\\% of posts",                "Supervision"),
    ("full_no_cmc",        "CCEC-G, no CMC labels",                      "Supervision"),
    ("full_no_cmc_no_view", "CCEC-G, no CMC and no view labels",         "Supervision"),
    # earlier composition ablations (without correlation learning), if present
    ("no_regimes",         "CCEC w/o CMC regimes",                       "Composition (no corr.)"),
    ("no_view_sup",        "CCEC w/o view supervision",                  "Composition (no corr.)"),
    ("no_residual",        "CCEC w/o joint-evidence expert",             "Composition (no corr.)"),
]
MAIN = ["CMC", "Crisis_Combined", "Severity_Combined", "Informativeness_Combined",
        "Humanitarian_Combined", "Resource_Combined"]
HEAD = {"CMC": "CMC", "Crisis_Combined": "Crisis", "Severity_Combined": "Sev.",
        "Informativeness_Combined": "Inf.", "Humanitarian_Combined": "Hum.", "Resource_Combined": "Res.",
        "avg6": "Avg.", "avg5": "Avg.$_{5}$"}
ALIGN_COLS = [("learned.spearman_score_vs_cmc", r"$\rho$(score, CMC)"),
              ("learned.t2i_R@1", "T$\\to$I R@1"), ("learned.t2i_R@10", "T$\\to$I R@10"),
              ("learned.crossling_P@10", "X-ling P@10"),
              ("encoder.spearman_score_vs_cmc", r"$\rho$ (SigLIP)"),
              ("encoder.crossling_P@10", "X-ling (SigLIP)")]


def load_runs(root):
    rows, eff, align = [], [], []
    for path in sorted(glob.glob(os.path.join(root, "*", "report", "metrics_summary.csv"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = re.match(r"(.+)_s(\d+)$", run)
        if not m:
            print(f"skipping '{run}': folder name has no _s<seed> suffix")
            continue
        variant, seed = m.group(1), int(m.group(2))
        df = pd.read_csv(path)
        wide = df.set_index("field")["macro_f1"]
        rec = {"variant": variant, "seed": seed, **{k: wide.get(k, np.nan) for k in MAIN}}
        rec["avg6"] = np.nanmean([rec[k] for k in MAIN])
        rec["avg5"] = np.nanmean([rec[k] for k in MAIN[1:]])
        rows.append(rec)
        rep = os.path.dirname(path)
        if os.path.exists(os.path.join(rep, "efficiency.json")):
            eff.append({"variant": variant, "seed": seed, **json.load(open(os.path.join(rep, "efficiency.json")))})
        if os.path.exists(os.path.join(rep, "alignment.json")):
            a = pd.json_normalize(json.load(open(os.path.join(rep, "alignment.json"))), sep=".")
            align.append({"variant": variant, "seed": seed, **a.iloc[0].to_dict()})
    if not rows:
        raise SystemExit(f"no runs found under {root}/*/report/")
    long = pd.concat([pd.read_csv(p).assign(run=os.path.basename(os.path.dirname(os.path.dirname(p))))
                      for p in glob.glob(os.path.join(root, "*_s*", "report", "metrics_summary.csv"))])
    return pd.DataFrame(rows), pd.DataFrame(eff), pd.DataFrame(align), long


def ordered(variants):
    known = [v for v, _, _ in VARIANTS if v in variants]
    return known + sorted(set(variants) - set(known))


def display(v):
    return next((d for k, d, _ in VARIANTS if k == v), v.replace("_", r"\_"))


def group(v):
    return next((g for k, _, g in VARIANTS if k == v), "Other")


def paired_p(df, ref, v, col):
    if ttest_rel is None or v == ref:
        return np.nan
    a = df[df.variant == ref].set_index("seed")[col]
    b = df[df.variant == v].set_index("seed")[col]
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    return float(ttest_rel(a[common], b[common]).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/summary")
    ap.add_argument("--ref", default="full", help="reference variant for significance tests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, eff, align, long = load_runs(a.runs)
    long.to_csv(os.path.join(a.out, "all_metrics_long.csv"), index=False)
    cols = MAIN + ["avg6", "avg5"]
    vs = ordered(df.variant.unique())
    ref = a.ref if a.ref in vs else vs[0]

    mean = df.groupby("variant")[cols].mean().loc[vs] * 100
    std = df.groupby("variant")[cols].std().fillna(0).loc[vs] * 100
    n = df.groupby("variant").size().loc[vs]
    table = mean.round(2).astype(str) + " ± " + std.round(2).astype(str)
    table.insert(0, "n_seeds", n)
    table.to_csv(os.path.join(a.out, "macro_f1_mean_std.csv"))

    sig = pd.DataFrame({c: [paired_p(df, ref, v, c) for v in vs] for c in cols}, index=vs)
    delta = mean - mean.loc[ref]
    sig_out = pd.concat({"delta_vs_ref": delta.round(2), "p_value": sig.round(4)}, axis=1)
    sig_out.to_csv(os.path.join(a.out, f"significance_vs_{ref}.csv"))

    if len(eff):
        eff.drop(columns=["seed"]).groupby("variant").mean(numeric_only=True).reindex(
            [v for v in vs if v in set(eff.variant)]).round(3).to_csv(os.path.join(a.out, "efficiency.csv"))

    # ── LaTeX: grouped ablation table ──
    hdr = [HEAD[c] for c in cols]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Macro-F1 (\%, mean $\pm$ std over " + f"{int(n.max())}" +
         r" seeds) on the combined labels and CMC. Avg.: mean over the six columns; Avg.$_5$: the five"
         r" crisis tasks only (the fair comparison when CMC is not supervised). Bold: best mean per column."
         r" $^{*}$: significantly different from " + display(ref) + r" (paired $t$-test over seeds, $p<0.05$).}",
         r"\label{tab:ablations}", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Variant & " + " & ".join(hdr) + r" \\"]
    last_group = None
    for v in vs:
        if group(v) != last_group:
            last_group = group(v)
            L += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{last_group}}}}} \\"]
        cells = []
        for c in cols:
            txt = f"{mean.loc[v, c]:.2f}$\\pm${std.loc[v, c]:.2f}"
            if np.isclose(mean.loc[v, c], mean[c].max()):
                txt = r"\textbf{" + txt + "}"
            if not np.isnan(sig.loc[v, c]) and sig.loc[v, c] < 0.05:
                txt += r"$^{*}$"
            cells.append(txt)
        L.append(display(v) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.out, "ablation_table.tex"), "w").write("\n".join(L) + "\n")

    # ── alignment metrics ──
    if len(align):
        acols = [c for c, _ in ALIGN_COLS if c in align.columns]
        am = align.groupby("variant")[acols].mean()
        asd = align.groupby("variant")[acols].std().fillna(0)
        av = [v for v in vs if v in am.index]
        (am.loc[av].round(4).astype(str) + " ± " + asd.loc[av].round(4).astype(str)).to_csv(
            os.path.join(a.out, "alignment_mean_std.csv"))
        heads = [h for c, h in ALIGN_COLS if c in acols]
        T = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Cross-modal correlation analysis on the test set (mean over seeds). $\rho$: Spearman"
             r" correlation between the matched-pair score and the annotated CMC level. T$\to$I R@k: text-to-image"
             r" retrieval. X-ling P@10: share of the top-10 images retrieved from posts in the \emph{other}"
             r" language that have the same crisis type. SigLIP columns use the encoder space.}",
             r"\label{tab:alignment}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{0.85\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(acols) + "}", r"\toprule",
             "Variant & " + " & ".join(heads) + r" \\", r"\midrule"]
        for v in av:
            T.append(display(v) + " & " + " & ".join(
                "--" if np.isnan(am.loc[v, c]) else f"{am.loc[v, c]:.3f}" for c in acols) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
        open(os.path.join(a.out, "alignment_table.tex"), "w").write("\n".join(T) + "\n")

    show = table.copy()
    show.index = [display(v).replace(r"\_", "_").replace(r"\%", "%") for v in show.index]
    pd.set_option("display.width", 250)
    print(show.to_string())
    print(f"\nreference for significance: {ref}")
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
