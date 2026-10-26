import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
NAMES = {"mbert": "mBERT", "indicbert": "IndicBERT", "muril": "MuRIL", "xlmr": "XLM-R",
         "vgg19": "VGG-19", "resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext": "ConvNeXt-B",
         "fusion_baseline": "Fusion", "ccec": "CCEC", "fusion_full": "Fusion + corr. learning",
         "full": "CCEC-G", "fusion_full_vlm": "Fusion + corr. + VLM view", "full_vlm": "CCEC-G + VLM view",
         "full_vlm_comp": "CCEC-G + VLM view (trained comp.)",
         "vlm_direct": "VLM zero-shot (direct)", "vlm_cot": "VLM zero-shot (CoT)"}
GROUPS = [("Zero-shot VLM", ["vlm_direct", "vlm_cot"]),
          ("Text-only (fine-tuned)", ["mbert", "indicbert", "muril", "xlmr"]),
          ("Image-only (fine-tuned)", ["vgg19", "resnet50", "vit_b16", "convnext"]),
          ("Multimodal (fine-tuned)", ["fusion_baseline", "ccec", "fusion_full", "full",
                                       "fusion_full_vlm", "full_vlm", "full_vlm_comp"])]
LANGS = [("en", "English"), ("hi", "Hindi"), ("all", "All")]


def from_ccec(r):
    out = {"all": {t: r["test"][f"{t}_Combined"]["macro_f1"] for t in TASKS}}
    for lg in ("en", "hi"):
        k = f"test_lang={lg}"
        if k in r:
            out[lg] = {t: r[k][f"{t}_Combined"]["macro_f1"] for t in TASKS}
    return out


def from_baseline(r):
    out = {"all": {t: r["test"][f"{t}_Combined"]["macro_f1"] for t in TASKS}}
    for row in r.get("test_lang", []):
        out[row["language"]] = {t: row[f"{t}_Combined"] for t in TASKS}
    return out


def from_vlm(path, spaces):
    d = pd.read_csv(path, encoding="utf-8-sig")
    out = {}
    for lg in ("en", "hi", "all"):
        m = np.ones(len(d), bool) if lg == "all" else (d["lang"] == lg).values
        out[lg] = {t: f1_score(d[f"{t}_Combined"][m], d[f"{t}_combined"][m], labels=spaces[t],
                               average="macro", zero_division=0) for t in TASKS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--vlm", default=None, help="pilot output folder with direct/ and cot/ predictions")
    ap.add_argument("--csv", default="IndicCrisis_clean.csv", help="only needed for the VLM label spaces")
    ap.add_argument("--out", default="language_summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*_s*", "results.json"))):
        run = os.path.basename(os.path.dirname(p))
        m = re.match(r"(.+)_s(\d+)$", run)
        if not m:
            continue
        name, seed = m.group(1), int(m.group(2))
        r = json.load(open(p))
        if name.startswith(("txt_", "img_")):
            name, res = name[4:], from_baseline(r)
        else:
            res = from_ccec(r)
        if name not in NAMES:
            continue
        for lg, sc in res.items():
            rows.append({"model": name, "seed": seed, "lang": lg, **sc, "avg5": np.mean(list(sc.values()))})
    if a.vlm:
        import types
        import train_ccec as T
        args = types.SimpleNamespace(text_col="description", image_col="image_name", keep_source=False,
                                     allow_lost_text=False, test_size=0.1, val_size=0.1, seed=42)
        spaces = T.build_label_spaces(T.load_dataframe(a.csv, args))
        for mode in ("direct", "cot"):
            p = os.path.join(a.vlm, mode, "predictions.csv")
            if os.path.exists(p):
                for lg, sc in from_vlm(p, spaces).items():
                    rows.append({"model": f"vlm_{mode}", "seed": 42, "lang": lg, **sc,
                                 "avg5": np.mean(list(sc.values()))})
    d = pd.DataFrame(rows)
    cols = TASKS + ["avg5"]
    M = (d.groupby(["model", "lang"])[cols].mean() * 100)
    S = (d.groupby(["model", "lang"])[cols].std().fillna(0) * 100)
    N = d.groupby(["model", "lang"]).size()
    long = M.round(2).astype(str) + " ± " + S.round(2).astype(str)
    long.insert(0, "n_seeds", N)
    long.to_csv(os.path.join(a.out, "language_breakdown.csv"))

    present = set(d["model"])
    pd.set_option("display.width", 250)
    wide = M["avg5"].unstack("lang").reindex(columns=["en", "hi", "all"])
    order = [m for _, ms in GROUPS for m in ms if m in present]
    wide = wide.loc[order].round(2)
    wide.index = [NAMES[m] for m in order]
    print("Avg_5 macro-F1 (%) by language (mean over seeds):\n" + wide.to_string())

    # LaTeX: three blocks (English / Hindi / All), each with the five tasks + Avg_5
    head = " & ".join(["Cri.", "Sev.", "Inf.", "Hum.", "Res.", "Avg.$_5$"])
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Macro-F1 (\%) on the English test posts, the Hindi test posts and all test posts (combined"
         r" labels). Fine-tuned models: mean over three seeds, trained on the bilingual training split; zero-shot"
         r" VLM: test split of seed 42. Values: mean $\\pm$ standard deviation over seeds. Bold: best mean per column.}",
         r"\label{tab:language}", r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "cccccc" * 3 + "}", r"\toprule",
         r"\multirow{2}{*}{Model} & " + " & ".join(rf"\multicolumn{{6}}{{c}}{{{n}}}" for _, n in LANGS) + r" \\",
         r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}",
         " & " + " & ".join([head] * 3) + r" \\"]
    best = {(lg, c): M.xs(lg, level="lang")[c].max() for lg, _ in LANGS for c in cols}
    for title, ms in GROUPS:
        ms = [m for m in ms if m in present]
        if not ms:
            continue
        L += [r"\midrule", rf"\multicolumn{{19}}{{l}}{{\textit{{{title}}}}} \\"]
        for m in ms:
            cells = []
            for lg, _ in LANGS:
                for c in cols:
                    v = M[c].get((m, lg), np.nan)
                    sd = S[c].get((m, lg), 0)
                    txt = "--" if pd.isna(v) else (f"{v:.2f}" if m.startswith("vlm") else f"{v:.2f}$\\pm${sd:.2f}")
                    cells.append(r"\textbf{" + txt + "}" if not pd.isna(v) and np.isclose(v, best[(lg, c)]) else txt)
            L.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.out, "language_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\nwritten to {a.out}/ (language_breakdown.csv, language_table.tex)")


if __name__ == "__main__":
    main()
