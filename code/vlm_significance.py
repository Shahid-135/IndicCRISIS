"""
vlm_significance.py - paired bootstrap tests behind the VLM paragraphs of the paper (test split of seed 42).

    python vlm_significance.py --runs runs_xlmr --vlm pilot_test pilot_test_internvl3 pilot_test_gemma3 \
                               --names Qwen2.5-VL-7B InternVL3-8B Gemma-3-12B --out vlm_significance

<runs>/<variant>_s42/predictions.csv  (or .../report/predictions.csv): CCEC-G (full) and Fusion (fusion_baseline)
<vlm folder>/<mode>/predictions.csv  for mode in direct, cot, fewshot, finetuned

Writes to --out:
    avg5_vs_ccecg.csv     CCEC-G minus model, combined Avg5 macro-F1, 95% interval, two-sided bootstrap p
    cmc_vs_ccecg.csv      the same for CMC macro-F1 (fine-tuned VLMs)
    cot_by_language.csv   chain-of-thought minus direct, Avg5 per language and VLM
    cot_imgdep.csv        chain-of-thought minus direct, accuracy on image-dependent / text-sufficient decisions
Bootstrap: resample test posts with replacement (paired: the same posts for both models), default 10,000 times.
Label sets are those of modality_reliance.py, so the point estimates equal the numbers in the paper.
"""
import argparse
import os

import numpy as np
import pandas as pd

from modality_reliance import TASKS, from_run, from_vlm


def find_run(root, variant, seed=42):
    for p in (f"{root}/{variant}_s{seed}/predictions.csv", f"{root}/{variant}_s{seed}/report/predictions.csv"):
        if os.path.exists(p):
            return p
    raise SystemExit(f"no predictions for {variant}_s{seed} in {root}")


def encode(df, labels, col):
    """integer-encode gold/pred of one column group per task; unknown predictions -> id K (never correct)."""
    out = []
    for t in TASKS:
        m = {l: i for i, l in enumerate(labels[t])}
        K = len(labels[t])
        out.append((np.array([m[x] for x in df[f"gold_{t}_{col}"]]),
                    np.array([m.get(x, K) for x in df[f"pred_{t}_{col}"]]), K))
    return out


def macro_f1(g, p, K):
    tp = np.bincount(g[g == p], minlength=K + 1)[:K]
    gc = np.bincount(g, minlength=K + 1)[:K]
    pc = np.bincount(p, minlength=K + 1)[:K]
    return np.mean(np.where(gc + pc > 0, 2 * tp / np.maximum(gc + pc, 1), 0.0))


def avg5(E, idx):
    return np.mean([macro_f1(g[idx], p[idx], K) for g, p, K in E])


def task_labels(df):
    return {t: sorted(set(df[f"gold_{t}_text"]) | set(df[f"gold_{t}_image"]) | set(df[f"gold_{t}_combined"]))
            for t in TASKS}


def paired(stat_a, stat_b, n, B, rng):
    """point estimate, 95% interval and two-sided p of stat_a - stat_b; stat_* map an index array to a number."""
    full = np.arange(n)
    d0 = stat_a(full) - stat_b(full)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = stat_a(i) - stat_b(i)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d0, lo, hi, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="fine-tuned runs (runs_xlmr)")
    ap.add_argument("--vlm", nargs="+", required=True, help="VLM output folders")
    ap.add_argument("--names", nargs="+", default=None, help="display names of the VLM folders")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default="vlm_significance")
    a = ap.parse_args()
    names = a.names or [os.path.basename(v.rstrip("/")) for v in a.vlm]
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    ref = from_run(find_run(a.runs, "full")).set_index("image_name")
    order = ref.index
    align = lambda df: df.set_index("image_name").loc[order].reset_index()
    ref = ref.reset_index()
    labels = task_labels(ref)
    n = len(ref)
    E_ref = encode(ref, labels, "combined")

    models = {"Fusion": align(from_run(find_run(a.runs, "fusion_baseline")))}
    for name, folder in zip(names, a.vlm):
        for mode in ("direct", "cot", "fewshot", "finetuned"):
            p = os.path.join(folder, mode, "predictions.csv")
            if os.path.exists(p):
                models[f"{name}|{mode}"] = align(from_vlm(p))

    # 1) Avg5 vs CCEC-G
    rows = []
    for k, df in models.items():
        for t in TASKS:
            assert (df[f"gold_{t}_combined"].values == ref[f"gold_{t}_combined"].values).all(), k
        E = encode(df, labels, "combined")
        d0, lo, hi, p = paired(lambda i: avg5(E_ref, i), lambda i, E=E: avg5(E, i), n, a.B, rng)
        rows.append({"model": k, "avg5": 100 * avg5(E, np.arange(n)), "ccecg_minus_model": 100 * d0,
                     "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p": p})
        print(f"{k:28s} Avg5 {rows[-1]['avg5']:6.2f}  CCEC-G - model {100*d0:+6.2f} [{100*lo:+.2f}, {100*hi:+.2f}] p={p:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "avg5_vs_ccecg.csv"), index=False)

    # 2) CMC vs CCEC-G (fine-tuned VLMs and Fusion)
    cl = sorted(ref.gold_cmc.unique())
    cm = {l: i for i, l in enumerate(cl)}
    K = len(cl)
    enc_cmc = lambda d: (np.array([cm[x] for x in d.gold_cmc]), np.array([cm.get(x, K) for x in d.pred_cmc]))
    g0, p0 = enc_cmc(ref)
    rows = []
    for k, df in models.items():
        if not (k == "Fusion" or k.endswith("|finetuned")):
            continue
        g1, p1 = enc_cmc(df)
        d0, lo, hi, p = paired(lambda i: macro_f1(g0[i], p0[i], K), lambda i: macro_f1(g1[i], p1[i], K), n, a.B, rng)
        rows.append({"model": k, "cmc_f1": 100 * macro_f1(g1, p1, K), "ccecg_minus_model": 100 * d0,
                     "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p": p,
                     "share_pred_strong": (df.pred_cmc == "Strong").mean()})
        print(f"CMC {k:24s} {rows[-1]['cmc_f1']:6.2f}  CCEC-G - model {100*d0:+6.2f} [{100*lo:+.2f}, {100*hi:+.2f}] p={p:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cmc_vs_ccecg.csv"), index=False)

    # 3) chain-of-thought minus direct, per language;  4) on image-dependent / text-sufficient decisions
    rows3, rows4 = [], []
    for name in names:
        if f"{name}|direct" not in models or f"{name}|cot" not in models:
            continue
        d, c = models[f"{name}|direct"], models[f"{name}|cot"]
        for lang in sorted(d.lang.unique()):
            m = (d.lang == lang).values
            sd, sc = d[m].reset_index(drop=True), c[m].reset_index(drop=True)
            lab = task_labels(sd)
            Ed, Ec = encode(sd, lab, "combined"), encode(sc, lab, "combined")
            d0, lo, hi, p = paired(lambda i: avg5(Ec, i), lambda i: avg5(Ed, i), len(sd), a.B, rng)
            rows3.append({"vlm": name, "lang": lang, "direct": 100 * avg5(Ed, np.arange(len(sd))),
                          "cot": 100 * avg5(Ec, np.arange(len(sd))), "cot_minus_direct": 100 * d0,
                          "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p": p})
        dep = np.stack([(d[f"gold_{t}_combined"] != d[f"gold_{t}_text"]).values for t in TASKS], 1)
        ok_d = np.stack([(d[f"pred_{t}_combined"] == d[f"gold_{t}_combined"]).values for t in TASKS], 1)
        ok_c = np.stack([(c[f"pred_{t}_combined"] == c[f"gold_{t}_combined"]).values for t in TASKS], 1)
        for kind, mask in (("image-dependent", dep), ("text-sufficient", ~dep)):
            acc = lambda ok, i, mask=mask: np.mean([ok[i, j][mask[i, j]].mean() for j in range(len(TASKS))])
            d0, lo, hi, p = paired(lambda i: acc(ok_c, i), lambda i: acc(ok_d, i), n, a.B, rng)
            rows4.append({"vlm": name, "decisions": kind, "cot_minus_direct": 100 * d0,
                          "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p": p})
    pd.DataFrame(rows3).to_csv(os.path.join(a.out, "cot_by_language.csv"), index=False)
    pd.DataFrame(rows4).to_csv(os.path.join(a.out, "cot_imgdep.csv"), index=False)
    print(pd.DataFrame(rows3).round(2).to_string(index=False))
    print(pd.DataFrame(rows4).round(2).to_string(index=False))
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
