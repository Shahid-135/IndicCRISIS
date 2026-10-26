"""
check_image_overlap.py - are test images also in the training set?

Several stories appear in an English and a Hindi version with the SAME photo, and generic images
(logos, stock photos) are reused across stories. The split keeps identical headlines together, but
not translations or reused photos. This script measures how many test posts have a near-identical
image in the training split (frozen SigLIP-2 cosine >= --threshold) and re-scores every model on
the "clean" test posts without such an image.

    python check_image_overlap.py --csv IndicCrisis_clean.csv --image_dir "Merged Images" \
        --runs runs_v2 --vlm pilot_test
Outputs (overlap_out/): overlap_per_post.csv, rescored.csv, overlap_table.tex
"""
import argparse
import glob
import os
import re
import types

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]


def embed(a, names):
    if a.mock:
        rng = np.random.default_rng(0)
        e = rng.normal(size=(len(names), 32))
        e[1::7] = e[0::7][:len(e[1::7])]                     # inject duplicates for the test
    else:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(a.siglip)
        model = AutoModel.from_pretrained(a.siglip).to(dev).eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(names), 64):
                imgs = []
                for n in names[b:b + 64]:
                    try:
                        imgs.append(Image.open(os.path.join(a.image_dir, n)).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (224, 224)))
                out.append(model.get_image_features(**proc(images=imgs, return_tensors="pt").to(dev)).float().cpu().numpy())
                if (b // 64) % 20 == 0:
                    print(f"  embedded {min(b + 64, len(names))}/{len(names)} images", flush=True)
        e = np.concatenate(out)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def avg5(df, mask, prefix_true, prefix_pred, spaces):
    return float(np.mean([f1_score(df[prefix_true(t)][mask], df[prefix_pred(t)][mask], labels=spaces[t],
                                   average="macro", zero_division=0) for t in TASKS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--variants", nargs="+", default=["ccec", "full", "full_vlm", "fusion_baseline",
                                                      "fusion_full", "fusion_full_vlm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--vlm", default=None, help="pilot output folder of the zero-shot VLM (seed-42 test split)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="overlap_out")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import train_ccec as T
    args = types.SimpleNamespace(text_col="description", image_col="image_name", keep_source=False,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=42)
    df = T.load_dataframe(a.csv, args)
    spaces = T.build_label_spaces(df)
    names = df["image_name"].tolist()
    E = embed(a, names)
    pos = {n: i for i, n in enumerate(names)}

    per_post, rows = [], []
    for seed in a.seeds:
        args.seed = seed
        train, _, test = T.split(df, args)
        tr = np.array([pos[n] for n in train["image_name"]])
        te = np.array([pos[n] for n in test["image_name"]])
        best = (E[te] @ E[tr].T).max(1)
        dup = best >= a.threshold
        for n, b, l in zip(test["image_name"], best, test["lang"]):
            per_post.append({"seed": seed, "image_name": n, "lang": l, "max_sim_to_train": round(float(b), 4),
                             "near_duplicate_in_train": bool(b >= a.threshold)})
        print(f"seed {seed}: {dup.sum()} of {len(te)} test posts ({100 * dup.mean():.1f}%) have a near-identical "
              f"image in train (cos >= {a.threshold})")
        clean = set(test["image_name"][~dup])
        dups = set(test["image_name"][dup])
        for v in a.variants:
            p = os.path.join(a.runs, f"{v}_s{seed}", "report", "predictions.csv")
            if not os.path.exists(p):
                continue
            pr = pd.read_csv(p, encoding="utf-8-sig")
            known = pr["image_name"].isin(clean | dups).values
            if known.mean() < 0.99:
                print(f"  WARNING {v}_s{seed}: only {known.mean():.0%} of its test posts are in this split - "
                      f"different scikit-learn version? Run this script where the models were trained.")
            m_all = np.ones(len(pr), bool)
            m_clean = pr["image_name"].isin(clean).values
            m_dup = pr["image_name"].isin(dups).values
            f = lambda m: avg5(pr, m, lambda t: f"{t}_Combined_true", lambda t: f"{t}_Combined_pred", spaces)
            fi = lambda m: avg5(pr, m, lambda t: f"{t}_Image_true", lambda t: f"{t}_Image_pred", spaces)
            rows.append({"model": v, "seed": seed, "n_clean": int(m_clean.sum()), "n_dup": int(m_dup.sum()),
                         "avg5_all": f(m_all), "avg5_clean": f(m_clean), "avg5_dup": f(m_dup) if m_dup.any() else np.nan,
                         "image_step_all": fi(m_all), "image_step_clean": fi(m_clean)})
        if a.vlm and seed == 42:
            for mode in ("direct", "cot"):
                p = os.path.join(a.vlm, mode, "predictions.csv")
                if os.path.exists(p):
                    pr = pd.read_csv(p, encoding="utf-8-sig")
                    m_clean = pr["image_name"].isin(clean).values
                    m_dupv = pr["image_name"].isin(dups).values
                    f = lambda m: avg5(pr, m, lambda t: f"{t}_Combined", lambda t: f"{t}_combined", spaces)
                    rows.append({"model": f"vlm_{mode}", "seed": 42, "n_clean": int(m_clean.sum()),
                                 "n_dup": int(m_dupv.sum()), "avg5_all": f(np.ones(len(pr), bool)),
                                 "avg5_clean": f(m_clean), "avg5_dup": f(m_dupv) if m_dupv.any() else np.nan})
    pd.DataFrame(per_post).to_csv(os.path.join(a.out, "overlap_per_post.csv"), index=False)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "rescored.csv"), index=False)
    S = R.groupby("model")[["n_clean", "n_dup", "avg5_all", "avg5_clean", "avg5_dup", "image_step_all",
                            "image_step_clean"]].mean()
    S["delta_clean_minus_all"] = S["avg5_clean"] - S["avg5_all"]
    order = [m for m in a.variants + ["vlm_direct", "vlm_cot"] if m in S.index]
    S = S.loc[order]
    for c in [c for c in S.columns if c.startswith(("avg5", "image", "delta"))]:
        S[c] = (100 * S[c]).round(2)
    pd.set_option("display.width", 220)
    print("\n" + S.round(2).to_string())
    pp = pd.DataFrame(per_post)
    print("\nnear-duplicate share by language:", pp.groupby("lang")["near_duplicate_in_train"].mean().round(3).to_dict())
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Avg.$_5$ macro-F1 (\%) on all test posts and on the test posts without a near-identical"
         rf" image in the training split (frozen SigLIP-2 cosine $\geq {a.threshold}$); mean over seeds.}}",
         r"\label{tab:overlap}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
         r"Model & All & Without image overlap & $\Delta$ \\", r"\midrule"]
    for m, r in S.iterrows():
        L.append(f"{m.replace('_', ' ')} & {r.avg5_all:.2f} & {r.avg5_clean:.2f} & {r.delta_clean_minus_all:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(a.out, "overlap_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
