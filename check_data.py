"""
check_data.py - run this on YOUR machine, where the images are.

    python check_data.py --csv IndicCrisis.xlsx --image_dir "Merged Images" --out data_report

It checks everything that could not be checked without the images, and writes:
    data_report/report.txt              summary
    data_report/missing_images.csv      rows whose image is missing or unreadable
    data_report/duplicate_images.csv    groups of (near-)identical images
    data_report/text_label_conflicts.csv  identical headlines with different TEXT labels
                                          (text labels depend only on the text, so these
                                          are annotation inconsistencies to review)
Only needs pandas, numpy, Pillow and openpyxl.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]


def ahash(path, size=8):
    """Average hash: 64-bit fingerprint that is identical for resized / re-compressed copies."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return "".join("1" if x else "0" for x in (a > a.mean()).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--image_col", default="image_name")
    ap.add_argument("--out", default="data_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_excel(a.csv) if a.csv.endswith((".xlsx", ".xls")) else pd.read_csv(a.csv, encoding="utf-8-sig")
    txt = df[a.text_col].astype(str)
    lines = [f"rows: {len(df)}"]

    # ── 1. text encoding ──
    deva = txt.str.contains(r"[\u0900-\u097F]")
    lost = txt.map(lambda s: s.replace(" ", "").count("?") / max(1, len(s.replace(" ", "")))) > 0.3
    moj = txt.str.contains("Â|â€|Ã")
    lines += [f"Hindi rows with intact Devanagari : {int(deva.sum())}",
              f"rows whose text is mostly '?'     : {int(lost.sum())}   <- must be 0",
              f"rows with mojibake (e.g. 'Â’')    : {int(moj.sum())}   (train_ccec.py repairs these)"]

    # ── 2. images: exist, open, size ──
    bad, hashes, sizes = [], {}, []
    for i, name in enumerate(df[a.image_col].astype(str)):
        p = os.path.join(a.image_dir, name)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append(im.size)
            hashes[i] = ahash(p)
        except Exception as e:
            bad.append({"row": i, "image_name": name, "error": type(e).__name__})
        if (i + 1) % 1000 == 0:
            print(f"  checked {i + 1} images")
    pd.DataFrame(bad).to_csv(os.path.join(a.out, "missing_images.csv"), index=False)
    lines.append(f"missing / unreadable images       : {len(bad)}")
    if sizes:
        w, h = np.array(sizes).T
        lines.append(f"image size (median WxH)           : {int(np.median(w))}x{int(np.median(h))}; "
                     f"smaller than 100px on a side: {int(((w < 100) | (h < 100)).sum())}")

    # ── 3. duplicate images (same fingerprint) ──
    hs = pd.Series(hashes)
    groups = hs[hs.duplicated(keep=False)]
    rows = []
    for gid, (hv, idx) in enumerate(groups.groupby(groups).groups.items()):
        for i in idx:
            rows.append({"group": gid, "row": i, "image_name": df.at[i, a.image_col],
                         "description": df.at[i, a.text_col], "CMC": df.at[i, "CMC"],
                         **{f"{t}_Image": df.at[i, f"{t}_Image"] for t in TASKS}})
    dup_img = pd.DataFrame(rows)
    dup_img.to_csv(os.path.join(a.out, "duplicate_images.csv"), index=False)
    n_groups = dup_img["group"].nunique() if len(dup_img) else 0
    lines.append(f"duplicate-image groups            : {n_groups} ({len(dup_img)} rows)")
    if len(dup_img):
        incons = dup_img.groupby("group")[[f"{t}_Image" for t in TASKS]].nunique().gt(1).sum()
        lines.append("  groups whose IMAGE labels differ: " + ", ".join(f"{k}={v}" for k, v in incons.items()))

    # ── 4. identical headlines with different text labels ──
    tail = re.compile(r"\s+[-|]\s+[^-|\u0900-\u097F]{2,40}$")      # outlet names, Latin script only
    norm = txt.map(lambda s: tail.sub("", s)).str.lower().str.split().str.join(" ")
    ok = ~lost
    g = df[ok & norm.duplicated(keep=False)].assign(_n=norm).groupby("_n")
    conflict = []
    for key, sub in g:
        diff = [f"{t}_Text" for t in TASKS if sub[f"{t}_Text"].nunique() > 1]
        if diff:
            for i, r in sub.iterrows():
                conflict.append({"headline": key, "row": i, "image_name": r[a.image_col],
                                 "fields_in_conflict": ";".join(diff),
                                 **{f"{t}_Text": r[f"{t}_Text"] for t in TASKS}})
    cdf = pd.DataFrame(conflict)
    cdf.to_csv(os.path.join(a.out, "text_label_conflicts.csv"), index=False)
    lines.append(f"duplicate-headline groups         : {g.ngroups}; with conflicting TEXT labels: "
                 f"{cdf['headline'].nunique() if len(cdf) else 0}")

    report = "\n".join(lines)
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write(report)
    print("\n" + report + f"\n\nfiles written to {a.out}/")


if __name__ == "__main__":
    main()
