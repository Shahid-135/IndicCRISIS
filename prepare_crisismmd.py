"""
prepare_crisismmd.py - CrisisMMD benchmark splits (Ofli et al., 2020) -> the CSV format of train_ccec.py.

    python prepare_crisismmd.py --tsv_dir /home/nagendra/cafunet/crisismmd \
                                --image_root /path/to/CrisisMMD_v2.0 --out_dir crisismmd_csv

Input:  <tsv_dir>/task_<task>_text_img_{train,dev,test}.tsv   (task: informative, humanitarian, damage)
        image paths in the TSVs are relative to --image_root (the folder that contains data_image/);
        if --image_root is omitted, a few folders around --tsv_dir are searched.
Output: <out_dir>/crisismmd_<task>.csv  with
          split (train/val/test), event_name, tweet_id, image_id, description, image_name,
          <Task>_Text, <Task>_Image, <Task>_Combined        (Task: Informativeness, Humanitarian, Damage)
        <out_dir>/crisismmd_<task>_summary.json   (row counts, dropped rows, label counts per split)

Labels. If the TSVs have separate text and image labels (label_text, label_image), only pairs whose two labels
agree are kept (the "agreed" benchmark setting); then text, image and combined label are the same. If they
only have one label column, it is used for all three views and every row is kept.
Humanitarian classes are merged as in prior work (injured/dead and missing/found people -> affected
individuals, vehicle damage -> infrastructure and utility damage) unless --no_merge; check the printed counts
against the paper whose numbers you compare with.
Text: URLs removed, "RT @user:" prefixes removed, @mentions -> @user, '#' dropped, HTML entities unescaped
(--no_clean keeps the raw tweet).
"""
import argparse
import csv
import html
import json
import os
import re
import sys

import pandas as pd

TASK_NAMES = {"informative": "Informativeness", "humanitarian": "Humanitarian", "damage": "Damage"}
SPLITS = {"train": "train", "dev": "val", "test": "test"}
HUMANITARIAN_MERGE = {"injured_or_dead_people": "affected_individuals",
                      "missing_or_found_people": "affected_individuals",
                      "vehicle_damage": "infrastructure_and_utility_damage"}
TEXT_COLS = ["tweet_text", "text", "tweet"]
IMAGE_COLS = ["image", "image_path", "image_name"]


def first_col(df, options, what, path):
    for c in options:
        if c in df.columns:
            return c
    raise SystemExit(f"{path}: no {what} column (looked for {options}); columns are {list(df.columns)}")


def clean_tweet(s: str) -> str:
    s = html.unescape(str(s))
    s = re.sub(r"^RT\s+@\w+:?\s*", "", s)
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@\w+", "@user", s)
    s = s.replace("#", "")
    return " ".join(s.split())


def read_tsv(path):
    # tweets contain unbalanced quotes: no quote handling, every field as text
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                       encoding="utf-8", on_bad_lines="warn")


def guess_image_root(tsv_dir, rel_path):
    """Folders near the TSVs that contain rel_path (the first image of the data)."""
    base = os.path.abspath(tsv_dir)
    cands = [base, os.path.dirname(base), os.path.dirname(os.path.dirname(base))]
    seen = []
    for c in cands:
        for root, dirs, _ in os.walk(c):
            depth = root[len(c):].count(os.sep)
            if depth > 2:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "data_image"]
            if root not in seen:
                seen.append(root)
                if os.path.exists(os.path.join(root, rel_path)):
                    return root
    return None


def convert_task(task, a):
    T = TASK_NAMES[task]
    frames = []
    for fname_split, split in SPLITS.items():
        path = os.path.join(a.tsv_dir, f"task_{task}_text_img_{fname_split}.tsv")
        if not os.path.exists(path):
            raise SystemExit(f"missing file: {path}")
        d = read_tsv(path)
        d["split"] = split
        d["_file"] = os.path.basename(path)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    path0 = frames[0]["_file"].iloc[0]
    print(f"\n=== {task} ({T}) ===")
    print(f"columns: {[c for c in raw.columns if c not in ('split', '_file')]}")
    tcol = first_col(raw, TEXT_COLS, "tweet text", path0)
    icol = first_col(raw, IMAGE_COLS, "image path", path0)

    has_views = {"label_text", "label_image"} <= set(raw.columns)
    if has_views:
        lt, li = raw["label_text"].str.strip(), raw["label_image"].str.strip()
    elif "label" in raw.columns:
        lt = li = raw["label"].str.strip()
    else:
        raise SystemExit(f"{path0}: no label columns (need label_text + label_image, or label)")
    merge = HUMANITARIAN_MERGE if (task == "humanitarian" and not a.no_merge) else {}
    if a.merge_before_agree:
        lt, li = lt.replace(merge), li.replace(merge)
    agree = lt == li
    n_raw = raw.groupby("split").size().to_dict()
    if has_views:
        print(f"separate text/image labels: keeping the {int(agree.sum())} of {len(raw)} pairs whose labels agree "
              f"(dropped per split: {raw[~agree].groupby('split').size().to_dict()})")
        if "label" in raw.columns:
            off = (raw["label"].str.strip() != lt) & agree
            if off.any():
                print(f"  note: {int(off.sum())} agreed rows have a 'label' value different from label_text")
    df = raw[agree].copy()
    lab = lt[agree].replace(merge) if not a.merge_before_agree else lt[agree]

    out = pd.DataFrame({
        "split": df["split"],
        "event_name": df.get("event_name", ""),
        "tweet_id": df.get("tweet_id", ""),
        "image_id": df.get("image_id", ""),
        "description": df[tcol].map(clean_tweet) if not a.no_clean else df[tcol],
        "image_name": df[icol].str.strip(),
    })
    for v in ("Text", "Image", "Combined"):
        out[f"{T}_{v}"] = lab.values
    empty = out["description"].str.len() == 0
    if empty.any():
        print(f"  note: {int(empty.sum())} tweets are empty after cleaning (kept; the model sees 'no description')")

    # images
    root = a.image_root or guess_image_root(a.tsv_dir, out["image_name"].iloc[0])
    if root is None:
        raise SystemExit(f"could not find the image folder: pass --image_root (the folder that contains "
                         f"{out['image_name'].iloc[0].split('/')[0]}/). Try:\n"
                         f"  find / -type d -name data_image 2>/dev/null | head")
    exists = out["image_name"].map(lambda p: os.path.exists(os.path.join(root, p)))
    n_miss = int((~exists).sum())
    print(f"image root: {root} | missing images: {n_miss} of {len(out)}")
    if n_miss:
        for p in out.loc[~exists, "image_name"].head(5):
            print(f"    missing: {os.path.join(root, p)}")
        if n_miss > a.max_missing * len(out):
            raise SystemExit("too many missing images (train_ccec.py would silently use black images); "
                             "fix --image_root or pass --max_missing")

    counts = out.groupby("split").size().reindex(["train", "val", "test"]).to_dict()
    dist = out.groupby(["split", f"{T}_Combined"]).size().unstack(0).reindex(columns=["train", "val", "test"])
    print(f"rows: {counts}")
    print(dist.fillna(0).astype(int).to_string())
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, f"crisismmd_{task}.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {"task": task, "column_prefix": T, "image_root": root, "raw_rows": n_raw, "rows": counts,
               "agreed_filter": bool(has_views), "humanitarian_merge": merge, "merge_before_agree": a.merge_before_agree,
               "missing_images": n_miss, "labels": {c: dist[c].dropna().astype(int).to_dict() for c in dist.columns},
               "text_cleaning": not a.no_clean}
    with open(os.path.join(a.out_dir, f"crisismmd_{task}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {csv_path}")
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv_dir", required=True)
    ap.add_argument("--image_root", default=None, help="folder that contains data_image/ (default: searched)")
    ap.add_argument("--out_dir", default="crisismmd_csv")
    ap.add_argument("--tasks", nargs="+", default=["informative", "humanitarian"], choices=list(TASK_NAMES))
    ap.add_argument("--no_merge", action="store_true", help="keep all humanitarian classes")
    ap.add_argument("--merge_before_agree", action="store_true",
                    help="merge humanitarian classes before checking text-image agreement (default: after)")
    ap.add_argument("--no_clean", action="store_true", help="keep the raw tweet text")
    ap.add_argument("--max_missing", type=float, default=0.005, help="allowed share of missing images")
    a = ap.parse_args()
    for t in a.tasks:
        convert_task(t, a)


if __name__ == "__main__":
    sys.exit(main())
