"""
crisismmd_table.py - mean (and std) accuracy and weighted F1 of the CrisisMMD runs (run_crisismmd.sh).

    python crisismmd_table.py --runs runs_crisismmd                       # print the table
    python crisismmd_table.py --runs runs_crisismmd --patch ../paper/tables/crisismmd_table.tex

<runs>/<task>/<model>_s<seed>/results.json, model in xlmr | fusion | ccecg.
--patch fills the table rows whose first cell is exactly one of the ROWS labels below (other rows, e.g. published
numbers with a citation, are left untouched). Writes <runs>/crisismmd_summary.csv as well.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

TASK_NAMES = {"informative": "Informativeness", "humanitarian": "Humanitarian", "damage": "Damage"}
ROWS = {"mann": "MANN", "camm": "CAMM", "robertamft": "RoBERTaMFT", "dmcc": "DMCC", "clip": "CLIP", "align": "ALIGN",
        "xlmr": "XLM-R (text only)", "fusion": "Fusion", "ccecg": "CCEC-G (latent CMC)"}


def metrics(res, T):
    r = res["test"][f"{T}_Combined"]
    return 100 * r.get("acc", r.get("accuracy")), 100 * r["weighted_f1"], 100 * r["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_crisismmd")
    ap.add_argument("--tasks", nargs="+", default=["informative", "humanitarian"])
    ap.add_argument("--patch", default=None, help="tables/crisismmd_table.tex to fill in")
    a = ap.parse_args()

    rows = []
    for task in a.tasks:
        T = TASK_NAMES[task]
        for f in sorted(glob.glob(os.path.join(a.runs, task, "*_s*", "results.json"))):
            m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
            acc, wf1, mf1 = metrics(json.load(open(f)), T)
            rows.append({"task": task, "model": m.group(1), "seed": int(m.group(2)), "acc": acc,
                         "weighted_f1": wf1, "macro_f1": mf1})
    if not rows:
        raise SystemExit(f"no results.json under {a.runs}/<task>/")
    df = pd.DataFrame(rows)
    agg = df.groupby(["task", "model"]).agg(n=("seed", "count"), acc=("acc", "mean"), acc_std=("acc", "std"),
                                            weighted_f1=("weighted_f1", "mean"), wf1_std=("weighted_f1", "std"),
                                            macro_f1=("macro_f1", "mean")).reset_index()
    agg.to_csv(os.path.join(a.runs, "crisismmd_summary.csv"), index=False)
    print(agg.round(2).to_string(index=False))

    if a.patch:
        tex = open(a.patch).read()
        for m, label in ROWS.items():
            cells = []
            for task in ("informative", "humanitarian"):
                r = agg[(agg.task == task) & (agg.model == m)]
                cells += ([f"{r.acc.iloc[0]:.2f}", f"{r.weighted_f1.iloc[0]:.2f}"] if len(r) else ["\\tbd", "\\tbd"])
            pat = re.compile(r"^" + re.escape(label) + r"\s*&.*?\\\\", re.M)
            if not pat.search(tex):
                print(f"row {label!r} not found in {a.patch} - skipped")
                continue
            tex = pat.sub(lambda _: label + " & " + " & ".join(cells) + r" \\", tex, count=1)
        n_seeds = sorted(df.seed.unique())
        open(a.patch, "w").write(tex)
        print(f"patched {a.patch} (seeds {n_seeds}); check that the caption says 'mean over {len(n_seeds)} seeds'")


if __name__ == "__main__":
    main()
