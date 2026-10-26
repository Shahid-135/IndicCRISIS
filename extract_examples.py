"""
extract_examples.py - qualitative examples for the paper.

(A) VLM knowledge view: test posts (seed-42 split) where adding the VLM view changed the combined
    prediction, compared with the same model without it - wins and failures - with an excerpt of
    the VLM reasoning.
(B) CMC-weighted batch graph: pairs of TRAINING posts whose images are similar under the frozen
    SigLIP-2 image encoder, with the edge weights the graph-contrastive loss would give them:
        omega_i = w(CMC_i)  (Strong 1.0, Partial 0.5, Low 0.1)
        e_ij    = max(0, (sim_ij - kappa) / (1 - kappa))
        W_ij    = beta * omega_i * omega_j * e_ij          (two-hop soft positive)
    - cross-lingual pairs (Hindi <-> English) of trusted posts: strong two-hop positives
    - pairs involving a Low-CMC post: similar image, but the edge is almost switched off
    plus an image montage of the pairs (examples_out/graph_pairs.png).

Outputs (examples_out/): vlm_examples.csv/.tex, graph_examples.csv/.tex, graph_pairs.png
Hindi text in the .tex tables uses \\hi{...} (XeLaTeX + Devanagari font, as in the dataset section).

    python extract_examples.py --csv IndicCrisis_clean.csv --image_dir "Merged Images" \
        --runs runs_v2 --vlm_views vlm_views.csv
"""
import argparse
import os
import re
import types

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
W_CMC = {"Strong": 1.0, "Partial": 0.5, "Low": 0.1}


def tex(s, n_words=None):
    """Escape for LaTeX; wrap Devanagari runs in \\hi{}; optionally truncate to n words."""
    s = str(s)
    if n_words:
        w = s.split()
        s = " ".join(w[:n_words]) + (" \\dots" if len(w) > n_words else "")
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        if a == "\\" and "\\dots" in s:
            s = s.replace("\\dots", "@@DOTS@@")
        s = s.replace(a, b)
    s = s.replace("@@DOTS@@", r"\dots")
    s = s.replace(r"\textbackslash{}dots", r"\dots")
    return re.sub(r"([\u0900-\u097F][\u0900-\u097F\s,.\-।0-9]*[\u0900-\u097F])", r"\\hi{\1}", s)


def short(label):
    return {"Flood / Water Disaster": "Flood", "Cyclone / Storm / Extreme Weather": "Cyclone/Storm",
            "Earthquake & Geological": "Earthquake", "Fire & Industrial Accident": "Fire/Industrial",
            "Human Casualties": "Casualties", "Infrastructure Damage": "Infra. damage"}.get(label, label)


# ════════════════════════════════════════════════════════════════════
# (A) VLM knowledge view
# ════════════════════════════════════════════════════════════════════
def vlm_examples(a, data, out):
    base = pd.read_csv(os.path.join(a.runs, f"{a.base}_s{a.seed}", "report", "predictions.csv"), encoding="utf-8-sig")
    vlm = pd.read_csv(os.path.join(a.runs, f"{a.with_vlm}_s{a.seed}", "report", "predictions.csv"), encoding="utf-8-sig")
    views = pd.read_csv(a.vlm_views, encoding="utf-8-sig")
    m = (base.merge(vlm, on="image_name", suffixes=("_b", "_v"))
         .merge(views, on="image_name").merge(data[["image_name", "description", "lang"]], on="image_name"))
    rows = []
    for _, r in m.iterrows():
        for t in TASKS:
            gold = r[f"{t}_Combined_true_b"]
            pb, pv = r[f"{t}_Combined_pred_b"], r[f"{t}_Combined_pred_v"]
            if pb != pv and gold in (pb, pv):
                rows.append({"image_name": r["image_name"], "lang": r["lang"], "task": t, "gold": gold,
                             "without_vlm": pb, "with_vlm": pv, "outcome": "win" if pv == gold else "loss",
                             "gold_text_label": r[f"{t}_Text_true_b"], "gold_CMC": r["CMC_true_b"],
                             "text": r["description"], "reasoning": r["reasoning"]})
    ex = pd.DataFrame(rows)
    if ex.empty:
        print("(A) no changed decisions found")
        return
    n_win, n_loss = (ex.outcome == "win").sum(), (ex.outcome == "loss").sum()
    print(f"(A) decisions changed by the VLM view (seed {a.seed}): {n_win} corrected, {n_loss} broken")
    ex.to_csv(os.path.join(out, "vlm_examples_all.csv"), index=False, encoding="utf-8-sig")
    # pick: wins in both languages and different tasks, then failures
    picks = []
    for outcome, k in (("win", a.n_win), ("loss", a.n_loss)):
        sub = ex[ex.outcome == outcome].sample(frac=1, random_state=0)
        seen_task, seen = set(), set()
        for lang_first in ("hi", "en", "hi", "en", "hi", "en"):
            for _, r in sub[sub.lang == lang_first].iterrows():
                if r.image_name in seen or (r.task in seen_task and len(seen_task) < len(TASKS)):
                    continue
                picks.append(r)
                seen.add(r.image_name)
                seen_task.add(r.task)
                break
            if sum(p.outcome == outcome for p in picks) >= k:
                break
    sel = pd.DataFrame(picks)
    sel.to_csv(os.path.join(out, "vlm_examples.csv"), index=False, encoding="utf-8-sig")
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Test posts in which the VLM knowledge view changed the combined prediction of CCEC-G"
         r" (seed 42). Without / With: prediction without and with the VLM view; the excerpt is the"
         r" frozen zero-shot VLM analysis that the model receives. Upper part: corrected decisions; lower"
         r" part: decisions the VLM view broke.}",
         r"\label{tab:vlm_examples}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{p{4.1cm} l l l l p{7.4cm}}", r"\toprule",
         r"News text & Task & Gold & Without & With & VLM analysis (excerpt) \\", r"\midrule"]
    for i, (_, r) in enumerate(sel.iterrows()):
        if i > 0 and r.outcome == "loss" and sel.iloc[i - 1].outcome == "win":
            L.append(r"\midrule")
        L.append(f"{tex(r.text, 22)} & {r.task} & {tex(short(r.gold))} & {tex(short(r.without_vlm))} & "
                 f"{tex(short(r.with_vlm))} & {tex(r.reasoning, 55)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "vlm_examples.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ════════════════════════════════════════════════════════════════════
# (B) CMC-weighted batch graph
# ════════════════════════════════════════════════════════════════════
def image_embeddings(a, names):
    if a.mock:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(40, 64))
        e = base[np.arange(len(names)) % 40] + 0.3 * rng.normal(size=(len(names), 64))
    else:
        import torch
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
                inp = proc(images=imgs, return_tensors="pt").to(dev)
                out.append(model.get_image_features(**inp).float().cpu().numpy())
                if (b // 64) % 20 == 0:
                    print(f"  embedded {min(b + 64, len(names))}/{len(names)} images", flush=True)
        e = np.concatenate(out)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def graph_examples(a, train, out):
    names = train["image_name"].tolist()
    E = image_embeddings(a, names)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    S[train["group"].values[:, None] == train["group"].values[None, :]] = -1     # never the same headline
    cmc, lang = train["CMC"].values, train["lang"].values
    om = np.array([W_CMC.get(c, 1.0) for c in cmc])
    pairs = []
    k = min(5, len(names) - 1)
    nbr = np.argsort(-S, axis=1)[:, :k]
    for i in range(len(names)):
        for j in nbr[i]:
            if i < j or i not in nbr[j]:
                sim = S[i, j]
                e = max(0.0, (sim - a.kappa) / (1 - a.kappa))
                pairs.append((i, j, sim, e, a.beta * om[i] * om[j] * e))
    P = pd.DataFrame(pairs, columns=["i", "j", "sim", "e", "W"]).drop_duplicates(subset=["i", "j"])
    P = P[P.sim >= a.kappa]
    crossling = P[(lang[P.i] != lang[P.j]) & (cmc[P.i] == "Strong") & (cmc[P.j] == "Strong")]
    lowpair = P[((cmc[P.i] == "Low") | (cmc[P.j] == "Low"))]
    sel = []
    for df, kind, n in ((crossling, "cross-lingual two-hop positive", a.n_cross), (lowpair, "down-weighted (Low CMC)", a.n_low)):
        seen_crisis = set()
        for _, r in df.sort_values("sim", ascending=False).iterrows():
            c = train["Crisis_Combined"].values[int(r.i)]
            if c in seen_crisis:
                continue
            seen_crisis.add(c)
            sel.append({**r.to_dict(), "kind": kind})
            if sum(s["kind"] == kind for s in sel) >= n:
                break
    print(f"(B) candidate pairs: {len(P)} with sim >= {a.kappa}; cross-lingual trusted pairs {len(crossling)}; "
          f"pairs with a Low-CMC post {len(lowpair)}")
    rows = []
    for s in sel:
        i, j = int(s["i"]), int(s["j"])
        ri, rj = train.iloc[i], train.iloc[j]
        rows.append({"kind": s["kind"], "image_i": ri.image_name, "image_j": rj.image_name,
                     "lang_i": ri.lang, "lang_j": rj.lang, "CMC_i": ri.CMC, "CMC_j": rj.CMC,
                     "omega_i": W_CMC[ri.CMC], "omega_j": W_CMC[rj.CMC], "sim": round(s["sim"], 3),
                     "e_ij": round(s["e"], 3), "W_ij": round(s["W"], 3),
                     "crisis_i": ri.Crisis_Combined, "crisis_j": rj.Crisis_Combined,
                     "text_i": ri.description, "text_j": rj.description})
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(out, "graph_examples.csv"), index=False, encoding="utf-8-sig")

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Edges of the CMC-weighted batch graph for pairs of training posts with similar images"
         r" (frozen SigLIP-2 cosine similarity $c_{ij}$). $\omega$: text--image edge weight from the CMC level;"
         rf" $e_{{ij}}$: image--image edge ($\kappa={a.kappa}$); $W_{{ij}}=\beta\,\omega_i\omega_j e_{{ij}}$"
         rf" ($\beta={a.beta}$): weight of post $j$'s image as a soft positive for post $i$'s text. Images"
         r" link posts across languages when both pairs are trustworthy (upper part); a Low-correlation post"
         r" switches the edge almost off (lower part).}",
         r"\label{tab:graph_examples}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{p{5.6cm} p{5.6cm} c c c c}", r"\toprule",
         r"Post $i$ (CMC, $\omega_i$) & Post $j$ (CMC, $\omega_j$) & $c_{ij}$ & $e_{ij}$ & $W_{ij}$ & Crisis \\",
         r"\midrule"]
    prev = None
    for _, r in G.iterrows():
        if prev is not None and r.kind != prev:
            L.append(r"\midrule")
        prev = r.kind
        crisis = short(r.crisis_i) if r.crisis_i == r.crisis_j else f"{short(r.crisis_i)} / {short(r.crisis_j)}"
        # braces: a row must not start with "[" (LaTeX would read it as the optional argument of \\)
        L.append(f"{{[{r.lang_i.upper()}]}} {tex(r.text_i, 18)} ({r.CMC_i}, {r.omega_i}) & "
                 f"{{[{r.lang_j.upper()}]}} {tex(r.text_j, 18)} ({r.CMC_j}, {r.omega_j}) & "
                 f"{r.sim:.2f} & {r.e_ij:.2f} & {r.W_ij:.3f} & {tex(crisis)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "graph_examples.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # montage: one row per pair, the two images side by side
    if len(G):
        th, pad = 200, 10
        canvas = Image.new("RGB", (2 * th + 3 * pad + 180, len(G) * (th + pad) + pad), "white")
        dr = ImageDraw.Draw(canvas)
        for k, (_, r) in enumerate(G.iterrows()):
            for c, n in enumerate((r.image_i, r.image_j)):
                try:
                    im = Image.open(os.path.join(a.image_dir, n)).convert("RGB")
                except Exception:
                    im = Image.new("RGB", (th, th), (200, 200, 200))
                im.thumbnail((th, th))
                canvas.paste(im, (pad + c * (th + pad), pad + k * (th + pad)))
            dr.text((2 * th + 3 * pad, pad + k * (th + pad) + 10),
                    f"({k + 1}) {r.lang_i.upper()}-{r.lang_j.upper()}\nCMC {r.CMC_i}/{r.CMC_j}\n"
                    f"c={r.sim:.2f}\nW={r.W_ij:.3f}", fill="black")
        canvas.save(os.path.join(out, "graph_pairs.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--vlm_views", default="vlm_views.csv")
    ap.add_argument("--base", default="full", help="model without the VLM view")
    ap.add_argument("--with_vlm", default="full_vlm", help="the same model with the VLM view")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--kappa", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_win", type=int, default=4)
    ap.add_argument("--n_loss", type=int, default=2)
    ap.add_argument("--n_cross", type=int, default=4)
    ap.add_argument("--n_low", type=int, default=2)
    ap.add_argument("--out", default="examples_out")
    ap.add_argument("--mock", action="store_true", help="random image embeddings (pipeline test only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import train_ccec as T                                    # same cleaning, language tags and split
    args = types.SimpleNamespace(text_col="description", image_col="image_name", keep_source=False,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=a.seed)
    df = T.load_dataframe(a.csv, args)
    train, _, _ = T.split(df, args)
    vlm_examples(a, df, a.out)
    graph_examples(a, train.reset_index(drop=True), a.out)
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
