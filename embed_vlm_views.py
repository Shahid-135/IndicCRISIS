"""
embed_vlm_views.py - turn zero-shot VLM chain-of-thought into a frozen "knowledge view" for CCEC-G.

Input : the output folder of
          pilot_reasoning_graph.py --split all --modes cot ...
        (cot/raw.jsonl + sample.csv). The VLM never sees any label, so using its reasoning for
        training, validation AND test posts is leak-free.
Output: vlm_views.npz   names (image_name) + emb (frozen SigLIP-2 text embedding of the reasoning)
        vlm_views.csv   the extracted reasoning text per post, for inspection

SigLIP-2 reads at most 64 tokens, but a reasoning is ~150 words, so it is split into sentence
chunks of at most --chunk_words words; every chunk is embedded and the normalised chunk
embeddings are averaged. Embeddings are computed once; training then just looks them up.

    python embed_vlm_views.py --pilot_out vlm_all --out vlm_views.npz
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from pilot_reasoning_graph import extract_json


def reasoning_of(response):
    """The free-text reasoning of a CoT answer (falls back to the raw text if JSON is broken)."""
    d = extract_json(response, required=("reasoning", "combined", "cmc"))
    if d and isinstance(d.get("reasoning"), str) and d["reasoning"].strip():
        return d["reasoning"].strip(), True
    m = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"', response, flags=re.S)
    if m:
        return m.group(1).strip(), True
    return "", False


def chunks(text, max_words):
    """Sentence-aware chunks of at most max_words words."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
    out, cur = [], []
    for s in sents:
        w = s.split()
        while len(w) > max_words:                       # very long sentence: hard split
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(" ".join(w[:max_words]))
            w = w[max_words:]
        if len(cur) + len(w) > max_words and cur:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return out or ["no analysis available"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_out", required=True, help="folder written by pilot_reasoning_graph.py --split all")
    ap.add_argument("--out", default="vlm_views.npz")
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--chunk_words", type=int, default=35)
    ap.add_argument("--max_chunks", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_col", default="image_name")
    ap.add_argument("--mock", action="store_true", help="random embeddings (pipeline test only)")
    a = ap.parse_args()

    sample = pd.read_csv(os.path.join(a.pilot_out, "sample.csv"), encoding="utf-8-sig")
    raw = {}
    for line in open(os.path.join(a.pilot_out, "cot", "raw.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        raw[r["i"]] = r["response"]
    names, texts, ok = [], [], 0
    for i, row in sample.iterrows():
        txt, good = reasoning_of(raw.get(i, ""))
        ok += good
        names.append(str(row[a.image_col]))
        texts.append(txt)
    missing = len(sample) - len(raw)
    print(f"{len(sample)} posts | answers {len(raw)} (missing {missing}) | reasoning extracted for {ok}")
    pd.DataFrame({a.image_col: names, "reasoning": texts}).to_csv(
        a.out.replace(".npz", ".csv"), index=False, encoding="utf-8-sig")

    all_chunks, owner = [], []
    for k, t in enumerate(texts):
        cs = chunks(t, a.chunk_words)[:a.max_chunks] if t else ["no analysis available"]
        all_chunks += cs
        owner += [k] * len(cs)
    print(f"{len(all_chunks)} chunks ({len(all_chunks) / len(texts):.1f} per post)")

    if a.mock:
        rng = np.random.default_rng(0)
        chunk_emb = rng.normal(size=(len(all_chunks), 64)).astype(np.float32)
    else:
        import torch
        from transformers import AutoModel, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(a.siglip)
        model = AutoModel.from_pretrained(a.siglip).to(dev).eval()
        outs = []
        with torch.no_grad():
            for b in range(0, len(all_chunks), a.batch_size):
                inp = proc(text=all_chunks[b:b + a.batch_size], padding="max_length", truncation=True,
                           max_length=64, return_tensors="pt").to(dev)
                outs.append(model.get_text_features(**inp).float().cpu().numpy())
                if (b // a.batch_size) % 20 == 0:
                    print(f"  embedded {min(b + a.batch_size, len(all_chunks))}/{len(all_chunks)}", flush=True)
        chunk_emb = np.concatenate(outs)

    chunk_emb /= np.linalg.norm(chunk_emb, axis=1, keepdims=True) + 1e-8
    emb = np.zeros((len(texts), chunk_emb.shape[1]), np.float32)
    np.add.at(emb, np.array(owner), chunk_emb)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    np.savez(a.out, names=np.array(names), emb=emb)
    print(f"saved {a.out}: {emb.shape} | reasoning texts in {a.out.replace('.npz', '.csv')}")


if __name__ == "__main__":
    main()
