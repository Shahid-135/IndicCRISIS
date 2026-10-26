"""
pilot_reasoning_graph.py - does structured multimodal reasoning help, and can we verify it?

Compares three zero-shot prompting modes of an open vision-language model on a sample of the
VALIDATION split (the test split is never touched):

    direct : answer immediately (JSON)
    cot    : free-text reasoning, then the answer (JSON)
    graph  : a reasoning GRAPH - text evidence nodes, image evidence nodes, cross-modal links,
             a text-only conclusion, an image-only conclusion and a combined conclusion whose
             labels cite the evidence nodes they rest on

Every mode returns the same fields (text-only labels, image-only labels, CMC, combined labels),
so each intermediate step can be VERIFIED against the IndicCrisis annotations:

    step 1  text-only conclusion   vs  *_Text labels
    step 2  image-only conclusion  vs  *_Image labels
    step 3  cross-modal correlation vs CMC      (+ graph: are the links consistent with it?)
    step 4  combined conclusion    vs  *_Combined labels
    graph structure: cited nodes exist; a combined label that departs from the text-only label
                     must cite image evidence

Backends
    --backend vllm    offline vLLM (one process, fastest)
    --backend hf      Hugging Face transformers (slower; works with the PyTorch that trains CCEC)
    --backend openai  any OpenAI-compatible server, e.g.  `vllm serve <model>`
    --backend mock    fake answers built from the gold labels + noise - ONLY to test the pipeline

Examples
    python pilot_reasoning_graph.py --csv IndicCrisis_clean.csv --image_dir "Merged Images" \
        --model Qwen/Qwen2.5-VL-7B-Instruct --backend vllm --modes direct cot graph --n 200
    python pilot_reasoning_graph.py ... --backend openai --api_base http://localhost:8000/v1
Results: <out>/<mode>/raw.jsonl, predictions.csv, metrics.json and <out>/pilot_summary.csv
"""
import argparse
import base64
import difflib
import io
import json
import os
import re
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
INVALID = "__invalid__"

# short label definitions from the annotation rulebook (Table: annotation guidelines)
DEFINITIONS = {
    "CMC": {
        "Strong": "the image directly shows the event, place or impact described in the text",
        "Partial": "the image is about the same kind of disaster or context, but not the specific event in the text",
        "Low": "the image has little or no relation to the text (generic, symbolic or unrelated)",
    },
    "Crisis": {
        "Flood / Water Disaster": "floods, flash floods, torrential rain, water-related disasters",
        "Cyclone / Storm / Extreme Weather": "cyclones, hurricanes, tornadoes, storms, severe weather",
        "Earthquake & Geological": "earthquakes, landslides, other geological events",
        "Fire & Industrial Accident": "fires, explosions, industrial or infrastructure accidents",
        "Human Casualties": "focus on deaths, injuries or missing people, no single hazard type dominates",
        "Infrastructure Damage": "focus on physical damage, no single hazard type dominates",
        "Other": "crisis events not covered above",
    },
    "Severity": {
        "Critical": "multiple confirmed casualties, major damage, active threat, large-scale emergency or evacuation",
        "Serious": "isolated casualties, localised damage, disrupted services",
        "Moderate": "few or no casualties, limited damage, alerts, preparedness or resolved situations",
    },
    "Informativeness": {
        "Informative": "gives useful situational information about the crisis, even if not fully specific",
        "Low Informativeness": "background, vague, emotional, repeated, advertorial or decorative content",
    },
    "Humanitarian": {
        "Affected Individuals": "casualties, injured, missing, survivors, displaced people",
        "Infrastructure & Utilities": "damage to roads, bridges, power, hospitals, schools, dams, pipelines",
        "Rescue & Medical": "search and rescue, treatment of casualties, hospital capacity, health risks",
        "Not Humanitarian": "no humanitarian dimension, e.g. purely political commentary",
    },
    "Resource": {
        "Medical": "medicines, blood, medical equipment, ambulances, ICU capacity",
        "Rescue Equipment": "helicopters, boats, excavators, breathing apparatus",
        "Shelter & Relief Materials": "tents, blankets, relief for displaced people",
        "Other": "food and water, logistics and access, financial needs",
        "No Resource Need": "no resource need is reported or implied",
    },
}


# ════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════
def label_block(spaces):
    lines = []
    for task in ["CMC"] + TASKS:
        lines.append(f"{task}:")
        for lab in spaces[task]:
            d = DEFINITIONS.get(task, {}).get(lab, "")
            lines.append(f'  - "{lab}"' + (f": {d}" if d else ""))
    return "\n".join(lines)


def json_labels(indent="    "):
    return ",\n".join(f'{indent}"{t}": "<label>"' for t in TASKS)


SYSTEM = ("You are an expert annotator of disaster news for humanitarian response. You read a news "
          "headline (English or Hindi) together with the photo published with it. You answer only "
          "with valid JSON and use label names exactly as given.")


def build_prompt(mode, text, spaces):
    head = (f"News text:\n\"\"\"{text}\"\"\"\n\nThe attached image is the photo published with this article.\n\n"
            f"Label sets and definitions:\n{label_block(spaces)}\n\n"
            "Give three judgements for every task: from the TEXT ONLY (ignore the image), from the IMAGE "
            "ONLY (ignore the text), and COMBINED (the article as a whole). Also judge the cross-modal "
            "correlation (CMC) between image and text.\n\n")
    common = (f'  "text_only": {{\n{json_labels()}\n  }},\n'
              f'  "image_only": {{\n{json_labels()}\n  }},\n'
              '  "cmc": "<Strong | Partial | Low>",\n')
    if mode == "direct":
        return head + ("Answer immediately, without explanation, as JSON:\n{\n" + common +
                       f'  "combined": {{\n{json_labels()}\n  }}\n}}')
    if mode == "cot":
        return head + ("First think step by step in a field \"reasoning\" (at most 150 words, in English): "
                       "what the text says, what the image shows, whether they describe the same event, "
                       "and what follows for each task. Then answer. Output JSON:\n{\n"
                       '  "reasoning": "<your reasoning>",\n' + common +
                       f'  "combined": {{\n{json_labels()}\n  }}\n}}')
    if mode == "graph":
        comb = ",\n".join(f'    "{t}": {{"label": "<label>", "because": ["<node ids>"]}}' for t in TASKS)
        return head + (
            "Reason by building a small reasoning graph, written in English:\n"
            "1. text_evidence: 2-5 short facts stated in the text (ids t1, t2, ...).\n"
            "2. image_evidence: 2-5 short facts visible in the image (ids v1, v2, ...).\n"
            "3. links: connect text facts to image facts with relation \"same_event\" (the image shows exactly "
            "this), \"related\" (same kind of situation) or \"unrelated\". Use only existing ids.\n"
            "4. text_only / image_only: conclusions using ONLY text facts / ONLY image facts.\n"
            "5. cmc: must agree with your links (Strong needs a same_event link; Low has no same_event link).\n"
            "6. combined: for each task a label and the evidence ids it rests on. If the combined label "
            "differs from the text-only label, cite the image evidence that changes it.\n"
            "Output JSON:\n{\n"
            '  "text_evidence": [{"id": "t1", "fact": "<fact>"}],\n'
            '  "image_evidence": [{"id": "v1", "fact": "<fact>"}],\n'
            '  "links": [{"text": "t1", "image": "v1", "relation": "<same_event | related | unrelated>"}],\n'
            + common + f'  "combined": {{\n{comb}\n  }}\n}}')
    raise ValueError(mode)


def encode_image(path, max_side):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224))
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages(prompt, image_url):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": prompt}]}]


# ════════════════════════════════════════════════════════════════════
# Backends
# ════════════════════════════════════════════════════════════════════
class VLLMBackend:
    def __init__(self, a):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                       limit_mm_per_prompt={"image": 1}, trust_remote_code=True, seed=a.seed)
        self.SP = SamplingParams

    def generate(self, convs, max_tokens):
        sp = self.SP(temperature=0.0, max_tokens=max_tokens)
        outs = self.llm.chat(convs, sp, use_tqdm=True)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


class HFBackend:
    """Plain Hugging Face transformers generation - slower than vLLM, but uses the same PyTorch
    that already trains CCEC on this machine, so it works with the installed GPU driver."""

    def __init__(self, a):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.bs = torch, a.hf_batch
        # Qwen takes one flat image list for a batch; Gemma 3, Llama 3.2 Vision, InternVL want one list per prompt
        self.nested = "qwen" not in a.model.lower()
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        tok = getattr(self.proc, "tokenizer", self.proc)
        tok.padding_side = "left"                                  # required for batched generation
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()

    @staticmethod
    def _decode_image(url):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")

    def _convert(self, conv):
        """OpenAI-style message (image as data URI) -> transformers chat format + PIL images."""
        msgs, imgs = [], []
        for m in conv:
            if isinstance(m["content"], str):
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
                continue
            content = []
            for c in m["content"]:
                if c["type"] == "image_url":
                    content.append({"type": "image"})
                    imgs.append(self._decode_image(c["image_url"]["url"]))
                else:
                    content.append({"type": "text", "text": c["text"]})
            msgs.append({"role": m["role"], "content": content})
        return msgs, imgs

    def generate(self, convs, max_tokens):
        results = []
        for b0 in range(0, len(convs), self.bs):
            texts, imgs = [], []
            for conv in convs[b0:b0 + self.bs]:
                msgs, im = self._convert(conv)
                texts.append(self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                if self.nested:
                    imgs.append(im)
                else:
                    imgs += im
            inp = self.proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                out = self.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
            gen = out[:, inp["input_ids"].shape[1]:]
            dec = self.proc.batch_decode(gen, skip_special_tokens=True)
            results += [(d, int((g != self.pad_id).sum())) for d, g in zip(dec, gen)]
            print(f"  generated {min(b0 + self.bs, len(convs))}/{len(convs)}", flush=True)
        return results


class OpenAIBackend:
    def __init__(self, a):
        import requests
        self.req, self.a = requests, a

    def _one(self, conv, max_tokens):
        r = self.req.post(f"{self.a.api_base.rstrip('/')}/chat/completions", timeout=600, json={
            "model": self.a.model, "messages": conv, "temperature": 0.0, "max_tokens": max_tokens})
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], j.get("usage", {}).get("completion_tokens", 0)

    def generate(self, convs, max_tokens):
        with ThreadPoolExecutor(self.a.workers) as ex:
            return list(ex.map(lambda c: self._one(c, max_tokens), convs))


class MockBackend:
    """Builds answers from the gold labels with noise. Tests the pipeline - never a result."""

    def __init__(self, a, rows, spaces):
        self.rows, self.spaces, self.rng = rows, spaces, np.random.default_rng(a.seed)

    def _lab(self, task, gold, p):
        return gold if self.rng.random() < p else str(self.rng.choice(self.spaces[task]))

    def generate(self, convs, max_tokens, mode=None, ids=None):
        out = []
        for i in ids:
            r = self.rows[i]
            d = {"text_only": {t: self._lab(t, r[f"{t}_Text"], .6) for t in TASKS},
                 "image_only": {t: self._lab(t, r[f"{t}_Image"], .5) for t in TASKS},
                 "cmc": self._lab("CMC", r["CMC"], .5)}
            if mode == "graph":
                d = {"text_evidence": [{"id": "t1", "fact": "x"}, {"id": "t2", "fact": "y"}],
                     "image_evidence": [{"id": "v1", "fact": "z"}],
                     "links": [{"text": "t1", "image": "v1",
                                "relation": {"Strong": "same_event", "Partial": "related"}.get(d["cmc"], "unrelated")}],
                     **d, "combined": {t: {"label": self._lab(t, r[f"{t}_Combined"], .6),
                                           "because": ["t1"] if self.rng.random() < .7 else ["t1", "v1"]}
                                       for t in TASKS}}
            else:
                d["combined"] = {t: self._lab(t, r[f"{t}_Combined"], .6) for t in TASKS}
                if mode == "cot":
                    d = {"reasoning": "mock reasoning", **d}
            txt = json.dumps(d, ensure_ascii=False)
            if self.rng.random() < 0.05:
                txt = "Sure! " + txt[:-3]                          # simulate a broken answer
            out.append((txt, len(txt) // 4))
        return out


# ════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════
def _balanced_blocks(text):
    """Yield (start, end) of every balanced {...} block, outermost first."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield start, i + 1
                    break
        start = text.find("{", start + 1)


def extract_json(text, required=("combined", "text_only", "cmc")):
    """The answer object: the first JSON block containing an expected top-level key.
    A truncated answer (e.g. max_tokens reached) is repaired by closing open brackets."""
    text = re.sub(r"```(?:json)?", "", text)
    for a, b in _balanced_blocks(text):
        try:
            d = json.loads(text[a:b])
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and any(k in d for k in required):
            return d
    start = text.find("{")                                    # repair a truncated answer
    if start != -1:
        frag = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"{}\[\]]*$', "", text[start:])   # drop a half-written field
        opens = []
        in_str = esc = False
        for ch in frag:
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                opens.append("}" if ch == "{" else "]")
            elif ch in "}]" and opens:
                opens.pop()
        try:
            d = json.loads(frag + ('"' if in_str else "") + "".join(reversed(opens)))
            if isinstance(d, dict) and any(k in d for k in required):
                return d
        except json.JSONDecodeError:
            pass
    return None


def normalise(value, labels):
    """Map a free-form label to the closest valid label (exact > case-insensitive > fuzzy)."""
    if isinstance(value, dict):
        value = value.get("label", "")
    if not isinstance(value, str) or not value.strip():
        return INVALID
    v = value.strip().strip('"').strip()
    if v in labels:
        return v
    low = {l.lower(): l for l in labels}
    if v.lower() in low:
        return low[v.lower()]
    for l in labels:                                   # "Flood" -> "Flood / Water Disaster"
        if l.lower().startswith(v.lower()) or v.lower().startswith(l.lower()):
            return l
    m = difflib.get_close_matches(v.lower(), list(low), n=1, cutoff=0.6)
    return low[m[0]] if m else INVALID


def parse(mode, raw, spaces):
    d = extract_json(raw)
    rec = {"parsed": d is not None}
    d = d or {}
    for view in ("text_only", "image_only", "combined"):
        block = d.get(view, {}) if isinstance(d.get(view), dict) else {}
        for t in TASKS:
            rec[f"{t}_{view}"] = normalise(block.get(t), spaces[t])
    rec["cmc"] = normalise(d.get("cmc"), spaces["CMC"])
    if mode == "graph":
        rec.update(graph_checks(d, rec))
    return rec


def graph_checks(d, rec):
    """Structural verification of a reasoning graph (independent of the gold labels)."""
    tids = {n.get("id") for n in d.get("text_evidence", []) if isinstance(n, dict)}
    vids = {n.get("id") for n in d.get("image_evidence", []) if isinstance(n, dict)}
    links = [l for l in d.get("links", []) if isinstance(l, dict)]
    rels = [l.get("relation") for l in links]
    valid_links = all(l.get("text") in tids and l.get("image") in vids for l in links) if links else False
    cmc = rec["cmc"]
    cmc_consistent = (("same_event" in rels) if cmc == "Strong" else
                      ("same_event" not in rels) if cmc == "Low" else True)
    comb = d.get("combined", {}) if isinstance(d.get("combined"), dict) else {}
    cites_ok, change_grounded, n_change = True, True, 0
    for t in TASKS:
        c = comb.get(t, {})
        because = c.get("because", []) if isinstance(c, dict) else []
        because = because if isinstance(because, list) else []
        if not because or any(b not in tids | vids for b in because):
            cites_ok = False
        if rec[f"{t}_combined"] != rec[f"{t}_text_only"] and rec[f"{t}_combined"] != INVALID:
            n_change += 1
            if not any(b in vids for b in because):
                change_grounded = False
    return {"g_n_text_nodes": len(tids), "g_n_image_nodes": len(vids), "g_n_links": len(links),
            "g_links_valid": valid_links, "g_cmc_consistent_with_links": cmc_consistent,
            "g_citations_valid": cites_ok, "g_changes_grounded_in_image": change_grounded,
            "g_n_label_changes": n_change}


# ════════════════════════════════════════════════════════════════════
# Verification against the annotations + metrics
# ════════════════════════════════════════════════════════════════════
def macro_f1(gold, pred, labels):
    return f1_score(gold, pred, labels=labels, average="macro", zero_division=0)


def evaluate(df, spaces):
    m = {"n": len(df), "parse_rate": float(df["parsed"].mean())}
    step = {"text_only": "Text", "image_only": "Image", "combined": "Combined"}
    for view, gv in step.items():
        f1s = []
        for t in TASKS:
            f = macro_f1(df[f"{t}_{gv}"], df[f"{t}_{view}"], spaces[t])
            m[f"{view}.{t}.macro_f1"] = f
            m[f"{view}.{t}.acc"] = float((df[f"{t}_{gv}"] == df[f"{t}_{view}"]).mean())
            f1s.append(f)
        m[f"{view}.avg_macro_f1"] = float(np.mean(f1s))
    m["cmc.macro_f1"] = macro_f1(df["CMC"], df["cmc"], spaces["CMC"])
    m["avg6_macro_f1"] = float(np.mean([m["cmc.macro_f1"]] + [m[f"combined.{t}.macro_f1"] for t in TASKS]))
    # step-wise verification: share of posts whose step is fully correct (all five tasks)
    for view, gv in step.items():
        ok = np.all([df[f"{t}_{gv}"] == df[f"{t}_{view}"] for t in TASKS], axis=0)
        m[f"verified.{view}_all5"] = float(ok.mean())
    m["verified.cmc"] = float((df["CMC"] == df["cmc"]).mean())
    ok_all = np.all([df[f"{t}_{gv}"] == df[f"{t}_{v}"] for t in TASKS for v, gv in step.items()]
                    + [df["CMC"] == df["cmc"]], axis=0)
    m["verified.all_steps"] = float(ok_all.mean())
    for c in [c for c in df.columns if c.startswith("g_")]:
        m[f"graph.{c[2:]}"] = float(df[c].astype(float).mean())
    for lang in sorted(df["lang"].unique()):
        s = df[df["lang"] == lang]
        m[f"lang={lang}.combined.avg_macro_f1"] = float(np.mean(
            [macro_f1(s[f"{t}_Combined"], s[f"{t}_combined"], spaces[t]) for t in TASKS]))
    return m


# ════════════════════════════════════════════════════════════════════
# Data: same validation split as train_ccec.py
# ════════════════════════════════════════════════════════════════════
def load_val_sample(a):
    import train_ccec as T                                     # same cleaning + split as training
    args = types.SimpleNamespace(text_col=a.text_col, image_col=a.image_col, keep_source=False,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=a.split_seed)
    df = T.load_dataframe(a.csv, args)
    spaces = T.build_label_spaces(df)
    if a.split == "all":                                   # every post: inputs for the VLM knowledge view
        return df.reset_index(drop=True), spaces
    _, val, test = T.split(df, args)
    val = test if a.split == "test" else val              # prompts were designed on val only
    # stratified by language x CMC, at least `min_cell` per cell so Low/Partial are represented
    cells = val.groupby(["lang", "CMC"])
    n_target = min(a.n, len(val))
    take = {k: max(min(len(g), a.min_cell), round(n_target * len(g) / len(val))) for k, g in cells}
    parts = [g.sample(min(len(g), take[k]), random_state=a.seed) for k, g in cells]
    sample = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    return sample, spaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", default="pilot_out")
    ap.add_argument("--modes", nargs="+", default=["direct", "cot", "graph"], choices=["direct", "cot", "graph"])
    ap.add_argument("--n", type=int, default=200, help="posts to sample; 0 = the whole split")
    ap.add_argument("--split", choices=["val", "test", "all"], default="val",
                    help="val while designing prompts; test ONCE for the final baseline numbers; "
                         "all = every post (zero-shot, no labels used) for the VLM knowledge view")
    ap.add_argument("--min_cell", type=int, default=8)
    ap.add_argument("--backend", choices=["vllm", "hf", "openai", "mock"], default="vllm")
    ap.add_argument("--hf_batch", type=int, default=8, help="batch size for the hf backend")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--image_max_side", type=int, default=512)
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--image_col", default="image_name")
    ap.add_argument("--split_seed", type=int, default=42, help="must match the training seed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.n <= 0:
        a.n = 10 ** 9
    sample, spaces = load_val_sample(a)
    print(f"pilot sample: {len(sample)} {a.split} posts | "
          f"languages {sample['lang'].value_counts().to_dict()} | CMC {sample['CMC'].value_counts().to_dict()}")
    sample.to_csv(os.path.join(a.out, "sample.csv"), index=False, encoding="utf-8-sig")
    images = [encode_image(os.path.join(a.image_dir, str(n)), a.image_max_side) for n in sample[a.image_col]]
    rows = sample.to_dict("records")

    backend = (VLLMBackend(a) if a.backend == "vllm" else HFBackend(a) if a.backend == "hf"
               else OpenAIBackend(a) if a.backend == "openai" else MockBackend(a, rows, spaces))
    max_tokens = {"direct": 400, "cot": 900, "graph": 1400}
    summary = []
    for mode in a.modes:
        mdir = os.path.join(a.out, mode)
        os.makedirs(mdir, exist_ok=True)
        raw_path = os.path.join(mdir, "raw.jsonl")
        done = {}
        if os.path.exists(raw_path):                              # resume
            for line in open(raw_path, encoding="utf-8"):
                r = json.loads(line)
                done[r["i"]] = r
        todo = [i for i in range(len(rows)) if i not in done]
        print(f"\n[{mode}] {len(done)} cached, {len(todo)} to generate")
        t0 = time.time()
        for b0 in range(0, len(todo), 64):                        # chunks: progress survives interruptions
            ids = todo[b0:b0 + 64]
            convs = [messages(build_prompt(mode, rows[i][a.text_col], spaces), images[i]) for i in ids]
            outs = (backend.generate(convs, max_tokens[mode], mode=mode, ids=ids) if a.backend == "mock"
                    else backend.generate(convs, max_tokens[mode]))
            with open(raw_path, "a", encoding="utf-8") as f:
                for i, (txt, ntok) in zip(ids, outs):
                    rec = {"i": i, "image_name": rows[i][a.image_col], "response": txt, "tokens": ntok}
                    done[i] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        gen_time = time.time() - t0

        recs = []
        for i in range(len(rows)):
            p = parse(mode, done[i]["response"], spaces)
            recs.append({**{k: rows[i][k] for k in [a.image_col, "lang", "CMC"] +
                            [f"{t}_{v}" for t in TASKS for v in ("Text", "Image", "Combined")]},
                         **p, "tokens": done[i]["tokens"]})
        pdf = pd.DataFrame(recs)
        pdf.to_csv(os.path.join(mdir, "predictions.csv"), index=False, encoding="utf-8-sig")
        m = evaluate(pdf, spaces)
        m.update({"mode": mode, "mean_output_tokens": float(pdf["tokens"].mean()),
                  "generation_s_this_session": round(gen_time, 1)})
        json.dump(m, open(os.path.join(mdir, "metrics.json"), "w"), indent=2)
        summary.append(m)

    cols = ["mode", "parse_rate", "combined.avg_macro_f1", "cmc.macro_f1", "avg6_macro_f1",
            "text_only.avg_macro_f1", "image_only.avg_macro_f1", "verified.text_only_all5",
            "verified.image_only_all5", "verified.cmc", "verified.combined_all5", "verified.all_steps",
            "graph.links_valid", "graph.cmc_consistent_with_links", "graph.citations_valid",
            "graph.changes_grounded_in_image", "mean_output_tokens"]
    s = pd.DataFrame(summary)
    s = s[[c for c in cols if c in s.columns] + [c for c in s.columns if c.startswith("lang=")]]
    s.to_csv(os.path.join(a.out, "pilot_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n" + s.round(4).T.to_string(header=False))
    print(f"\nwritten to {a.out}/")


if __name__ == "__main__":
    main()
