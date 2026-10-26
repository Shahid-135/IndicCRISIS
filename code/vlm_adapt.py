"""
vlm_adapt.py - Qwen2.5-VL beyond zero-shot: few-shot prompting with retrieved examples and LoRA
fine-tuning. Uses EXACTLY the zero-shot "direct" prompt, JSON format, parser, metrics and test posts of
pilot_reasoning_graph.py, so the rows are directly comparable with the zero-shot results.

Stages (run them one after the other; every stage is resumable):
    index     frozen SigLIP-2 text+image embeddings of all posts  -> <out>/retrieval_index.npz
    fewshot   k retrieved training posts as demonstrations         -> <out>/fewshot/{raw.jsonl,predictions.csv,metrics.json}
    train     LoRA on the language model (vision encoder frozen)    -> <out>/lora_adapter/
    finetuned inference with the adapter                            -> <out>/finetuned/{...}

    python vlm_adapt.py --stage index     --csv IndicCrisis_clean.csv --image_dir "Merged Images" --out pilot_test
    python vlm_adapt.py --stage fewshot   --csv ... --image_dir ... --out pilot_test --k 4
    python vlm_adapt.py --stage train     --csv ... --image_dir ... --out pilot_test --epochs 2
    python vlm_adapt.py --stage finetuned --csv ... --image_dir ... --out pilot_test
    (or simply: bash run_vlm_all.sh)

Test posts: the full test split of seed 42 (same as the zero-shot run, --split test --n 0).
Training / demonstration posts: the training split of seed 42 only (never validation or test).
"""
import argparse
import json
import math
import os
import random
import time
import types

import numpy as np
import pandas as pd

import pilot_reasoning_graph as P   # prompts, parser, metrics of the zero-shot run

TASKS = P.TASKS


# ════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════
def load_splits(a):
    import train_ccec as T
    args = types.SimpleNamespace(text_col=a.text_col, image_col=a.image_col, keep_source=False,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=a.split_seed)
    df = T.load_dataframe(a.csv, args)
    spaces = T.build_label_spaces(df)
    train, _, _ = T.split(df, args)
    # identical test sample and order as the zero-shot run (pilot_reasoning_graph --split test --n 0)
    pa = types.SimpleNamespace(**{**vars(a), "split": "test", "n": 10 ** 9, "min_cell": 8, "seed": 0})
    test, _ = P.load_val_sample(pa)
    return train.reset_index(drop=True), test, spaces


def gold_json(row):
    """The annotated answer in exactly the JSON structure the prompt asks for."""
    block = lambda v: {t: row[f"{t}_{v}"] for t in TASKS}
    return json.dumps({"text_only": block("Text"), "image_only": block("Image"), "cmc": row["CMC"],
                       "combined": block("Combined")}, ensure_ascii=False)


def image_path(a, row):
    return os.path.join(a.image_dir, str(row[a.image_col]))


# ════════════════════════════════════════════════════════════════════
# Stage: retrieval index (frozen SigLIP-2)
# ════════════════════════════════════════════════════════════════════
def stage_index(a, train, test):
    path = os.path.join(a.out, "retrieval_index.npz")
    if os.path.exists(path):
        print(f"[index] {path} exists - skipping")
        return
    if a.index_from and os.path.exists(a.index_from):          # same posts, same SigLIP-2: just copy
        import shutil
        shutil.copy(a.index_from, path)
        print(f"[index] copied {a.index_from} -> {path}")
        return
    names = list(train[a.image_col]) + list(test[a.image_col])
    texts = list(train[a.text_col]) + list(test[a.text_col])
    if a.mock:
        rng = np.random.default_rng(0)
        ti, te = rng.normal(size=(len(names), 32)), rng.normal(size=(len(names), 32))
    else:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(a.siglip)
        model = AutoModel.from_pretrained(a.siglip).to(dev).eval()
        ti, te = [], []
        for b in range(0, len(names), 64):
            imgs = []
            for n in names[b:b + 64]:
                try:
                    imgs.append(Image.open(os.path.join(a.image_dir, str(n))).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224)))
            with torch.no_grad():
                pi = proc(images=imgs, return_tensors="pt").to(dev)
                pt = proc(text=[str(t) for t in texts[b:b + 64]], padding="max_length", truncation=True,
                          max_length=64, return_tensors="pt").to(dev)
                fi, ft = model.get_image_features(**pi), model.get_text_features(**pt)
                # newer transformers return an output object instead of a tensor
                fi = fi if torch.is_tensor(fi) else fi.pooler_output
                ft = ft if torch.is_tensor(ft) else ft.pooler_output
                ti.append(fi.float().cpu().numpy())
                te.append(ft.float().cpu().numpy())
            print(f"  embedded {min(b + 64, len(names))}/{len(names)}", flush=True)
        ti, te = np.concatenate(ti), np.concatenate(te)
    norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    np.savez(path, names=np.array(names, dtype=object), image=norm(ti).astype(np.float32),
             text=norm(te).astype(np.float32), n_train=len(train))
    print(f"[index] wrote {path}")


def retrieve(a, train, test, k):
    z = np.load(os.path.join(a.out, "retrieval_index.npz"), allow_pickle=True)
    n = int(z["n_train"])
    sim = 0.5 * (z["text"][n:] @ z["text"][:n].T) + 0.5 * (z["image"][n:] @ z["image"][:n].T)
    # never use a demonstration whose image is a near-duplicate of the test image: the label would leak
    sim[(z["image"][n:] @ z["image"][:n].T) > a.dup_threshold] = -9
    return np.argsort(-sim, axis=1)[:, :k]


# ════════════════════════════════════════════════════════════════════
# HF model helpers
# ════════════════════════════════════════════════════════════════════
def load_model(a, adapter=None, train=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = {"min_pixels": a.min_pixels, "max_pixels": a.max_pixels} if "qwen" in a.model.lower() else {}
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, **kw)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda",
                                                        trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    if not train:
        model.eval()
    return proc, model


def nested_images(a):
    """Qwen takes one flat image list for a batch; Gemma 3, Llama 3.2 Vision and InternVL want one list per prompt."""
    return "qwen" not in a.model.lower()


def to_chat(conv):
    """OpenAI-style messages (images as data URIs) -> transformers chat format + PIL images.
    Works for multi-turn conversations (few-shot demonstrations with assistant answers)."""
    msgs, imgs = [], []
    for m in conv:
        if isinstance(m["content"], str):
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
            continue
        content = []
        for c in m["content"]:
            if c["type"] == "image_url":
                content.append({"type": "image"})
                imgs.append(P.HFBackend._decode_image(c["image_url"]["url"]))
            else:
                content.append({"type": "text", "text": c["text"]})
        msgs.append({"role": m["role"], "content": content})
    return msgs, imgs


def generate(a, proc, model, convs, max_tokens):
    import torch
    tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    texts, imgs = [], []
    for conv in convs:
        msgs, im = to_chat(conv)
        texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        if nested_images(a):
            imgs.append(im)
        else:
            imgs += im
    inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    gen = out[:, inp["input_ids"].shape[1]:]
    return [(d, int((g != pad).sum())) for d, g in zip(proc.batch_decode(gen, skip_special_tokens=True), gen)]


# ════════════════════════════════════════════════════════════════════
# Inference stages (few-shot, fine-tuned) - same bookkeeping as the zero-shot script
# ════════════════════════════════════════════════════════════════════
def run_inference(a, mode, test, spaces, build_conv, proc=None, model=None, mock_rows=None):
    mdir = os.path.join(a.out, mode)
    os.makedirs(mdir, exist_ok=True)
    raw_path = os.path.join(mdir, "raw.jsonl")
    done = {}
    if os.path.exists(raw_path):
        for line in open(raw_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r
    rows = test.to_dict("records")
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"[{mode}] {len(done)} cached, {len(todo)} to generate")
    mock = P.MockBackend(a, rows, spaces) if a.mock else None
    t0 = time.time()
    for b0 in range(0, len(todo), a.batch):
        ids = todo[b0:b0 + a.batch]
        convs = [build_conv(i) for i in ids]
        outs = (mock.generate(convs, 400, mode="direct", ids=ids) if a.mock
                else generate(a, proc, model, convs, 400))
        with open(raw_path, "a", encoding="utf-8") as f:
            for i, (txt, ntok) in zip(ids, outs):
                rec = {"i": i, "image_name": rows[i][a.image_col], "response": txt, "tokens": ntok}
                done[i] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {mode}: {len(done)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    recs = []
    for i in range(len(rows)):
        p = P.parse("direct", done[i]["response"], spaces)
        recs.append({**{k: rows[i][k] for k in [a.image_col, "lang", "CMC"] +
                        [f"{t}_{v}" for t in TASKS for v in ("Text", "Image", "Combined")]},
                     **p, "tokens": done[i]["tokens"]})
    pdf = pd.DataFrame(recs)
    pdf.to_csv(os.path.join(mdir, "predictions.csv"), index=False, encoding="utf-8-sig")
    m = P.evaluate(pdf, spaces)
    m.update({"mode": mode, "mean_output_tokens": float(pdf["tokens"].mean())})
    json.dump(m, open(os.path.join(mdir, "metrics.json"), "w"), indent=2)
    print(f"[{mode}] combined Avg5 {100 * m['combined.avg_macro_f1']:.2f} | CMC {100 * m['cmc.macro_f1']:.2f} | "
          f"parse rate {m['parse_rate']:.3f}")


def single_image(a):
    """Models that accept only one image per prompt: few-shot demonstrations are given as text only."""
    return "llama" in a.model.lower() and "vision" in a.model.lower()


def stage_fewshot(a, train, test, spaces):
    demo_idx = retrieve(a, train, test, a.k)
    trows, rows = train.to_dict("records"), test.to_dict("records")
    small = lambda r: P.encode_image(image_path(a, r), a.demo_image_side)
    with open(os.path.join(a.out, "fewshot_demonstrations.json"), "w") as f:      # for inspection / the paper
        json.dump({rows[i][a.image_col]: [trows[j][a.image_col] for j in demo_idx[i]] for i in range(len(rows))}, f)

    def build_conv(i):
        conv = [{"role": "system", "content": P.SYSTEM}]
        for j in demo_idx[i]:
            d = trows[j]
            demo = [{"type": "text", "text": f'Example news text:\n"""{d[a.text_col]}"""\n'
                                             "Answer as JSON with text_only, image_only, cmc and combined."}]
            if not single_image(a):
                demo = [{"type": "image_url", "image_url": {"url": small(d)}}] + demo
            conv.append({"role": "user", "content": demo})
            conv.append({"role": "assistant", "content": gold_json(d)})
        # the test question is the unchanged zero-shot "direct" prompt
        conv += P.messages(P.build_prompt("direct", rows[i][a.text_col], spaces),
                           P.encode_image(image_path(a, rows[i]), a.image_max_side))[1:]
        return conv

    proc, model = (None, None) if a.mock else load_model(a)
    run_inference(a, "fewshot", test, spaces, build_conv, proc, model)


def stage_finetuned(a, test, spaces):
    adapter = os.path.join(a.out, "lora_adapter")
    if not a.mock and not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        raise SystemExit(f"no adapter in {adapter} - run --stage train first")
    rows = test.to_dict("records")

    def build_conv(i):
        return P.messages(P.build_prompt("direct", rows[i][a.text_col], spaces),
                          P.encode_image(image_path(a, rows[i]), a.image_max_side))

    proc, model = (None, None) if a.mock else load_model(a, adapter=adapter)
    run_inference(a, "finetuned", test, spaces, build_conv, proc, model)


# ════════════════════════════════════════════════════════════════════
# Stage: LoRA fine-tuning
# ════════════════════════════════════════════════════════════════════
# language model only: every attention / MLP projection outside the vision tower and the projector
# (Qwen: "visual", Gemma 3 / InternVL: "vision_tower", Llama 3.2: "vision_model", projectors: "multi_modal_projector")
LORA_TARGETS = (r"^(?!.*(visual|vision_tower|vision_model|multi_modal_projector|mm_projector)).*"
                r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


def build_example(a, proc, row, spaces):
    """input_ids / labels for one post: loss only on the assistant's JSON answer."""
    conv = P.messages(P.build_prompt("direct", row[a.text_col], spaces), P.encode_image(image_path(a, row), a.image_max_side))
    msgs, imgs = to_chat(conv)
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": gold_json(row)}]}],
                                    tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):                 # template differs from the generation prompt: fall back
        full = prompt + gold_json(row) + proc.tokenizer.eos_token
    im = [imgs] if nested_images(a) else imgs
    enc = proc(text=[full], images=im, return_tensors="pt")
    n_prompt = proc(text=[prompt], images=im, return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    enc["labels"] = labels
    return enc


def stage_train(a, train, spaces):
    import torch
    from peft import LoraConfig, get_peft_model
    adapter = os.path.join(a.out, "lora_adapter")
    if os.path.exists(os.path.join(adapter, "adapter_config.json")):
        print(f"[train] adapter exists in {adapter} - skipping (delete it to retrain)")
        return
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    proc, model = load_model(a, train=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                     # language model only: every attention / MLP projection outside the vision tower
                     target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    rows = train.to_dict("records")
    if a.max_train:
        rows = rows[:a.max_train]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(rows) * a.epochs / a.grad_accum)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log = open(os.path.join(a.out, "lora_train_log.txt"), "a")
    model.train()
    step, t0 = 0, time.time()
    for ep in range(1, a.epochs + 1):
        order = list(range(len(rows)))
        random.shuffle(order)
        run, n = 0.0, 0
        for k, idx in enumerate(order, 1):
            enc = build_example(a, proc, rows[idx], spaces).to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc).loss / a.grad_accum
            loss.backward()
            run += loss.item() * a.grad_accum
            n += 1
            if k % a.grad_accum == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    msg = (f"epoch {ep} step {step}/{steps} | loss {run / n:.4f} | "
                           f"{(time.time() - t0) / 60:.1f} min | lr {sched.get_last_lr()[0]:.2e}")
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    run, n = 0.0, 0
        model.save_pretrained(os.path.join(a.out, f"lora_adapter_epoch{ep}"))
    model.save_pretrained(adapter)
    print(f"[train] adapter saved to {adapter} ({(time.time() - t0) / 3600:.2f} h)")


# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["index", "fewshot", "train", "finetuned", "check"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", default="pilot_test", help="same folder as the zero-shot test run")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--k", type=int, default=4, help="few-shot demonstrations")
    ap.add_argument("--index_from", default="pilot_test/retrieval_index.npz", help="reuse this retrieval index")
    ap.add_argument("--dup_threshold", type=float, default=0.95, help="exclude near-duplicate images as demos")
    ap.add_argument("--batch", type=int, default=4, help="inference batch size")
    ap.add_argument("--image_max_side", type=int, default=512, help="as in the zero-shot run")
    ap.add_argument("--demo_image_side", type=int, default=336, help="smaller demonstration images")
    ap.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_train", type=int, default=0, help="debug: use only the first N training posts")
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--image_col", default="image_name")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="no model: test the pipeline with fake answers")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, test, spaces = load_splits(a)
    print(f"train {len(train)} posts (demonstrations / fine-tuning) | test {len(test)} posts")
    if a.stage == "check":          # the gold JSON must parse back to the gold labels
        recs = [{**{k: r[k] for k in ["lang", "CMC"] + [f"{t}_{v}" for t in TASKS for v in ("Text", "Image", "Combined")]},
                 **P.parse("direct", gold_json(r), spaces)} for r in test.to_dict("records")]
        m = P.evaluate(pd.DataFrame(recs), spaces)
        print(f"[check] gold answers re-parsed: parse rate {m['parse_rate']:.3f}, combined Avg5 "
              f"{100 * m['combined.avg_macro_f1']:.2f}, all steps verified {m['verified.all_steps']:.3f}")
    elif a.stage == "index":
        stage_index(a, train, test)
    elif a.stage == "fewshot":
        stage_index(a, train, test)
        stage_fewshot(a, train, test, spaces)
    elif a.stage == "train":
        stage_train(a, train, spaces)
    else:
        stage_finetuned(a, test, spaces)


if __name__ == "__main__":
    main()
