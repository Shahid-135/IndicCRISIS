"""
train_baselines.py - standard single-modality baselines on the FINAL IndicCrisis split.

Uses exactly the same data cleaning, label spaces and group-aware split as train_ccec.py (it imports
them), so the results are directly comparable with CCEC-G and the zero-shot VLM.

    text-only : mbert, indicbert, muril, xlmr        (fine-tuned transformer, mean pooling)
    image-only: vgg19, resnet50, vit_b16, convnext   (fine-tuned ImageNet backbone)

Each model has one shared encoder and two heads per task:
    * combined head  -> the article-level label  (main comparison, Avg_5 in the paper)
    * own-view head  -> the text label (text models) / the image label (image models)
Training loss = class-weighted cross-entropy summed over all heads; model selection on the validation
mean macro-F1 of the combined heads.

Outputs (same layout as train_ccec.py, so aggregate_runs.py can read them):
    <save_dir>/results.json, <save_dir>/report/metrics_summary.csv, per_language.csv, predictions.csv

    python train_baselines.py --model muril --csv IndicCrisis_clean.csv --image_dir "Merged Images" \
        --seed 42 --save_dir runs_v2/txt_muril_s42
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
TEXT_MODELS = {"mbert": "bert-base-multilingual-cased", "indicbert": "ai4bharat/indic-bert",
               "muril": "google/muril-base-cased", "xlmr": "xlm-roberta-base"}
IMAGE_MODELS = ["vgg19", "resnet50", "vit_b16", "convnext"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(TEXT_MODELS) + IMAGE_MODELS)
    p.add_argument("--csv", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="encoder learning rate (default: 2e-5 text, 1e-4 image)")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class_weight", choices=["none", "sqrt", "inv"], default="sqrt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hf_model", default=None, help="override the Hugging Face checkpoint of a text model")
    p.add_argument("--mock", action="store_true", help="tiny random encoders, no downloads (pipeline test only)")
    # other datasets (CrisisMMD)
    p.add_argument("--tasks", default="Crisis,Severity,Informativeness,Humanitarian,Resource")
    p.add_argument("--split_col", default=None, help="column with train/val(dev)/test (fixed benchmark splits)")
    p.add_argument("--keep_source", action="store_true", help="do not strip ' - <outlet>' tails (tweets)")
    p.add_argument("--select_metric", default="avg_combined_macro_f1",
                   choices=["avg_combined_macro_f1", "avg_combined_weighted_f1", "avg_combined_acc"])
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════
class BaselineDataset(Dataset):
    def __init__(self, df, a, spaces, modality, train):
        self.df, self.a, self.modality = df.reset_index(drop=True), a, modality
        self.c2i = {t: {c: i for i, c in enumerate(spaces[t])} for t in TASKS}
        self.view = "Text" if modality == "text" else "Image"
        self.tf = None
        if modality == "image" and not a.mock:
            import torchvision.transforms as T
            norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.tf = (T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
                       if train else T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = {}
        for t in TASKS:
            y[f"{t}_Combined"] = self.c2i[t][r[f"{t}_Combined"]]
            y[f"{t}_{self.view}"] = self.c2i[t][r[f"{t}_{self.view}"]]
        if self.modality == "text":
            x = r["description"] or "no description"
        else:
            try:
                img = Image.open(os.path.join(self.a.image_dir, str(r["image_name"]))).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            x = self.tf(img) if self.tf else torch.from_numpy(
                np.asarray(img.resize((32, 32)), dtype=np.float32)).permute(2, 0, 1) / 255.0
        return str(r["image_name"]), r["lang"], x, y


def collate(batch):
    names, langs, xs, ys = zip(*batch)
    x = list(xs) if isinstance(xs[0], str) else torch.stack(xs)
    return list(names), list(langs), x, {k: torch.tensor([y[k] for y in ys]) for k in ys[0]}


# ════════════════════════════════════════════════════════════════════
# Encoders
# ════════════════════════════════════════════════════════════════════
class TextEncoder(nn.Module):
    def __init__(self, a):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        name = a.hf_model or TEXT_MODELS[a.model]
        # converted copies (convert_to_safetensors.py / convert_indicbert.py) work with torch < 2.6
        for local in (f"{a.model}-safetensors", "indic-bert-safetensors" if a.model == "indicbert" else None):
            if not a.hf_model and local and os.path.isdir(local):
                name = local
                break
        print(f"text encoder: {name}")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name)
        self.max_len, self.dim = a.max_len, self.enc.config.hidden_size

    def forward(self, texts):
        dev = next(self.enc.parameters()).device
        b = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(dev)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1)                    # mean pooling


class ImageEncoder(nn.Module):
    def __init__(self, a):
        super().__init__()
        import torchvision.models as M
        if a.model == "vgg19":
            m = M.vgg19(weights=M.VGG19_Weights.IMAGENET1K_V1)
            self.dim = m.classifier[-1].in_features
            m.classifier[-1] = nn.Identity()
        elif a.model == "resnet50":
            m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
            self.dim = m.fc.in_features
            m.fc = nn.Identity()
        elif a.model == "vit_b16":
            m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
            self.dim = m.heads.head.in_features
            m.heads = nn.Identity()
        else:                                                            # convnext
            m = M.convnext_base(weights=M.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.dim = m.classifier[-1].in_features
            m.classifier[-1] = nn.Identity()
        self.enc = m

    def forward(self, x):
        return self.enc(x)


class MockEncoder(nn.Module):
    """Tiny encoders for testing the pipeline without downloads."""

    def __init__(self, modality):
        super().__init__()
        self.modality, self.dim = modality, 32
        if modality == "text":
            self.emb = nn.EmbeddingBag(5003, 32)
        else:
            self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(8, 32))

    def forward(self, x):
        if self.modality == "text":
            dev = self.emb.weight.device
            ids = [torch.tensor([hash(w) % 5003 for w in s.split()] or [0]) for s in x]
            off = torch.tensor([0] + [len(i) for i in ids[:-1]]).cumsum(0)
            return self.emb(torch.cat(ids).to(dev), off.to(dev))
        return self.cnn(x)


class Baseline(nn.Module):
    def __init__(self, a, spaces, modality):
        super().__init__()
        self.enc = MockEncoder(modality) if a.mock else (TextEncoder(a) if modality == "text" else ImageEncoder(a))
        self.view = "Text" if modality == "text" else "Image"
        d = self.enc.dim
        self.drop = nn.Dropout(a.dropout)
        self.heads = nn.ModuleDict()
        for t in TASKS:
            K = len(spaces[t])
            self.heads[f"{t}_Combined"] = nn.Linear(d, K)
            self.heads[f"{t}_{self.view}"] = nn.Linear(d, K)

    def forward(self, x):
        h = self.drop(self.enc(x))
        return {k: head(h) for k, head in self.heads.items()}


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, loader, device, full=False):
    model.eval()
    P, G, names, langs = {}, {}, [], []
    t_fwd, n_fwd = 0.0, 0
    for b, (nm, lg, x, y) in enumerate(loader):
        x = x if isinstance(x, list) else x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if b > 0:
            t_fwd += time.perf_counter() - t0
            n_fwd += len(nm)
        names += nm
        langs += lg
        for k, v in out.items():
            P.setdefault(k, []).append(v.float().argmax(-1).cpu())
            G.setdefault(k, []).append(y[k])
    P = {k: torch.cat(v).numpy() for k, v in P.items()}
    G = {k: torch.cat(v).numpy() for k, v in G.items()}
    if not full:
        return P, G
    return P, G, names, np.array(langs), 1000 * t_fwd / max(n_fwd, 1)


def score(P, G, spaces, mask=None):
    res = {}
    for k in P:
        t = k.rsplit("_", 1)[0]
        g, p = (G[k], P[k]) if mask is None else (G[k][mask], P[k][mask])
        res[k] = {"macro_f1": f1_score(g, p, labels=list(range(len(spaces[t]))), average="macro", zero_division=0),
                  "weighted_f1": f1_score(g, p, average="weighted", zero_division=0),
                  "accuracy": accuracy_score(g, p)}
    res["avg_combined_macro_f1"] = float(np.mean([res[f"{t}_Combined"]["macro_f1"] for t in TASKS]))
    res["avg_combined_weighted_f1"] = float(np.mean([res[f"{t}_Combined"]["weighted_f1"] for t in TASKS]))
    res["avg_combined_acc"] = float(np.mean([res[f"{t}_Combined"]["accuracy"] for t in TASKS]))
    return res


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    a = get_args()
    set_seed(a.seed)
    os.makedirs(os.path.join(a.save_dir, "report"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality = "text" if a.model in TEXT_MODELS else "image"
    a.lr = a.lr or (2e-5 if modality == "text" else 1e-4)

    import types
    import train_ccec as T                                  # identical cleaning and split
    T.set_tasks(a.tasks)
    TASKS[:] = T.TASKS
    args = types.SimpleNamespace(text_col="description", image_col="image_name", keep_source=a.keep_source,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=a.seed,
                                 split_col=a.split_col, cmc_optional=True)
    df = T.load_dataframe(a.csv, args)
    spaces = T.build_label_spaces(df)
    train_df, val_df, test_df = T.split(df, args)
    print(f"{a.model} ({modality}) | train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    mk = lambda d, tr: DataLoader(BaselineDataset(d, a, spaces, modality, tr), batch_size=a.batch_size, shuffle=tr,
                                  num_workers=a.num_workers, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(train_df, True), mk(val_df, False), mk(test_df, False)
    w = {k: (v.to(device) if v is not None else None) for k, v in T.class_weights(train_df, spaces, a.class_weight).items()}

    model = Baseline(a, spaces, modality).to(device)
    enc_p = list(model.enc.parameters())
    head_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.lr_head}],
                            weight_decay=a.weight_decay)
    total = a.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    n_par = sum(p.numel() for p in model.parameters())
    ckpt = os.path.join(a.save_dir, "best.pt")
    best, hist, t_start = -1.0, [], time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for _, _, x, y in train_dl:
            x = x if isinstance(x, list) else x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            loss = sum(F.cross_entropy(out[k].float(), y[k], weight=w[k]) for k in out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        v = score(*predict(model, val_dl, device), spaces)[a.select_metric]
        hist.append({"epoch": ep, "loss": run / len(train_dl), f"val_{a.select_metric}": v})
        print(f"epoch {ep:02d} | loss {run / len(train_dl):.4f} | val {a.select_metric} {v:.4f} | {time.time() - t0:.0f}s")
        if v > best:
            best = v
            torch.save(model.state_dict(), ckpt)
    train_min = (time.time() - t_start) / 60

    model.load_state_dict(torch.load(ckpt, map_location=device))
    P, G, names, langs, ms = predict(model, test_dl, device, full=True)
    res = score(P, G, spaces)
    view = "Text" if modality == "text" else "Image"
    rows = [{"field": k, **v} for k, v in res.items() if isinstance(v, dict)]
    rows.append({"field": "AVERAGE (combined)", "macro_f1": res["avg_combined_macro_f1"]})
    pd.DataFrame(rows).to_csv(os.path.join(a.save_dir, "report", "metrics_summary.csv"), index=False)
    lang_rows = []
    for lg in sorted(set(langs)):
        r = score(P, G, spaces, mask=langs == lg)
        lang_rows.append({"language": lg, "avg_combined_macro_f1": r["avg_combined_macro_f1"],
                          **{f"{t}_Combined": r[f"{t}_Combined"]["macro_f1"] for t in TASKS}})
    pd.DataFrame(lang_rows).to_csv(os.path.join(a.save_dir, "report", "per_language.csv"), index=False)
    pred = pd.DataFrame({"image_name": names, "language": langs})
    for k in P:
        t = k.rsplit("_", 1)[0]
        pred[f"{k}_true"] = [spaces[t][i] for i in G[k]]
        pred[f"{k}_pred"] = [spaces[t][i] for i in P[k]]
    pred.to_csv(os.path.join(a.save_dir, "report", "predictions.csv"), index=False, encoding="utf-8-sig")
    eff = {"total_parameters": n_par, "trainable_parameters": n_par, "training_time_min": round(train_min, 2),
           "inference_ms_per_sample": round(ms, 3), "test_samples": len(names),
           "peak_gpu_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None}
    json.dump({"args": vars(a), "modality": modality, "own_view": view, "test": res, "history": hist,
               "test_lang": lang_rows, "efficiency": eff},
              open(os.path.join(a.save_dir, "results.json"), "w"), indent=2, default=float)
    os.remove(ckpt)                                          # baselines are large; keep only the results
    print(f"\nTEST {a.model}: avg macro-F1 (combined) {res['avg_combined_macro_f1']:.4f} | "
          + " | ".join(f"{t} {res[f'{t}_Combined']['macro_f1']:.3f}" for t in TASKS))
    print(f"own-view ({view}) avg macro-F1: {np.mean([res[f'{t}_{view}']['macro_f1'] for t in TASKS]):.4f}")


if __name__ == "__main__":
    main()
