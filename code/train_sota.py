"""
train_sota.py - re-implementations of published multimodal crisis classifiers, trained and evaluated on
the FINAL IndicCrisis split with exactly the same data cleaning, labels, model selection and metrics as
train_ccec.py / train_baselines.py.

    mann      Gao et al., "Detecting Disaster-Related Tweets via Multimodal Adversarial Neural Network",
              IEEE MultiMedia 2020. BERT [CLS] + frozen VGG-19 -> FC(32) each -> concat -> detector
              FC64-FC32; adversarial discriminator (gradient reversal, lambda = 1).
    camm      Khattar & Quadri, "CAMM: Cross-Attention Multimodal Classification of Disaster-Related
              Tweets", IEEE Access 2022. VGG-16 conv map (7x7x512 -> 1x1 conv 1024, tanh) and a 2-layer
              BiLSTM over static word embeddings (1024 per word); text-to-image cross-attention,
              attended image features -> 3 FC layers.
    robertamft Yoon et al., "Multimedia analysis of robustly optimized multimodal transformer based on
              vision and language co-learning", Information Fusion 2023. ViT patch/CLS outputs become a
              multimodal class token and a distillation token inside the RoBERTa input sequence;
              score-level late fusion of the two output tokens.
    dmcc      Rezk et al., "Categorizing Crises From Social Media Feeds via Multimodal Channel
              Attention", IEEE Access 2023. EfficientNetV2 + BERT, projections D=1000 (FC-BN-ReLU),
              Multimodal Channel Attention (squeeze-and-excitation over the two modality channels) and
              score-level fusion y = y_V + y_T + y_MCA.
    clip / align  Mandal et al., "Contrastive Learning for Multimodal Classification of Crisis related
              Tweets", WWW 2024. Pre-trained CLIP (ViT-B/32) or ALIGN (kakaobrain/align-base) image and
              text embeddings, concatenated -> dense softmax layer (encoders frozen, as in the paper's
              linear fine-tuning; --finetune unfreezes them).

Adaptations to IndicCrisis (state them in the paper):
  * all models predict the five combined labels with one head per task (multi-task) instead of one task;
  * English-only text encoders are replaced by multilingual counterparts by default so that Hindi posts can
    be read (RoBERTa -> XLM-R, BERT -> mBERT, GloVe -> frozen mBERT word-piece embeddings);
    --faithful keeps the original English encoders. CLIP and ALIGN are used as published;
  * MANN's disaster-event discriminator has no event labels here; it discriminates the post language,
    i.e. it learns language-invariant features (crisis type cannot be used: it is a target task);
  * training protocol (10 epochs, AdamW, model selection on validation mean macro-F1, sqrt class weights)
    is the same for every model in the paper.

    python train_sota.py --model dmcc --csv IndicCrisis_clean.csv --image_dir "Merged Images" \
        --seed 42 --save_dir runs_v2/sota_dmcc_s42

Other datasets (CrisisMMD, see run_crisismmd.sh): --tasks, --split_col (fixed benchmark splits), --keep_source,
--select_metric, --faithful (English tweets: the original English encoders) and --adv_col event_name (MANN's
discriminator on the disaster event, as in the original paper).
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
MODELS = ["mann", "camm", "robertamft", "dmcc", "clip", "align"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--csv", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_text", type=float, default=2e-5, help="fine-tuned transformer encoders")
    p.add_argument("--lr_image", type=float, default=1e-4, help="fine-tuned CNN / ViT encoders")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--class_weight", choices=["none", "sqrt", "inv"], default="sqrt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--faithful", action="store_true", help="original English text encoders")
    p.add_argument("--finetune", action="store_true", help="clip/align: also fine-tune the encoders")
    p.add_argument("--l2norm", action="store_true",
                   help="clip/align: L2-normalise the image and the text embedding before concatenation")
    p.add_argument("--mock", action="store_true", help="tiny random encoders, no downloads (pipeline test)")
    # other datasets (CrisisMMD)
    p.add_argument("--tasks", default="Crisis,Severity,Informativeness,Humanitarian,Resource")
    p.add_argument("--split_col", default=None, help="column with train/val(dev)/test (fixed benchmark splits)")
    p.add_argument("--keep_source", action="store_true", help="do not strip ' - <outlet>' tails (tweets)")
    p.add_argument("--select_metric", default="avg_combined_macro_f1",
                   choices=["avg_combined_macro_f1", "avg_combined_weighted_f1", "avg_combined_acc"])
    p.add_argument("--adv_col", default="lang",
                   help="MANN: column the adversarial discriminator predicts (IndicCrisis: lang; CrisisMMD: event_name)")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def local_or(name, short):
    """Use a converted safetensors copy if present (torch < 2.6, see convert_to_safetensors.py)."""
    return f"{short}-safetensors" if os.path.isdir(f"{short}-safetensors") else name


TEXT_NAMES = {  # (multilingual default, faithful original)
    "bert": (("bert-base-multilingual-cased", "mbert"), ("bert-base-uncased", "bert")),
    "roberta": (("xlm-roberta-base", "xlmr"), ("roberta-base", "roberta")),
}


# ════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════
class PairDataset(Dataset):
    def __init__(self, df, a, spaces, img_tf):
        self.df, self.a, self.tf = df.reset_index(drop=True), a, img_tf
        self.c2i = {t: {c: i for i, c in enumerate(spaces[t])} for t in TASKS}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = Image.open(os.path.join(self.a.image_dir, str(r["image_name"]))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        x = self.tf(img) if self.tf else img
        y = {t: self.c2i[t][r[f"{t}_Combined"]] for t in TASKS}
        y["_lang"] = (1 if r["lang"] == "hi" else 0) if self.a.adv_col == "lang" else self.a.adv_map[r[self.a.adv_col]]
        return str(r["image_name"]), r["lang"], x, r["description"] or "no description", y


def collate(batch):
    names, langs, xs, texts, ys = zip(*batch)
    x = torch.stack(xs) if torch.is_tensor(xs[0]) else list(xs)
    return list(names), list(langs), x, list(texts), {k: torch.tensor([y[k] for y in ys]) for k in ys[0]}


def imagenet_tf(train, size=224):
    import torchvision.transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), norm])


def mock_tf(img):
    return torch.from_numpy(np.asarray(img.resize((32, 32)), dtype=np.float32)).permute(2, 0, 1) / 255.0


# ════════════════════════════════════════════════════════════════════
# Shared building blocks
# ════════════════════════════════════════════════════════════════════
class _MockTok:
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        ids = [[hash(w) % 297 + 3 for w in t.split()][:max_length] or [3] for t in texts]
        L = max(map(len, ids))
        return {"input_ids": torch.tensor([i + [1] * (L - len(i)) for i in ids]),
                "attention_mask": torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids])}


class TextTransformer(nn.Module):
    """HF encoder with access to word embeddings (for inputs_embeds / static-embedding use)."""

    def __init__(self, family, a, trainable=True):
        super().__init__()
        if a.mock:
            from transformers import XLMRobertaConfig, XLMRobertaModel
            self.tok = _MockTok()
            self.enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=300, hidden_size=48, num_hidden_layers=2,
                                                        num_attention_heads=2, intermediate_size=96))
        else:
            from transformers import AutoModel, AutoTokenizer
            name, short = TEXT_NAMES[family][1 if a.faithful else 0]
            name = local_or(name, short)
            print(f"text encoder: {name}")
            self.tok = AutoTokenizer.from_pretrained(name)
            self.enc = AutoModel.from_pretrained(name)
        self.max_len, self.dim = a.max_len, self.enc.config.hidden_size
        for p_ in self.enc.parameters():
            p_.requires_grad = trainable

    def batch(self, texts):
        dev = next(self.enc.parameters()).device
        b = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {k: v.to(dev) for k, v in b.items() if k in ("input_ids", "attention_mask")}

    def forward(self, texts):
        b = self.batch(texts)
        h = self.enc(**b).last_hidden_state
        return h, b["attention_mask"]

    def word_embeddings(self, input_ids):
        return self.enc.get_input_embeddings()(input_ids)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def heads(dim, spaces):
    return nn.ModuleDict({t: nn.Linear(dim, len(spaces[t])) for t in TASKS})


class MockCNN(nn.Module):
    """Stands in for a CNN backbone: returns a (B, C, 7, 7) map or a vector."""

    def __init__(self, c=16):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(7))

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════
# Models
# ════════════════════════════════════════════════════════════════════
class MANN(nn.Module):
    def __init__(self, a, spaces):
        super().__init__()
        self.text = TextTransformer("bert", a)
        if a.mock:
            self.vgg, vdim = nn.Sequential(MockCNN(), nn.Flatten()), 16 * 49
        else:
            import torchvision.models as M
            v = M.vgg19(weights=M.VGG19_Weights.IMAGENET1K_V1)
            v.classifier = nn.Sequential(*list(v.classifier.children())[:-1])    # 4096-d penultimate layer
            self.vgg, vdim = v, 4096
        for p_ in self.vgg.parameters():                                          # frozen, as in the paper
            p_.requires_grad = False
        self.fv = nn.Sequential(nn.Linear(vdim, 32), nn.ReLU())
        self.ft = nn.Sequential(nn.Linear(self.text.dim, 32), nn.ReLU())
        self.det = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5))
        self.out = heads(32, spaces)
        self.disc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, getattr(a, "n_adv", 2)))

    def forward(self, imgs, texts):
        h, _ = self.text(texts)
        with torch.no_grad():
            v = self.vgg(imgs)
        fm = torch.cat([self.fv(v), self.ft(h[:, 0])], dim=-1)
        z = self.det(fm)
        out = {t: self.out[t](z) for t in TASKS}
        out["_adv"] = self.disc(GradReverse.apply(fm, 1.0))
        return out


class CAMM(nn.Module):
    def __init__(self, a, spaces):
        super().__init__()
        self.emb = TextTransformer("bert", a, trainable=False)                   # static word embeddings
        edim = self.emb.dim
        if a.mock:
            self.cnn, cdim = MockCNN(), 16
        else:
            import torchvision.models as M
            self.cnn, cdim = M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1).features, 512   # (B,512,7,7)
        self.proj = nn.Conv2d(cdim, 1024, 1)
        self.lstm = nn.LSTM(edim, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.Wq, self.Wk, self.Wv = nn.Linear(1024, 1024), nn.Linear(1024, 1024), nn.Linear(1024, 1024)
        self.cls = nn.Sequential(nn.Linear(49 * 1024, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU())
        self.out = heads(256, spaces)

    def forward(self, imgs, texts):
        b = self.emb.batch(texts)
        with torch.no_grad():
            e = self.emb.word_embeddings(b["input_ids"])
        ft, _ = self.lstm(e)                                                     # (B, n, 1024)
        fi = torch.tanh(self.proj(self.cnn(imgs))).flatten(2).transpose(1, 2)    # (B, 49, 1024)
        s = self.Wq(fi) @ self.Wk(ft).transpose(1, 2) / math.sqrt(1024)          # (B, 49, n)
        s = s.masked_fill(b["attention_mask"][:, None, :] == 0, -1e4)
        # each word attends over the spatial regions; average the words' attention -> region weights
        att = torch.softmax(s, dim=1)
        m = b["attention_mask"][:, None, :].to(att.dtype)
        w = (att * m).sum(-1) / m.sum(-1).clamp_min(1)                           # (B, 49)
        fca = w.unsqueeze(-1) * self.Wv(fi) * 49                                 # attended image features
        z = self.cls(fca.flatten(1))
        return {t: self.out[t](z) for t in TASKS}


class RoBERTaMFT(nn.Module):
    def __init__(self, a, spaces):
        super().__init__()
        self.text = TextTransformer("roberta", a)
        d = self.text.dim
        if a.mock:
            self.vit, vdim = None, 16
            self.mock_vit = MockCNN()
        else:
            import torchvision.models as M
            self.vit, vdim = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1), 768
        self.m_class = nn.Linear(vdim, d)     # multimodal class token from the ViT patch outputs, Eq. (5)
        self.m_distill = nn.Linear(vdim, d)   # distillation token from the ViT class token, Eq. (5)
        self.out_c, self.out_d = heads(d, spaces), heads(d, spaces)

    def vit_tokens(self, imgs):
        if self.vit is None:
            m = self.mock_vit(imgs).flatten(2).transpose(1, 2)                  # (B, 49, 16)
            return m.mean(1), m
        v = self.vit
        x = v._process_input(imgs)
        x = torch.cat([v.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = v.encoder(x)                                                         # (B, 197, 768)
        return x[:, 0], x[:, 1:]

    def forward(self, imgs, texts):
        cls_v, patches = self.vit_tokens(imgs)
        b = self.text.batch(texts)
        e = self.text.word_embeddings(b["input_ids"])
        mc = self.m_class(patches.mean(1)).unsqueeze(1)
        md = self.m_distill(cls_v).unsqueeze(1)
        x = torch.cat([mc, e, md], dim=1)
        ones = torch.ones(e.shape[0], 1, dtype=b["attention_mask"].dtype, device=e.device)
        mask = torch.cat([ones, b["attention_mask"], ones], dim=1)
        h = self.text.enc(inputs_embeds=x, attention_mask=mask).last_hidden_state
        hc, hd = h[:, 0], h[:, -1]           # class token first, distillation token always last (after padding)
        out = {}
        for t in TASKS:
            out[f"{t}__c"], out[f"{t}__d"] = self.out_c[t](hc), self.out_d[t](hd)
            # score-level late fusion: mean of the two class-probability vectors (Algorithm 1)
            out[t] = torch.log((F.softmax(out[f"{t}__c"], -1) + F.softmax(out[f"{t}__d"], -1)) / 2 + 1e-8)
        return out


class DMCC(nn.Module):
    def __init__(self, a, spaces, D=1000):
        super().__init__()
        self.text = TextTransformer("bert", a)
        if a.mock:
            self.img, vdim = nn.Sequential(MockCNN(), nn.Flatten()), 16 * 49
        else:
            import torchvision.models as M
            self.img, vdim = M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1), 1000
        proj = lambda i: nn.Sequential(nn.Linear(i, D), nn.BatchNorm1d(D), nn.ReLU())
        self.pv, self.pt = proj(vdim), proj(self.text.dim)
        self.hv, self.ht = heads(vdim, spaces), heads(self.text.dim, spaces)
        self.se = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.Sigmoid())   # MCA, Eq. (6)-(8)
        self.hm = heads(2 * D, spaces)

    def forward(self, imgs, texts):
        v = self.img(imgs)
        h, _ = self.text(texts)
        tvec = h[:, 0]
        f = torch.stack([self.pv(v), self.pt(tvec)], dim=-1)                     # (B, D, 2) = channel concat
        alpha = self.se(f.mean(1))                                               # GAP over D -> (B, 2)
        fa = (f * alpha.unsqueeze(1)).flatten(1)
        out = {}
        for t in TASKS:
            out[f"{t}__v"], out[f"{t}__t"], out[f"{t}__m"] = self.hv[t](v), self.ht[t](tvec), self.hm[t](fa)
            out[t] = out[f"{t}__v"] + out[f"{t}__t"] + out[f"{t}__m"]            # score-level fusion, Eq. (9)
        return out


class ContrastiveLinear(nn.Module):
    """Mandal et al.: frozen CLIP / ALIGN embeddings, concatenated, dense softmax layer."""

    def __init__(self, a, spaces, kind):
        super().__init__()
        self.mock, self.ft, self.l2 = a.mock, a.finetune, getattr(a, "l2norm", False)
        if a.mock:
            self.mi, self.mt, dim = nn.Sequential(MockCNN(), nn.Flatten(), nn.Linear(16 * 49, 32)), nn.EmbeddingBag(301, 32), 32
        else:
            from transformers import AlignModel, AlignProcessor, CLIPModel, CLIPProcessor
            if kind == "clip":
                name, Mdl, Proc = "openai/clip-vit-base-patch32", CLIPModel, CLIPProcessor
            else:
                name, Mdl, Proc = "kakaobrain/align-base", AlignModel, AlignProcessor
            name = local_or(name, kind)       # align-safetensors / clip-safetensors if converted (torch < 2.6)
            print(f"{kind} model: {name}")
            self.proc, self.m = Proc.from_pretrained(name), Mdl.from_pretrained(name)
            dim = self.m.config.projection_dim if kind == "clip" else self.m.config.projection_dim
            for p_ in self.m.parameters():
                p_.requires_grad = self.ft
        self.out = heads(2 * dim, spaces)

    def forward(self, imgs, texts):
        if self.mock:
            dev = next(self.mi.parameters()).device
            ids = [torch.tensor([hash(w) % 300 + 1 for w in s.split()] or [0]) for s in texts]
            off = torch.tensor([0] + [len(i) for i in ids[:-1]]).cumsum(0)
            z = torch.cat([self.mi(imgs), self.mt(torch.cat(ids).to(dev), off.to(dev))], -1)
        else:
            dev = next(self.m.parameters()).device
            inp = self.proc(text=list(texts), images=imgs, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=64 if "Align" in type(self.m).__name__ else 77).to(dev)
            with torch.set_grad_enabled(self.ft and self.training):
                zi = self.m.get_image_features(pixel_values=inp["pixel_values"])
                zt = self.m.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            if self.l2:                       # embeddings live on the unit sphere in CLIP / ALIGN training
                zi, zt = F.normalize(zi.float(), dim=-1), F.normalize(zt.float(), dim=-1)
            z = torch.cat([zi, zt], -1).float()
        return {t: self.out[t](z) for t in TASKS}


def build(a, spaces):
    if a.model == "mann":
        return MANN(a, spaces)
    if a.model == "camm":
        return CAMM(a, spaces)
    if a.model == "robertamft":
        return RoBERTaMFT(a, spaces)
    if a.model == "dmcc":
        return DMCC(a, spaces)
    return ContrastiveLinear(a, spaces, a.model)


def loss_fn(a, out, y, w):
    if a.model == "robertamft":       # train both token classifiers; predict with the fused scores
        loss = sum(F.cross_entropy(out[f"{t}__{k}"].float(), y[t], weight=w[t]) for t in TASKS for k in "cd")
    elif a.model == "dmcc":           # each branch and the fused score are supervised
        loss = sum(F.cross_entropy(out[f"{t}__{k}"].float(), y[t], weight=w[t]) for t in TASKS for k in "vtm")
        loss = loss + sum(F.cross_entropy(out[t].float(), y[t], weight=w[t]) for t in TASKS)
    else:
        loss = sum(F.cross_entropy(out[t].float(), y[t], weight=w[t]) for t in TASKS)
    if "_adv" in out:                 # MANN: discriminator loss, reversed for the features (Eq. 11-12)
        loss = loss + F.cross_entropy(out["_adv"].float(), y["_lang"])
    return loss


# ════════════════════════════════════════════════════════════════════
# Evaluation (same measures as train_baselines.py)
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, loader, device, full=False):
    model.eval()
    P, G, names, langs, t_fwd, n_fwd = {t: [] for t in TASKS}, {t: [] for t in TASKS}, [], [], 0.0, 0
    for b, (nm, lg, x, txt, y) in enumerate(loader):
        x = x.to(device) if torch.is_tensor(x) else x
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, txt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if b > 0:
            t_fwd += time.perf_counter() - t0
            n_fwd += len(nm)
        names += nm
        langs += lg
        for t in TASKS:
            P[t].append(out[t].float().argmax(-1).cpu())
            G[t].append(y[t])
    P = {t: torch.cat(v).numpy() for t, v in P.items()}
    G = {t: torch.cat(v).numpy() for t, v in G.items()}
    return (P, G, names, np.array(langs), 1000 * t_fwd / max(n_fwd, 1)) if full else (P, G)


def score(P, G, spaces, mask=None):
    r = {}
    for t in TASKS:
        g, p = (G[t], P[t]) if mask is None else (G[t][mask], P[t][mask])
        r[f"{t}_Combined"] = {"macro_f1": f1_score(g, p, labels=list(range(len(spaces[t]))), average="macro",
                                                   zero_division=0), "accuracy": accuracy_score(g, p),
                              "weighted_f1": f1_score(g, p, average="weighted", zero_division=0)}
    r["avg_combined_macro_f1"] = float(np.mean([r[f"{t}_Combined"]["macro_f1"] for t in TASKS]))
    r["avg_combined_weighted_f1"] = float(np.mean([r[f"{t}_Combined"]["weighted_f1"] for t in TASKS]))
    r["avg_combined_acc"] = float(np.mean([r[f"{t}_Combined"]["accuracy"] for t in TASKS]))
    return r


def main():
    a = get_args()
    set_seed(a.seed)
    os.makedirs(os.path.join(a.save_dir, "report"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import train_ccec as T                                    # identical cleaning and split
    T.set_tasks(a.tasks)
    TASKS[:] = T.TASKS
    args = types.SimpleNamespace(text_col="description", image_col="image_name", keep_source=a.keep_source,
                                 allow_lost_text=False, test_size=0.10, val_size=0.10, seed=a.seed,
                                 split_col=a.split_col, cmc_optional=True)
    df = T.load_dataframe(a.csv, args)
    spaces = T.build_label_spaces(df)
    train_df, val_df, test_df = T.split(df, args)
    if a.adv_col != "lang":                                   # MANN's discriminator target (e.g. disaster event)
        if a.adv_col not in df.columns:
            raise SystemExit(f"--adv_col {a.adv_col!r} is not a column of the CSV")
        levels = sorted(df[a.adv_col].astype(str).unique())
        a.adv_map, a.n_adv = {v: i for i, v in enumerate(levels)}, len(levels)
        df[a.adv_col] = df[a.adv_col].astype(str)
        for d_ in (train_df, val_df, test_df):
            d_[a.adv_col] = d_[a.adv_col].astype(str)
        if a.model == "mann":
            print(f"MANN discriminator: {a.adv_col} ({a.n_adv} classes)")
    print(f"{a.model} | train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    contrastive = a.model in ("clip", "align") and not a.mock
    tf_train = mock_tf if a.mock else (None if contrastive else imagenet_tf(True))
    tf_eval = mock_tf if a.mock else (None if contrastive else imagenet_tf(False))
    mk = lambda d, tr, tf: DataLoader(PairDataset(d, a, spaces, tf), batch_size=a.batch_size, shuffle=tr,
                                      num_workers=a.num_workers, collate_fn=collate,
                                      drop_last=tr)       # BatchNorm (DMCC) cannot train on a batch of one
    train_dl, val_dl, test_dl = mk(train_df, True, tf_train), mk(val_df, False, tf_eval), mk(test_df, False, tf_eval)
    cw = T.class_weights(train_df, spaces, a.class_weight)
    w = {t: (cw[f"{t}_Combined"].to(device) if cw.get(f"{t}_Combined") is not None else None) for t in TASKS}

    model = build(a, spaces).to(device)
    text_p, img_p, head_p = [], [], []
    for n, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if n.startswith(("text.", "emb.", "m.text_model", "m.text_encoder")):
            text_p.append(p_)
        elif n.startswith(("vgg.", "cnn.", "vit.", "img.", "m.vision_model", "m.vision_encoder")):
            img_p.append(p_)
        else:
            head_p.append(p_)
    groups = [g for g in ({"params": text_p, "lr": a.lr_text}, {"params": img_p, "lr": a.lr_image},
                          {"params": head_p, "lr": a.lr_head}) if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    total = a.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    n_par = sum(p_.numel() for p_ in model.parameters())
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"parameters: total {n_par / 1e6:.1f}M | trainable {n_train / 1e6:.1f}M")
    ckpt = os.path.join(a.save_dir, "best.pt")
    best, hist, t_start = -1.0, [], time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for _, _, x, txt, y in train_dl:
            x = x.to(device) if torch.is_tensor(x) else x
            y = {k: v.to(device) for k, v in y.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, txt)
            loss = loss_fn(a, out, y, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        v = score(*predict(model, val_dl, device), spaces)[a.select_metric]
        hist.append({"epoch": ep, "loss": run / len(train_dl), f"val_{a.select_metric}": v})
        print(f"epoch {ep:02d} | loss {run / len(train_dl):.4f} | val {a.select_metric} {v:.4f} | {time.time() - t0:.0f}s")
        if v > best:
            best = v
            torch.save({k: v_ for k, v_ in model.state_dict().items()}, ckpt)
    train_min = (time.time() - t_start) / 60

    model.load_state_dict(torch.load(ckpt, map_location=device))
    P, G, names, langs, ms = predict(model, test_dl, device, full=True)
    res = score(P, G, spaces)
    rows = [{"field": k, **v} for k, v in res.items() if isinstance(v, dict)]
    rows.append({"field": "AVERAGE (combined)", "macro_f1": res["avg_combined_macro_f1"]})
    pd.DataFrame(rows).to_csv(os.path.join(a.save_dir, "report", "metrics_summary.csv"), index=False)
    lang_rows = []
    for lg in sorted(set(langs)):
        r = score(P, G, spaces, mask=langs == lg)
        lang_rows.append({"language": lg, "avg_combined_macro_f1": r["avg_combined_macro_f1"],
                          **{f"{t}_Combined": r[f"{t}_Combined"]["macro_f1"] for t in TASKS}})
    pred = pd.DataFrame({"image_name": names, "language": langs})
    for t in TASKS:
        pred[f"{t}_Combined_true"] = [spaces[t][i] for i in G[t]]
        pred[f"{t}_Combined_pred"] = [spaces[t][i] for i in P[t]]
    pred.to_csv(os.path.join(a.save_dir, "report", "predictions.csv"), index=False, encoding="utf-8-sig")
    eff = {"total_parameters": n_par, "trainable_parameters": n_train, "training_time_min": round(train_min, 2),
           "inference_ms_per_sample": round(ms, 3), "test_samples": len(names),
           "peak_gpu_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None}
    json.dump({"args": vars(a), "test": res, "history": hist, "test_lang": lang_rows, "efficiency": eff},
              open(os.path.join(a.save_dir, "results.json"), "w"), indent=2, default=float)
    os.remove(ckpt)
    print(f"\nTEST {a.model}: avg macro-F1 (combined) {res['avg_combined_macro_f1']:.4f} | "
          + " | ".join(f"{t} {res[f'{t}_Combined']['macro_f1']:.3f}" for t in TASKS))


if __name__ == "__main__":
    main()
