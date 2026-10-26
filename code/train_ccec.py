"""
CCEC: Correlation-Conditioned Evidence Composition
===================================================

One idea, one objective.

Annotators of this dataset gave three judgements per task: one for the text, one
for the image, and one for the combined post, plus a cross-modal correlation (CMC)
level.  The combined judgement is not an independent label; it is *composed* from
the two single-modality judgements, and HOW it is composed depends on how well the
image matches the text.

CCEC models exactly this annotation process:

    p(y_c | x) = sum_z q(z | x) * sum_{i,j} p_T(i | x_text) * p_V(j | x_image) * C_z[i, j, :]

    z      : CMC regime (Strong / Partial / Low), predicted from the pair
    p_T    : text-only judgement  (sees ONLY the text)
    p_V    : image-only judgement (sees ONLY the image)
    C_z    : a learned, interpretable composition tensor per task and regime,
             C_z[i, j, k] = p(combined = k | text says i, image says j, regime z)

Every annotation field is an observed variable of this one model, so training is
a single joint negative log-likelihood - no hand-tuned auxiliary-loss weights:

    L = -log q(z*) + sum_tasks [ -log p_T(y_T*) - log p_V(y_V*) - log p(y_c*) ]

An optional joint-evidence expert (product of experts) lets the combined judgement
use cross-modal evidence that neither single modality shows on its own.

Ablations (command-line flags) correspond one-to-one to the model's assumptions:
    --no_regimes       one composition tensor for all CMC levels  (is CMC needed?)
    --no_view_sup      text/image judgements are latent            (is tri-view supervision needed?)
    --no_residual      pure composition, no joint-evidence expert
    --fusion_baseline  replace composition by a standard fused classifier
At test time we also report an *oracle-CMC* score (true z plugged in) as an upper bound.

Correlation learning (CCEC-G): graph-contrastive CMC learning
-------------------------------------------------------------
The router q(z|x) needs a good estimate of how well the image matches the text.
CCEC-G learns an alignment space in which the matched-pair score s_i = cos(g_T(t_i), g_V(v_i))
feeds the router, trained by two terms that are both likelihoods of correlation evidence:

  * graph contrastive  (--corr graph): per batch, a graph over texts and images whose
    edges are  text_i -- image_i  weighted by CMC (Strong 1.0 / Partial 0.5 / Low 0.1) and
    image_i -- image_j  weighted by frozen SigLIP image similarity. Two-hop paths
    text_i -> image_i -> image_j give soft positives  W_ij = beta * a_i * a_j * sim(v_i, v_j),
    so similar trusted posts - also across languages - are pulled together, while
    low-correlation pairs barely count. Soft-target InfoNCE, rows weighted by a_i.
      --corr clip : uniform weights, no graph      (standard CLIP-style contrastive)
      --corr cmcw : CMC weights, no graph           (does CMC weighting help?)
  * ordinal CMC (--ordinal_cmc): matched-pair scores must respect
    Strong > Partial > Low > mismatched pair (in-batch negatives), with learned thresholds.
    Its level probabilities are multiplied into the router (product of experts).
  * --fine_align: the score also uses token-patch late interaction (each text token
    matched to its best image patch).
  * --nbr_sampler knn: batches contain nearest image neighbours, so two-hop edges exist.

Supervision ablations: --no_cmc_sup (CMC latent), --cmc_frac 0.25 (partial CMC labels).

Usage
-----
python train_ccec.py --csv Merged_csv.xlsx --image_dir "Merged Images" --save_dir runs/ccec
python train_ccec.py ... --no_regimes --save_dir runs/abl_no_regimes
"""

import argparse
import json
import math
import os
import random
import re
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from ccec_report import write_full_report

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
VIEWS = ["Text", "Image", "Combined"]
CMC_COL = "CMC"
CMC_IS_PLACEHOLDER = False      # True when the data has no CMC labels (e.g. CrisisMMD): never scored or reported


def set_tasks(names):
    """Select the annotated tasks (default: the five IndicCrisis tasks). Changes TASKS in place so that
    modules holding a reference to this list (train_baselines.py, train_sota.py) see the same tasks."""
    names = [n.strip() for n in (names.split(",") if isinstance(names, str) else names) if n.strip()]
    if not names:
        raise SystemExit("--tasks is empty")
    TASKS[:] = names


# ════════════════════════════════════════════════════════════════════
# Arguments
# ════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="CCEC training")
    # data
    p.add_argument("--csv", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--save_dir", default="runs/ccec")
    p.add_argument("--text_col", default="description")
    p.add_argument("--image_col", default="image_name")
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--keep_source", action="store_true",
                   help="keep the trailing ' - <news source>' in headlines (removed by default)")
    p.add_argument("--allow_lost_text", action="store_true",
                   help="run even if many texts are '????' (Hindi lost to an encoding error)")
    # encoder
    p.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--lora_rank", type=int, default=4, help="0 = frozen encoder")
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--text_encoder", choices=["siglip", "xlmr", "muril", "mbert", "indicbert"], default="siglip",
                   help="siglip = SigLIP-2 text tower (frozen + LoRA); otherwise a fully fine-tuned multilingual "
                        "text encoder is combined with the SigLIP-2 image tower (frozen + LoRA)")
    p.add_argument("--max_len_text", type=int, default=128, help="text tokens for --text_encoder != siglip")
    p.add_argument("--lr_text", type=float, default=2e-5, help="learning rate of a fine-tuned text encoder")
    p.add_argument("--delete_ckpt", action="store_true",
                   help="delete best_trainable.pt after testing (a fine-tuned XLM-R makes it ~1 GB per run)")
    p.add_argument("--mock_encoder", action="store_true", help="random features, for debugging")
    # model
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--comp_lr", type=float, default=None,
                   help="learning rate of the composition tables (default: --lr). The tables are small and "
                        "receive weak gradients through the marginalisation, so they need a larger rate to move")
    p.add_argument("--comp_init", type=float, default=2.0,
                   help="initial logit bonus for 'combined = text label' and 'combined = image label'")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--class_weight", choices=["none", "sqrt", "inv"], default="sqrt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # ablations
    p.add_argument("--no_regimes", action="store_true")
    p.add_argument("--no_view_sup", action="store_true")
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--fusion_baseline", action="store_true")
    # correlation learning (graph-contrastive CMC learning)
    p.add_argument("--corr", choices=["none", "clip", "cmcw", "graph"], default="none",
                   help="contrastive alignment: none | clip (uniform) | cmcw (CMC-weighted) | graph (CMC graph)")
    p.add_argument("--ordinal_cmc", action="store_true", help="ordinal CMC loss on the alignment score")
    p.add_argument("--fine_align", action="store_true", help="add token-patch late-interaction score")
    p.add_argument("--align_dim", type=int, default=256)
    p.add_argument("--cmc_weights", default="Low:0.1,Partial:0.5,Strong:1.0",
                   help="edge weight of a text-image pair for each CMC level")
    p.add_argument("--cmc_order", default="Low,Partial,Strong", help="CMC levels from weakest to strongest")
    p.add_argument("--beta", type=float, default=0.5, help="weight of two-hop (image-image) edges")
    p.add_argument("--graph_topk", type=int, default=3, help="image neighbours kept per node in the batch graph")
    p.add_argument("--graph_min_sim", type=float, default=0.6,
                   help="frozen image similarity below which no image-image edge is created")
    p.add_argument("--dup_sim", type=float, default=0.95,
                   help="image pairs above this frozen similarity are duplicates and never used as negatives")
    p.add_argument("--nbr_sampler", choices=["none", "knn", "random"], default="none",
                   help="build batches from anchors + nearest image neighbours (knn) or random partners")
    p.add_argument("--nbr_k", type=int, default=5)
    # supervision
    p.add_argument("--no_cmc_sup", action="store_true",
                   help="CMC labels are not used (router latent); data without a CMC column then gets placeholder "
                        "levels that are never supervised, scored or reported (CrisisMMD)")
    p.add_argument("--cmc_frac", type=float, default=1.0, help="fraction of training posts with CMC labels")
    # VLM knowledge view (precomputed by embed_vlm_views.py)
    p.add_argument("--vlm_view", default=None,
                   help="npz with frozen embeddings of zero-shot VLM reasoning per post; used ONLY by the "
                        "pair-level components (CMC router and joint-evidence expert), never by the text/image heads")
    # evaluation only
    # other datasets (CrisisMMD): tasks, fixed benchmark splits, model selection
    p.add_argument("--tasks", default="Crisis,Severity,Informativeness,Humanitarian,Resource",
                   help="comma-separated tasks; the CSV needs <task>_Text, <task>_Image, <task>_Combined")
    p.add_argument("--split_col", default=None,
                   help="column with train/val(dev)/test: use these fixed splits instead of the random group split")
    p.add_argument("--select_metric", default="avg_macro_f1",
                   choices=["avg_macro_f1", "avg_combined_macro_f1", "avg_combined_weighted_f1", "avg_combined_acc"],
                   help="validation metric for checkpoint selection and early stopping")
    p.add_argument("--eval_only", action="store_true", help="skip training, evaluate a saved checkpoint")
    p.add_argument("--ckpt", default=None, help="checkpoint for --eval_only (default: save_dir/best_trainable.pt)")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════
_C1 = {0x91: "'", 0x92: "'", 0x93: '"', 0x94: '"', 0x96: "-", 0x97: "-", 0x85: "..."}
_SEQ = {"â€œ": '"', "â€\x9d": '"', "â€™": "'", "â€˜": "'", "â€“": "-", "â€”": "-", "â€¦": "..."}
SOURCE_SUFFIX = re.compile(r"\s+[-|]\s+[^-|]{2,40}$")


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252 (e.g. "PradeshÂ’s" -> "Pradesh's")."""
    if not re.search("Â|â€|Ã", s):
        return s
    try:
        return s.encode("cp1252").decode("utf-8").translate(_C1)
    except (UnicodeEncodeError, UnicodeDecodeError):
        for bad, good in _SEQ.items():          # rows containing bytes cp1252 cannot encode
            s = s.replace(bad, good)
        return s


def lost_fraction(s: str) -> float:
    """Share of non-space characters that are '?' - Devanagari destroyed by a non-UTF-8 export."""
    chars = s.replace(" ", "")
    return chars.count("?") / max(1, len(chars))


def clean_text(s, strip_source: bool) -> str:
    s = "" if not isinstance(s, str) else " ".join(fix_mojibake(s).split())
    if strip_source:
        m = SOURCE_SUFFIX.search(s)
        # English headlines end in " - <news outlet>"; in Hindi rows the same pattern is real
        # content ("... - असम भीषण बाढ़ की चपेट में है."), so only strip Latin-script tails
        if m and not re.search(r"[\u0900-\u097F]", m.group(0)):
            s = s[:m.start()].strip()
    return s


def detect_language(s: str) -> str:
    if re.search(r"[\u0900-\u097F]", s):
        return "hi"
    return "hi_lost" if lost_fraction(s) > 0.3 else "en"


def check_language_confound(df: pd.DataFrame, thresh: float = 0.99):
    """Warn about classes that exist in only one language (e.g. 'Wildfire' only in English and
    'Fire' only in Hindi). Such classes are naming artefacts of merging two annotation files:
    a model learns the language instead of the event, and scores become meaningless."""
    if df["lang"].nunique() < 2:
        return
    found = []
    for c in [CMC_COL] + [f"{t}_{v}" for t in TASKS for v in VIEWS]:
        share = pd.crosstab(df[c], df["lang"], normalize="index")
        for cls, row in share.iterrows():
            if row.max() >= thresh and (df[c] == cls).sum() >= 20:
                found.append(f"{c}: '{cls}' is {row.max():.0%} {row.idxmax()}")
    if found:
        print("\nWARNING - classes that occur in only one language (merge the equivalent names):")
        for f_ in found:
            print("   ", f_)
        print()


def load_dataframe(path: str, args) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")        # handles the BOM Excel adds
    global CMC_IS_PLACEHOLDER
    if CMC_COL not in df.columns and getattr(args, "cmc_optional", False):
        levels = [c.strip() for c in getattr(args, "cmc_order", "Low,Partial,Strong").split(",")]
        df[CMC_COL] = [levels[i % len(levels)] for i in range(len(df))]
        CMC_IS_PLACEHOLDER = True
        print(f"note: no {CMC_COL} column - placeholder levels {levels} (latent regimes; never supervised, "
              "scored or reported)")
    label_cols = [CMC_COL] + [f"{t}_{v}" for t in TASKS for v in VIEWS]
    missing = [c for c in label_cols + [args.text_col, args.image_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    n0 = len(df)
    df = df.dropna(subset=label_cols + [args.image_col]).reset_index(drop=True)
    for c in label_cols:
        df[c] = df[c].astype(str).str.strip()
    df[args.text_col] = df[args.text_col].map(lambda s: clean_text(s, not args.keep_source))
    df["lang"] = df[args.text_col].map(detect_language)
    counts = df["lang"].value_counts().to_dict()
    print(f"Loaded {len(df)} samples ({n0 - len(df)} dropped for missing labels) | languages: {counts}")
    n_lost = counts.get("hi_lost", 0)
    if n_lost > 0.05 * len(df) and not args.allow_lost_text:
        raise SystemExit(
            f"\nERROR: {n_lost} texts consist mostly of '?' - the Hindi (Devanagari) text was lost when the\n"
            "file was saved in a non-UTF-8 encoding. Re-export the data from the original source as .xlsx\n"
            "or as CSV with UTF-8 encoding, then rerun. (Use --allow_lost_text only for debugging.)")
    check_language_confound(df)
    # headlines that are identical after cleaning form one group; a group never crosses splits
    df["group"] = df[args.text_col].str.lower().factorize()[0]
    return df


def build_label_spaces(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Text, image and combined views of a task share ONE label space
    (required so that the composition tensor C[i, j, k] is well defined)."""
    spaces = {CMC_COL: sorted(df[CMC_COL].unique())}
    for t in TASKS:
        per_view = {v: set(df[f"{t}_{v}"].unique()) for v in VIEWS}
        union = sorted(set.union(*per_view.values()))
        for v in VIEWS:
            absent = set(union) - per_view[v]
            if absent:
                print(f"  note: {t}_{v} never uses {sorted(absent)}")
        spaces[t] = union
    return spaces


def _group_holdout(df, frac, seed):
    """One fold of a stratified *group* split: ~frac of the data, stratified by crisis type and
    language, with identical headlines always kept on the same side."""
    k = max(2, round(1 / frac))
    strat = df[("Crisis" if "Crisis" in TASKS else TASKS[0]) + "_Combined"] + "|" + df["lang"]
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    rest_idx, hold_idx = next(sgkf.split(df, strat, groups=df["group"]))
    return df.iloc[rest_idx], df.iloc[hold_idx]


def split(df: pd.DataFrame, args):
    col = getattr(args, "split_col", None)
    if col:                                     # fixed benchmark splits (e.g. CrisisMMD, Ofli et al. 2020)
        if col not in df.columns:
            raise SystemExit(f"--split_col {col!r} is not a column of the CSV")
        s = df[col].astype(str).str.strip().str.lower().replace({"dev": "val", "valid": "val", "validation": "val"})
        parts = [df[s == k].reset_index(drop=True) for k in ("train", "val", "test")]
        if any(len(d_) == 0 for d_ in parts):
            raise SystemExit(f"--split_col {col!r} needs the values train, val (or dev) and test; found "
                             f"{sorted(s.unique())}")
        return parts
    trval, test = _group_holdout(df, args.test_size, args.seed)
    train, val = _group_holdout(trval, args.val_size / (1 - args.test_size), args.seed)
    shared = set(train["group"]) & set(test["group"])
    assert not shared, "group leakage between train and test"
    return [d.reset_index(drop=True) for d in (train, val, test)]


class CrisisDataset(Dataset):
    """Returns the RAW PIL image; the SigLIP processor does all resizing and
    normalisation (no double normalisation)."""

    def __init__(self, df, image_dir, spaces, args):
        self.df, self.image_dir, self.args = df, image_dir, args
        self.c2i = {k: {c: i for i, c in enumerate(v)} for k, v in spaces.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(os.path.join(self.image_dir, str(row[self.args.image_col]))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        text = row[self.args.text_col] or "no description"
        y = {"cmc": self.c2i[CMC_COL][row[CMC_COL]], "_lang": row["lang"], "_idx": idx,
             "cmc_sup": int(row["cmc_sup"]) if "cmc_sup" in row else 1}
        for t in TASKS:
            for v in VIEWS:
                y[f"{t}_{v}"] = self.c2i[t][row[f"{t}_{v}"]]
        return str(row[self.args.image_col]), img, text, y


def collate(batch):
    names, imgs, texts, ys = zip(*batch)
    labels = {k: torch.tensor([y[k] for y in ys], dtype=torch.long) for k in ys[0] if k != "_lang"}
    labels["_lang"] = [y["_lang"] for y in ys]
    return list(names), list(imgs), list(texts), labels


def class_weights(df, spaces, mode):
    """Inverse-frequency weights (normalised to mean 1) for each label field."""
    out = {}
    fields = {"cmc": (CMC_COL, spaces[CMC_COL])}
    fields.update({f"{t}_{v}": (f"{t}_{v}", spaces[t]) for t in TASKS for v in VIEWS})
    for key, (col, classes) in fields.items():
        if mode == "none":
            out[key] = None
            continue
        counts = df[col].value_counts().reindex(classes).fillna(0).values.astype(float)
        w = 1.0 / np.maximum(counts, 1.0)
        w = np.sqrt(w) if mode == "sqrt" else w
        w = w * (counts > 0)                      # classes absent from a view get weight 0
        w = w / (w[counts > 0].mean() + 1e-12)
        out[key] = torch.tensor(w, dtype=torch.float32)
    return out


# ════════════════════════════════════════════════════════════════════
# Encoder: SigLIP-2 (multilingual) with optional LoRA
# ════════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        for p_ in self.base.parameters():
            p_.requires_grad = False
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())


def inject_lora(model: nn.Module, rank: int, alpha: int,
                targets=("q_proj", "k_proj", "v_proj", "out_proj")) -> int:
    n = 0
    for _, parent in model.named_modules():
        # nn.MultiheadAttention (used in SigLIP's pooling head) reads out_proj.weight directly
        # instead of calling out_proj(x), so its projection must stay a plain nn.Linear
        if isinstance(parent, nn.MultiheadAttention):
            continue
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha))
                n += 1
    return n


class SigLIPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(args.siglip)
        self.model = AutoModel.from_pretrained(args.siglip)
        for p_ in self.model.parameters():
            p_.requires_grad = False
        self.use_lora = args.lora_rank > 0
        if self.use_lora:
            n = inject_lora(self.model, args.lora_rank, args.lora_alpha)
            print(f"LoRA injected into {n} linear layers")
        self.max_len = args.max_len
        tok = getattr(self.processor, "tokenizer", None)
        self.pad_id = getattr(tok, "pad_token_id", None)
        self.pad_id = 0 if self.pad_id is None else self.pad_id
        with torch.no_grad():                        # infer embedding sizes from one dummy forward
            e = self.forward([Image.new("RGB", (224, 224))], ["probe"], tokens=True)
        self.dims = {"text": e["t"].shape[-1], "image": e["v"].shape[-1],
                     "text_tok": e["t_tok"].shape[-1], "image_tok": e["v_tok"].shape[-1]}

    def forward(self, images, texts, tokens=False):
        dev = next(self.model.parameters()).device
        inp = self.processor(images=images, text=texts, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.set_grad_enabled(self.use_lora and self.training):
            out = self.model(**inp)
        res = {"t": out.text_embeds.float(), "v": out.image_embeds.float()}
        if tokens:
            res["t_tok"] = out.text_model_output.last_hidden_state.float()
            res["v_tok"] = out.vision_model_output.last_hidden_state.float()
            res["t_mask"] = (inp["attention_mask"].bool() if "attention_mask" in inp
                             else inp["input_ids"] != self.pad_id)
        return res


TEXT_ENCODERS = {"xlmr": "xlm-roberta-base", "muril": "google/muril-base-cased",
                 "mbert": "bert-base-multilingual-cased", "indicbert": "ai4bharat/indic-bert"}


class HybridEncoder(nn.Module):
    """SigLIP-2 image tower (frozen + LoRA) + a fully fine-tuned multilingual text encoder.
    The SigLIP-2 text tower is dropped; text features come from e.g. XLM-R (mean pooling)."""

    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        proc = AutoProcessor.from_pretrained(args.siglip)
        self.image_processor = getattr(proc, "image_processor", proc)
        siglip = AutoModel.from_pretrained(args.siglip)
        self.vision = siglip.vision_model
        del siglip                                            # free the unused text tower
        for p_ in self.vision.parameters():
            p_.requires_grad = False
        self.use_lora = args.lora_rank > 0
        if self.use_lora:
            n = inject_lora(self.vision, args.lora_rank, args.lora_alpha)
            print(f"LoRA injected into {n} linear layers of the image tower")
        name = TEXT_ENCODERS[args.text_encoder]
        for local in (f"{args.text_encoder}-safetensors", "indic-bert-safetensors"):
            if os.path.isdir(local) and (local.startswith(args.text_encoder) or args.text_encoder == "indicbert"):
                name = local                                  # converted copy (torch < 2.6)
                break
        print(f"text encoder: {name} (fully fine-tuned, {args.max_len_text} tokens)")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.text_model = AutoModel.from_pretrained(name)
        self.max_len_text = args.max_len_text
        with torch.no_grad():
            e = self.forward([Image.new("RGB", (224, 224))], ["probe"], tokens=True)
        self.dims = {"text": e["t"].shape[-1], "image": e["v"].shape[-1],
                     "text_tok": e["t_tok"].shape[-1], "image_tok": e["v_tok"].shape[-1]}

    def forward(self, images, texts, tokens=False):
        dev = next(self.text_model.parameters()).device
        px = self.image_processor(images=images, return_tensors="pt")["pixel_values"].to(dev)
        with torch.set_grad_enabled(self.use_lora and self.training):
            vo = self.vision(pixel_values=px)
        b = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len_text,
                     return_tensors="pt").to(dev)
        h = self.text_model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        res = {"t": ((h * m).sum(1) / m.sum(1).clamp_min(1)).float(), "v": vo.pooler_output.float()}
        if tokens:
            res["t_tok"] = h.float()
            res["v_tok"] = vo.last_hidden_state.float()
            res["t_mask"] = b["attention_mask"].bool()
        return res


class MockEncoder(nn.Module):
    """Deterministic random features - only for testing the pipeline."""

    def __init__(self, dim=64):
        super().__init__()
        self.dims = {"text": dim, "image": dim, "text_tok": dim, "image_tok": dim}
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images, texts, tokens=False):
        import zlib
        def emb(s, n=1):
            g = torch.Generator().manual_seed(zlib.crc32(s.encode("utf-8")))
            return torch.randn(n, self.dims["text"], generator=g)
        dev = self.dummy.device
        ikeys = ["v" + str(int(np.asarray(im.resize((4, 4))).sum())) for im in images]
        res = {"t": torch.cat([emb("t" + s) for s in texts]).to(dev),
               "v": torch.cat([emb(k) for k in ikeys]).to(dev)}
        if tokens:
            res["t_tok"] = torch.stack([emb("tt" + s, 8) for s in texts]).to(dev)
            res["v_tok"] = torch.stack([emb("vv" + k, 16) for k in ikeys]).to(dev)
            res["t_mask"] = torch.ones(len(texts), 8, dtype=torch.bool, device=dev)
            res["t_mask"][:, 6:] = False
        return res


# ════════════════════════════════════════════════════════════════════
# CCEC model
# ════════════════════════════════════════════════════════════════════
def mlp(i, h, o, p):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


def cmc_levels(spaces, args):
    """Ordinal level (1 = weakest ... L = strongest) of every CMC class, in label-space order."""
    order = [c.strip() for c in args.cmc_order.split(",")]
    missing = [c for c in spaces[CMC_COL] if c not in order]
    if missing:
        raise SystemExit(f"CMC classes {missing} are not in --cmc_order {order}; pass the right order")
    return torch.tensor([order.index(c) + 1 for c in spaces[CMC_COL]], dtype=torch.long)


def cmc_edge_weights(spaces, args):
    """Text-image edge weight a(z) of every CMC class, in label-space order."""
    w = dict(kv.split(":") for kv in args.cmc_weights.split(","))
    return torch.tensor([float(w[c]) for c in spaces[CMC_COL]], dtype=torch.float32)


class CCEC(nn.Module):
    def __init__(self, dims, spaces, args):
        super().__init__()
        self.args = args
        text_dim, image_dim = dims["text"], dims["image"]
        d, h, p = args.dim, args.hidden, args.dropout
        self.K = {t: len(spaces[t]) for t in TASKS}
        self.Z = len(spaces[CMC_COL])
        self.Zc = 1 if args.no_regimes else self.Z
        self.use_align = args.corr != "none" or args.ordinal_cmc or args.fine_align
        # VLM reasoning mentions both text and image, so it may only enter pair-level components
        self.use_vlm = "vlm" in dims
        if self.use_vlm:
            self.proj_r = nn.Sequential(nn.LayerNorm(dims["vlm"]), nn.Linear(dims["vlm"], d), nn.GELU(), nn.Dropout(p))
        pair_dim = 4 * d + (d if self.use_vlm else 0)

        # modality-specific evidence encoders (text head never sees the image and vice versa)
        self.proj_t = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.GELU(), nn.Dropout(p))
        self.proj_v = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, d), nn.GELU(), nn.Dropout(p))
        self.head_t = nn.ModuleDict({t: mlp(d, h, self.K[t], p) for t in TASKS})
        self.head_v = nn.ModuleDict({t: mlp(d, h, self.K[t], p) for t in TASKS})

        # ── correlation learning: alignment space whose matched-pair score feeds the router ──
        if self.use_align:
            da = args.align_dim
            self.align_t = nn.Linear(text_dim, da)
            self.align_v = nn.Linear(image_dim, da)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            if args.fine_align:
                self.tok_t = nn.Linear(dims["text_tok"], da)
                self.tok_v = nn.Linear(dims["image_tok"], da)
        if args.ordinal_cmc:
            self.register_buffer("level_of_z", cmc_levels(spaces, args))
            n_lv = int(self.level_of_z.max())
            self.ord_base = nn.Parameter(torch.tensor(-0.2))
            self.ord_raw = nn.Parameter(torch.full((n_lv,), -1.0))      # thresholds = base + cumsum(softplus)
            self.ord_log_gamma = nn.Parameter(torch.tensor(math.log(10.0)))
            # tempered product of experts: weight of the ordinal evidence in the router, starts at 0
            self.ord_mix = nn.Parameter(torch.tensor(0.0))

        # correlation regime q(z | x) from the pair (+ alignment score features)
        self.head_z = mlp(pair_dim + (2 if self.use_align else 0), h, self.Z, p)

        if args.fusion_baseline:
            self.head_fused = nn.ModuleDict({t: mlp(pair_dim, h, self.K[t], p) for t in TASKS})
        else:
            # composition tensors C_z[i, j, k] (logits); init: combined agrees with text or image
            self.comp = nn.ParameterDict()
            for t in TASKS:
                K = self.K[t]
                eye = torch.eye(K)
                init = args.comp_init * (eye.view(K, 1, K) + eye.view(1, K, K))      # [i, j, k]
                self.comp[t] = nn.Parameter(init.unsqueeze(0).repeat(self.Zc, 1, 1, 1))
            if not args.no_residual:
                # joint-evidence expert: product of experts with the composed prior
                self.head_joint = nn.ModuleDict({t: mlp(pair_dim, h, self.K[t], p) for t in TASKS})
                for t in TASKS:                                            # start as pure composition
                    nn.init.zeros_(self.head_joint[t][-1].weight)
                    nn.init.zeros_(self.head_joint[t][-1].bias)

    def composition(self, task):
        return F.softmax(self.comp[task], dim=-1)                          # [Zc, K, K, K]

    # ── alignment ──
    def similarity(self, enc):
        """Pairwise text-image alignment scores S[i, j] for all pairs in the batch."""
        gt = F.normalize(self.align_t(enc["t"].float()), dim=-1)
        gv = F.normalize(self.align_v(enc["v"].float()), dim=-1)
        S = gt @ gv.t()
        if self.args.fine_align:                    # token-patch late interaction
            tt = F.normalize(self.tok_t(enc["t_tok"].float()), dim=-1)
            vv = F.normalize(self.tok_v(enc["v_tok"].float()), dim=-1)
            m = enc["t_mask"].float()
            best = torch.einsum("itd,jpd->ijtp", tt, vv).amax(-1)                     # [B, B, T]
            fine = (best * m[:, None, :]).sum(-1) / m.sum(-1).clamp_min(1)[:, None]
            S = 0.5 * (S + fine)
        return S, gt, gv

    def ordinal_logits(self, s):
        """logit P(level >= k | score s), k = 1..L; thresholds are increasing by construction."""
        theta = self.ord_base + torch.cumsum(F.softplus(self.ord_raw), 0)
        return self.ord_log_gamma.exp() * (s.unsqueeze(-1) - theta)

    def ordinal_level_logprob(self, s):
        """log P(level = l | matched pair, s), l = 1..L."""
        c = torch.sigmoid(self.ordinal_logits(s))                                    # [B, L]
        nxt = torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], dim=-1)
        return torch.log((c - nxt).clamp_min(1e-6))

    def forward(self, enc, z_override=None):
        t, v = self.proj_t(enc["t"]), self.proj_v(enc["v"])
        joint = torch.cat([t, v, t * v, (t - v).abs()], dim=-1)
        if self.use_vlm:                            # pair-level only: router, joint-evidence / fused head
            joint = torch.cat([joint, self.proj_r(enc["r"].to(joint.dtype))], dim=-1)
        aux = {"t_raw": F.normalize(enc["t"].float(), dim=-1), "v_raw": F.normalize(enc["v"].float(), dim=-1)}
        z_in = joint
        if self.use_align:
            S, gt, gv = self.similarity(enc)
            sd = S.diagonal()
            aux.update(S=S, s=sd, gt=gt, gv=gv)
            z_in = torch.cat([joint, sd[:, None].to(joint.dtype), (sd ** 2)[:, None].to(joint.dtype)], -1)
        z_logits = self.head_z(z_in)
        if self.args.ordinal_cmc:                   # tempered product of experts: classifier x ordinal^w
            z_logits = z_logits.float() + self.ord_mix * self.ordinal_level_logprob(aux["s"])[:, self.level_of_z - 1]
        q_z = F.softmax(z_logits, dim=-1) if z_override is None else z_override

        out = {"cmc": z_logits}
        for task in TASKS:
            lt, lv = self.head_t[task](t), self.head_v[task](v)
            if self.args.fusion_baseline:
                lc = self.head_fused[task](joint)
            else:
                pair = F.softmax(lt, -1).unsqueeze(2) * F.softmax(lv, -1).unsqueeze(1)   # [B, K, K]
                per_z = torch.einsum("bij,zijk->bzk", pair, self.composition(task))       # [B, Zc, K]
                pc = per_z[:, 0] if self.Zc == 1 else torch.einsum("bz,bzk->bk", q_z, per_z)
                lc = torch.log(pc.clamp_min(1e-8))
                if not self.args.no_residual:
                    lc = lc + self.head_joint[task](joint)     # renormalised by log_softmax in the loss
            out[f"{task}_Text"], out[f"{task}_Image"], out[f"{task}_Combined"] = lt, lv, lc
        return out, aux


class Model(nn.Module):
    def __init__(self, spaces, args):
        super().__init__()
        self.args = args
        self.encoder = (MockEncoder() if args.mock_encoder else
                        SigLIPEncoder(args) if args.text_encoder == "siglip" else HybridEncoder(args))
        dims = dict(self.encoder.dims)
        self.vlm_index = None
        if args.vlm_view:
            z = np.load(args.vlm_view, allow_pickle=True)
            names, emb = [str(n) for n in z["names"]], z["emb"].astype(np.float32)
            self.vlm_index = {n: i for i, n in enumerate(names)}
            table = np.concatenate([emb, np.zeros((1, emb.shape[1]), np.float32)])   # last row = missing
            self.register_buffer("vlm_table", torch.from_numpy(table))
            dims["vlm"] = emb.shape[1]
            print(f"VLM view: {len(names)} posts, dim {emb.shape[1]} ({args.vlm_view})")
        self.ccec = CCEC(dims, spaces, args)
        self.vlm_missing = 0

    def forward(self, images, texts, z_override=None, names=None):
        """Returns (classification logits, auxiliary tensors for correlation learning)."""
        enc = self.encoder(images, texts, tokens=self.args.fine_align)
        if self.vlm_index is not None:
            if names is None:
                raise ValueError("--vlm_view needs the image names of the batch")
            miss = len(self.vlm_index)
            idx = [self.vlm_index.get(str(n), miss) for n in names]
            self.vlm_missing += sum(i == miss for i in idx)
            enc["r"] = self.vlm_table[torch.tensor(idx, device=self.vlm_table.device)]
        return self.ccec(enc, z_override)


# ════════════════════════════════════════════════════════════════════
# Objective: joint negative log-likelihood of all annotation fields
# ════════════════════════════════════════════════════════════════════
def masked_ce(logits, y, weight, mask):
    """Class-weighted cross-entropy averaged over the rows where mask == 1 (0 if none)."""
    loss = F.cross_entropy(logits, y, weight=weight, reduction="none")
    w = (weight[y] if weight is not None else torch.ones_like(loss)) * mask.float()
    return (loss * w).sum() / w.sum().clamp_min(1e-8)


def joint_nll(out, labels, weights, args):
    keys = [f"{t}_Combined" for t in TASKS]
    if not args.no_view_sup:
        keys += [f"{t}_{v}" for t in TASKS for v in ("Text", "Image")]
    parts = {k: F.cross_entropy(out[k], labels[k], weight=weights[k]) for k in keys}
    parts["cmc"] = masked_ce(out["cmc"], labels["cmc"], weights["cmc"], labels["cmc_sup"])
    return sum(parts.values()), parts


def soft_infonce(logits, targets, row_w):
    """InfoNCE with soft target distributions (rows of `targets`), rows weighted by row_w."""
    tgt = targets / targets.sum(1, keepdim=True).clamp_min(1e-8)
    loss = -(tgt * F.log_softmax(logits, dim=1)).sum(1)
    return (row_w * loss).sum() / row_w.sum().clamp_min(1e-8)


def correlation_losses(ccec, aux, labels, frozen_v, args, edge_w, level_of_z):
    """Graph-contrastive and ordinal CMC terms for one batch.
    frozen_v: frozen SigLIP image embeddings of the batch (define image-image edges)."""
    parts = {}
    S = aux["S"].float()
    B = S.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=S.device)
    sup = labels["cmc_sup"].float()
    z = labels["cmc"]
    simv = frozen_v @ frozen_v.t()
    dup = (simv >= args.dup_sim) & ~eye                     # same photo in two posts: never a negative

    # CMC edge weight of each text-image pair (1.0 when CMC is unknown or not used)
    if args.corr in ("cmcw", "graph"):
        a = torch.where(sup > 0, edge_w[z], torch.ones_like(sup))
    else:
        a = torch.ones_like(sup)

    if args.corr != "none":
        W = torch.diag(a)
        logits = S * ccec.logit_scale.exp().clamp(max=100)
        if args.corr == "graph":
            e = ((simv - args.graph_min_sim) / (1 - args.graph_min_sim)).clamp(min=0).masked_fill(eye, 0)
            k = min(args.graph_topk, B - 1)
            if k > 0:
                kth = e.topk(k, dim=1).values[:, -1:]
                e = torch.where(e >= kth, e, torch.zeros_like(e))
                e = torch.maximum(e, e.t())                                 # undirected graph
            W = W + args.beta * a[:, None] * a[None, :] * e                  # two-hop soft positives
            parts["graph_edge_density"] = (e > 0).float().sum().detach() / max(B * (B - 1), 1)
        else:
            logits = logits.masked_fill(dup, -1e4)                           # duplicates removed from negatives
        parts["contrastive"] = 0.5 * (soft_infonce(logits, W, a) + soft_infonce(logits.t(), W.t(), a))

    if args.ordinal_cmc:
        L = int(level_of_z.max())
        ks = torch.arange(1, L + 1, device=S.device)
        lvl = level_of_z[z]                                                   # 1..L
        diag_logit = ccec.ordinal_logits(S.diagonal())                       # [B, L]
        tgt = (lvl[:, None] >= ks[None]).float()
        known = torch.ones_like(tgt)
        known[sup == 0, 1:] = 0          # unlabelled pair: only "matched >= weakest level" is known
        bce = F.binary_cross_entropy_with_logits(diag_logit, tgt, reduction="none")
        l_pos = (bce * known).sum() / known.sum().clamp_min(1)
        off = ~eye & ~dup
        if off.any():
            off_logit = ccec.ordinal_logits(S[off])                           # mismatched pairs: level 0
            l_neg = F.binary_cross_entropy_with_logits(off_logit, torch.zeros_like(off_logit))
        else:
            l_neg = S.new_zeros(())
        parts["ordinal"] = l_pos + l_neg
    return parts


# ════════════════════════════════════════════════════════════════════
# Frozen image keys and neighbour-aware batches
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def frozen_image_keys(model, dataset, args, device):
    """Normalised SigLIP image embeddings of the training set, computed BEFORE training
    (LoRA's B matrices start at zero, so these are exactly the frozen base model's)."""
    dl = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    was = model.training
    model.eval()
    keys = [F.normalize(model.encoder(imgs, texts)["v"].float(), dim=-1) for _, imgs, texts, _ in dl]
    model.train(was)
    return torch.cat(keys).to(device)


class NeighbourBatchSampler(torch.utils.data.Sampler):
    """Each batch = batch_size/2 random anchors + one partner per anchor: a nearest image
    neighbour (knn) or a random post (random, the control). Posts with the same headline
    are never partners. An epoch has the same number of samples as a normal epoch."""

    def __init__(self, keys, groups, batch_size, k, mode, seed):
        self.N, self.bs, self.mode, self.k = len(keys), batch_size, mode, k
        self.rng = np.random.default_rng(seed)
        self.n_batches = math.ceil(self.N / batch_size)
        if mode == "knn":
            g = torch.as_tensor(np.asarray(groups), device=keys.device)
            nbrs = []
            for i0 in range(0, self.N, 1024):
                sim = keys[i0:i0 + 1024] @ keys.t()
                rows = torch.arange(i0, min(i0 + 1024, self.N), device=keys.device)
                sim[torch.arange(len(rows)), rows] = -2                         # not itself
                sim[g[rows][:, None] == g[None, :]] = -2                        # not the same headline
                nbrs.append(sim.topk(k, dim=1).indices.cpu())
            self.nbrs = torch.cat(nbrs).numpy()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        perm = self.rng.permutation(self.N)
        half = max(1, self.bs // 2)
        for b in range(self.n_batches):
            anchors = perm[(b * half) % self.N: (b * half) % self.N + half]
            batch, used = list(map(int, anchors)), set(map(int, anchors))
            for a in anchors:
                cand = self.nbrs[a] if self.mode == "knn" else self.rng.integers(0, self.N, self.k)
                cand = [int(c) for c in cand if int(c) not in used]
                if cand:
                    c = int(self.rng.choice(cand))
                    batch.append(c)
                    used.add(c)
            while len(batch) < self.bs:
                c = int(self.rng.integers(0, self.N))
                if c not in used:
                    batch.append(c)
                    used.add(c)
            yield batch


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, loader, device, oracle=False, n_z=None, full=False):
    """Returns (preds, gold) or, with full=True, (preds, gold, probs, names, timing, emb).
    Timing covers the model forward pass (incl. SigLIP preprocessing), excludes data loading,
    and skips the first batch as warm-up."""
    model.eval()
    preds, gold, probs, langs, names, emb = {}, {}, {}, [], [], {}
    fwd_time, fwd_n = 0.0, 0
    for b, (nm, imgs, texts, labels) in enumerate(loader):
        zo = F.one_hot(labels["cmc"], n_z).float().to(device) if oracle else None
        _sync(device)
        t0 = time.perf_counter()
        out, aux = model(imgs, texts, z_override=zo, names=nm)
        _sync(device)
        if full:
            for k in ("t_raw", "v_raw", "gt", "gv", "s"):
                if k in aux:
                    emb.setdefault(k, []).append(aux[k].float().cpu())
        if b > 0:
            fwd_time += time.perf_counter() - t0
            fwd_n += len(imgs)
        langs += labels["_lang"]
        names += nm
        for k, logits in out.items():
            logits = logits.float()
            preds.setdefault(k, []).append(logits.argmax(-1).cpu())
            probs.setdefault(k, []).append(F.softmax(logits, -1).cpu())
            gold.setdefault(k, []).append(labels[k])
    preds = {k: torch.cat(v).numpy() for k, v in preds.items()}
    gold = {k: torch.cat(v).numpy() for k, v in gold.items()}
    gold["_lang"] = np.array(langs)
    if not full:
        return preds, gold
    probs = {k: torch.cat(v).numpy() for k, v in probs.items()}
    emb = {k: torch.cat(v).numpy() for k, v in emb.items()}
    timing = {"inference_samples_timed": fwd_n,
              "inference_total_s": round(fwd_time, 3),
              "inference_ms_per_sample": round(1000 * fwd_time / max(fwd_n, 1), 3),
              "inference_samples_per_s": round(fwd_n / max(fwd_time, 1e-9), 2)}
    return preds, gold, probs, names, timing, emb


def score(preds, gold):
    gold = {k: v for k, v in gold.items() if k != "_lang"}
    res = {k: {"acc": accuracy_score(gold[k], preds[k]),
               "macro_f1": f1_score(gold[k], preds[k], average="macro", zero_division=0),
               "weighted_f1": f1_score(gold[k], preds[k], average="weighted", zero_division=0)}
           for k in preds if not (k == "cmc" and CMC_IS_PLACEHOLDER)}
    main = ([] if CMC_IS_PLACEHOLDER else ["cmc"]) + [f"{t}_Combined" for t in TASKS]
    res["avg_macro_f1"] = float(np.mean([res[k]["macro_f1"] for k in main]))
    res["avg_combined_weighted_f1"] = float(np.mean([res[f"{t}_Combined"]["weighted_f1"] for t in TASKS]))
    res["avg_combined_acc"] = float(np.mean([res[f"{t}_Combined"]["acc"] for t in TASKS]))
    # without CMC: the fair average when CMC is not supervised (--no_cmc_sup)
    res["avg_combined_macro_f1"] = float(np.mean([res[f"{t}_Combined"]["macro_f1"] for t in TASKS]))
    return res


def alignment_analysis(emb, gold, test_df, spaces, args):
    """How well does an embedding space capture cross-modal correlation?
    For the encoder space (SigLIP, every variant) and the learned alignment space (CCEC-G):
      * mean matched-pair score per CMC level and Spearman rho(score, CMC level)
      * text->image retrieval Recall@1/5/10 on the test set, overall and per CMC level
      * cross-lingual P@10: for a Hindi (English) text, the top-10 images from ENGLISH (Hindi)
        posts - share of them with the same crisis type (the image-as-bridge effect)."""
    levels = cmc_levels(spaces, args).numpy()[gold["cmc"]]
    names = spaces[CMC_COL]
    lang = gold["_lang"]
    crisis = gold.get("Crisis_Combined")
    # the raw encoder space is a shared image-text space only for SigLIP-2 itself
    spaces_ = {"encoder": ("t_raw", "v_raw")} if getattr(args, "text_encoder", "siglip") == "siglip" else {}
    if "gt" in emb:
        spaces_["learned"] = ("gt", "gv")
    res = {}
    for name, (kt, kv) in spaces_.items():
        T, V = emb[kt], emb[kv]
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        sim = T @ V.T
        diag = np.diag(sim)
        r = {} if CMC_IS_PLACEHOLDER else \
            {"spearman_score_vs_cmc": float(pd.Series(diag).corr(pd.Series(levels), method="spearman"))}
        for zi, zn in enumerate([] if CMC_IS_PLACEHOLDER else names):
            m = gold["cmc"] == zi
            if m.any():
                r[f"mean_score_CMC={zn}"] = float(diag[m].mean())
        rank = (sim > diag[:, None]).sum(1)                                  # 0 = matched image first
        for kk in (1, 5, 10):
            r[f"t2i_R@{kk}"] = float((rank < kk).mean())
            for zi, zn in enumerate([] if CMC_IS_PLACEHOLDER else names):
                m = gold["cmc"] == zi
                if m.any():
                    r[f"t2i_R@{kk}_CMC={zn}"] = float((rank[m] < kk).mean())
        langs = [l for l in np.unique(lang) if l in ("en", "hi")]
        if len(langs) == 2 and crisis is not None:
            p_all = []
            for src, dst in (("hi", "en"), ("en", "hi")):
                qi, ci = np.where(lang == src)[0], np.where(lang == dst)[0]
                top = ci[np.argsort(-sim[np.ix_(qi, ci)], axis=1)[:, :10]]
                p = (crisis[top] == crisis[qi][:, None]).mean()
                r[f"crossling_P@10_{src}->{dst}"] = float(p)
                p_all.append(p)
            r["crossling_P@10"] = float(np.mean(p_all))
        res[name] = r
    return res


def subset_score(preds, gold, mask):
    return {t: f1_score(gold[f"{t}_Combined"][mask], preds[f"{t}_Combined"][mask],
                        average="macro", zero_division=0) for t in TASKS}


def export_composition(model, spaces, path, top=8):
    """Human-readable composition rules for the paper: cases where text and image
    DISAGREE, and what the combined judgement becomes under each CMC regime."""
    lines = []
    regimes = spaces[CMC_COL] if model.ccec.Zc > 1 else ["all"]
    tensors = {}
    for task in TASKS:
        C = model.ccec.composition(task).detach().cpu().numpy()          # [Zc, K, K, K]
        tensors[task] = C
        cls = spaces[task]
        lines.append(f"\n=== {task} ===")
        for zi, zname in enumerate(regimes):
            rules = []
            for i in range(len(cls)):
                for j in range(len(cls)):
                    if i == j:
                        continue
                    k = int(C[zi, i, j].argmax())
                    rules.append((C[zi, i, j, k], cls[i], cls[j], cls[k]))
            rules.sort(reverse=True)
            lines.append(f"  CMC = {zname}")
            for pr, ti, vj, ck in rules[:top]:
                side = "text" if ck == ti else ("image" if ck == vj else "other")
                lines.append(f"    text={ti!r:32} image={vj!r:32} -> {ck!r} ({pr:.2f}, follows {side})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    np.savez(path.replace(".txt", ".npz"), **tensors)


# ════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_tasks(args.tasks)
    args.cmc_optional = args.no_cmc_sup
    df = load_dataframe(args.csv, args)
    spaces = build_label_spaces(df)
    train_df, val_df, test_df = split(df, args)
    print(f"tasks {TASKS} | train {len(train_df)} | val {len(val_df)} | test {len(test_df)} "
          + (f"(fixed splits from column {args.split_col!r})" if args.split_col else
             "(group split: identical headlines never cross splits)"))
    frac = 0.0 if args.no_cmc_sup else args.cmc_frac
    rng = np.random.default_rng(args.seed)
    train_df["cmc_sup"] = (rng.random(len(train_df)) < frac).astype(int)
    val_df["cmc_sup"], test_df["cmc_sup"] = 1, 1
    if frac < 1:
        print(f"CMC supervision on {train_df['cmc_sup'].mean():.0%} of training posts")
    if args.corr == "graph" and args.nbr_sampler == "none":
        print("note: --corr graph without --nbr_sampler knn - few two-hop edges will exist in a batch")

    mk = lambda d, sh: DataLoader(CrisisDataset(d, args.image_dir, spaces, args), batch_size=args.batch_size,
                                  shuffle=sh, num_workers=args.num_workers, collate_fn=collate)
    val_dl, test_dl = mk(val_df, False), mk(test_df, False)

    weights = {k: (w.to(device) if w is not None else None)
               for k, w in class_weights(train_df, spaces, args.class_weight).items()}

    model = Model(spaces, args).to(device)
    edge_w = cmc_edge_weights(spaces, args).to(device)
    level_of_z = cmc_levels(spaces, args).to(device)

    # frozen image keys (image-image graph edges, neighbour batches, duplicate detection)
    train_ds = CrisisDataset(train_df, args.image_dir, spaces, args)
    need_keys = args.corr != "none" or args.ordinal_cmc or args.nbr_sampler != "none"
    frozen_v = None
    if need_keys and not args.eval_only:
        t0 = time.time()
        frozen_v = frozen_image_keys(model, train_ds, args, device)
        print(f"frozen image keys: {tuple(frozen_v.shape)} in {time.time() - t0:.0f}s")
    if args.nbr_sampler != "none" and not args.eval_only:
        sampler = NeighbourBatchSampler(frozen_v, train_df["group"].values, args.batch_size,
                                        args.nbr_k, args.nbr_sampler, args.seed)
        train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    text_ft = [p_ for n, p_ in model.encoder.named_parameters() if p_.requires_grad and n.startswith("text_model.")]
    lora = [p_ for n, p_ in model.encoder.named_parameters()
            if p_.requires_grad and not n.startswith("text_model.") and (".A" in n or ".B" in n)]
    comp = list(model.ccec.comp.parameters()) if hasattr(model.ccec, "comp") else []
    comp_ids = {id(p_) for p_ in comp}
    head = [p_ for p_ in model.ccec.parameters() if p_.requires_grad and id(p_) not in comp_ids]
    groups = [{"params": head, "lr": args.lr}]
    if comp:
        groups.append({"params": comp, "lr": args.comp_lr or args.lr, "weight_decay": 0.0})
    if lora:
        groups.append({"params": lora, "lr": args.lr_lora})
    if text_ft:
        groups.append({"params": text_ft, "lr": args.lr_text})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = int(args.warmup_ratio * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    n_total = sum(p_.numel() for p_ in model.parameters())
    params = {"total_parameters": n_total,
              "trainable_parameters": n_train,
              "trainable_percent": round(100 * n_train / n_total, 4),
              "trainable_lora_parameters": sum(p_.numel() for p_ in lora),
              "trainable_text_encoder_parameters": sum(p_.numel() for p_ in text_ft),
              "trainable_head_parameters": sum(p_.numel() for p_ in head) + sum(p_.numel() for p_ in comp),
              "frozen_parameters": n_total - n_train}
    print(f"parameters: total {n_total / 1e6:.2f}M | trainable {n_train / 1e6:.2f}M "
          f"({params['trainable_percent']:.2f}%) = LoRA {params['trainable_lora_parameters'] / 1e6:.2f}M "
          f"+ text encoder {params['trainable_text_encoder_parameters'] / 1e6:.2f}M "
          f"+ CCEC head {params['trainable_head_parameters'] / 1e6:.2f}M")

    use_amp = device.type == "cuda"
    best, bad, history = -1.0, 0, []
    ckpt = args.ckpt or os.path.join(args.save_dir, "best_trainable.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start, epochs_run = time.time(), 0
    for epoch in range(1, (0 if args.eval_only else args.epochs) + 1):
        epochs_run = epoch
        model.train()
        t0, running, comp_log = time.time(), 0.0, {}
        for nm, imgs, texts, labels in train_dl:
            labels = {k: v.to(device) for k, v in labels.items() if k != "_lang"}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out, aux = model(imgs, texts, names=nm)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = joint_nll(out, labels, weights, args)
            if frozen_v is not None and (args.corr != "none" or args.ordinal_cmc):
                cparts = correlation_losses(model.ccec, aux, labels, frozen_v[labels["_idx"]],
                                            args, edge_w, level_of_z)
                for k in ("contrastive", "ordinal"):
                    if k in cparts:
                        loss = loss + cparts[k]
                        comp_log[k] = comp_log.get(k, 0.0) + cparts[k].item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        val = score(*predict(model, val_dl, device))
        history.append({"epoch": epoch, "loss": running / len(train_dl), "val_avg_macro_f1": val["avg_macro_f1"],
                        f"val_{args.select_metric}": val[args.select_metric]})
        extra = "".join(f" | {k} {v / len(train_dl):.3f}" for k, v in comp_log.items())
        print(f"epoch {epoch:02d} | loss {running / len(train_dl):.4f}{extra} | "
              f"val {args.select_metric} {val[args.select_metric]:.4f} | {time.time() - t0:.0f}s")
        if val[args.select_metric] > best + 1e-4:
            best, bad = val[args.select_metric], 0
            torch.save({n: p_.detach().cpu() for n, p_ in model.named_parameters() if p_.requires_grad}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (best val {best:.4f})")
                break

    train_time = time.time() - train_start

    # ── final test with best checkpoint ──
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    missing = model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    n_loaded = len(torch.load(ckpt, map_location="cpu"))
    print(f"loaded {n_loaded} trainable tensors from {ckpt}")
    preds, gold, probs, names, timing, emb = predict(model, test_dl, device, full=True)
    results = {"args": vars(args), "test": score(preds, gold), "history": history}
    results["alignment"] = alignment_analysis(emb, gold, test_df, spaces, args)
    if args.vlm_view:
        results["vlm_missing_lookups"] = int(model.vlm_missing)
        if model.vlm_missing:
            print(f"WARNING: {model.vlm_missing} post lookups had no VLM reasoning (zero vector used)")
    for lang in np.unique(gold["_lang"]):                                # per-language breakdown
        m = gold["_lang"] == lang
        sub_p = {k: v[m] for k, v in preds.items()}
        sub_g = {k: v[m] for k, v in gold.items()}
        results[f"test_lang={lang}"] = score(sub_p, sub_g)
        print(f"  [{lang}] n={int(m.sum())} avg macro-F1 {results[f'test_lang={lang}']['avg_macro_f1']:.4f}")
    for zi, zname in enumerate([] if CMC_IS_PLACEHOLDER else spaces[CMC_COL]):   # per-regime breakdown
        m = gold["cmc"] == zi
        if m.sum() > 0:
            results[f"test_combined_macro_f1_CMC={zname}"] = subset_score(preds, gold, m)
    if not args.fusion_baseline and not args.no_regimes and not args.no_cmc_sup:
        results["test_oracle_cmc"] = score(*predict(model, test_dl, device, oracle=True,
                                                    n_z=len(spaces[CMC_COL])))
    if not args.fusion_baseline:
        export_composition(model, spaces, os.path.join(args.save_dir, "composition_rules.txt"))

    efficiency = dict(params)
    efficiency.update({
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "batch_size": args.batch_size,
        "epochs_run": epochs_run,
        "training_time_min": None if args.eval_only else round(train_time / 60, 2),
        "time_per_epoch_s": None if args.eval_only else round(train_time / max(epochs_run, 1), 1),
        "peak_gpu_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        "checkpoint_size_MB": round(os.path.getsize(ckpt) / 1e6, 2),
        "test_samples": len(names),
        **timing})
    results["efficiency"] = efficiency
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    extra = {k: v for k, v in results.items() if k.startswith(("test_oracle", "test_combined"))}
    write_full_report(args.save_dir, preds, gold, probs, names, spaces, efficiency, extra,
                      report_cmc=not CMC_IS_PLACEHOLDER)
    with open(os.path.join(args.save_dir, "report", "alignment.json"), "w") as f:
        json.dump(results["alignment"], f, indent=2)
    if args.delete_ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    print("\nALIGNMENT (test):")
    for sp, r in results["alignment"].items():
        keys = ["spearman_score_vs_cmc", "t2i_R@1", "t2i_R@10", "crossling_P@10"]
        print(f"  [{sp:7s}] " + "  ".join(f"{k} {r[k]:.4f}" for k in keys if k in r))
    print("\nTEST (combined macro-F1):")
    for k in ([] if CMC_IS_PLACEHOLDER else ["cmc"]) + [f"{t}_Combined" for t in TASKS]:
        r_ = results['test'][k]
        print(f"  {k:26s} macro-F1 {r_['macro_f1']:.4f} | acc {r_['acc']:.4f} | weighted F1 {r_['weighted_f1']:.4f}")
    print(f"  {'average':26s} {results['test']['avg_macro_f1']:.4f}")
    print(f"  {f'average ({len(TASKS)} tasks, no CMC)':26s} {results['test']['avg_combined_macro_f1']:.4f}")
    if "test_oracle_cmc" in results:
        print(f"  oracle-CMC average          {results['test_oracle_cmc']['avg_macro_f1']:.4f}")
    print(f"saved to {args.save_dir}")


if __name__ == "__main__":
    main()
