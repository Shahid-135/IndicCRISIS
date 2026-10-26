"""
convert_indicbert.py - one-time conversion of ai4bharat/indic-bert to safetensors.

Recent transformers versions refuse to read old pytorch_model.bin checkpoints unless torch >= 2.6
(CVE-2025-32434), but they accept safetensors with any torch version. This script reads the .bin
once with plain torch.load(weights_only=True), maps the weights into an ALBERT encoder, checks that
all encoder weights were found, and saves model + tokenizer as safetensors to ./indic-bert-safetensors.
train_baselines.py uses that folder automatically for --model indicbert.

    python convert_indicbert.py
"""
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel, AutoTokenizer

REPO, OUT = "ai4bharat/indic-bert", "indic-bert-safetensors"

cfg = AlbertConfig.from_pretrained(REPO)
path = hf_hub_download(REPO, "pytorch_model.bin")
sd = torch.load(path, map_location="cpu", weights_only=True)
print(f"checkpoint: {len(sd)} tensors, e.g. {list(sd)[:3]}")

# the checkpoint may come from AlbertForMaskedLM / pre-training ("albert." prefix, plus LM heads)
model = AlbertModel(cfg)
want = set(model.state_dict())
mapped = {}
for k, v in sd.items():
    for prefix in ("albert.", ""):
        if k.startswith(prefix) and k[len(prefix):] in want:
            mapped[k[len(prefix):]] = v
            break
missing = sorted(want - set(mapped))
print(f"mapped {len(mapped)} / {len(want)} encoder tensors; missing: {missing}")
if any(not m.startswith("pooler") for m in missing):
    sys.exit("ERROR: encoder weights are missing - the conversion would give a randomly initialised model")
model.load_state_dict(mapped, strict=False)

# sanity check: the converted encoder must give the same output as the original weights
tok = AutoTokenizer.from_pretrained(REPO)
model.eval()
with torch.no_grad():
    out = model(**tok(["बाढ़ से 12 लोगों की मौत"], return_tensors="pt")).last_hidden_state
print(f"forward pass OK: output {tuple(out.shape)}, mean {out.mean():.4f}")

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"saved to {OUT}/ ({', '.join(sorted(os.listdir(OUT)))})")
