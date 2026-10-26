"""
convert_to_safetensors.py - one-time conversion of Hugging Face checkpoints that only ship
pytorch_model.bin into safetensors, so they load with torch < 2.6 (CVE-2025-32434 check in recent
transformers versions; safetensors files are exempt).

For each model the weights are read once with torch.load(weights_only=True), mapped into the bare
encoder (stripping "bert." / "roberta." / "albert." style prefixes of pre-training heads), checked for
completeness, run once as a sanity check, and saved with the tokenizer to ./<short>-safetensors.
train_baselines.py picks these folders up automatically.

    python convert_to_safetensors.py                 # muril, xlmr, indicbert
    python convert_to_safetensors.py muril           # just one
"""
import os
import sys

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODELS = {"muril": "google/muril-base-cased", "xlmr": "xlm-roberta-base", "indicbert": "ai4bharat/indic-bert",
          "mbert": "bert-base-multilingual-cased", "align": "kakaobrain/align-base", "clip": "openai/clip-vit-base-patch32"}


def convert(short):
    repo, out = MODELS[short], f"{short}-safetensors"
    files = list_repo_files(repo)
    if "model.safetensors" in files:
        print(f"[{short}] {repo} already ships safetensors - nothing to do")
        return
    if "pytorch_model.bin" not in files:
        sys.exit(f"[{short}] no pytorch_model.bin in {repo}: {files}")
    cfg = AutoConfig.from_pretrained(repo)
    model = AutoModel.from_config(cfg)
    want = set(model.state_dict())
    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    prefixes = (f"{model.base_model_prefix}.", "")
    mapped = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p) and k[len(p):] in want:
                mapped[k[len(p):]] = v
                break
    missing = sorted(want - set(mapped))
    critical = [m for m in missing if not m.startswith("pooler") and "position_ids" not in m]
    print(f"[{short}] mapped {len(mapped)}/{len(want)} tensors; missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if critical:
        sys.exit(f"[{short}] ERROR: encoder weights missing {critical[:5]} - not saving a half-random model")
    model.load_state_dict(mapped, strict=False)
    tok = AutoTokenizer.from_pretrained(repo)
    model.eval()
    os.makedirs(out, exist_ok=True)
    if short in ("align", "clip"):                     # vision-language models: save the full processor too
        from transformers import AutoProcessor
        with torch.no_grad():
            z = model.get_text_features(**tok(["Floods kill 12 in Assam"], return_tensors="pt"))
        print(f"[{short}] text features OK {tuple(z.shape)}")
        AutoProcessor.from_pretrained(repo).save_pretrained(out)
    else:
        with torch.no_grad():
            h = model(**tok(["बाढ़ से 12 लोगों की मौत", "Floods kill 12 in Assam"], padding=True,
                            return_tensors="pt")).last_hidden_state
        print(f"[{short}] forward pass OK {tuple(h.shape)}")
        tok.save_pretrained(out)
    model.save_pretrained(out, safe_serialization=True)
    print(f"[{short}] saved to {out}/")


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["muril", "xlmr", "indicbert"]):
        convert(m)
