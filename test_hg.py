import os
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")  

import torch
if hasattr(torch, "compile"):
    def _no_compile(fn=None, *args, **kwargs):
        if fn is None:
            def deco(f): return f
            return deco
        return fn
    torch.compile = _no_compile

from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_DIR = "/home/zhaol/projects/huggingface_crossdna/crossdna" # This directory must contain either the `model.safetensors` or `pytorch_model.bin` file.

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model.eval()

# ---- Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Base mapping (logits indexed 0..4 <-> A/C/G/T/N)
labels = ["A", "C", "G", "T", "N"]
base_map = {ch: i for i, ch in enumerate(labels)}

def dna_to_base_ids(seq: str, device=None):
    t = torch.tensor([base_map.get(ch.upper(), base_map["N"]) for ch in seq], dtype=torch.long)
    return t.to(device) if device is not None else t

# ========== Test (MaskedLM Forward) ==========
x = torch.full((2, 16), base_map['N'], dtype=torch.long, device=device)
mask_id = getattr(model.config, "mask_token_id", 3)
x[:, 3] = mask_id
x[:, 9] = mask_id

with torch.no_grad():
    out = model(input_ids=x)

logits = out.logits.detach().cpu()
print("logits.shape =", tuple(logits.shape))

dna = "TGATGTGACTCACATAGGCGGTGGCGTGATATGTTGTGACTCATTTCCCGGAAACGGATGACTAATGCCATATGTTATCAGTTTCCTGGAAATTTGATCACGCCATATTGTGAAATCATGCGATTCCCGGATCACGTGACGGCCGGACGTGACAAGTATGAGTCACTAAGTGGCGTGATCTTACGAATCACGTGATGGTCAATGTCACGTGATCGGCTGGTGAGTCAGCAATATCGTGTGATTCATTC"
inp = dna_to_base_ids(dna, device=device).unsqueeze(0)
with torch.no_grad():
    out2 = model(input_ids=inp)
pred2 = out2.logits.argmax(dim=-1).squeeze(0).cpu().tolist()
print("argmax base:", "".join(labels[i] for i in pred2))

# ==========  Representations (Embedding) ==========

max_len_cfg = getattr(model.config, "max_position_embeddings", 1024)
max_length = int(min(512, max_len_cfg))  # You can change it to a longer value, but do not exceed max_position_embeddings.

# 2) Construct sample DNA text with base IDs (note: not tokenizer IDs)
sequence = "ACTG" * (max_length // 4)
seq_base = dna_to_base_ids(sequence, device=device).unsqueeze(0)  # [1, L] in 0..4

# 3) Temporarily switch the backbone to “characterization mode”.
was_pretrain = getattr(model.backbone, "pretrain", False)
was_for_repr = getattr(model.backbone, "for_representation", False)
model.backbone.pretrain = False                 # Let Backbone accept a single sequence
model.backbone.for_representation = True        # Let forward return the fused representation.

with torch.inference_mode():
    embeddings, _ = model.backbone(seq_base)    # [B, L, H]
embeddings_cpu = embeddings.detach().cpu()
print("embeddings.shape =", tuple(embeddings_cpu.shape))

# 4) Restore original settings
model.backbone.pretrain = was_pretrain
model.backbone.for_representation = was_for_repr
