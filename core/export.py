# tools/pt_to_safetensors.py
import torch
from safetensors.torch import save_file

ckpt = torch.load("saved_models/model_final.pt", map_location="cpu")

# adjust depending on how you saved it:
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

# If trained with DDP, remove "module."
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

# IMPORTANT: wrapper uses self.model.<keys>, so prefix weights with "model."
out = {"model." + k: v for k, v in state_dict.items()}

save_file(out, "../hf_model/model.safetensors")
print("saved ../hf_model/model.safetensors")
