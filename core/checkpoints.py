import json
import glob
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dataclasses import dataclass, field
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from tokenizer import BPETokenizer
from model import MoEGPTConfig, MoEGPT

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config.__dict__,
    }
    
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save latest checkpoint
    latest_path = f"{checkpoint_dir}/checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['loss']


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pt")
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    def extract_epoch(path):
        return int(Path(path).stem.split('_')[-1])
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return latest_checkpoint


def save_model(model, model_dir="saved_models"):
    """Save complete model for inference"""
    Path(model_dir).mkdir(exist_ok=True)
    
    model_path = f"{model_dir}/model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__,
    }, model_path)
    
    # Save config separately for easy inspection
    config_path = f"{model_dir}/model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model.config.__dict__, f, indent=2)
    
    print(f"Model saved: {model_path}")
    return model_path


def load_model(model_class, model_path, device='cuda'):
    """Load complete model for inference or further training"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate config
    config_dict = checkpoint['config']
    config = MoEGPTConfig(**config_dict)
    
    # Create model and load weights
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded: {model_path}")
    return model


def save_tokenizer(tokenizer, tokenizer_dir="saved_models"):
    """Save tokenizer"""
    Path(tokenizer_dir).mkdir(exist_ok=True)
    tokenizer_path = f"{tokenizer_dir}/tokenizer.json"
    tokenizer.tk.save(tokenizer_path)
    print(f"Tokenizer saved: {tokenizer_path}")
    return tokenizer_path


def load_tokenizer(tokenizer_dir="saved_models"):
    """Load tokenizer"""
    tokenizer_path = f"{tokenizer_dir}/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = BPETokenizer(tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer_path}")
    return tokenizer


# --- NEW: precision helper for inference_from_saved ---

def _get_precision_for_inference(precision: str):
    """
    Map a string precision name to (param_dtype, amp_dtype, use_autocast).
    This is only used for pure inference_from_saved.
    """
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16, torch.float16, True
    elif precision == "bf16":
        return torch.bfloat16, torch.bfloat16, True
    elif precision == "fp32":
        return torch.float32, torch.float32, False
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def inference_from_saved(
    model_path,
    tokenizer_path,
    prompt="",
    max_new_tokens=100,
    temperature=0.3,
    precision: str = "fp16",
):
    """
    Load model and tokenizer for inference.

    precision: "fp32", "fp16", or "bf16"
      - "fp16": weights -> float16, autocast(float16)
      - "bf16": weights -> bfloat16, autocast(bfloat16)
      - "fp32": weights -> float32, no autocast
    """
    # Load tokenizer
    tokenizer = BPETokenizer(tokenizer_path)
    
    # Choose precision
    param_dtype, amp_dtype, use_autocast = _get_precision_for_inference(precision)
    
    # Load model
    model = load_model(MoEGPT, model_path)
    model = model.to(device='cuda', dtype=param_dtype)
    model.eval()
    
    # Run inference
    ids = tokenizer.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device='cuda')

    if use_autocast and amp_dtype != torch.float32:
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            out = model.generate(ctx, max_new_tokens, temperature=temperature)[0].tolist()
    
    generated = tokenizer.decode(out)
    return generated
