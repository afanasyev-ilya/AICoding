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

from checkpoints import *
from tokenizer import BPETokenizer

from model import CONTEXT_SIZE, create_moegpt_deepseek_style, MoEGPTConfig, MoEGPT

########################################################################################################

class StreamingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_length=CONTEXT_SIZE, max_samples=1000):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples
        self._current_buffer = []
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Fill buffer with tokenized samples"""
        self._current_buffer = []
        for i, row in enumerate(self.hf_dataset):
            if i >= self.max_samples:
                break
            # Tokenize and split into sequences
            tokens = self.tokenizer.encode(row["content"])
            # Split into chunks of seq_length
            for i in range(0, len(tokens), self.seq_length):
                chunk = tokens[i:i + self.seq_length]
                if len(chunk) == self.seq_length:  # Only use complete sequences
                    self._current_buffer.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self._current_buffer)
    
    def __getitem__(self, idx):
        return self._current_buffer[idx]

def get_batch_from_dataloader(dataloader):
    """Get batch from DataLoader instead of random sampling"""
    for batch in dataloader:
        x = batch[:, :-1]  # Input sequence
        y = batch[:, 1:]   # Target sequence (shifted by one)
        return x, y
    return None, None


def analyze_memory_usage(model, batch_size=16, seq_length=512):
    """Analyze memory usage by layer and provide scaling recommendations"""
    print("=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    total_params = 0
    layer_breakdown = {}
    
    # Analyze embedding layer
    emb_params = sum(p.numel() for p in model.tok_emb.parameters())
    if model.pos_emb is not None:
        emb_params += sum(p.numel() for p in model.pos_emb.parameters())
    total_params += emb_params
    layer_breakdown["Embedding"] = emb_params
    print(f"Embedding layers: {emb_params:,} parameters")
    
    # Analyze blocks (now MHA+MoE blocks)
    block_params = 0
    for i, block in enumerate(model.blocks):
        block_param_count = sum(p.numel() for p in block.parameters())
        block_params += block_param_count
        layer_breakdown[f"Block {i+1} (MHA+MoE)"] = block_param_count
        print(f"Block {i+1} (MHA+MoE): {block_param_count:,} parameters")
    
    total_params += block_params
    print(f"Total blocks: {block_params:,} parameters")
    
    # Final layers
    final_params = sum(p.numel() for p in model.ln_f.parameters()) + sum(p.numel() for p in model.head.parameters())
    total_params += final_params
    layer_breakdown["Final Layers"] = final_params
    print(f"Final layers: {final_params:,} parameters")
    
    print("-" * 60)
    print(f"TOTAL MODEL: {total_params:,} parameters")
    
    return total_params, layer_breakdown


def get_precision_config(precision: str):
    """
    Returns (param_dtype, amp_dtype, use_autocast, use_scaler) for training/inference.
    - param_dtype: dtype of model weights
    - amp_dtype:   dtype used by autocast for activations
    """
    if precision == "fp32":
        return torch.float32, torch.float32, False, False
    elif precision == "fp16":
        return torch.float16, torch.float16, True, True
    elif precision == "bf16":
        return torch.bfloat16, torch.bfloat16, True, False
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def train(model, dataset, batch_size=16, epochs=3, lr=3e-4, grad_accum_steps=2, 
          checkpoint_dir="checkpoints", save_every=10, resume_from=None,
          amp_dtype=torch.float16, use_autocast=True, use_scaler=True):
    """Training with checkpoint saving and resuming"""
    
    stream_dataset = StreamingDataset(dataset, tok, seq_length=CONTEXT_SIZE)
    dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    
    start_epoch = 1
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from == "latest":
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)
            start_epoch += 1  # Start from next epoch
            print(f"Resuming training from epoch {start_epoch}")
    elif resume_from and os.path.exists(resume_from):
        start_epoch, best_loss = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
    
    print(f"[STATUS] Training started from epoch {start_epoch}...")
    model.train()
    
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            xb = batch[:, :-1].to('cuda', non_blocking=True)
            yb = batch[:, 1:].to('cuda', non_blocking=True)

            if use_autocast and amp_dtype != torch.float32:
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                logits, loss_ce, aux_loss = model(xb, yb)
                loss = loss_ce + model.config.aux_loss_weight * aux_loss
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"epoch {epoch:3d}/{epochs} | step {step:4d} | loss {avg_loss:.4f}")

        avg_epoch_loss = total_loss / max(1, num_batches)
        print(f"epoch {epoch:3d}/{epochs} | avg_loss {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            print("saving....")
            print(f"{epoch} - {save_every}")
            save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_dir)
            
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_dir)
            # Also save as best model
            best_path = f"{checkpoint_dir}/checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': model.config.__dict__,
            }, best_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Generate sample to monitor progress
        if epoch % 20 == 0:
            model.eval()
            sample = inference(
                model,
                tok,
                "def binary_search(arr, target):\n",
                max_new_tokens=100,
                temperature=0.8,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
            )
            print(f"\n--- Epoch {epoch} Sample ---")
            print(sample[:500] + "..." if len(sample) > 500 else sample)
            print("---" + "-" * 20)
            model.train()


def inference(model, tok, prompt="", max_new_tokens=100, temperature=0.8,
              amp_dtype=torch.float32, use_autocast=False):
    model.eval()

    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device='cuda')

    if use_autocast and amp_dtype != torch.float32:
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    with torch.no_grad():
        with autocast_ctx:
            out = model.generate(ctx, max_new_tokens, temperature=temperature)[0].tolist()

    generated = tok.decode(out)
    return generated


def generate_multiple_samples(model, tok, prompt, num_samples=3, max_new_tokens=150, temperature=0.7,
                              amp_dtype=torch.float32, use_autocast=False):
    """Generate multiple samples and pick the best one"""
    samples = []
    
    for i in range(num_samples):
        # Vary temperature slightly for diversity
        current_temp = temperature * (0.8 + 0.4 * (i / num_samples))
        sample = inference(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=current_temp,
            amp_dtype=amp_dtype,
            use_autocast=use_autocast,
        )
        samples.append(sample)
        
        print(f"--- Sample {i+1} (temp={current_temp:.2f}) ---")
        print(sample)
        print()
    
    # Simple heuristic: pick the one with most complete function structure
    best_sample = max(samples, key=lambda s: (
        s.count('def '),
        s.count('return '),
        s.count(':') - s.count('":'),  # Count colons but not in strings
        len(s)
    ))
    
    return best_sample, samples


def print_torch_stats():
    # Check if FlashAttention is available in your PyTorch installation
    print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

    # List available backends
    from torch.backends.cuda import SDPBackend
    print("\nAvailable SDP backends:")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        print("All backends enabled for context")


# --- Main ---
if __name__ == "__main__":
    print_torch_stats()

    parser = argparse.ArgumentParser()
    # LLM settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--pos_encoding", type=str, choices=["rope", "learned"], default="rope")
    parser.add_argument("--model_arch", type=str, choices=["deepseek", "optimized"], default="deepseek")

    # Precision settings
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Compute precision for weights + activations: fp32, fp16, or bf16",
    )
    
    # Checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Resume from 'latest', or path to specific checkpoint")
    parser.add_argument("--save_final_model", action="store_true", default=True, 
                       help="Save final model after training")
    
    # tokenizer settings
    parser.add_argument("--tok_path", type=str, default="./tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    # Load data
    print("[STATUS] preparing dataset...")
    dataset = load_dataset("/home/i.afanasyev/codeparrot-clean", split="train", streaming=True)
    print("[STATUS] data loaded.")

    # tokenize data
    tok = BPETokenizer(tokenizer_path=args.tok_path if os.path.exists(args.tok_path) else None)
    if tok.tk is None:
        print(f"[STATUS] Training byte-level BPE tokenizer (vocab={args.vocab_size}) on dataset...")
        tok.train(
            dataset=dataset,
            vocab_size=args.vocab_size, 
            save_path=args.tok_path,
            max_samples=1000
        )
        print(f"Saved tokenizer to {args.tok_path}")
    print("[STATUS] tokenizer prepared.")

    # Load or create model
    if args.resume_from and args.resume_from != "latest" and os.path.exists(args.resume_from):
        # Load existing model from checkpoint (dtype will be converted below)
        model = load_model(MoEGPT, args.resume_from)
        print(f"[STATUS] Model loaded from checkpoint: {args.resume_from}")
    else:
        # Create new model
        if args.model_arch == "deepseek":
            cfg = create_moegpt_deepseek_style(vocab_size=tok.vocab_size)
            print("[ARCHITECTURE] Using DeepSeek-style: alternating MHA -> MoE blocks")
        else:
            cfg = create_moegpt_a5000_optimized(vocab_size=tok.vocab_size)
            print("[ARCHITECTURE] Using optimized single-MoE architecture")
        
        model = MoEGPT(cfg)
        print("[STATUS] New model created.")

    # Precision configuration (weights + activations)
    param_dtype, amp_dtype, use_autocast, use_scaler = get_precision_config(args.precision)
    model = model.to(device="cuda", dtype=param_dtype)
    print(f"[PRECISION] Using {args.precision}: weights={param_dtype}, activations={amp_dtype}, "
          f"autocast={use_autocast}, scaler={use_scaler}")

    # Print model info
    total_params, breakdown = analyze_memory_usage(model, batch_size=args.batch_size, seq_length=CONTEXT_SIZE)
    
    print(f"\nARCHITECTURE SUMMARY:")
    print(f"• Total blocks: {model.config.n_layer}")
    print(f"• Each block: MHA -> MoE")
    print(f"• Attention: {model.config.n_head} heads, {model.config.n_embd} dim")
    print(f"• MoE per block: {model.config.num_experts} experts, {model.config.expert_dim} dim")
    print(f"• Context: {model.config.block_size} tokens")

    # Train the model
    train(model, dataset, 
          batch_size=args.batch_size, 
          epochs=args.epochs, 
          lr=args.lr,
          checkpoint_dir=args.checkpoint_dir,
          save_every=args.save_every,
          resume_from=args.resume_from,
          amp_dtype=amp_dtype,
          use_autocast=use_autocast,
          use_scaler=use_scaler)

    # Save final model
    if args.save_final_model:
        model_dir = "saved_models"
        save_model(model, model_dir)
        save_tokenizer(tok, model_dir)
        print(f"Final model and tokenizer saved to {model_dir}")

    # Generate text with the trained model
    print("\n" + "="*50)
    print("GENERATION EXAMPLES")
    print("="*50)
    
    prompts = [
        "def binary_search(arr, target):\n",
        "def quicksort(arr):\n",
        "def fibonacci(n):\n",
        "class LinkedList:\n    def __init__(self):\n",
    ]
    
    for prompt in prompts:
        print(f"\n--- Generating for: {prompt.strip()} ---")
        output = inference(
            model,
            tok,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            amp_dtype=amp_dtype,
            use_autocast=use_autocast,
        )
        print(output)
        print("-" * 40)

    # Also try multiple samples for the main prompt
    print("\n--- Multiple samples for binary_search ---")
    best_code, all_samples = generate_multiple_samples(
        model,
        tok, 
        "def binary_search(arr, target):\n",
        num_samples=3,
        max_new_tokens=args.max_new_tokens,
        amp_dtype=amp_dtype,
        use_autocast=use_autocast,
    )
    
    print("\n--- BEST SAMPLE ---")
    print(best_code)
