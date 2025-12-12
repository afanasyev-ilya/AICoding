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

from torch.utils.data import IterableDataset  # add this import


class StreamingDataset(IterableDataset):
    """
    Stream the Hugging Face dataset and yield fixed-length token chunks.

    - Uses the *whole* streaming dataset (unless max_files_per_epoch is set).
    - Tokenizes on the fly, so memory usage stays reasonable.
    - Tracks how many source rows/files were processed in this epoch.
    """
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_length=CONTEXT_SIZE,
        max_files_per_epoch: Optional[int] = None,
        text_key: str = "content",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_files_per_epoch = max_files_per_epoch
        self.text_key = text_key

        # Stats per epoch
        self.files_yielded = 0

    def __iter__(self):
        """Each epoch, iterate over the HF streaming dataset and yield chunks."""
        self.files_yielded = 0
        num_files = 0

        for row in self.hf_dataset:
            if self.max_files_per_epoch is not None and num_files >= self.max_files_per_epoch:
                break
            num_files += 1
            self.files_yielded += 1

            # Tokenize one file
            text = row[self.text_key]
            tokens = self.tokenizer.encode(text)

            # Chunk into non-overlapping seq_length segments
            for i in range(0, len(tokens), self.seq_length):
                chunk = tokens[i:i + self.seq_length]
                if len(chunk) == self.seq_length:
                    yield torch.tensor(chunk, dtype=torch.long)


class TokenizerStreamingDataset(IterableDataset):
    """
    Wrapper to train the tokenizer on a mixture of datasets.

    Each yielded example has key 'content', because BPETokenizer.train expects
    example["content"] to be a string.
    """
    def __init__(self, datasets_and_keys):
        """
        datasets_and_keys: list of (hf_dataset, text_key) pairs.
        """
        self.datasets_and_keys = datasets_and_keys

    def __iter__(self):
        for hf_ds, text_key in self.datasets_and_keys:
            for row in hf_ds:
                yield {"content": row[text_key]}


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


def load_training_dataset(dataset_name: str, python_large_path: str):
    """
    Load the training dataset (streaming + optional map-style) and return:
      - dataset_stream: HF streaming dataset
      - dataset_map: map-style dataset or None
      - text_key: column containing code text
    """
    if dataset_name == "python_large":
        dataset_stream = load_dataset(python_large_path, split="train", streaming=True)
        dataset_map = load_dataset(python_large_path, split="train")
        text_key = "content"
        print(f"[DATASET] Using python_large from local path: {python_large_path}")
    elif dataset_name == "cpp_large":
        dataset_stream = load_dataset(
            "TempestTeam/dataset-the-stack-v2-dedup-sub",
            name="C++",
            split="train",
            streaming=True,
        )
        dataset_map = None  # too large to load fully
        text_key = "content"
        print("[DATASET] Using cpp_large: TempestTeam/dataset-the-stack-v2-dedup-sub (C++ subset)")
    elif dataset_name == "cpp_small":
        # we delibaretly do not stream tiny dataset
        dataset_stream = load_dataset("shibing624/source_code", "cpp", split="train")
        dataset_map = load_dataset("shibing624/source_code", "cpp", split="train")
        text_key = "text"
        print("[DATASET] Using cpp_small: shibing624/source_code (subset='cpp')")
    elif dataset_name == "python_small":
        dataset_stream = load_dataset("shibing624/source_code", "python", split="train", streaming=True)
        dataset_map = load_dataset("shibing624/source_code", "python", split="train")
        text_key = "text"
        print("[DATASET] Using python_small: shibing624/source_code (subset='python')")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return dataset_stream, dataset_map, text_key


def build_tokenizer_training_dataset(python_large_path: str) -> TokenizerStreamingDataset:
    """
    Build a mixed dataset (Python + C++) to train the tokenizer on.

    Uses:
      - python_large (codeparrot-clean) if available
      - shibing624/source_code (subset='python')
      - shibing624/source_code (subset='cpp')
    """
    datasets_and_keys = []

    if os.path.exists(python_large_path):
        try:
            ds_py_large = load_dataset(python_large_path, split="train", streaming=True)
            datasets_and_keys.append((ds_py_large, "content"))
            print(f"[TOKENIZER] Added python_large from {python_large_path} for tokenizer training.")
        except Exception as e:
            print(f"[TOKENIZER] Failed to load python_large from {python_large_path}: {e}")
    else:
        print(f"[TOKENIZER] python_large_path not found ({python_large_path}), skipping it for tokenizer training.")

    ds_py_small = load_dataset("shibing624/source_code", "python", split="train", streaming=True)
    datasets_and_keys.append((ds_py_small, "text"))
    print("[TOKENIZER] Added shibing624/source_code (python) for tokenizer training.")

    ds_cpp_small = load_dataset("shibing624/source_code", "cpp", split="train", streaming=True)
    datasets_and_keys.append((ds_cpp_small, "text"))
    print("[TOKENIZER] Added shibing624/source_code (cpp) for tokenizer training.")

    return TokenizerStreamingDataset(datasets_and_keys)


def train(
    model,
    dataset,
    batch_size=16,
    epochs=3,
    lr=3e-4,
    grad_accum_steps=2,
    checkpoint_dir="checkpoints",
    save_every=10,  # kept for compatibility, but not used for epoch-based saving now
    resume_from=None,
    amp_dtype=torch.float16,
    use_autocast=True,
    use_scaler=True,
    total_rows: Optional[int] = None,
    save_every_minutes: Optional[float] = None,
    text_key: str = "content",
):
    """Training with checkpoint saving and resuming.

    - Shows row-level progress: processed X / Y rows (Z%).
    - Saves a checkpoint every `save_every_minutes` minutes (time-based).
    - Removes previous checkpoint file when a new one is saved.
    - Keeps and updates a best-model checkpoint based on running avg loss.
    """
    
    stream_dataset = StreamingDataset(
        dataset,
        tok,
        seq_length=CONTEXT_SIZE,
        max_files_per_epoch=None,  # or an int like 100_000 if you want to cap
        text_key=text_key,
    )
    # IterableDataset does not support shuffle=True; we also require num_workers=0
    # so that stats (files_yielded) are visible in this process.
    dataloader = DataLoader(stream_dataset, batch_size=batch_size, num_workers=0)

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

    # Global stats
    global_step = 0

    # Time-based checkpointing
    last_ckpt_time = time.time()
    last_ckpt_path: Optional[str] = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")

    def save_time_based_checkpoint(current_epoch: int, current_loss: float):
        nonlocal last_ckpt_path, last_ckpt_time
        # Use your existing helper; it returns the epoch-specific path.
        ckpt_path = save_checkpoint(model, optimizer, current_epoch, current_loss, checkpoint_dir)

        # Remove previous epoch checkpoint if different
        if last_ckpt_path is not None and last_ckpt_path != ckpt_path and os.path.exists(last_ckpt_path):
            try:
                os.remove(last_ckpt_path)
                print(f"[CHECKPOINT] Removed previous checkpoint: {last_ckpt_path}")
            except OSError as e:
                print(f"[CHECKPOINT] Failed to remove previous checkpoint {last_ckpt_path}: {e}")

        last_ckpt_path = ckpt_path
        last_ckpt_time = time.time()
        print(f"[CHECKPOINT] Saved checkpoint: {ckpt_path}")

    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        print(f"[STATUS] Starting epoch {epoch}/{epochs}")

        for step, batch in enumerate(dataloader):
            global_step += 1

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

            # Time-based checkpointing
            if save_every_minutes is not None and save_every_minutes > 0:
                elapsed_minutes = (time.time() - last_ckpt_time) / 60.0
                if elapsed_minutes >= save_every_minutes:
                    avg_loss_so_far = total_loss / max(1, num_batches)
                    print(f"[CHECKPOINT] {elapsed_minutes:.2f} minutes elapsed since last checkpoint. "
                          f"Saving at epoch {epoch}, step {step}, global_step {global_step}.")
                    save_time_based_checkpoint(epoch, avg_loss_so_far)

            # Periodic logging + best model updates
            if step % 10 == 0:
                avg_loss = total_loss / max(1, num_batches)

                # Row progress: processed X / total_rows rows
                if total_rows is not None and total_rows > 0:
                    rows_seen = getattr(stream_dataset, "files_yielded", 0)
                    row_pct = rows_seen / total_rows * 100.0
                    row_info = f" | processed {rows_seen:,}/{total_rows:,} rows ({row_pct:5.2f}%)"
                else:
                    row_info = ""

                print(
                    f"epoch {epoch:3d}/{epochs} | "
                    f"step {step:5d} | "
                    f"loss {avg_loss:.4f}{row_info}"
                )

                # Best-model checkpoint based on running avg loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                            "config": model.config.__dict__,
                        },
                        best_path,
                    )
                    print(f"[BEST] New best model saved with running avg loss: {best_loss:.4f} "
                          f"at epoch {epoch}, step {step}, global_step {global_step}")

        # End of epoch summary
        avg_epoch_loss = total_loss / max(1, num_batches)
        print(f"epoch {epoch:3d}/{epochs} | avg_loss {avg_epoch_loss:.4f}")

        # Optionally, also update best at epoch boundary (keeps old behavior)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "step": -1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": model.config.__dict__,
                },
                best_path,
            )
            print(f"[BEST] New best model (epoch avg) saved with loss: {best_loss:.4f}")

        # Final safety: at the very end of training, we can ensure a checkpoint exists
        if epoch == epochs:
            avg_loss_for_save = avg_epoch_loss
            print("[CHECKPOINT] Final epoch reached, saving final checkpoint.")
            save_time_based_checkpoint(epoch, avg_loss_for_save)


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
    parser.add_argument("--model_arch", type=str, choices=["deepseek"], default="deepseek")

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["python_large", 
                 "cpp_small", 
                 "cpp_large", 
                 "python_small"],
        default="python_large",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--python_large_path",
        type=str,
        default="/home/i.afanasyev/codeparrot-clean",
        help="Local path to codeparrot-clean (used for python_large and tokenizer training)",
    )

    # Precision settings
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Compute precision for weights + activations: fp32, fp16, or bf16",
    )
    
    # Checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=50,
                        help="(Unused for epochs now, kept for compatibility)")
    parser.add_argument(
        "--save_every_minutes",
        type=float,
        default=10.0,
        help="Save checkpoint every N minutes (time-based). Set <=0 to disable.",
    )
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Resume from 'latest', or path to specific checkpoint")
    parser.add_argument("--save_final_model", action="store_true", default=True, 
                       help="Save final model after training")
    
    # tokenizer settings
    parser.add_argument("--tok_path", type=str, default="./tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument(
        "--tok_train_samples",
        type=int,
        default=2000,
        help="Max samples for tokenizer training across mixed Python+C++ datasets",
    )
    args = parser.parse_args()

    # --- Tokenizer (Python + C++) ---
    if os.path.exists(args.tok_path):
        tok = BPETokenizer(tokenizer_path=args.tok_path)
        print(f"[STATUS] Loaded existing tokenizer from {args.tok_path}")
    else:
        print(f"[STATUS] Training byte-level BPE tokenizer (vocab={args.vocab_size}) on mixed Python+C++ datasets...")
        tokenizer_dataset = build_tokenizer_training_dataset(args.python_large_path)
        tok = BPETokenizer()
        tok.train(
            dataset=tokenizer_dataset,
            vocab_size=args.vocab_size, 
            save_path=args.tok_path,
            max_samples=args.tok_train_samples,
        )
        print(f"Saved tokenizer to {args.tok_path}")
    print("[STATUS] tokenizer prepared.")

    # --- Load data for training (selected dataset) ---
    print("[STATUS] preparing dataset...")
    dataset_stream, dataset_map, text_key = load_training_dataset(args.dataset, args.python_large_path)
    print("[STATUS] streaming dataset loaded.")

    if dataset_map is not None:
        total_rows = len(dataset_map)
        print(f"[STATUS] map-style dataset loaded. total_rows = {total_rows:,}")
    else:
        total_rows = None
        print("[STATUS] no map-style dataset (skipping row count).")

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
            print("Incorrect architecture of model provided")
            exit(1)
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
    train(
        model,
        dataset_stream, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        amp_dtype=amp_dtype,
        use_autocast=use_autocast,
        use_scaler=use_scaler,
        total_rows=total_rows,
        save_every_minutes=args.save_every_minutes,
        text_key=text_key,
    )

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
