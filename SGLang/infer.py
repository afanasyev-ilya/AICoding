#!/usr/bin/env python3
import gc
import sys
import time
import threading

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import sglang as sgl
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from transformers import AutoTokenizer

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None
    print("[WARN] transformers not installed -> model param count will be skipped.")
    print("       Install with: pip install transformers")

MODEL_ID = "unsloth/Qwen3-0.6B"
# Force dtype so our size estimate is meaningful
DTYPE_STR = "float16"   # you can change to "bfloat16" or "float32" if you want

GB = 1024 ** 3


# ---------- Helpers ----------
def bytes_per_param(dtype_str: str) -> int:
    dtype_str = dtype_str.lower()
    if dtype_str in ["float16", "half", "fp16", "bfloat16", "bf16"]:
        return 2
    if dtype_str in ["float32", "float", "fp32"]:
        return 4
    if dtype_str in ["fp8", "fp8_e5m2", "fp8_e4m3"]:
        return 1
    # fallback guess
    return 2


def print_model_size(model_id: str, dtype_str: str):
    """Load model once with transformers to estimate param count & size."""
    if AutoModelForCausalLM is None:
        print("[info] Skipping model param count (transformers not available).")
        return

    print(f"[model] Loading '{model_id}' with transformers to estimate size...")
    start = time.time()
    # This loads to CPU by default – fine for 0.6B
    model = AutoModelForCausalLM.from_pretrained(model_id)
    load_t = time.time() - start

    num_params = sum(p.numel() for p in model.parameters())
    bpp = bytes_per_param(dtype_str)
    size_bytes = num_params * bpp
    size_gb = size_bytes / GB

    print(f"[model] Loaded in {load_t:.1f}s")
    print(f"[model] Number of parameters: {num_params:,}")
    print(f"[model] Assumed dtype: {dtype_str} ({bpp} bytes/param)")
    print(f"[model] Theoretical weight size: {size_gb:.3f} GB\n")

    # Free CPU RAM
    del model
    gc.collect()


def print_gpu_overview():
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available.")
        return

    num_devices = torch.cuda.device_count()
    print(f"[gpu] Detected {num_devices} CUDA device(s):")
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / GB
        print(f"  GPU {i}: {props.name} ({total_gb:.2f} GB)")


def print_gpu_memory(tag: str = ""):
    """Use global CUDA mem_get_info so it sees usage from all processes."""
    if not torch.cuda.is_available():
        return

    print(f"[gpu] Memory usage {tag}")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        print(
            f"  GPU {i}: used {used/GB:.3f} GB / total {total/GB:.3f} GB "
            f"(free {free/GB:.3f} GB)"
        )
    print("")


def spinner(message: str, stop_event: threading.Event):
    symbols = "|/-\\"
    i = 0
    while not stop_event.is_set():
        sys.stdout.write("\r" + message + " " + symbols[i % len(symbols)])
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + message + " done.\n")
    sys.stdout.flush()


# ---------- Main ----------
def main():
    enable_progress_bars()

    print_gpu_overview()
    print_gpu_memory("[before anything]")

    # 1) Hugging Face download with progress bar
    print(f"[1/4] Checking / downloading model '{MODEL_ID}' from Hugging Face...")
    local_model_path = snapshot_download(repo_id=MODEL_ID)
    print(f"[1/4] Model snapshot ready at: {local_model_path}\n")

    # 2) Model param count + theoretical size
    print("[2/4] Estimating model parameter count & theoretical size...")
    print_model_size(MODEL_ID, DTYPE_STR)

    # 3) Initialize SGLang engine (this is where real GPU usage jumps)
    print_gpu_memory("[right before Engine init]")

    print("[3/4] Initializing SGLang engine (this can take a bit)...")
    stop_event = threading.Event()
    spin_thread = threading.Thread(
        target=spinner,
        args=("[3/4] Initializing SGLang engine...", stop_event),
        daemon=True,
    )
    spin_thread.start()

    start_time = time.time()
    engine = sgl.Engine(
        model_path=local_model_path,
        random_seed=42,
        dtype=DTYPE_STR,   # force float16 for now
        # tp_size=1,       # single-GPU; see below for tp_size=2 example
    )
    elapsed = time.time() - start_time

    stop_event.set()
    spin_thread.join()
    print(f"[3/4] Engine initialized in {elapsed:.1f} seconds.")
    print_gpu_memory("[after Engine init]")

    # 4) Inference
    print("[4/4] Running inference...")
    prompt = "What is the capital of Russia?"
    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": 64,
    }

    t0 = time.time()
    result = engine.generate(prompt, sampling_params)
    t1 = time.time()
    gen_time = t1 - t0

    print_gpu_memory("[after first generate()]")

    text = result["text"]

    # --- token accounting ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    out_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    num_prompt_tokens = prompt_ids.shape[-1]
    num_output_tokens = out_ids.shape[-1]

    # In many setups text == full completion *without* re-including the prompt.
    # If it ever includes the prompt, this still works as "total tokens emitted".
    tokens_per_sec_total = num_output_tokens / gen_time if gen_time > 0 else float("inf")

    print("\n=== OUTPUT ===")
    print(text)

    print(f"\n[stats] Prompt tokens:      {num_prompt_tokens}")
    print(f"[stats] Output tokens:      {num_output_tokens}")
    print(f"[stats] Generation time:    {gen_time:.3f} s")
    print(f"[stats] Tokens/sec (output): {tokens_per_sec_total:.2f}")

    engine.shutdown()
    print("\nAll done. ✅")


if __name__ == "__main__":
    main()
