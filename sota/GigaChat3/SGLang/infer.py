#!/usr/bin/env python3
import gc
import sys
import time
import threading
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Recommended in the GigaChat docs for tooling that may run custom code
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

import torch
import sglang as sgl
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# --- GIGACHAT 3 LIGHTNING (BF16) ---
# This is the ~10B MoE dialog model, "GigaChat 3 Lightning"
# Model card: ai-sage/GigaChat3-10B-A1.8B-bf16
MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"

# We'll assume BF16 weights for size estimate & SGLang
DTYPE_STR = "bfloat16"

GB = 1024 ** 3

# Known param counts for big models to avoid loading full weights on CPU
KNOWN_NUM_PARAMS = {
    "ai-sage/GigaChat3-10B-A1.8B-bf16": 11_000_000_000,  # from HF card
    "ai-sage/GigaChat3-10B-A1.8B": 11_000_000_000,
}


# ---------- Helpers ----------
def bytes_per_param(dtype_str: str) -> int:
    dtype_str = dtype_str.lower()
    if dtype_str in ["float16", "half", "fp16", "bfloat16", "bf16"]:
        return 2
    if dtype_str in ["float32", "float", "fp32"]:
        return 4
    if dtype_str in ["fp8", "fp8_e5m2", "fp8_e4m3", "f8_e4m3"]:
        return 1
    # fallback guess
    return 2


def print_model_size(model_id: str, dtype_str: str):
    """Estimate param count & size without blowing up CPU RAM for 11B models."""
    bpp = bytes_per_param(dtype_str)

    if model_id in KNOWN_NUM_PARAMS:
        num_params = KNOWN_NUM_PARAMS[model_id]
        size_bytes = num_params * bpp
        size_gb = size_bytes / GB

        print(f"[model] Using known param count for {model_id}")
        print(f"[model] Number of parameters: {num_params:,}")
        print(f"[model] Assumed dtype: {dtype_str} ({bpp} bytes/param)")
        print(f"[model] Theoretical weight size: {size_gb:.3f} GB\n")
        return

    if AutoModelForCausalLM is None:
        print("[info] Skipping exact model param count (transformers not available).")
        return

    print(f"[model] Loading '{model_id}' with transformers to estimate size...")
    start = time.time()
    # NOTE: trust_remote_code=True is needed for GigaChat/other custom archs
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cpu",
    )
    load_t = time.time() - start

    num_params = sum(p.numel() for p in model.parameters())
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

    # 2) Model param count + theoretical size (using known 11B for GigaChat3)
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

    # Use BOTH A5000s via tensor parallelism
    # If you want to force single-GPU, change tp_size=1.
    engine = sgl.Engine(
        model_path=local_model_path,
        random_seed=42,
        dtype=DTYPE_STR,          # "bfloat16"
        tp_size=2,                # 2 GPUs: split weights across both A5000s
        mem_fraction_static=0.8,  # can move toward 0.88 later if you want max throughput
        trust_remote_code=True,   # required for GigaChat's custom MLA/MoE code
        allow_auto_truncate=True, # safer for long prompts
    )

    elapsed = time.time() - start_time

    stop_event.set()
    spin_thread.join()
    print(f"[3/4] Engine initialized in {elapsed:.1f} seconds.")
    print_gpu_memory("[after Engine init]")

    # 4) Inference
    print("[4/4] Running inference with GigaChat 3 Lightning...")

    # Use the official chat template for better behavior
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    user_message = "Напиши простейшее матричное умножение на С++."
    messages = [
        {"role": "user", "content": user_message},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": 1000,
    }

    t0 = time.time()
    result = engine.generate(chat_prompt, sampling_params)
    t1 = time.time()
    gen_time = t1 - t0

    print_gpu_memory("[after first generate()]")

    text = result["text"]

    # --- token accounting ---
    # Count prompt tokens as SGLang sees them (i.e., after chat template)
    prompt_ids = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    out_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    num_prompt_tokens = prompt_ids.shape[-1]
    num_output_tokens = out_ids.shape[-1]

    tokens_per_sec = (
        num_output_tokens / gen_time if gen_time > 0 else float("inf")
    )

    print("\n=== RAW MODEL OUTPUT ===")
    print(text)

    print(f"\n[stats] Prompt tokens:        {num_prompt_tokens}")
    print(f"[stats] Output tokens:        {num_output_tokens}")
    print(f"[stats] Generation time:      {gen_time:.3f} s")
    print(f"[stats] Tokens/sec (output):  {tokens_per_sec:.2f}")

    engine.shutdown()
    print("\nAll done. ✅")


if __name__ == "__main__":
    main()
