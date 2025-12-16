# infer.py
import argparse
import time

import torch
import torch.nn as nn  # noqa: F401  (if you don't use it, you can remove)
from contextlib import nullcontext

from tokenizer import BPETokenizer
from model import MoEGPTConfig, MoEGPT
from tests.prompts import CPP_PROMPTS
from checkpoints import _get_precision_for_inference, load_model

# ---------- ANSI COLORS ----------
COLOR_PROMPT = "\033[92m"   # bright green
COLOR_GEN    = "\033[96m"   # cyan
COLOR_RESET  = "\033[0m"
# -------------------------------


def generate(
    model,
    tokenizer,
    precision,
    prompt="",
    max_new_tokens=100,
    temperature=0.3,
    measure_speed: bool = True,
):
    # Tokenize prompt
    ids = tokenizer.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device="cuda")

    _, amp_dtype, use_autocast = _get_precision_for_inference(precision)

    if use_autocast and amp_dtype != torch.float32:
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    start = time.perf_counter()
    with torch.no_grad():
        with autocast_ctx:
            out = model.generate(
                ctx,
                max_new_tokens,
                temperature=temperature,
            )[0].tolist()
    elapsed = time.perf_counter() - start

    total_tokens = len(out)
    prompt_tokens = len(ids)
    new_tokens = max(total_tokens - prompt_tokens, 0)

    if measure_speed:
        toks_per_sec = 0.0 if elapsed <= 0.0 else new_tokens / elapsed
        print(
            f"[speed] generated {new_tokens} tokens in {elapsed:.3f}s "
            f"({toks_per_sec:.2f} tok/s) "
            f"(prompt={prompt_tokens}, total={total_tokens})"
        )

    # Decode full text and only the newly generated part
    full_text = tokenizer.decode(out)
    gen_text = tokenizer.decode(out[prompt_tokens:]) if new_tokens > 0 else ""

    return full_text, gen_text


def main():
    parser = argparse.ArgumentParser(description="Inference from saved MoEGPT model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/model_final.pt",
        help="Path to saved model file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="saved_models/tokenizer.json",
        help="Path to saved tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt string (if not set, uses CPP_PROMPTS list)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Precision for inference: fp32, fp16, or bf16",
    )

    args = parser.parse_args()

    print("[STATUS] loading model and preparing tokenizer...")
    tokenizer = BPETokenizer(args.tokenizer_path)

    # Choose precision for weights
    param_dtype, _, _ = _get_precision_for_inference(args.precision)

    model = load_model(MoEGPT, args.model_path)
    model = model.to(device="cuda", dtype=param_dtype)
    model.eval()
    print("[STATUS] model loaded.")

    # If user passed a single custom prompt, just run that
    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        prompts = CPP_PROMPTS

    for idx, prompt in enumerate(prompts):
        print("\n" + "=" * 80)
        print(f"[TEST] Prompt #{idx}")
        print("-" * 80)

        full_text, gen_text = generate(
            model,
            tokenizer,
            args.precision,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            measure_speed=True,
        )

        # Colored display: prompt in one color, completion in another
        # We print the *original* prompt string, then only gen_text.
        print(f"{COLOR_PROMPT}{prompt}{COLOR_GEN}{gen_text}{COLOR_RESET}")

        # If you also want the raw full text for logging/debug, you can print:
        # print("\n[DEBUG full text]")
        # print(full_text)


if __name__ == "__main__":
    main()
