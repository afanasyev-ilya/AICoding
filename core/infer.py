# infer.py
from tokenizer import BPETokenizer
from model import MoEGPTConfig, MoEGPT
import argparse
from tests.prompts import *
from checkpoints import _get_precision_for_inference, load_model
import torch
import torch.nn as nn
import time


def generate(model, 
             tokenizer,
             precision,
             prompt="",
             max_new_tokens=100,
             temperature=0.3,
             measure_speed: bool = True):
    # Run inference
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

    if measure_speed:
        total_tokens = len(out)
        prompt_tokens = len(ids)
        new_tokens = max(total_tokens - prompt_tokens, 0)
        toks_per_sec = 0.0 if elapsed <= 0.0 else new_tokens / elapsed
        print(
            f"[speed] generated {new_tokens} tokens in {elapsed:.3f}s "
            f"({toks_per_sec:.2f} tok/s) "
            f"(prompt={prompt_tokens}, total={total_tokens})"
        )

    generated = tokenizer.decode(out)
    return generated

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
        help="Custom prompt string (if not set, uses binary_search prompt)",
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
        default=0.3,
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

    """
    Load model and tokenizer for inference.

    precision: "fp32", "fp16", or "bf16"
      - "fp16": weights -> float16, autocast(float16)
      - "bf16": weights -> bfloat16, autocast(bfloat16)
      - "fp32": weights -> float32, no autocast
    """
    # Load tokenizer
    print("[STATUS] loading model and preparing tokenizer...")
    tokenizer = BPETokenizer(args.tokenizer_path)

    # Choose precision
    param_dtype, _, _ = _get_precision_for_inference(args.precision)

    # Load model
    model = load_model(MoEGPT, args.model_path)
    model = model.to(device="cuda", dtype=param_dtype)
    model.eval()
    print("[STATUS] model loaded.")

    for prompt in CPP_PROMPTS:
        print(f"=== Prompt ===\n{prompt}\n")

        output = generate(model, tokenizer, args.precision, prompt=prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

        print("=== Output ===")
        print(output)


if __name__ == "__main__":
    main()
