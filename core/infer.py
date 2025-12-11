# infer.py

import argparse
from checkpoints import inference_from_saved
from tests.prompts import *

DEFAULT_PROMPT = "def binary_search(arr, target):\n"

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

    prompt = "You need to implement binary search in python"
    print(f"=== Prompt ===\n{prompt}\n")

    output = inference_from_saved(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        precision=args.precision,
    )

    print("=== Output ===")
    print(output)


if __name__ == "__main__":
    main()
