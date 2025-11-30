#!/usr/bin/env python
import argparse
from pprint import pprint

from datasets import load_dataset


def load_apps(split: str):
    """
    Load APPS (codeparrot/apps) without using the deprecated dataset script.
    We directly read the JSONL files from the dataset repo.
    """
    if split == "train":
        data_files = "hf://datasets/codeparrot/apps/train.jsonl"
    elif split == "test":
        data_files = "hf://datasets/codeparrot/apps/test.jsonl"
    else:
        raise ValueError(f"Unsupported split for APPS: {split!r} (use 'train' or 'test')")

    ds = load_dataset("json", data_files=data_files, split="train")
    return ds


def load_mbpp(subset: str):
    """
    Load MBPP (Muennighoff/mbpp). This one is already converted to Parquet
    so the standard load_dataset call works fine.
    """
    if subset not in {"full", "sanitized"}:
        raise ValueError(f"Unsupported MBPP subset: {subset!r} (use 'full' or 'sanitized')")

    # MBPP has only a 'test' split (974 rows for 'full', 427 for 'sanitized')
    ds = load_dataset("Muennighoff/mbpp", subset, split="test")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["apps", "mbpp_full", "mbpp_sanitized"],
        default="apps",
        help="Which dataset to inspect",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split for apps (train/test). Ignored for MBPP.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="How many samples to print",
    )
    args = parser.parse_args()

    if args.dataset == "apps":
        print(f"Loading dataset: codeparrot/apps (split={args.split}) via raw JSONL...")
        ds = load_apps(args.split)
    elif args.dataset == "mbpp_full":
        print("Loading dataset: Muennighoff/mbpp (subset=full, split=test)...")
        ds = load_mbpp("full")
    else:  # mbpp_sanitized
        print("Loading dataset: Muennighoff/mbpp (subset=sanitized, split=test)...")
        ds = load_mbpp("sanitized")

    print(f"Loaded {len(ds)} rows.\n")

    n = min(args.num_samples, len(ds))
    for i in range(n):
        print("=" * 80)
        print(f"SAMPLE {i}")
        print("-" * 80)
        pprint(ds[i])
        print()


if __name__ == "__main__":
    main()
