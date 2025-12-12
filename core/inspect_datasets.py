#!/usr/bin/env python
import argparse
from pprint import pprint
import json
import os

from datasets import load_dataset


# ------------------------- Loaders -------------------------


def load_apps(split: str, streaming: bool):
    """
    Load APPS (codeparrot/apps) without using the deprecated dataset script.
    We directly read the JSONL files from the dataset repo.
    Supports streaming / non-streaming via flag.
    """
    if split == "train":
        data_files = "hf://datasets/codeparrot/apps/train.jsonl"
    elif split == "test":
        data_files = "hf://datasets/codeparrot/apps/test.jsonl"
    else:
        raise ValueError(
            f"Unsupported split for APPS: {split!r} (use 'train' or 'test')"
        )

    ds = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=streaming,
    )
    return ds


def load_stack_cpp(split: str, streaming: bool):
    """
    Load C++ subset of TempestTeam/dataset-the-stack-v2-dedup-sub.
    Supports streaming / non-streaming via flag.
    Only 'train' split exists.
    """
    if split != "train":
        raise ValueError(
            f"Unsupported split for stack_cpp: {split!r} (only 'train' is available)"
        )

    ds = load_dataset(
        "TempestTeam/dataset-the-stack-v2-dedup-sub",
        name="C++",
        split="train",
        streaming=streaming,
    )
    return ds


def load_source_code_cpp(split: str, streaming: bool):
    """
    Load cpp subset of shibing624/source_code.
    Supports streaming / non-streaming via flag.
    Splits: train / validation / test.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"Unsupported split for source_code_cpp: {split!r} "
            "(use 'train', 'validation' or 'test')"
        )

    ds = load_dataset(
        "shibing624/source_code",
        "cpp",
        split=split,
        streaming=streaming,
    )
    return ds


# ------------------------- HF size metadata -------------------------


def get_hf_size_metadata(repo_id: str):
    """
    Query the Hugging Face Dataset Viewer `/size` endpoint to get:
    - total num_rows
    - bytes (original/parquet/memory)
    - per-config and per-split stats

    This DOES NOT download the dataset itself; it only pulls metadata.

    Returns:
        dict or None (if something goes wrong)
    """
    try:
        import requests
    except ImportError:
        print(
            "Size metadata: 'requests' not installed. "
            "Run `pip install requests` to enable metadata lookup."
        )
        return None

    base_url = "https://datasets-server.huggingface.co/size"
    params = {"dataset": repo_id}

    headers = {}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        print(f"Size metadata: could not query /size endpoint for {repo_id}: {e}")
        return None


def extract_split_stats(size_json, config: str | None, split: str | None):
    """
    From the /size JSON, extract:
    - dataset-level stats
    - config-level stats (if config given)
    - split-level stats (if split given, optionally filtered by config)

    Returns a dict with keys: 'dataset', 'config', 'split', 'partial'
    (each of dataset/config/split may be None if not found).
    """
    if not size_json:
        return {"dataset": None, "config": None, "split": None, "partial": False}

    size_obj = size_json.get("size", {})
    dataset_stats = size_obj.get("dataset")
    cfg_stats = None
    split_stats = None

    configs = size_obj.get("configs", []) or []
    splits = size_obj.get("splits", []) or []

    if config is not None:
        for cfg in configs:
            if cfg.get("config") == config:
                cfg_stats = cfg
                break

    if split is not None:
        for s in splits:
            if s.get("split") == split and (config is None or s.get("config") == config):
                split_stats = s
                break

    return {
        "dataset": dataset_stats,
        "config": cfg_stats,
        "split": split_stats,
        "partial": size_json.get("partial", False),
    }


def print_meta_info(size_json, dataset_name: str, config: str | None, split: str | None,
                    chars_per_token: float):
    """
    Pretty-print metadata-derived:
    - num_rows
    - size in GiB
    - approx tokens (from parquet bytes / chars_per_token)
    """

    stats = extract_split_stats(size_json, config=config, split=split)
    partial = stats["partial"]

    def _fmt_bytes(num_bytes: int | None):
        if num_bytes is None:
            return None, None
        gib = float(num_bytes) / (1024**3)
        return gib, num_bytes

    # Dataset-level
    ds = stats["dataset"]
    if ds is not None:
        ds_rows = ds.get("num_rows")
        ds_bytes = ds.get("num_bytes_parquet_files") or ds.get("num_bytes_original_files")
        ds_gib, ds_raw_bytes = _fmt_bytes(ds_bytes)

        print(f"HF viewer (dataset={dataset_name}):")
        if ds_rows is not None:
            print(f"  total rows (all configs/splits): {ds_rows:,}")
        if ds_bytes is not None:
            print(
                f"  total parquet size: {ds_gib:.3f} GiB ({ds_raw_bytes:,} bytes)"
            )

    # Config-level
    if stats["config"] is not None:
        cfg = stats["config"]
        cfg_rows = cfg.get("num_rows")
        cfg_bytes = cfg.get("num_bytes_parquet_files") or cfg.get("num_bytes_original_files")
        cfg_gib, cfg_raw_bytes = _fmt_bytes(cfg_bytes)

        print(f"HF viewer (config={config}):")
        if cfg_rows is not None:
            print(f"  rows: {cfg_rows:,}")
        if cfg_bytes is not None:
            print(
                f"  parquet size: {cfg_gib:.3f} GiB ({cfg_raw_bytes:,} bytes)"
            )

    # Split-level
    split_rows_meta = None
    split_bytes_meta = None
    if stats["split"] is not None:
        sp = stats["split"]
        split_rows_meta = sp.get("num_rows")
        split_bytes_meta = (
            sp.get("num_bytes_parquet_files") or sp.get("num_bytes_original_files")
        )
        sp_gib, sp_raw_bytes = _fmt_bytes(split_bytes_meta)

        label = f"config={config}, split={split}" if config else f"split={split}"
        print(f"HF viewer ({label}):")
        if split_rows_meta is not None:
            print(f"  rows: {split_rows_meta:,}")
        if split_bytes_meta is not None:
            print(
                f"  parquet size: {sp_gib:.3f} GiB ({sp_raw_bytes:,} bytes)"
            )

        # Approx tokens from bytes (assuming ~1 byte per char)
        if split_bytes_meta is not None:
            approx_tokens = split_bytes_meta / chars_per_token
            print(
                f"  approx tokens ≈ {approx_tokens / 1e9:.3f}B "
                f"(parquet_bytes / chars_per_token={chars_per_token})"
            )

    if partial:
        print(
            "  [NOTE] HF viewer reports partial=true; "
            "rows/bytes may be lower than the real values."
        )

    # Return split-level stats so caller can reuse
    return split_rows_meta, split_bytes_meta


# ------------------------- Pretty printing -------------------------


def pretty_print_sample(sample, max_string_chars: int = 2000):
    """
    Nicely print a single dataset sample.
    - Special handling for APPS 'solutions' (JSON-encoded list).
    - For large strings, print only a prefix and mention truncation.
    """
    print("{")
    for key, value in sample.items():
        print(f"  {key}:")
        if key == "solutions" and isinstance(value, str):
            # Parse JSON-encoded list of solutions
            try:
                sols = json.loads(value)
            except json.JSONDecodeError:
                print("    [!! could not decode solutions JSON !!]")
                print(f"    {value}")
                print()
                continue

            for idx, sol in enumerate(sols):
                print(f"    --- solution #{idx} ---")
                print(sol)  # real newlines and tabs
                print()
        else:
            if isinstance(value, str):
                total_len = len(value)
                shown = value
                truncated = False
                if total_len > max_string_chars:
                    shown = value[:max_string_chars]
                    truncated = True

                for line in shown.splitlines():
                    print(f"    {line}")

                if truncated:
                    print(f"    ... [truncated, {total_len} characters total]")
            else:
                # Non-strings – pretty-print the Python object
                pprint(value, width=80, compact=False)
        print()
    print("}")


# ------------------------- Token estimation (row-based, local) -------------------------


def estimate_total_tokens_row_based(
    ds,
    text_fields,
    sample_size: int = 1000,
    chars_per_token: float = 4.0,
) -> float:
    """
    Very rough token count estimate for NON-STREAMING datasets:
    - Take the first `sample_size` rows.
    - Sum length of selected text fields (in characters).
    - Convert chars -> tokens with `chars_per_token`.
    - Scale by total number of rows.

    Returns:
        Approx total tokens (float).
    """
    try:
        n_rows = len(ds)
    except TypeError:
        # Streaming datasets don't support len(); this function is not for them.
        return 0.0

    if n_rows == 0:
        return 0.0

    sample_size = min(sample_size, n_rows)

    total_chars = 0
    for i in range(sample_size):
        row = ds[i]
        for key in text_fields:
            val = row.get(key)
            if isinstance(val, str):
                total_chars += len(val)

    if sample_size == 0:
        return 0.0

    avg_chars_per_row = total_chars / sample_size
    total_chars_all_rows = avg_chars_per_row * n_rows
    total_tokens = total_chars_all_rows / chars_per_token
    return total_tokens


# ------------------------- Main -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["apps", "stack_cpp", "source_code_cpp"],
        default="apps",
        help="Which dataset to inspect",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help=(
            "Split to use. "
            "apps: train/test; "
            "stack_cpp: train; "
            "source_code_cpp: train/validation/test."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="How many samples to print",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=1000,
        help="How many rows to sample for rough token count (non-streaming only)",
    )
    parser.add_argument(
        "--chars_per_token",
        type=float,
        default=4.0,
        help="Approximate characters per token for estimating token count",
    )

    # streaming / non-streaming toggle, default = streaming
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Load dataset in streaming mode (no full download). [default]",
    )
    group.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming; load dataset fully in memory (when supported).",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    # Map dataset -> HF repo id + config name for metadata
    if args.dataset == "apps":
        repo_id = "codeparrot/apps"
        config_name = None  # single config
        split_for_meta = args.split
    elif args.dataset == "stack_cpp":
        repo_id = "TempestTeam/dataset-the-stack-v2-dedup-sub"
        config_name = "C++"
        split_for_meta = "train"
    elif args.dataset == "source_code_cpp":
        repo_id = "shibing624/source_code"
        config_name = "cpp"
        split_for_meta = args.split
    else:
        repo_id = None
        config_name = None
        split_for_meta = None

    # ---- HF metadata: rows & bytes from Dataset Viewer ----
    size_meta_json = None
    if repo_id is not None:
        size_meta_json = get_hf_size_metadata(repo_id)
        if size_meta_json is not None:
            print_meta_info(
                size_meta_json,
                dataset_name=repo_id,
                config=config_name,
                split=split_for_meta,
                chars_per_token=args.chars_per_token,
            )
        print()

    # ---- Load chosen dataset (streaming or not) ----
    if args.dataset == "apps":
        mode = "STREAMING" if args.streaming else "non-streaming"
        print(
            f"Loading dataset: codeparrot/apps (split={args.split}) "
            f"via raw JSONL in {mode} mode..."
        )
        ds = load_apps(args.split, streaming=args.streaming)
        token_fields = ["question", "solutions"]
    elif args.dataset == "stack_cpp":
        mode = "STREAMING" if args.streaming else "non-streaming"
        print(
            f"Loading dataset: TempestTeam/dataset-the-stack-v2-dedup-sub "
            f"(subset=C++, split={args.split}) in {mode} mode..."
        )
        ds = load_stack_cpp(args.split, streaming=args.streaming)
        token_fields = ["content"]
    elif args.dataset == "source_code_cpp":
        mode = "STREAMING" if args.streaming else "non-streaming"
        print(
            f"Loading dataset: shibing624/source_code "
            f"(subset=cpp, split={args.split}) in {mode} mode..."
        )
        ds = load_source_code_cpp(args.split, streaming=args.streaming)
        token_fields = ["text"]
    else:
        print("Unsupported dataset")
        raise SystemExit(1)

    # Determine if it's streaming based on type (and fallback to CLI flag)
    try:
        from datasets import IterableDataset

        is_streaming = isinstance(ds, IterableDataset)
    except ImportError:
        is_streaming = args.streaming

    # -------------------- Non-streaming path --------------------
    if not is_streaming:
        try:
            n_rows = len(ds)
            print(f"Loaded {n_rows} rows (non-streaming).")
        except TypeError:
            n_rows = None
            print("Loaded non-streaming dataset, but len(ds) is not available.")

        # Row-based token estimate using local sample (optional extra)
        if n_rows is not None and token_fields:
            try:
                total_tokens = estimate_total_tokens_row_based(
                    ds,
                    token_fields,
                    sample_size=args.token_sample_size,
                    chars_per_token=args.chars_per_token,
                )
                tokens_in_b = total_tokens / 1e9
                print(
                    f"Approx. total size (local row-based) ≈ {tokens_in_b:.3f}B tokens "
                    f"(chars_per_token={args.chars_per_token})"
                )
            except Exception as e:
                print(f"Could not estimate total tokens row-based: {e}")
        print()

        # Print samples via indexing or simple iteration
        if n_rows is None:
            n = args.num_samples
            for i, sample in enumerate(ds):
                if i >= n:
                    break
                print("=" * 80)
                print(f"SAMPLE {i}")
                print("-" * 80)
                pretty_print_sample(sample)
                print()
        else:
            n = min(args.num_samples, n_rows)
            for i in range(n):
                print("=" * 80)
                print(f"SAMPLE {i}")
                print("-" * 80)
                sample = ds[i]
                pretty_print_sample(sample)
                print()

    # -------------------- Streaming path --------------------
    else:
        print("Dataset is in streaming mode; rows are read lazily from HF.\n")

        # Print a few samples by iterating (no random access)
        n = args.num_samples
        for i, sample in enumerate(ds):
            if i >= n:
                break
            print("=" * 80)
            print(f"SAMPLE {i}")
            print("-" * 80)
            pretty_print_sample(sample)
            print()

    # Try to help with any mysterious crashes by forcing cleanup
    del ds


if __name__ == "__main__":
    main()

