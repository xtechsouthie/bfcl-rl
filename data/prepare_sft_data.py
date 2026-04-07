"""
Prepare the Salesforce/xlam-function-calling-60k dataset for SFT training.

- Shuffles the full dataset (mitigates model-generation bias)
- Splits into train (57,000) and eval (3,000) sets
- Parses conversations into proper chat format
- Saves processed dataset to data/sft_processed/

Run:
    python data/prepare_sft_data.py
    python data/prepare_sft_data.py --config configs/sft_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from project root or data/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import load_config, resolve_path


def parse_conversations(raw_conversations: str) -> List[Dict[str, str]]:
    """Parse the `conversations` field from xlam into chat message dicts.

    The xlam dataset stores conversations as a JSON string of
    [{"from": "system", "value": "..."}, {"from": "human", "value": "..."},
     {"from": "gpt", "value": "..."}, ...]

    We convert to:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}, ...]
    """
    role_map = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }

    if isinstance(raw_conversations, str):
        convs = json.loads(raw_conversations)
    else:
        convs = raw_conversations

    messages: List[Dict[str, str]] = []
    for turn in convs:
        role = role_map.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})

    return messages


def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single example into chat messages and count tool complexity."""
    messages = parse_conversations(example["conversations"])

    # Count number of tools for curriculum learning metadata
    tools_str = example.get("tools", "[]")
    try:
        tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        num_tools = len(tools) if isinstance(tools, list) else 0
    except (json.JSONDecodeError, TypeError):
        num_tools = 0

    return {
        "messages": json.dumps(messages),  # Store as JSON string for HF dataset
        "num_tools": num_tools,
    }


def prepare_sft_dataset(config: Dict[str, Any]) -> None:
    """Main data preparation pipeline for SFT."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    dataset_name: str = config.get("dataset_name", "Salesforce/xlam-function-calling-60k")
    cache_dir = resolve_path(config.get("dataset_cache_dir", "data/sft_processed"))
    raw_cache_dir = resolve_path("data/sft_raw_cache")
    train_size: int = config.get("train_size", 57000)
    eval_size: int = config.get("eval_size", 3000)
    seed: int = config.get("seed", 42)

    # ── Check for cached processed dataset ────────────────────────────────
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"✅ Processed dataset already exists at {cache_dir}")
        print("   Delete the directory to re-process.")
        return

    # ── Load raw dataset ──────────────────────────────────────────────────
    if raw_cache_dir.exists() and any(raw_cache_dir.iterdir()):
        print(f"Loading cached raw dataset from {raw_cache_dir}")
        raw_ds = load_from_disk(str(raw_cache_dir))
        # Handle DatasetDict vs Dataset
        if hasattr(raw_ds, "keys"):
            raw_ds = raw_ds["train"] if "train" in raw_ds else list(raw_ds.values())[0]
    else:
        print(f"Downloading dataset: {dataset_name}")
        raw_ds = load_dataset(dataset_name, split="train")
        raw_cache_dir.mkdir(parents=True, exist_ok=True)
        raw_ds.save_to_disk(str(raw_cache_dir))
        print(f"Saved raw dataset to {raw_cache_dir}")

    total = len(raw_ds)
    print(f"Total examples: {total}")

    # ── Shuffle (mitigate DeepSeek-V2 vs Mixtral generation bias) ─────────
    print("Shuffling dataset to mitigate model-generation bias…")
    raw_ds = raw_ds.shuffle(seed=seed)

    # ── Format examples ───────────────────────────────────────────────────
    print("Formatting examples into chat messages…")
    processed_ds = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        num_proc=4,
        desc="Formatting",
    )

    # ── Split ──────────────────────────────────────────────────────────────
    needed = train_size + eval_size
    if len(processed_ds) < needed:
        print(f"⚠️  Dataset has {len(processed_ds)} examples, "
              f"need {needed}. Using all available.")
        train_size = len(processed_ds) - eval_size

    # Stratified-ish split: shuffle is already done, so we just slice
    train_ds = processed_ds.select(range(train_size))
    eval_ds = processed_ds.select(range(train_size, train_size + eval_size))

    print(f"Train set: {len(train_ds)} examples")
    print(f"Eval set:  {len(eval_ds)} examples")

    # ── Save ──────────────────────────────────────────────────────────────
    ds_dict = DatasetDict({"train": train_ds, "eval": eval_ds})
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(cache_dir))
    print(f"✅ Saved processed dataset to {cache_dir}")

    # ── Print sample ──────────────────────────────────────────────────────
    sample = train_ds[0]
    msgs = json.loads(sample["messages"])
    print(f"\n── Sample (num_tools={sample['num_tools']}) ──")
    for msg in msgs[:3]:
        print(f"  [{msg['role']}]: {msg['content'][:120]}…")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "configs" / "sft_config.yaml"),
        help="Path to SFT config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_sft_dataset(config)


if __name__ == "__main__":
    main()
