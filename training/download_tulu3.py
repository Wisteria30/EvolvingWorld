#!/usr/bin/env python3
"""
Download tulu3 SFT mixture data and convert to sharegpt format.

The tulu-3-sft-mixture dataset from Allen AI contains general-domain SFT data.
We download it, convert to sharegpt format compatible with LLaMA-Factory,
and save to the dataset/ directory.

Usage:
    python training/download_tulu3.py [--max_samples 50000] [--seed 42]
    
References:
    - https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
    - CoSER (2024): Mixed general-domain data during SFT to prevent catastrophic forgetting
    - AdaMARP (2024): Similar approach with tulu3 data
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path

OUTPUT_DIR = Path("dataset")


def download_and_convert(max_samples: int = 50000, seed: int = 42):
    """Download tulu3-sft-mixture and convert to sharegpt format."""

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Downloading allenai/tulu-3-sft-mixture from HuggingFace...")
    print("(This may take a while on first run)")
    sys.stdout.flush()

    # Load the dataset - it's a large dataset, we use streaming to save memory
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

    print("Converting to sharegpt format...")
    sys.stdout.flush()

    all_data = []
    count = 0
    skipped = 0

    # Use explicit iterator so we can close it gracefully after break
    ds_iter = iter(ds)
    try:
        for example in ds_iter:
            if max_samples > 0 and count >= max_samples * 3:
                # Load 3x then sample down for diversity
                break

            messages = example.get("messages", [])
            if not messages:
                skipped += 1
                continue

            # Convert to sharegpt format
            conversations = []
            system_text = ""
            valid = True

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    system_text = content
                elif role == "user":
                    conversations.append({"from": "human", "value": content})
                elif role == "assistant":
                    conversations.append({"from": "assistant", "value": content})
                else:
                    # Skip unknown roles
                    pass

            # Validate sharegpt format: must start with human and alternate
            if len(conversations) < 2:
                skipped += 1
                continue
            if conversations[0]["from"] != "human":
                skipped += 1
                continue

            # Check alternating pattern
            for i in range(len(conversations)):
                expected = "human" if i % 2 == 0 else "assistant"
                if conversations[i]["from"] != expected:
                    valid = False
                    break

            if not valid:
                skipped += 1
                continue

            # Must end with assistant
            if conversations[-1]["from"] != "assistant":
                skipped += 1
                continue

            item = {"conversations": conversations}
            if system_text:
                item["system"] = system_text

            all_data.append(item)
            count += 1

            if count % 10000 == 0:
                print(f"  Processed {count} valid samples (skipped {skipped})...")
                sys.stdout.flush()
    finally:
        # Gracefully close the streaming iterator to avoid
        # 'Bad file descriptor' warnings from dangling HTTP connections
        if hasattr(ds_iter, 'close'):
            try:
                ds_iter.close()
            except Exception:
                pass
        del ds_iter

    print(f"  Total valid samples collected: {len(all_data)} (skipped: {skipped})")

    # Sample down to max_samples
    if max_samples > 0 and len(all_data) > max_samples:
        rng = random.Random(seed)
        all_data = rng.sample(all_data, max_samples)
        print(f"  Sampled down to {len(all_data)} samples")

    # Save
    output_path = OUTPUT_DIR / "tulu3_sft_sharegpt.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    fsize = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Samples: {len(all_data)}")
    print(f"   Size: {fsize:.1f} MB")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download tulu3 SFT data")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Maximum number of samples to keep (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    output_path = download_and_convert(
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(f"\nNext steps:")
    print(f"  1. Update train_config.yaml:")
    print(f"     tulu3_path: {output_path}")
    print(f"     tulu3_ratio: 1.0  # or adjust as needed")
    print(f"  2. Run training with --with-tulu3 flag")


if __name__ == "__main__":
    main()
