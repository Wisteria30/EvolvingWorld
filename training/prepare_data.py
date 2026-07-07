"""
Data preparation script for LLaMA-Factory SFT training.

Supports multiple training modes:
  1. balanced   - Equal number of samples per task within each model
  2. full       - Use all available data
  3. custom     - User-defined ratios per task

Additionally supports mixing with general-domain data (e.g., tulu3)
for preventing catastrophic forgetting, following CoSER & AdaMARP.

Usage:
    python training/prepare_data.py --config training/train_config.yaml
    python training/prepare_data.py --model model_a --mode balanced --tulu3_ratio 1.0
    python training/prepare_data.py --model model_b --mode full --tulu3_path dataset/tulu3_sft_sharegpt.json
"""

import os
import json
import random
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Default paths (relative to EvolvingWorld root)
# ============================================================
DEFAULT_DATA_DIR = Path("dataset/train")
DEFAULT_OUTPUT_DIR = Path("training/prepared_data")


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-root-relative path."""
    return Path(path_value).expanduser().resolve()

# Task definitions for each model
MODEL_A_TASKS = ["scene_cast", "location_scenario", "next_character", "world_update"]
MODEL_B_TASKS = ["interaction_gen", "character_update", "motivation_update"]

TASK_FILE_MAP = {
    # Model A tasks
    "scene_cast":          "model_a_scene_cast.json",
    "location_scenario":  "model_a_location_scenario.json",
    "next_character":    "model_a_next_character.json",
    "world_update":      "model_a_world_update.json",
    # Model B tasks
    "interaction_gen":   "model_b_interaction_gen.json",
    "character_update":  "model_b_character_update.json",
    "motivation_update": "model_b_motivation_update.json",
}


def _sample_to_messages(sample: dict) -> List[dict]:
    """Convert a ShareGPT sample to OpenAI-style messages for token counting."""
    messages = []
    for msg in sample.get("conversations", []):
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role == "human":
            mapped_role = "user"
        elif role == "assistant":
            mapped_role = "assistant"
        elif role == "system":
            mapped_role = "system"
        else:
            mapped_role = role or "user"
        messages.append({"role": mapped_role, "content": content})
    return messages


def _build_token_length_counter(model_name_or_path: str):
    """
    Build a callable that returns token length for a ShareGPT sample.
    Uses the base model tokenizer when available, and falls back to plain tokenization.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    def count_sample_tokens(sample: dict) -> int:
        messages = _sample_to_messages(sample)
        if not messages:
            return 0

        if getattr(tokenizer, "chat_template", None):
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(token_ids)
            except Exception:
                pass

        text = "\n".join(msg.get("content", "") for msg in messages)
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    return count_sample_tokens


def filter_overlong_samples(
    data: List[dict],
    cutoff_len: int,
    count_tokens_fn,
    label: str,
) -> tuple[List[dict], int]:
    """
    Drop samples whose tokenized length exceeds cutoff_len.
    """
    kept = []
    dropped = 0
    for sample in data:
        try:
            token_len = count_tokens_fn(sample)
        except Exception:
            token_len = None

        if token_len is not None and token_len > cutoff_len:
            dropped += 1
            continue
        kept.append(sample)

    if dropped > 0:
        print(
            f"  Filtered overlong samples from {label}: "
            f"dropped {dropped}, kept {len(kept)} (cutoff_len={cutoff_len})"
        )
    else:
        print(f"  No overlong samples dropped from {label} (cutoff_len={cutoff_len})")
    return kept, dropped


def load_task_data(data_dir: Path, task_name: str) -> List[dict]:
    """Load sharegpt data for a specific task."""
    file_path = data_dir / TASK_FILE_MAP[task_name]
    if not file_path.exists():
        raise FileNotFoundError(f"Task data not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {task_name}: {len(data)} samples")
    return data


def load_tulu3_data(tulu3_path: str) -> List[dict]:
    """
    Load tulu3 general-domain SFT data.
    Expected format: sharegpt (list of dicts with 'conversations' key).
    Supports both .json and .jsonl formats.
    """
    path = Path(tulu3_path)
    if not path.exists():
        raise FileNotFoundError(f"Tulu3 data not found: {path}")

    data = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(f"  Loaded tulu3 general-domain data: {len(data)} samples from {path}")
    return data


def normalize_sharegpt_sample(sample: dict) -> dict:
    """Normalize a ShareGPT sample so every JSONL row has the same top-level columns.

    Handles two system-prompt conventions:
      1. Top-level ``"system"`` field (e.g. tulu3 data)
      2. First conversation turn with ``"from": "system"`` (e.g. EvolvingWorld data)

    The output always uses the top-level ``"system"`` field and strips any
    leading system turn from ``conversations``.
    """
    conversations = list(sample.get("conversations", []))
    system = sample.get("system", "")

    # If the first conversation turn is a system message, extract it
    if conversations and conversations[0].get("from") == "system":
        # Only override if top-level system is not already set
        if not system:
            system = conversations[0].get("value", "")
        conversations = conversations[1:]

    return {
        "conversations": conversations,
        "system": system,
    }


def write_jsonl(path: Path, data: List[dict]) -> None:
    """Write ShareGPT samples as JSON Lines for large-dataset friendliness."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            normalized = normalize_sharegpt_sample(sample)
            f.write(json.dumps(normalized, ensure_ascii=False))
            f.write("\n")


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records without loading the whole file into memory."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def sample_to_count(data: List[dict], target_count: int, seed: int = 42) -> List[dict]:
    """
    Sample data to a target count.
    If target_count >= len(data), return all data.
    Otherwise, randomly sample target_count items.
    """
    if target_count >= len(data):
        return data[:]
    rng = random.Random(seed)
    return rng.sample(data, target_count)


def prepare_balanced(task_data: Dict[str, List[dict]], seed: int = 42) -> Dict[str, List[dict]]:
    """
    Balance tasks to have equal number of samples (1:1:1...).
    The minimum task size determines the count.
    """
    min_count = min(len(v) for v in task_data.values())
    print(f"  Balanced mode: min task size = {min_count}, sampling all tasks to {min_count}")
    result = {}
    for task_name, data in task_data.items():
        result[task_name] = sample_to_count(data, min_count, seed)
    return result


def prepare_custom_ratio(
    task_data: Dict[str, List[dict]],
    ratios: Dict[str, float],
    seed: int = 42
) -> Dict[str, List[dict]]:
    """
    Apply custom ratios to task data.
    Ratios are relative weights. The task with the smallest (data_size / ratio) 
    determines the base, and other tasks are scaled accordingly.

    Example: ratios = {"scene_cast": 1.0, "location_scenario": 1.0, "next_character": 2.0, "world_update": 1.5}
    """
    # Normalize: find the constraining task
    # For each task, max samples = data_size; desired ratio = ratio[task]
    # We need: count[task] = base * ratio[task], and count[task] <= data_size[task]
    # So base <= data_size[task] / ratio[task] for all tasks
    base = float("inf")
    has_positive_ratio = False
    for task_name, data in task_data.items():
        r = ratios.get(task_name, 1.0)
        if r > 0:
            has_positive_ratio = True
            base = min(base, len(data) / r)

    if not has_positive_ratio:
        raise ValueError(
            "Custom ratios must include at least one positive value. "
            f"Got: {ratios}"
        )

    result = {}
    for task_name, data in task_data.items():
        r = ratios.get(task_name, 1.0)
        target = int(base * r)
        result[task_name] = sample_to_count(data, target, seed)
        print(f"  Custom ratio: {task_name} -> {len(result[task_name])} samples (ratio={r})")
    return result


def prepare_full(task_data: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Use all available data (full mode)."""
    result = {}
    for task_name, data in task_data.items():
        result[task_name] = data[:]
        print(f"  Full mode: {task_name} -> {len(data)} samples")
    return result


def merge_and_shuffle(
    task_data: Dict[str, List[dict]],
    tulu3_data: Optional[List[dict]] = None,
    tulu3_ratio: float = 1.0,
    seed: int = 42
) -> List[dict]:
    """
    Merge all task data and optionally mix with tulu3 general-domain data.

    tulu3_ratio: ratio of tulu3 data relative to total domain-specific data.
                 e.g., 1.0 means equal amount of tulu3 data as domain data.
                       0.5 means half as much tulu3 data.
    """
    rng = random.Random(seed)

    # Merge all task data
    all_domain_data = []
    for task_name, data in task_data.items():
        all_domain_data.extend(data)
    total_domain = len(all_domain_data)

    # Mix with tulu3 if provided
    mixed = all_domain_data[:]
    if tulu3_data and tulu3_ratio > 0:
        target_tulu3_count = int(total_domain * tulu3_ratio)
        sampled_tulu3 = sample_to_count(tulu3_data, target_tulu3_count, seed)
        mixed.extend(sampled_tulu3)
        print(f"  Mixed {len(sampled_tulu3)} tulu3 samples (ratio={tulu3_ratio}) "
              f"with {total_domain} domain samples -> {len(mixed)} total")
    else:
        print(f"  No tulu3 mixing. Total: {len(mixed)} samples")

    rng.shuffle(mixed)
    return mixed


def split_train_eval(
    data: List[dict],
    eval_holdout_ratio: float,
    seed: int = 42
) -> tuple[List[dict], List[dict]]:
    """
    Split merged data into train and eval subsets.
    Keeps at least one eval sample when ratio > 0 and data is non-empty.
    """
    if not 0.0 <= eval_holdout_ratio < 1.0:
        raise ValueError(
            f"eval_holdout_ratio must be in [0.0, 1.0). Got: {eval_holdout_ratio}"
        )

    if not data or eval_holdout_ratio <= 0:
        return data[:], []

    eval_count = max(1, int(len(data) * eval_holdout_ratio))
    if eval_count >= len(data):
        eval_count = len(data) - 1

    rng = random.Random(seed)
    eval_indices = set(rng.sample(range(len(data)), eval_count))
    train_data = []
    eval_data = []
    for idx, sample in enumerate(data):
        if idx in eval_indices:
            eval_data.append(sample)
        else:
            train_data.append(sample)

    return train_data, eval_data


def generate_llamafactory_dataset_info(
    dataset_name: str,
    output_filename: str,
) -> dict:
    """
    Generate dataset_info.json entry for LLaMA-Factory.
    Returns the dataset info dict.
    """
    return {
        dataset_name: {
            "file_name": output_filename,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": ""
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }


def task_dataset_basename(
    model_name: str,
    task_name: str,
    mode: str,
    tulu3_suffix: str = "",
) -> str:
    """Build a per-task dataset basename without extension."""
    return f"{model_name}_{task_name}_{mode}{tulu3_suffix}"


def prepared_outputs_exist(
    output_dir: Path,
    model_name: str,
    mode: str,
    task_names: List[str],
    tulu3_ratio: float = 0.0,
    use_tulu3: bool = False,
    eval_holdout_ratio: float = 0.0,
) -> bool:
    """
    Return True when all expected prepared-data outputs for this model already exist.
    """
    suffix_parts = [model_name, mode]
    tulu3_suffix = ""
    if use_tulu3 and tulu3_ratio > 0:
        suffix_parts.append(f"tulu3_{tulu3_ratio}")
        tulu3_suffix = f"_tulu3_{tulu3_ratio}"

    dataset_suffix = "_".join(suffix_parts)
    required_paths = [
        output_dir / f"{dataset_suffix}_sharegpt.jsonl",
        output_dir / "dataset_info.json",
    ]

    for task_name in task_names:
        required_paths.append(
            output_dir / f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}.jsonl"
        )
        if eval_holdout_ratio > 0:
            required_paths.append(
                output_dir / f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}_eval.jsonl"
            )

    return all(path.exists() for path in required_paths)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LLaMA-Factory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides other args)")
    parser.add_argument("--model", type=str, required=False, choices=["model_a", "model_b"],
                        help="Which model to prepare data for")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["balanced", "full", "custom"],
                        help="Data mixing mode")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing sharegpt task data")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for prepared data")
    parser.add_argument("--tulu3_path", type=str, default=None,
                        help="Path to tulu3 general-domain SFT data")
    parser.add_argument("--tulu3_ratio", type=float, default=1.0,
                        help="Ratio of tulu3 data relative to domain data (0=no mixing)")
    parser.add_argument("--custom_ratios", type=str, default=None,
                        help='JSON string of custom ratios, e.g. \'{"scene_cast":1,"location_scenario":1,"next_character":2,"world_update":1.5}\'')
    parser.add_argument("--eval_holdout_ratio", type=float, default=0.0,
                        help="Fraction of merged data to hold out as eval data")
    parser.add_argument("--cutoff_len", type=int, default=None,
                        help="Drop samples longer than this token length before ratio mixing")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Base model path/name used to load tokenizer for overlong filtering")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # We may run for both models from config
        models_to_run = config.get("models", ["model_a", "model_b"])
        data_dir = resolve_project_path(config.get("data_dir", DEFAULT_DATA_DIR))
        output_dir = resolve_project_path(config.get("data_output_dir", DEFAULT_OUTPUT_DIR))
        raw_tulu3_path = config.get("tulu3_path", None)
        tulu3_path = resolve_project_path(raw_tulu3_path) if raw_tulu3_path else None
        tulu3_ratio = config.get("tulu3_ratio", 0.0)
        mode = config.get("mode", "full")
        eval_holdout_ratio = config.get("eval_holdout_ratio", 0.0)
        cutoff_len = config.get("cutoff_len", None)
        model_name_or_path = config.get("model_name_or_path", None)
        seed = config.get("seed", 42)
        custom_ratios = config.get("custom_ratios", {})
    else:
        if not args.model:
            parser.error("--model is required when not using --config")
        models_to_run = [args.model]
        data_dir = resolve_project_path(args.data_dir)
        output_dir = resolve_project_path(args.output_dir)
        tulu3_path = resolve_project_path(args.tulu3_path) if args.tulu3_path else None
        tulu3_ratio = args.tulu3_ratio
        mode = args.mode
        eval_holdout_ratio = args.eval_holdout_ratio
        cutoff_len = args.cutoff_len
        model_name_or_path = args.model_name_or_path
        seed = args.seed
        custom_ratios = json.loads(args.custom_ratios) if args.custom_ratios else {}

    os.makedirs(output_dir, exist_ok=True)

    # Load tulu3 data once if needed
    tulu3_data = None
    if tulu3_path and tulu3_ratio > 0:
        print(f"\n[Loading tulu3 data]")
        tulu3_data = load_tulu3_data(tulu3_path)

    # Dataset info for LLaMA-Factory (to be appended to dataset_info.json)
    all_dataset_info = {}

    count_tokens_fn = None
    if cutoff_len is not None:
        if not model_name_or_path:
            raise ValueError(
                "cutoff_len filtering requires model_name_or_path so the tokenizer can be loaded."
            )
        print(f"\n[Loading tokenizer for overlong filtering]")
        count_tokens_fn = _build_token_length_counter(str(model_name_or_path))

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Preparing data for: {model_name}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        # Determine task list
        task_names = MODEL_A_TASKS if model_name == "model_a" else MODEL_B_TASKS

        if prepared_outputs_exist(
            output_dir=output_dir,
            model_name=model_name,
            mode=mode,
            task_names=task_names,
            tulu3_ratio=tulu3_ratio,
            use_tulu3=bool(tulu3_data and tulu3_ratio > 0),
            eval_holdout_ratio=eval_holdout_ratio,
        ):
            print("\n[Skip]")
            print(
                f"  Prepared data already exists for {model_name} "
                f"({mode}{', tulu3' if tulu3_data and tulu3_ratio > 0 else ''})."
            )
            print(f"  Output directory: {output_dir}")
            print("  Skipping regeneration.")
            continue

        # Load all task data
        print(f"\n[Loading task data]")
        task_data = {}
        overlong_filter_stats = {}
        for task_name in task_names:
            task_samples = load_task_data(data_dir, task_name)
            original_count = len(task_samples)
            if count_tokens_fn is not None:
                task_samples, dropped_count = filter_overlong_samples(
                    task_samples,
                    cutoff_len=cutoff_len,
                    count_tokens_fn=count_tokens_fn,
                    label=f"{model_name}/{task_name}",
                )
            else:
                dropped_count = 0
            task_data[task_name] = task_samples
            overlong_filter_stats[task_name] = {
                "original": original_count,
                "dropped": dropped_count,
                "kept": len(task_samples),
            }

        if count_tokens_fn is not None:
            print(f"\n[Overlong sample filter summary: {model_name}]")
            total_original = 0
            total_dropped = 0
            total_kept = 0
            for task_name in task_names:
                stats = overlong_filter_stats[task_name]
                total_original += stats["original"]
                total_dropped += stats["dropped"]
                total_kept += stats["kept"]
                print(
                    f"  {task_name}: original={stats['original']}, "
                    f"dropped={stats['dropped']}, kept={stats['kept']}"
                )
            print(
                f"  TOTAL: original={total_original}, "
                f"dropped={total_dropped}, kept={total_kept}"
            )

        # Apply mode
        print(f"\n[Applying mode: {mode}]")
        if mode == "balanced":
            processed = prepare_balanced(task_data, seed)
        elif mode == "custom":
            model_ratios = custom_ratios.get(model_name, {})
            if not model_ratios:
                print(f"  WARNING: No custom ratios for {model_name}, using full mode")
                processed = prepare_full(task_data)
            else:
                processed = prepare_custom_ratio(task_data, model_ratios, seed)
        else:  # full
            processed = prepare_full(task_data)

        # Determine output filename with descriptive suffix
        suffix_parts = [model_name, mode]
        if tulu3_data and tulu3_ratio > 0:
            suffix_parts.append(f"tulu3_{tulu3_ratio}")
        dataset_suffix = "_".join(suffix_parts)
        tulu3_suffix = ""
        if tulu3_data and tulu3_ratio > 0:
            tulu3_suffix = f"_tulu3_{tulu3_ratio}"
        output_filename = f"{dataset_suffix}_sharegpt.jsonl"
        # Hold out eval data independently for each task so we can evaluate
        # the mixed model on per-task validation sets.
        print(f"\n[Splitting train/eval by task]")
        task_train_data = {}
        task_eval_data = {}
        for task_name, data in processed.items():
            train_split, eval_split = split_train_eval(
                data,
                eval_holdout_ratio=eval_holdout_ratio,
                seed=seed,
            )
            task_train_data[task_name] = train_split
            task_eval_data[task_name] = eval_split
            print(
                f"  {task_name}: train={len(train_split)}"
                f", eval={len(eval_split)}"
            )

        # Mix training data as before.
        print(f"\n[Merging and shuffling]")
        filtered_tulu3_data = tulu3_data
        if tulu3_data and count_tokens_fn is not None:
            filtered_tulu3_data, dropped_tulu3 = filter_overlong_samples(
                tulu3_data,
                cutoff_len=cutoff_len,
                count_tokens_fn=count_tokens_fn,
                label="tulu3",
            )
            print(
                f"  Tulu3 filter summary: original={len(tulu3_data)}, "
                f"dropped={dropped_tulu3}, kept={len(filtered_tulu3_data)}"
            )
        train_merged = merge_and_shuffle(
            task_train_data,
            tulu3_data=filtered_tulu3_data,
            tulu3_ratio=tulu3_ratio if filtered_tulu3_data else 0.0,
            seed=seed
        )

        # Save train split
        output_path = output_dir / output_filename
        write_jsonl(output_path, train_merged)
        print(f"\n✅ Saved train: {output_path} ({len(train_merged)} samples)")

        # Save per-task train/eval splits so eval can be tracked by task.
        for task_name in task_names:
            task_train_path = output_dir / f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}.jsonl"
            write_jsonl(task_train_path, task_train_data[task_name])
            print(f"  Saved per-task train: {task_train_path} ({len(task_train_data[task_name])} samples)")

            if task_eval_data[task_name]:
                task_eval_filename = f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}_eval.jsonl"
                task_eval_path = output_dir / task_eval_filename
                write_jsonl(task_eval_path, task_eval_data[task_name])
                print(f"  Saved per-task eval: {task_eval_path} ({len(task_eval_data[task_name])} samples)")

        # Generate dataset_info entry
        ds_info = generate_llamafactory_dataset_info(
            f"evolvingworld_{model_name}",
            output_filename,
        )
        all_dataset_info.update(ds_info)

        for task_name in task_names:
            if task_eval_data[task_name]:
                task_eval_filename = f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}_eval.jsonl"
                task_eval_ds_info = generate_llamafactory_dataset_info(
                    f"evolvingworld_{model_name}_{task_name}_eval",
                    task_eval_filename,
                )
                all_dataset_info.update(task_eval_ds_info)

    # Save dataset_info.json for LLaMA-Factory
    dataset_info_path = output_dir / "dataset_info.json"
    # Merge with existing if present
    existing_info = {}
    if dataset_info_path.exists():
        with open(dataset_info_path, "r") as f:
            existing_info = json.load(f)
    existing_info.update(all_dataset_info)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)
    print(f"\n📋 Updated dataset_info.json: {dataset_info_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for model_name in models_to_run:
        suffix_parts = [model_name, mode]
        if tulu3_data and tulu3_ratio > 0:
            suffix_parts.append(f"tulu3_{tulu3_ratio}")
        dataset_suffix = "_".join(suffix_parts)
        fname = f"{dataset_suffix}_sharegpt.jsonl"
        fpath = output_dir / fname
        if fpath.exists():
            count = count_jsonl_records(fpath)
            print(f"  {model_name}: train {fname} -> {count} samples")
        if eval_holdout_ratio > 0:
            task_names = MODEL_A_TASKS if model_name == "model_a" else MODEL_B_TASKS
            tulu3_suffix = f"_tulu3_{tulu3_ratio}" if tulu3_data and tulu3_ratio > 0 else ""
            for task_name in task_names:
                task_eval_fname = f"{task_dataset_basename(model_name, task_name, mode, tulu3_suffix)}_eval.jsonl"
                task_eval_fpath = output_dir / task_eval_fname
                if task_eval_fpath.exists():
                    task_eval_count = count_jsonl_records(task_eval_fpath)
                    print(f"  {model_name}: task eval {task_eval_fname} -> {task_eval_count} samples")
    print(f"\nOutput directory: {output_dir}")
    print(f"Ready for LLaMA-Factory training!")


if __name__ == "__main__":
    main()
