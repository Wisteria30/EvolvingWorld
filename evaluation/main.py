"""
EvolvingWorld Evaluation Framework - Main Entry (Per-Metric Independent Evaluation)

Each metric makes a separate LLM Judge call.
Supports sample-level concurrency.

Usage:
    python evaluation/main.py [OPTIONS]

Arguments:
    --input_dir        simulation output directory (default: simulation/outputs/example_run)
    --input_snapshots  Path to test_all.json containing initial states (default: dataset/test/test_all.json)
    --output_dir       Evaluation results output directory (default: evaluation/results)
    --num_workers      Number of parallel workers (default: 8)
    --judge_model      Judge model name (default: gpt-4o)
    --samples          Sample indices to evaluate, comma-separated (e.g. "0,1,2"), default: all
    --sample_ratio     Test sample ratio (0.0-1.0), randomly selects this fraction of samples (default: 1.0)
    --sample_seed      Random seed for reproducible sampling (default: 42)
    --resume           Skip samples with valid eval.json, only rerun missing or failed ones

Examples:
    # Evaluate all samples with default parameters
    python evaluation/main.py

    # Specify judge model and concurrency
    python evaluation/main.py --judge_model gpt-4o --num_workers 16

    # Evaluate only specific samples
    python evaluation/main.py --samples 0,1,2,3

    # Resume after interruption, skip completed samples
    python evaluation/main.py --resume

    # Specify input/output directories + resume mode
    python evaluation/main.py --input_dir simulation/outputs/example_run --output_dir evaluation/results --resume
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# Ensure CWD is in sys.path
sys.path.insert(0, "evaluation")

from data_loader import (
    load_sample_data,
    sanitize_error_sample,
    dedup_world_state_history,
    classify_error_attribution,
    compute_error_ic_penalty,
    build_per_scene_slice,
    build_cross_scene_character_slice,
    build_cross_scene_global_slice,
    get_active_characters,
)
from judge import (
    JudgeClient,
    evaluate_per_scene_all_metrics,
    evaluate_cross_scene_character_all_metrics,
    evaluate_cross_scene_global_all_metrics,
)
from aggregator import (
    aggregate_per_scene_scores,
    aggregate_cross_scene_character_scores,
    aggregate_cross_scene_global_scores,
    compute_character_weights,
    compute_final_scores,
)


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Configure logger, attach handlers to  'evaluation'  logger, 
    so judge.py / aggregator.py / data_loader.py logs share the same log file.
    """
    formatter_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter_console = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter_file)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter_console)
    
    # Configure main logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Sync "evaluation" logger config (shared by judge/aggregator/data_loader)
    eval_logger = logging.getLogger("evaluation")
    eval_logger.setLevel(logging.DEBUG)
    eval_logger.handlers.clear()
    eval_logger.addHandler(fh)
    eval_logger.addHandler(ch)
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='EvolvingWorld Evaluation Framework')
    parser.add_argument(
        '--input_dir', type=str,
        default='simulation/outputs/example_run',
        help='Directory containing sample output folders (default: simulation/outputs/example_run)'
    )
    parser.add_argument(
        '--input_snapshots', type=str,
        default='dataset/test/test_all.json',
        help='Path to test_all.json containing original snapshots with initial states'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='evaluation/results',
        help='Directory to save evaluation results (default: evaluation/results)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of parallel workers for sample-level concurrency (default: 8)'
    )
    parser.add_argument(
        '--judge_model', type=str, default='gpt-4o',
        help='Judge model name (default: gpt-4o)'
    )
    parser.add_argument(
        '--samples', type=str, default=None,
        help='Comma-separated list of sample indices to evaluate (e.g., "0,1,2"). Default: all samples.'
    )
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Skip samples with valid eval.json (CHARACTER_score & WORLD_score both non-null), only rerun missing/failed'
    )
    parser.add_argument(
        '--sample_ratio', type=float, default=1.0,
        help='Ratio of test samples to evaluate (0.0-1.0). Randomly selects this fraction of samples. Default: 1.0 (all samples)'
    )
    parser.add_argument(
        '--sample_seed', type=int, default=42,
        help='Random seed for sample ratio selection, ensures reproducibility (default: 42)'
    )
    return parser.parse_args()


def load_config() -> Dict[str, str]:
    """Load root project config."""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_samples(input_dir: str, sample_indices: Optional[List[int]] = None) -> List[str]:
    """Discover all sample directories.
    
    Args:
        input_dir: input directory
        sample_indices: optional list of sample indices
    
    Returns:
        list of sample directory paths
    """
    samples = []
    for entry in sorted(os.listdir(input_dir)):
        if entry.startswith("sample_") and os.path.isdir(os.path.join(input_dir, entry)):
            # Check required files
            sample_dir = os.path.join(input_dir, entry)
            required_files = ["meta.json", "all_scenes.json", "character_dynamic.json", "world_dynamic.json"]
            if all(os.path.exists(os.path.join(sample_dir, f)) for f in required_files):
                if sample_indices is not None:
                    # Extract index from dir name
                    try:
                        idx = int(entry.split("_")[-1])
                        if idx in sample_indices:
                            samples.append(sample_dir)
                    except ValueError:
                        pass
                else:
                    samples.append(sample_dir)
    
    return samples


def _round_optional(value: Optional[float]) -> Optional[float]:
    return round(value, 2) if value is not None else None


def _round_optional_dict(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: _round_optional(v) for k, v in scores.items()}


def evaluate_single_sample(
    sample_dir: str,
    output_dir: str,
    config: Dict[str, str],
    judge_model: str,
    snapshot: Optional[Dict] = None,
    speaking_style_examples: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """Complete single-sample evaluation flow (Per-Metric independent).
    
    Each metric calls the LLM Judge once.
    
    Args:
        sample_dir: sample directory path
        output_dir: output directory
        config: API config
        judge_model: judge model name
        snapshot: optional, original test_all.json snapshot with initial states
        speaking_style_examples: optional, original book speaking style examples per character
                                  {char_name: [interaction_content, ...]}
    
    Returns:
        evaluation result dict
    """
    sample_name = os.path.basename(sample_dir)
    
    # Configure logger
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        f"eval_{sample_name}",
        os.path.join(log_dir, f"{sample_name}.log")
    )
    
    logger.info("=" * 60)
    logger.info("Starting evaluation for %s (per-metric mode)", sample_name)
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Load data
        logger.info("Loading sample data...")
        data = load_sample_data(sample_dir, snapshot=snapshot)
        
        # Before sanitize, analyze error attribution & compute IC penalty
        error_attribution = classify_error_attribution(data["meta"])
        # Save original all_scenes for API call counting (sanitize modifies scenes)
        import copy
        all_scenes_original = copy.deepcopy(data["all_scenes"])
        ic_penalty = compute_error_ic_penalty(all_scenes_original, error_attribution)
        
        if error_attribution.get("has_error") and error_attribution.get("error_source") not in ("infra", None):
            logger.info(
                "Error attribution: source=%s, task=%s, IC penalties: world=%.2f, char=%.2f",
                error_attribution["error_source"],
                error_attribution["task_name"],
                ic_penalty.get("IC_world_penalty") or 0,
                ic_penalty.get("IC_char_penalty") or 0,
            )
        
        # Clean error-terminated sample (remove incomplete last scene)
        data = sanitize_error_sample(data)
        if data is None:
            logger.warning("Sample %s has no usable scenes after sanitization, skipping.", sample_name)
            sample_tag = snapshot.get("tag", "unknown") if snapshot else "unknown"
            
            # 0-scene error sample: IC metrics get 0 (IC penalty),
            # task-corresponding metric also gets 0 (task penalty),
            # other metrics excluded (no scene data to evaluate)
            from aggregator import (
                PER_SCENE_CHARACTER_METRICS, PER_SCENE_WORLD_METRICS,
                CROSS_SCENE_CHARACTER_METRICS, CROSS_SCENE_GLOBAL_METRICS,
                TASK_TO_METRICS,
            )
            zero_char_metrics = {m: None for m in PER_SCENE_CHARACTER_METRICS + CROSS_SCENE_CHARACTER_METRICS}
            zero_world_metrics = {m: None for m in PER_SCENE_WORLD_METRICS + CROSS_SCENE_GLOBAL_METRICS}
            zero_char_metrics["IC_char"] = 0.0
            zero_world_metrics["IC_world"] = 0.0
            
            # Set corresponding metric to 0
            task_name = error_attribution.get("task_name")
            task_penalized_metrics = []
            if task_name and error_attribution.get("error_source") not in ("infra", None):
                for m in TASK_TO_METRICS.get(task_name, []):
                    if m in zero_char_metrics:
                        zero_char_metrics[m] = 0.0
                        task_penalized_metrics.append(m)
                    elif m in zero_world_metrics:
                        zero_world_metrics[m] = 0.0
                        task_penalized_metrics.append(m)
            
            logger.info(
                "0-scene error sample: IC_char=0, IC_world=0, task_penalized=%s (error_source=%s, task=%s)",
                task_penalized_metrics,
                error_attribution.get("error_source"), error_attribution.get("task_name"),
            )
            
            zero_scene_result = {
                "sample_name": sample_name,
                "tag": sample_tag,
                "error": "No usable scenes after error sanitization",
                "error_attribution": error_attribution,
                "ic_penalty": {k: v for k, v in ic_penalty.items() if v is not None} if any(v is not None for v in ic_penalty.values()) else None,
                "final_scores": {
                    "CHARACTER_score": None,
                    "WORLD_score": None,
                    "character_metrics": zero_char_metrics,
                    "world_metrics": zero_world_metrics,
                },
                "stats": {"num_scenes": 0, "turns_per_scene": [], "avg_turns_per_scene": 0},
                "token_stats": {},
            }
            
            # Save eval json so offline summarization includes IC=0 scores
            result_path = os.path.join(output_dir, f"{sample_name}_eval.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(zero_scene_result, f, ensure_ascii=False, indent=2)
            logger.info("0-scene eval result saved to %s", result_path)
            
            return zero_scene_result
        
        all_scenes = data["all_scenes"]
        char_dynamic = data["character_dynamic"]
        world_dynamic = data["world_dynamic"]
        meta = data["meta"]
        initial_states = data.get("initial_states")
        
        # Remove duplicate consecutive world state history records
        dedup_stats = dedup_world_state_history(world_dynamic, initial_states)
        
        scenes = all_scenes.get("scenes", [])
        logger.info("Loaded %d scenes for book '%s' (stop_reason: %s)",
                    len(scenes), meta.get("book_name", ""), meta.get("stop_reason", "unknown"))
        
        # Statistics
        turns_per_scene = [len(s.get("interactions", [])) for s in scenes]
        avg_turns = sum(turns_per_scene) / len(turns_per_scene) if turns_per_scene else 0
        sample_stats = {
            "num_scenes": len(scenes),
            "turns_per_scene": turns_per_scene,
            "avg_turns_per_scene": round(avg_turns, 1),
            "stop_reason": meta.get("stop_reason", "unknown"),
            "original_stop_reason": meta.get("original_stop_reason"),
        }
        logger.info("Stats: %d scenes, avg %.1f turns/scene", len(scenes), avg_turns)
        
        # 2. Init Judge
        judge = JudgeClient(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=judge_model,
            extra_headers=config.get("extra_headers"),
        )
        
        # ================================================================ #
        # Layer 1: Per-Scene evaluation
        # Sequential scenes (sliding window for scene summary required)
        # Randomly sample up to SCENES_PER_METRIC scenes per metric for evaluation
        # ================================================================ #
        logger.info("--- Layer 1: Per-Scene Evaluation (per-metric mode) ---")
        
        # Randomly sample scenes per metric (reduce evaluation cost)
        import random
        from prompts import get_per_scene_metrics
        
        SCENES_PER_METRIC = 7
        all_scene_indices = [s["scene_index"] for s in scenes]
        num_scenes = len(all_scene_indices)
        
        # scene_index -> set of metrics to evaluate for that scene
        scene_metric_map: Dict[int, set] = {idx: set() for idx in all_scene_indices}
        
        if num_scenes <= SCENES_PER_METRIC:
            # Not enough scenes for sampling; evaluate all metrics on all scenes
            for idx in all_scene_indices:
                scene_metric_map[idx] = set(get_per_scene_metrics())
            logger.info(
                "Scene count (%d) <= %d, evaluating all metrics on all scenes",
                num_scenes, SCENES_PER_METRIC
            )
        else:
            # Randomly select SCENES_PER_METRIC scenes per metric
            for metric in get_per_scene_metrics():
                sampled = random.sample(all_scene_indices, SCENES_PER_METRIC)
                for idx in sampled:
                    scene_metric_map[idx].add(metric)
            logger.info(
                "Sampling %d scenes per metric (out of %d total scenes)",
                SCENES_PER_METRIC, num_scenes
            )
            # Print how many metrics are assigned to each scene
            for idx in all_scene_indices:
                logger.info(
                    "  Scene %d: %d metrics to evaluate", idx, len(scene_metric_map[idx])
                )
        
        per_scene_results = []
        scene_summaries = {}  # {scene_index: summary}
        prev_summary = ""
        
        for scene in scenes:
            scene_index = scene["scene_index"]
            metrics_for_this_scene = scene_metric_map.get(scene_index, set())
            num_metrics_expected = len(metrics_for_this_scene)
            logger.info(
                "Evaluating scene %d/%d (%d metrics + summary)...",
                scene_index + 1, len(scenes), num_metrics_expected
            )
            
            # Build evaluation slices
            scene_slice = build_per_scene_slice(
                scene=scene,
                all_scenes=all_scenes,
                char_dynamic=char_dynamic,
                world_dynamic=world_dynamic,
                prev_scene_summary=prev_summary,
                initial_states=initial_states,
            )
            # Inject original book speaking style examples (for SSF evaluation)
            if speaking_style_examples:
                scene_slice["speaking_style_examples"] = speaking_style_examples
            
            # Evaluate sampled metrics + generate summary (summary is always generated)
            result = evaluate_per_scene_all_metrics(
                judge, scene_slice,
                metrics_to_evaluate=metrics_for_this_scene,
            )
            
            per_scene_results.append(result)
            summary = result.get("scene_summary", "")
            scene_summaries[scene_index] = summary
            prev_summary = summary
            
            num_metrics_ok = len(result.get("metrics", {}))
            logger.info(
                "Scene %d: %d/%d metrics evaluated, summary length: %d chars",
                scene_index, num_metrics_ok, num_metrics_expected, len(summary)
            )
        
        # ================================================================ #
        # Layer 1.5: Failure compensation - supplement metrics with too few successes from remaining scenes
        # ================================================================ #
        if num_scenes > SCENES_PER_METRIC:
            from judge import evaluate_per_scene_metric
            
            # Safe mapping of scene_index -> list index
            scene_index_to_pos = {s["scene_index"]: i for i, s in enumerate(scenes)}
            
            # Count successful scenes and attempted scenes per metric
            metric_success_scenes: Dict[str, set] = {m: set() for m in get_per_scene_metrics()}
            metric_tried_scenes: Dict[str, set] = {m: set() for m in get_per_scene_metrics()}
            
            # Record attempted scenes from initial sampling
            for idx in all_scene_indices:
                for m in scene_metric_map.get(idx, set()):
                    metric_tried_scenes[m].add(idx)
            
            # Count successful scenes from results
            for i, r in enumerate(per_scene_results):
                if r is None:
                    continue
                scene_idx = scenes[i]["scene_index"]
                for m in r.get("metrics", {}):
                    metric_success_scenes[m].add(scene_idx)
            
            # Find metrics that need compensation
            metrics_needing_backfill = {
                m: SCENES_PER_METRIC - len(metric_success_scenes[m])
                for m in get_per_scene_metrics()
                if len(metric_success_scenes[m]) < SCENES_PER_METRIC
            }
            
            if metrics_needing_backfill:
                logger.info(
                    "--- Layer 1.5: Backfill for %d metrics with insufficient results ---",
                    len(metrics_needing_backfill)
                )
                for m, deficit in metrics_needing_backfill.items():
                    logger.info("  %s: need %d more (have %d/%d)",
                                m, deficit, len(metric_success_scenes[m]), SCENES_PER_METRIC)
                
                # Compensate per metric
                for metric_name in list(metrics_needing_backfill.keys()):
                    remaining_deficit = metrics_needing_backfill[metric_name]
                    available_scenes = [
                        idx for idx in all_scene_indices
                        if idx not in metric_tried_scenes[metric_name]
                    ]
                    random.shuffle(available_scenes)
                    
                    for backfill_scene_idx in available_scenes:
                        if remaining_deficit <= 0:
                            break
                        
                        metric_tried_scenes[metric_name].add(backfill_scene_idx)
                        
                        # Safely get list index via mapping
                        list_pos = scene_index_to_pos[backfill_scene_idx]
                        scene_obj = scenes[list_pos]
                        # Use already generated prev_summary
                        prev_sum = scene_summaries.get(backfill_scene_idx - 1, "") if backfill_scene_idx > 0 else ""
                        backfill_slice = build_per_scene_slice(
                            scene=scene_obj,
                            all_scenes=all_scenes,
                            char_dynamic=char_dynamic,
                            world_dynamic=world_dynamic,
                            prev_scene_summary=prev_sum,
                            initial_states=initial_states,
                        )
                        # Inject original book speaking style examples (for SSF evaluation)
                        if speaking_style_examples:
                            backfill_slice["speaking_style_examples"] = speaking_style_examples
                        
                        logger.info(
                            "  Backfill: evaluating %s on scene %d...",
                            metric_name, backfill_scene_idx
                        )
                        backfill_result = evaluate_per_scene_metric(
                            judge, metric_name, backfill_slice
                        )
                        
                        if backfill_result is not None:
                            # Write results to per_scene_results for the corresponding scene
                            if per_scene_results[list_pos] is not None:
                                per_scene_results[list_pos]["metrics"][metric_name] = backfill_result
                            metric_success_scenes[metric_name].add(backfill_scene_idx)
                            remaining_deficit -= 1
                            logger.info(
                                "  Backfill: %s on scene %d succeeded (%d/%d)",
                                metric_name, backfill_scene_idx,
                                len(metric_success_scenes[metric_name]), SCENES_PER_METRIC
                            )
                        else:
                            logger.warning(
                                "  Backfill: %s on scene %d also failed",
                                metric_name, backfill_scene_idx
                            )
                    
                    if remaining_deficit > 0:
                        logger.warning(
                            "  %s: exhausted all scenes, still %d short (final: %d/%d)",
                            metric_name, remaining_deficit,
                            len(metric_success_scenes[metric_name]), SCENES_PER_METRIC
                        )
            else:
                logger.info("All metrics have sufficient results, no backfill needed.")
        
        # ================================================================ #
        # Layer 2: Cross-Scene evaluation
        # ================================================================ #
        
        # --- 2a. Cross-Scene Per-Character ---
        logger.info("--- Layer 2a: Cross-Scene Per-Character Evaluation ---")
        active_chars = get_active_characters(char_dynamic)
        logger.info("Active characters: %d (%s)", len(active_chars), ", ".join(active_chars[:5]))
        
        char_eval_results = []
        for char_name in active_chars:
            logger.info("Evaluating character trajectory: %s (1 metric)", char_name)
            char_slice = build_cross_scene_character_slice(
                char_name=char_name,
                char_dynamic=char_dynamic,
                all_scenes=all_scenes,
                scene_summaries=scene_summaries,
                initial_states=initial_states,
            )
            result = evaluate_cross_scene_character_all_metrics(judge, char_slice)
            char_eval_results.append({
                "char_name": char_name,
                "result": result,
                "num_scenes": char_slice["num_scenes_participated"],
            })
        
        # --- 2b. Cross-Scene Global ---
        logger.info("--- Layer 2b: Cross-Scene Global Evaluation (1 metric: SCC) ---")
        global_slice = build_cross_scene_global_slice(
            all_scenes=all_scenes,
            scene_summaries=scene_summaries,
        )
        global_result = evaluate_cross_scene_global_all_metrics(judge, global_slice)
        
        # ================================================================ #
        # Layer 3: Aggregation
        # ================================================================ #
        logger.info("--- Layer 3: Score Aggregation ---")
        
        # Compute weights
        char_weights = compute_character_weights(char_dynamic, all_scenes)
        
        # Aggregate scores across layers (pass IC penalty)
        per_scene_scores = aggregate_per_scene_scores(
            [r for r in per_scene_results if r is not None],
            ic_penalty=ic_penalty,
        )
        cross_char_scores = aggregate_cross_scene_character_scores(
            char_eval_results, char_weights
        )
        cross_global_scores = aggregate_cross_scene_global_scores(
            global_result
        )
        
        # Compute final scores
        final_scores = compute_final_scores(
            per_scene_scores, cross_char_scores, cross_global_scores
        )
        
        elapsed = time.time() - start_time
        logger.info("Evaluation completed in %.1f seconds", elapsed)
        logger.info("CHARACTER Score: %s", final_scores["CHARACTER_score"])
        logger.info("WORLD Score: %s", final_scores["WORLD_score"])
        
        # Extract tag (id / ood) from snapshot
        sample_tag = snapshot.get("tag", "unknown") if snapshot else "unknown"
        
        # Build complete result
        full_result = {
            "sample_name": sample_name,
            "book_name": meta.get("book_name", ""),
            "tag": sample_tag,
            "num_scenes": len(scenes),
            "num_active_characters": len(active_chars),
            "judge_model": judge_model,
            "evaluation_mode": "per-metric",
            "scenes_per_metric": SCENES_PER_METRIC if num_scenes > SCENES_PER_METRIC else num_scenes,
            "elapsed_seconds": round(elapsed, 1),
            "stats": sample_stats,
            "error_attribution": error_attribution,
            "ic_penalty": {k: v for k, v in ic_penalty.items() if v is not None} if any(v is not None for v in ic_penalty.values()) else None,
            "final_scores": final_scores,
            "per_scene_scores": _round_optional_dict(per_scene_scores),
            "cross_scene_character_scores": _round_optional_dict(cross_char_scores),
            "cross_scene_global_scores": _round_optional_dict(cross_global_scores),
            "character_weights": {k: round(v, 4) for k, v in char_weights.items()},
            "per_scene_details": [
                {
                    "scene_index": scenes[i]["scene_index"],
                    "scene_summary": r.get("scene_summary", "") if r else "",
                    "metrics": r.get("metrics", {}) if r else {},
                }
                for i, r in enumerate(per_scene_results)
            ],
            "cross_scene_character_details": [
                {
                    "char_name": cr["char_name"],
                    "num_scenes": cr["num_scenes"],
                    "metrics": cr["result"].get("metrics", {}) if cr["result"] else {},
                }
                for cr in char_eval_results
            ],
            "cross_scene_global_details": global_result.get("metrics", {}) if global_result else {},
            "token_stats": dict(judge.token_stats),
        }
        
        # Save result
        result_path = os.path.join(output_dir, f"{sample_name}_eval.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)
        
        logger.info("Results saved to %s", result_path)
        
        return full_result
        
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", sample_name, str(e), exc_info=True)
        sample_tag = snapshot.get("tag", "unknown") if snapshot else "unknown"
        return {
            "sample_name": sample_name,
            "tag": sample_tag,
            "error": str(e),
            "final_scores": {
                "CHARACTER_score": None,
                "WORLD_score": None,
            },
            "stats": {"num_scenes": 0, "turns_per_scene": [], "avg_turns_per_scene": 0},
            "token_stats": {},
        }


def _kill_child_processes(parent_pid: int) -> None:
    """Force kill all child processes of the given parent via SIGKILL."""
    try:
        import subprocess
        # Use pgrep to find all child processes
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True, text=True, timeout=5,
        )
        child_pids = [int(pid) for pid in result.stdout.strip().split() if pid.strip()]
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # child process has exited
        if child_pids:
            print(f"    killed  {len(child_pids)}  child processes: {child_pids}")
    except Exception as e:
        # If pgrep is unavailable, fall back to pkill
        try:
            os.system(f"pkill -KILL -P {parent_pid}")
        except Exception:
            print(f"    Warning: unable to auto-kill, run manually: pkill -f  'multiprocessing-fork'")


def main():
    args = parse_args()
    
    # Load config
    config = load_config()
    judge_model = args.judge_model
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure main logger
    main_logger = setup_logger(
        "evaluation_main",
        os.path.join(args.output_dir, "evaluation.log"),
    )
    
    main_logger.info("=" * 60)
    main_logger.info("EvolvingWorld Evaluation Framework (Per-Metric Mode)")
    main_logger.info("=" * 60)
    main_logger.info("Input directory: %s", args.input_dir)
    main_logger.info("Output directory: %s", args.output_dir)
    main_logger.info("Judge model: %s", judge_model)
    main_logger.info("Num workers: %d", args.num_workers)
    
    # Parse sample indices
    sample_indices = None
    if args.samples:
        sample_indices = [int(s.strip()) for s in args.samples.split(",")]
        main_logger.info("Evaluating specific samples: %s", sample_indices)
    
    # Discover samples
    sample_dirs = discover_samples(args.input_dir, sample_indices)
    main_logger.info("Found %d samples in total", len(sample_dirs))
    
    # --sample_ratio: randomly sample a fraction of samples
    if args.sample_ratio != 1.0:
        if args.sample_ratio <= 0.0 or args.sample_ratio > 1.0:
            print(f"Error: --sample_ratio must be in (0.0, 1.0], got {args.sample_ratio}")
            return
        if sample_indices is not None:
            print("Error: --samples and --sample_ratio cannot be used together")
            return
        import random as _rand
        _rand_gen = _rand.Random(args.sample_seed)
        total_before_sampling = len(sample_dirs)
        num_to_select = max(1, int(total_before_sampling * args.sample_ratio))
        sample_dirs = sorted(_rand_gen.sample(sample_dirs, num_to_select))
        main_logger.info(
            "Sample ratio %.2f (seed=%d): selected %d / %d samples",
            args.sample_ratio, args.sample_seed, num_to_select, total_before_sampling
        )
        print(f"Sample ratio {args.sample_ratio:.2f} (seed={args.sample_seed}): selected {num_to_select} / {total_before_sampling} samples")
    
    main_logger.info("Samples to evaluate: %d", len(sample_dirs))
    print(f"Found {len(sample_dirs)} samples to evaluate")
    
    if not sample_dirs:
        print("No samples found. Check input directory.")
        return
    
    # --resume: skip samples with valid eval.json
    total_sample_count = len(sample_dirs)  # save original count (before resume filtering)
    skipped_results = []  # store results of completed samples (for final summary)
    if args.resume:
        filtered_dirs = []
        for sd in sample_dirs:
            sname = os.path.basename(sd)
            eval_path = os.path.join(args.output_dir, f"{sname}_eval.json")
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r', encoding='utf-8') as _f:
                        existing = json.load(_f)
                    fs = existing.get("final_scores", {})
                    if fs.get("CHARACTER_score") is not None and fs.get("WORLD_score") is not None:
                        skipped_results.append(existing)
                        continue  # skip this sample
                except Exception:
                    pass  # file corrupted, re-evaluate
            filtered_dirs.append(sd)
        skipped_count = len(sample_dirs) - len(filtered_dirs)
        main_logger.info("Resume mode: skipping %d already-completed samples, %d to evaluate",
                         skipped_count, len(filtered_dirs))
        print(f"Resume mode: skipping {skipped_count}  completed samples, remaining:  {len(filtered_dirs)} ")
        sample_dirs = filtered_dirs
        
        if not sample_dirs:
            print("All samples already evaluated; regenerating penalty post-processing and summary.")
            # Still runs summary flow (all_results will include skipped_results)
    
    # Loading original snapshots from test_all.json (initial states before simulation)
    all_snapshots = None
    if args.input_snapshots and os.path.exists(args.input_snapshots):
        main_logger.info("Loading snapshots from %s ...", args.input_snapshots)
        print(f"Loading snapshots from {args.input_snapshots} ...")
        with open(args.input_snapshots, 'r', encoding='utf-8') as f:
            all_snapshots = json.load(f)
        main_logger.info("Loaded %d snapshots", len(all_snapshots))
        print(f"Loaded {len(all_snapshots)} snapshots")
    else:
        main_logger.warning("No snapshots file provided or file not found. Initial states will use fallback logic.")
        print("Warning: No snapshots file found. Initial states for scene_index=0 may be inaccurate.")
    
    # Loading speaking style examples (original book style reference for SSF evaluation)
    all_speaking_style_examples = None
    _style_examples_path = os.path.join(
        os.path.dirname(os.path.abspath(args.input_snapshots)) if args.input_snapshots else "",
        "speaking_style_examples.json"
    )
    if os.path.exists(_style_examples_path):
        with open(_style_examples_path, 'r', encoding='utf-8') as f:
            all_speaking_style_examples = json.load(f)
        main_logger.info("Loaded speaking style examples for %d books", len(all_speaking_style_examples))
        print(f"Loaded speaking style examples for {len(all_speaking_style_examples)} books")
    else:
        main_logger.info("No speaking_style_examples.json found at %s, SSF evaluation will proceed without style examples.", _style_examples_path)
    
    def _get_snapshot_for_sample(sample_dir: str) -> Optional[Dict]:
        """Extract corresponding snapshot from sample directory name."""
        if all_snapshots is None:
            return None
        try:
            idx = int(os.path.basename(sample_dir).split("_")[-1])
            if 0 <= idx < len(all_snapshots):
                return all_snapshots[idx]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_style_examples_for_sample(sample_dir: str) -> Optional[Dict[str, list]]:
        """Get speaking style examples by the sample's book_name."""
        if all_speaking_style_examples is None:
            return None
        snapshot = _get_snapshot_for_sample(sample_dir)
        if snapshot is None:
            return None
        book_name = snapshot.get("book_name", "")
        return all_speaking_style_examples.get(book_name)
    
    # Concurrent evaluation
    all_results = list(skipped_results)  # in resume mode, includes completed sample results
    start_time = time.time()
    
    if args.num_workers > 1 and len(sample_dirs) > 1:
        print(f"Starting parallel evaluation with {args.num_workers} workers (per-metric mode)...")
        print(f"Main process PID: {os.getpid()}  |  Press Ctrl+C to kill all child processes")

        executor = ProcessPoolExecutor(max_workers=args.num_workers)

        futures = {}
        for sample_dir in sample_dirs:
            snapshot = _get_snapshot_for_sample(sample_dir)
            style_examples = _get_style_examples_for_sample(sample_dir)
            future = executor.submit(
                evaluate_single_sample,
                sample_dir=sample_dir,
                output_dir=args.output_dir,
                config=config,
                judge_model=judge_model,
                snapshot=snapshot,
                speaking_style_examples=style_examples,
            )
            sample_tag = snapshot.get("tag", "unknown") if snapshot else "unknown"
            futures[future] = (os.path.basename(sample_dir), sample_tag)

        def _shutdown_handler(signum, frame):
            """On SIGINT/SIGTERM, cancel all pending futures and force kill child processes."""
            sig_name = signal.Signals(signum).name
            print(f"\n>>> Received  {sig_name}, terminating all child processes...")
            # Cancel all futures that have not started
            for fut in futures:
                fut.cancel()
            # Force shutdown executor (do not wait for child processes)
            executor.shutdown(wait=False, cancel_futures=True)
            # Kill all remaining child processes
            _kill_child_processes(os.getpid())
            print(">>> All child processes terminated.")
            sys.exit(1)

        # Register signal handlers
        old_sigint = signal.signal(signal.SIGINT, _shutdown_handler)
        old_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)

        try:
            completed = 0
            for future in as_completed(futures):
                sample_name, sample_tag = futures[future]
                completed += 1
                try:
                    result = future.result()
                    all_results.append(result)
                    char_score = result.get("final_scores", {}).get("CHARACTER_score", "N/A")
                    world_score = result.get("final_scores", {}).get("WORLD_score", "N/A")
                    print(f"  [{completed}/{len(futures)}] ✓ {sample_name}: CHARACTER={char_score}, WORLD={world_score}")
                except Exception as e:
                    print(f"  [{completed}/{len(futures)}] ✗ {sample_name}: Error - {str(e)}")
                    all_results.append({
                        "sample_name": sample_name,
                        "tag": sample_tag,
                        "error": str(e),
                        "final_scores": {"CHARACTER_score": None, "WORLD_score": None},
                        "token_stats": {},
                    })
        except KeyboardInterrupt:
            _shutdown_handler(signal.SIGINT, None)
        finally:
            # Restore original signal handlers and close executor
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            executor.shutdown(wait=False)
    else:
        print("Starting sequential evaluation (per-metric mode)...")
        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            print(f"  Evaluating {sample_name}...")
            snapshot = _get_snapshot_for_sample(sample_dir)
            style_examples = _get_style_examples_for_sample(sample_dir)
            result = evaluate_single_sample(
                sample_dir=sample_dir,
                output_dir=args.output_dir,
                config=config,
                judge_model=judge_model,
                snapshot=snapshot,
                speaking_style_examples=style_examples,
            )
            all_results.append(result)
            char_score = result.get("final_scores", {}).get("CHARACTER_score", "N/A")
            world_score = result.get("final_scores", {}).get("WORLD_score", "N/A")
            print(f"  ✓ {sample_name}: CHARACTER={char_score}, WORLD={world_score}")
    
    total_time = time.time() - start_time
    
    # ================================================================ #
    # Penalty post-processing: iterate all _eval.json, generate _penalized.json
    # ================================================================ #
    from aggregator import compute_task_penalty
    
    main_logger.info("--- Task Penalty Post-Processing ---")
    print("\n--- Task Penalty Post-Processing ---")
    
    # Collect currently selected sample names (for summary statistics)
    selected_sample_names = set()
    for sd in sample_dirs:
        selected_sample_names.add(os.path.basename(sd))
    # In resume mode, samples in skipped_results are also selected
    for r in skipped_results:
        sname = r.get("sample_name", "")
        if sname:
            selected_sample_names.add(sname)
    
    # Iterate all _eval.json, generate _penalized.json for each
    eval_files = sorted([
        f for f in os.listdir(args.output_dir)
        if f.endswith("_eval.json") and f.startswith("sample_")
    ])
    penalty_count = 0
    for eval_file in eval_files:
        eval_path = os.path.join(args.output_dir, eval_file)
        try:
            with open(eval_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            penalized_result = compute_task_penalty(eval_data)
            
            # Save penalized json
            penalized_file = eval_file.replace("_eval.json", "_penalized.json")
            penalized_path = os.path.join(args.output_dir, penalized_file)
            with open(penalized_path, 'w', encoding='utf-8') as f:
                json.dump(penalized_result, f, ensure_ascii=False, indent=2)
            
            if penalized_result.get("penalized_metrics"):
                penalty_count += 1
                main_logger.info(
                    "Penalized %s: metrics=%s",
                    penalized_result["sample_name"],
                    penalized_result["penalized_metrics"],
                )
        except Exception as e:
            main_logger.error("Failed to process penalty for %s: %s", eval_file, str(e))
    
    main_logger.info("Penalty post-processing done: %d/%d samples had task penalty applied", penalty_count, len(eval_files))
    print(f"Penalty post-processing done: {penalty_count}/{len(eval_files)} samples had task penalty applied")
    
    # ================================================================ #
    # Summary: generate summary from penalized scores of selected samples
    # ================================================================ #
    # Collect selected sample results from penalized json
    all_penalized_results = []
    for sname in sorted(selected_sample_names):
        penalized_path = os.path.join(args.output_dir, f"{sname}_penalized.json")
        if os.path.exists(penalized_path):
            try:
                with open(penalized_path, 'r', encoding='utf-8') as f:
                    pdata = json.load(f)
                # Use penalized_final_scores as final_scores for summary
                all_penalized_results.append({
                    "sample_name": pdata.get("sample_name", sname),
                    "tag": pdata.get("tag", "unknown"),
                    "final_scores": pdata.get("penalized_final_scores", {}),
                    "stats": pdata.get("stats", {}),
                    "penalized_metrics": pdata.get("penalized_metrics", []),
                    "original_final_scores": pdata.get("original_final_scores", {}),
                })
            except Exception as e:
                main_logger.error("Failed to load penalized result for %s: %s", sname, str(e))
        else:
            # No penalized json (should not happen, but fallback)
            eval_path = os.path.join(args.output_dir, f"{sname}_eval.json")
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r', encoding='utf-8') as f:
                        edata = json.load(f)
                    all_penalized_results.append({
                        "sample_name": edata.get("sample_name", sname),
                        "tag": edata.get("tag", "unknown"),
                        "final_scores": edata.get("final_scores", {}),
                        "stats": edata.get("stats", {}),
                        "penalized_metrics": [],
                        "original_final_scores": edata.get("final_scores", {}),
                    })
                except Exception:
                    pass
    
    # Also supplement book_name, error, etc. from all_results (for per_sample_scores)
    result_info_map = {}
    for r in all_results:
        result_info_map[r.get("sample_name", "")] = r
    valid_char_results = [
        r for r in all_penalized_results
        if r.get("final_scores", {}).get("CHARACTER_score") is not None
    ]
    valid_world_results = [
        r for r in all_penalized_results
        if r.get("final_scores", {}).get("WORLD_score") is not None
    ]
    fully_valid_results = [
        r for r in all_penalized_results
        if r.get("final_scores", {}).get("CHARACTER_score") is not None
        and r.get("final_scores", {}).get("WORLD_score") is not None
    ]

    import numpy as np
    
    # Note: avg_char / avg_world will be computed after per_metric stats,
    # by averaging the per-metric means (Plan B), see code below.
    avg_char = None
    avg_world = None
    
    # Statistics: scene count and avg turns per sample
    all_stats = []
    for r in all_penalized_results:
        stats = r.get("stats", {})
        if stats and stats.get("num_scenes", 0) > 0:
            all_stats.append(stats)
    
    total_scenes = sum(s["num_scenes"] for s in all_stats) if all_stats else 0
    all_turns = [t for s in all_stats for t in s.get("turns_per_scene", [])]
    overall_avg_turns = round(sum(all_turns) / len(all_turns), 1) if all_turns else 0
    
    # Count stop_reason distribution
    from collections import Counter, defaultdict
    stop_reason_counts = Counter()
    for r in all_penalized_results:
        stats = r.get("stats", {})
        sr = stats.get("original_stop_reason") or stats.get("stop_reason", "unknown")
        stop_reason_counts[sr] += 1
    
    # Per-metric cross-sample average scores
    from aggregator import (
        PER_SCENE_CHARACTER_METRICS, PER_SCENE_WORLD_METRICS,
        CROSS_SCENE_CHARACTER_METRICS, CROSS_SCENE_GLOBAL_METRICS,
    )
    all_character_metrics = PER_SCENE_CHARACTER_METRICS + CROSS_SCENE_CHARACTER_METRICS
    all_world_metrics = PER_SCENE_WORLD_METRICS + CROSS_SCENE_GLOBAL_METRICS
    
    per_metric_scores = defaultdict(list)  # {metric_name: [score1, score2, ...]}
    for r in all_penalized_results:
        final = r.get("final_scores", {})
        for m in all_character_metrics:
            score = final.get("character_metrics", {}).get(m)
            if score is not None:
                per_metric_scores[m].append(score)
        for m in all_world_metrics:
            score = final.get("world_metrics", {}).get(m)
            if score is not None:
                per_metric_scores[m].append(score)
    
    per_metric_avg = {}
    per_metric_std = {}
    for m, scores in per_metric_scores.items():
        if scores:
            per_metric_avg[m] = round(float(np.mean(scores)), 2)
            per_metric_std[m] = round(float(np.std(scores)), 2)
        else:
            per_metric_avg[m] = None
            per_metric_std[m] = None
    
    # Group by CHARACTER / WORLD
    avg_character_metrics = {
        m: {"mean": per_metric_avg.get(m), "std": per_metric_std.get(m)}
        for m in all_character_metrics
    }
    avg_world_metrics = {
        m: {"mean": per_metric_avg.get(m), "std": per_metric_std.get(m)}
        for m in all_world_metrics
    }
    
    # Plan B: average_CHARACTER/WORLD_score = mean of per-metric means
    char_metric_means = [per_metric_avg[m] for m in all_character_metrics if per_metric_avg.get(m) is not None]
    world_metric_means = [per_metric_avg[m] for m in all_world_metrics if per_metric_avg.get(m) is not None]
    if char_metric_means:
        avg_char = round(float(np.mean(char_metric_means)), 2)
    if world_metric_means:
        avg_world = round(float(np.mean(world_metric_means)), 2)
    
    # ================================================================ #
    # Group stats by tag (id / ood)
    # ================================================================ #
    def _compute_split_scores(results_subset: List[Dict]) -> Dict[str, Any]:
        """Compute per-metric mean and CHARACTER/WORLD total for a group of results (Plan B)."""
        valid_char = [r for r in results_subset if r.get("final_scores", {}).get("CHARACTER_score") is not None]
        valid_world = [r for r in results_subset if r.get("final_scores", {}).get("WORLD_score") is not None]
        
        # Per-metric mean
        split_metric_scores = defaultdict(list)
        for r in results_subset:
            final = r.get("final_scores", {})
            for m in all_character_metrics:
                s = final.get("character_metrics", {}).get(m)
                if s is not None:
                    split_metric_scores[m].append(s)
            for m in all_world_metrics:
                s = final.get("world_metrics", {}).get(m)
                if s is not None:
                    split_metric_scores[m].append(s)
        
        split_char_metrics = {}
        for m in all_character_metrics:
            vals = split_metric_scores.get(m, [])
            split_char_metrics[m] = {
                "mean": round(float(np.mean(vals)), 2) if vals else None,
                "std": round(float(np.std(vals)), 2) if vals else None,
            }
        split_world_metrics = {}
        for m in all_world_metrics:
            vals = split_metric_scores.get(m, [])
            split_world_metrics[m] = {
                "mean": round(float(np.mean(vals)), 2) if vals else None,
                "std": round(float(np.std(vals)), 2) if vals else None,
            }
        
        # Plan B: total = mean of per-metric means
        char_means = [split_char_metrics[m]["mean"] for m in all_character_metrics if split_char_metrics[m]["mean"] is not None]
        world_means = [split_world_metrics[m]["mean"] for m in all_world_metrics if split_world_metrics[m]["mean"] is not None]
        avg_c = round(float(np.mean(char_means)), 2) if char_means else None
        avg_w = round(float(np.mean(world_means)), 2) if world_means else None
        
        return {
            "num_samples": len(results_subset),
            "num_valid_character": len(valid_char),
            "num_valid_world": len(valid_world),
            "average_CHARACTER_score": avg_c,
            "average_WORLD_score": avg_w,
            "character_metrics": split_char_metrics,
            "world_metrics": split_world_metrics,
        }
    
    id_results = [r for r in all_penalized_results if r.get("tag") == "id"]
    ood_results = [r for r in all_penalized_results if r.get("tag") == "ood"]
    
    split_scores = {}
    if id_results:
        split_scores["id"] = _compute_split_scores(id_results)
    if ood_results:
        split_scores["ood"] = _compute_split_scores(ood_results)
    
    summary = {
        "total_samples": total_sample_count,
        "successful_samples": len(fully_valid_results),
        "failed_samples": len(all_penalized_results) - len(fully_valid_results),
        "successful_character_samples": len(valid_char_results),
        "failed_character_samples": len(all_penalized_results) - len(valid_char_results),
        "successful_world_samples": len(valid_world_results),
        "failed_world_samples": len(all_penalized_results) - len(valid_world_results),
        "judge_model": judge_model,
        "evaluation_mode": "per-metric",
        "total_time_seconds": round(total_time, 1),
        "average_CHARACTER_score": round(avg_char, 2) if avg_char is not None else None,
        "average_WORLD_score": round(avg_world, 2) if avg_world is not None else None,
        "average_character_metrics": avg_character_metrics,
        "average_world_metrics": avg_world_metrics,
        "split_scores": split_scores,
        "simulation_stats": {
            "total_scenes": total_scenes,
            "overall_avg_turns_per_scene": overall_avg_turns,
            "stop_reason_distribution": dict(stop_reason_counts),
        },
        "per_sample_scores": [
            {
                "sample_name": r.get("sample_name", ""),
                "book_name": result_info_map.get(r.get("sample_name", ""), {}).get("book_name", ""),
                "tag": r.get("tag", "unknown"),
                "CHARACTER_score": r.get("final_scores", {}).get("CHARACTER_score"),
                "WORLD_score": r.get("final_scores", {}).get("WORLD_score"),
                "num_scenes": r.get("stats", {}).get("num_scenes", 0),
                "avg_turns_per_scene": r.get("stats", {}).get("avg_turns_per_scene", 0),
                "stop_reason": r.get("stats", {}).get("original_stop_reason") or r.get("stats", {}).get("stop_reason", ""),
                "error": result_info_map.get(r.get("sample_name", ""), {}).get("error"),
                "penalized_metrics": r.get("penalized_metrics", []),
            }
            for r in all_penalized_results
        ],
    }
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary (Per-Metric Mode)")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful (both CHARACTER and WORLD available): {summary['successful_samples']}")
    print(f"Failed (either CHARACTER or WORLD missing): {summary['failed_samples']}")
    print(f"Successful CHARACTER samples: {summary['successful_character_samples']}")
    print(f"Successful WORLD samples: {summary['successful_world_samples']}")
    print(f"Total time: {summary['total_time_seconds']}s")
    if avg_char is not None:
        print(f"Average CHARACTER Score: {round(avg_char, 2)}")
    else:
        print(f"Average CHARACTER Score: N/A")
    if avg_world is not None:
        print(f"Average WORLD Score: {round(avg_world, 2)}")
    else:
        print(f"Average WORLD Score: N/A")
    print(f"\n--- Per-Metric Average Scores (CHARACTER) ---")
    for m in all_character_metrics:
        info = avg_character_metrics.get(m, {})
        mean_val = info.get("mean", "N/A") if isinstance(info, dict) else "N/A"
        std_val = info.get("std", "N/A") if isinstance(info, dict) else "N/A"
        print(f"  {m}: {mean_val} ± {std_val}")
    print(f"\n--- Per-Metric Average Scores (WORLD) ---")
    for m in all_world_metrics:
        info = avg_world_metrics.get(m, {})
        mean_val = info.get("mean", "N/A") if isinstance(info, dict) else "N/A"
        std_val = info.get("std", "N/A") if isinstance(info, dict) else "N/A"
        print(f"  {m}: {mean_val} ± {std_val}")
    
    # Print id / ood group results
    for split_tag, split_label in [("id", "In-Distribution (ID)"), ("ood", "Out-of-Distribution (OOD)")]:
        if split_tag not in split_scores:
            continue
        ss = split_scores[split_tag]
        print(f"\n{'=' * 60}")
        print(f"  {split_label} — {ss['num_samples']} samples ({ss['num_valid_character']} valid CHARACTER, {ss['num_valid_world']} valid WORLD)")
        print(f"{'=' * 60}")
        if ss["average_CHARACTER_score"] is not None:
            print(f"  Average CHARACTER Score: {ss['average_CHARACTER_score']}")
        else:
            print(f"  Average CHARACTER Score: N/A")
        if ss["average_WORLD_score"] is not None:
            print(f"  Average WORLD Score: {ss['average_WORLD_score']}")
        else:
            print(f"  Average WORLD Score: N/A")
        print(f"  --- CHARACTER Metrics ---")
        for m in all_character_metrics:
            info = ss["character_metrics"].get(m, {})
            mean_val = info.get("mean", "N/A") if isinstance(info, dict) else "N/A"
            std_val = info.get("std", "N/A") if isinstance(info, dict) else "N/A"
            print(f"    {m}: {mean_val} ± {std_val}")
        print(f"  --- WORLD Metrics ---")
        for m in all_world_metrics:
            info = ss["world_metrics"].get(m, {})
            mean_val = info.get("mean", "N/A") if isinstance(info, dict) else "N/A"
            std_val = info.get("std", "N/A") if isinstance(info, dict) else "N/A"
            print(f"    {m}: {mean_val} ± {std_val}")
    
    print(f"\n--- Simulation Stats ---")
    print(f"Total scenes: {total_scenes}")
    print(f"Overall avg turns/scene: {overall_avg_turns}")
    print(f"Stop reason distribution: {dict(stop_reason_counts)}")
    # ================================================================ #
    # Token usage summary
    # ================================================================ #
    merged_token_stats: Dict[str, List[int]] = defaultdict(list)
    for r in all_results:
        ts = r.get("token_stats", {})
        for metric_name, token_list in ts.items():
            merged_token_stats[metric_name].extend(token_list)
    
    token_summary = {}
    if merged_token_stats:
        for metric_name in sorted(merged_token_stats.keys()):
            tokens = merged_token_stats[metric_name]
            token_summary[metric_name] = {
                "count": len(tokens),
                "min": min(tokens),
                "max": max(tokens),
                "avg": round(sum(tokens) / len(tokens), 1),
            }
    
    # Add token stats to summary and re-save
    summary["token_stats"] = token_summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n--- Input Token Stats (per metric) ---")
    if token_summary:
        print(f"  {'Metric':<20} {'Count':>6} {'Min':>8} {'Max':>8} {'Avg':>10}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        total_tokens_all = 0
        for metric_name, info in token_summary.items():
            print(f"  {metric_name:<20} {info['count']:>6} {info['min']:>8} {info['max']:>8} {info['avg']:>10.1f}")
            total_tokens_all += sum(merged_token_stats[metric_name])
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        print(f"  {'TOTAL':<20} {'':<6} {'':<8} {'':<8} {total_tokens_all:>10}")
    else:
        print("  No token stats available.")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
