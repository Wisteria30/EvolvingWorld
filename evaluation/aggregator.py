"""
Score aggregation module

Aggregates evaluation scores from different layers into final CHARACTER Score and WORLD Score.

Aggregation strategy:
- Per-Scene metrics: simple average across all scenes
- Cross-Scene Per-Character metrics: weighted average by participation (scene count)
- Cross-Scene Global metrics: no weighting needed (single instance)
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("evaluation")

# ============================================================================ #
# Metric classification
# ============================================================================ #

# CHARACTER metrics from Per-Scene layer
PER_SCENE_CHARACTER_METRICS = ["PF", "SSF", "MDB", "PUF", "EA", "EU", "CR", "NP", "MQ", "IC_char"]

# WORLD metrics from Per-Scene layer
PER_SCENE_WORLD_METRICS = ["CSR", "LSR", "TSO", "GUS", "GSA", "LUS", "LSA", "IC_world"]

# Cross-Scene Per-Character metrics
CROSS_SCENE_CHARACTER_METRICS = ["PES"]

# Cross-Scene Global metrics
CROSS_SCENE_GLOBAL_METRICS = ["SCC"]

# Task -> Metric mapping (excluding IC_char / IC_world, which are penalized separately during eval)
TASK_TO_METRICS = {
    "scene_cast": ["CSR", "SCC"],
    "location_scenario": ["LSR", "SCC"],
    "next_character": ["TSO"],
    "world_update": ["GUS", "GSA", "LUS", "LSA"],
    "interaction_gen": ["PF", "SSF", "MDB", "EA", "EU", "CR", "NP"],
    "character_update": ["PUF", "PES"],
    "motivation_update": ["MQ"],
}


def safe_get_score(metrics: Dict, metric_name: str) -> Optional[float]:
    """Safely retrieve a score from the metrics dict."""
    if metric_name in metrics and isinstance(metrics[metric_name], dict):
        score = metrics[metric_name].get("score")
        if score is not None:
            return float(score)
    return None


def aggregate_per_scene_scores(
    per_scene_results: List[Dict[str, Any]],
    ic_penalty: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """Aggregate Per-Scene evaluation scores across all scenes (simple average).
    
    Args:
        per_scene_results: list of evaluation results per scene
        ic_penalty: optional, penalty dict from compute_error_ic_penalty,
                    containing IC_world_penalty and IC_char_penalty.
                    penalty values will be deducted from judge scores.
    
    Returns:
        dict of {metric_name: average_score_or_none}
    """
    all_metrics = PER_SCENE_CHARACTER_METRICS + PER_SCENE_WORLD_METRICS
    aggregated = {}
    
    for metric_name in all_metrics:
        scores = []
        for result in per_scene_results:
            if result is None:
                continue
            metrics = result.get("metrics", {})
            score = safe_get_score(metrics, metric_name)
            if score is not None:
                scores.append(score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            
            # Apply error penalty to IC_world and IC_char
            if ic_penalty is not None:
                if metric_name == "IC_world" and ic_penalty.get("IC_world_penalty") is not None:
                    penalty = ic_penalty["IC_world_penalty"]
                    penalized = max(0.0, avg_score - penalty)
                    logger.info(
                        "IC_world penalty applied: %.2f - %.2f = %.2f",
                        avg_score, penalty, penalized
                    )
                    avg_score = penalized
                elif metric_name == "IC_char" and ic_penalty.get("IC_char_penalty") is not None:
                    penalty = ic_penalty["IC_char_penalty"]
                    penalized = max(0.0, avg_score - penalty)
                    logger.info(
                        "IC_char penalty applied: %.2f - %.2f = %.2f",
                        avg_score, penalty, penalized
                    )
                    avg_score = penalized
            
            aggregated[metric_name] = avg_score
        else:
            logger.warning("No valid scores for Per-Scene metric: %s", metric_name)
            aggregated[metric_name] = None
    
    return aggregated


def aggregate_cross_scene_character_scores(
    char_results: List[Dict[str, Any]],
    char_weights: Dict[str, float],
) -> Dict[str, Optional[float]]:
    """Aggregate Cross-Scene evaluation scores across all characters (participation-weighted average).
    
    Args:
        char_results: list of (char_name, eval_result) tuples
        char_weights: dict of {char_name: normalized_weight}
    
    Returns:
        dict of {metric_name: weighted_average_score_or_none}
    """
    aggregated = {}
    
    for metric_name in CROSS_SCENE_CHARACTER_METRICS:
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for char_result in char_results:
            char_name = char_result.get("char_name", "")
            result = char_result.get("result")
            if result is None:
                continue
            
            metrics = result.get("metrics", {})
            score = safe_get_score(metrics, metric_name)
            weight = char_weights.get(char_name, 1.0)
            
            if score is not None:
                weighted_sum += score * weight
                weight_sum += weight
        
        if weight_sum > 0:
            aggregated[metric_name] = weighted_sum / weight_sum
        else:
            logger.warning("No valid scores for Cross-Scene Character metric: %s", metric_name)
            aggregated[metric_name] = None
    
    return aggregated


def aggregate_cross_scene_global_scores(
    global_result: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Extract Cross-Scene evaluation scores for global state."""
    aggregated = {}
    
    for metric_name in CROSS_SCENE_GLOBAL_METRICS:
        if global_result is None:
            aggregated[metric_name] = None
            continue
        
        metrics = global_result.get("metrics", {})
        score = safe_get_score(metrics, metric_name)
        aggregated[metric_name] = score if score is not None else None
    
    return aggregated


def compute_character_weights(
    char_dynamic: Dict,
    all_scenes: Dict,
) -> Dict[str, float]:
    """Compute character weights based on scene participation count.
    
    Returns:
        dict of {char_name: normalized_weight}
    """
    from data_loader import get_active_characters
    
    active_chars = get_active_characters(char_dynamic)
    scenes = all_scenes.get("scenes", [])
    
    weights = {}
    for char_name in active_chars:
        count = sum(
            1 for scene in scenes
            if char_name in scene.get("involved_characters", [])
        )
        weights[char_name] = max(count, 1)  # minimum weight of 1
    
    # normalization
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights


def compute_task_penalty(
    eval_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply penalty to the metric corresponding to the task_name in error_attribution.
    
    For error samples with scenes:
        penalized_score = existing_avg * num_scenes / (num_scenes + 1)
        (equivalent to adding a 0-scored scene to the existing N scenes, then averaging)
    
    For 0-scene error samples:
        set the corresponding metric directly to 0
    
    IC_char / IC_world are not handled here (already penalized separately during eval).
    
    Args:
        eval_result: eval json content of a single sample
    
    Returns:
        penalized result dict, containing sample_name, penalized_final_scores, etc.
    """
    error_attribution = eval_result.get("error_attribution", {})
    final_scores = eval_result.get("final_scores", {})
    num_scenes = eval_result.get("stats", {}).get("num_scenes", 0)
    if num_scenes == 0:
        num_scenes = eval_result.get("num_scenes", 0)
    
    # Deep copy final_scores as penalized version
    import copy
    penalized_scores = copy.deepcopy(final_scores)
    
    has_error = error_attribution.get("has_error", False)
    error_source = error_attribution.get("error_source")
    task_name = error_attribution.get("task_name")
    
    # Only penalize non-infra task errors
    penalized_metrics = []  # Record penalized metrics
    if has_error and error_source not in ("infra", None) and task_name:
        target_metrics = TASK_TO_METRICS.get(task_name, [])
        
        if num_scenes == 0:
            # 0-scene error sample: set corresponding metric to 0
            for m in target_metrics:
                if m in (PER_SCENE_CHARACTER_METRICS + CROSS_SCENE_CHARACTER_METRICS):
                    if penalized_scores.get("character_metrics") is not None:
                        penalized_scores["character_metrics"][m] = 0.0
                        penalized_metrics.append(m)
                elif m in (PER_SCENE_WORLD_METRICS + CROSS_SCENE_GLOBAL_METRICS):
                    if penalized_scores.get("world_metrics") is not None:
                        penalized_scores["world_metrics"][m] = 0.0
                        penalized_metrics.append(m)
        else:
            # error sample with scenes: penalized = avg * N / (N + 1)
            for m in target_metrics:
                # Skip IC metric (already handled separately)
                if m in ("IC_char", "IC_world"):
                    continue
                
                current_score = None
                metric_group = None
                if m in (PER_SCENE_CHARACTER_METRICS + CROSS_SCENE_CHARACTER_METRICS):
                    current_score = penalized_scores.get("character_metrics", {}).get(m)
                    metric_group = "character_metrics"
                elif m in (PER_SCENE_WORLD_METRICS + CROSS_SCENE_GLOBAL_METRICS):
                    current_score = penalized_scores.get("world_metrics", {}).get(m)
                    metric_group = "world_metrics"
                
                if current_score is not None and metric_group:
                    penalized_score = round(current_score * num_scenes / (num_scenes + 1), 2)
                    penalized_scores[metric_group][m] = penalized_score
                    penalized_metrics.append(m)
    
    # Recalculate CHARACTER_score and WORLD_score
    char_metrics = penalized_scores.get("character_metrics", {})
    world_metrics = penalized_scores.get("world_metrics", {})
    
    char_values = [v for v in char_metrics.values() if v is not None]
    world_values = [v for v in world_metrics.values() if v is not None]
    
    penalized_scores["CHARACTER_score"] = (
        round(sum(char_values) / len(char_values), 2)
        if len(char_values) == len(char_metrics) and char_values
        else None
    )
    penalized_scores["WORLD_score"] = (
        round(sum(world_values) / len(world_values), 2)
        if len(world_values) == len(world_metrics) and world_values
        else None
    )
    
    return {
        "sample_name": eval_result.get("sample_name", ""),
        "tag": eval_result.get("tag", "unknown"),
        "error_attribution": error_attribution,
        "num_scenes": num_scenes,
        "penalized_metrics": penalized_metrics,
        "original_final_scores": final_scores,
        "penalized_final_scores": penalized_scores,
        "stats": eval_result.get("stats", {}),
    }


def compute_final_scores(
    per_scene_scores: Dict[str, Optional[float]],
    cross_char_scores: Dict[str, Optional[float]],
    cross_global_scores: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Compute the final CHARACTER Score and WORLD Score.
    
    CHARACTER Score = mean of all CHARACTER metrics
    WORLD Score = mean of all WORLD metrics
    
    If any required metric is missing (None), the total returns None to avoid silently recording failures as 50.
    """
    def _round_optional(value: Optional[float]) -> Optional[float]:
        return round(value, 2) if value is not None else None
    
    # Collect all CHARACTER metric scores
    character_scores: Dict[str, Optional[float]] = {}
    for m in PER_SCENE_CHARACTER_METRICS:
        character_scores[m] = per_scene_scores.get(m)
    for m in CROSS_SCENE_CHARACTER_METRICS:
        character_scores[m] = cross_char_scores.get(m)
    
    # Collect all WORLD metric scores
    world_scores: Dict[str, Optional[float]] = {}
    for m in PER_SCENE_WORLD_METRICS:
        world_scores[m] = per_scene_scores.get(m)
    for m in CROSS_SCENE_GLOBAL_METRICS:
        world_scores[m] = cross_global_scores.get(m)
    
    char_values = [v for v in character_scores.values() if v is not None]
    world_values = [v for v in world_scores.values() if v is not None]
    
    char_avg = None if len(char_values) != len(character_scores) or not char_values else sum(char_values) / len(char_values)
    world_avg = None if len(world_values) != len(world_scores) or not world_values else sum(world_values) / len(world_values)
    
    return {
        "CHARACTER_score": _round_optional(char_avg),
        "WORLD_score": _round_optional(world_avg),
        "character_metrics": {k: _round_optional(v) for k, v in character_scores.items()},
        "world_metrics": {k: _round_optional(v) for k, v in world_scores.items()},
    }
