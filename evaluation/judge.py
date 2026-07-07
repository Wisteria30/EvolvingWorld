"""
LLM Judge call module (Per-Metric independent evaluation)

Each metric calls the LLM once, including:
- API call wrapper
- Response parsing
- Retry logic
- Per-Scene / Cross-Scene evaluation functions
"""

import json
import logging
import re
import time
from collections import defaultdict
from typing import Dict, Any, Optional, List, Set, Tuple

import openai
import tiktoken
from openai import OpenAI

logger = logging.getLogger("evaluation")


class JudgeClient:
    """LLM Judge client, wrapping API call logic."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=600,
        )
        # Token stats: {metric_name: [token_count1, token_count2, ...]}
        self.token_stats: Dict[str, List[int]] = defaultdict(list)
        # Some models do not support custom temperature (e.g. gpt-5.1 series only supports default 1)
        self._supports_temperature = not self.model.startswith('gpt-5')
        try:
            self._encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Use cl100k_base as fallback for unknown models
            self._encoder = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoder.encode(text))

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        max_retries: int = 3,
        metric_name: str = "unknown",
        suppress_parse_warning: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM Judge and parse JSON response.
        
        Args:
            system_prompt: system prompt
            user_prompt: user prompt
            max_tokens: max output tokens (single-metric eval does not need many)
            temperature: temperature parameter
            max_retries: max retry attempts
            metric_name: metric name being evaluated, for token stats
        
        Returns:
            Parsed JSON dict, returns None on failure
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Count input tokens
        input_tokens = self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        self.token_stats[metric_name].append(input_tokens)
        logger.info("  [Token Stats] metric=%s, input_tokens=%d", metric_name, input_tokens)
        
        last_response_text = None  # Record last raw response for debugging
        for attempt in range(1, max_retries + 1):
            try:
                # Choose parameter name based on model type
                use_completion_tokens = self.model.startswith('o') or self.model.startswith('gpt-5')
                token_param = 'max_completion_tokens' if use_completion_tokens else 'max_tokens'
                
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    token_param: max_tokens,
                }
                if self._supports_temperature:
                    api_params["temperature"] = temperature
                completion = self.client.chat.completions.create(**api_params)
                
                response_text = completion.choices[0].message.content or ""
                finish_reason = completion.choices[0].finish_reason
                last_response_text = response_text
                
                # Check if truncated
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (attempt %d/%d, metric=%s, "
                        "finish_reason=length, response_len=%d chars)",
                        attempt, max_retries, metric_name, len(response_text)
                    )
                    # Truncated response likely has incomplete JSON; retry directly
                    continue
                
                # Parse JSON
                result = self._parse_json_response(response_text)
                if result is not None:
                    return result
                
                if not suppress_parse_warning:
                    logger.warning(
                        "Failed to parse JSON response (attempt %d/%d, finish_reason=%s), retrying...\n"
                        "  [Raw Response] %s",
                        attempt, max_retries, finish_reason, response_text
                    )
                
            except openai.RateLimitError as e:
                retry_after_match = re.search(r'retry_after[\'"]?\s*:\s*(\d+)', str(e))
                wait_time = int(retry_after_match.group(1)) if retry_after_match else 60
                logger.warning(
                    "Rate limit exceeded (attempt %d/%d). Waiting %ds...",
                    attempt, max_retries, wait_time
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
                    
            except openai.InternalServerError as e:
                logger.warning(
                    "Internal server error (attempt %d/%d): %s",
                    attempt, max_retries, str(e)
                )
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                logger.warning(
                    "Unexpected error (attempt %d/%d): %s",
                    attempt, max_retries, str(e)
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue
        
        logger.error(
            "All %d attempts failed for judge call (metric=%s).\n"
            "  [Last Raw Response] %s",
            max_retries, metric_name,
            (last_response_text if last_response_text else "<no response>")
        )
        return None
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response.
        
        Parsing strategy (by priority):
        1. Direct json.loads
        2. Extract from ```json ... ``` blocks
        3. Find outermost { ... } and extract
        4. Preprocess by escaping newlines in strings, then scan with raw_decode
           to keep the longest JSON fragment.
        """
        # 1. Attempt direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 2. Attempt extraction from ```json ... ```
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # Attempt retry after fixing bare newlines
                try:
                    fixed = json_match.group(1).replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
        
        # 3. Attempt finding outermost { ... }
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                # Attempt retry after fixing bare newlines
                try:
                    fixed = text[brace_start:brace_end + 1].replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
        
        # 4. Preprocess + raw_decode. Scan character by character and keep the longest JSON fragment.
        # Replace all bare newlines/carriage returns with JSON-legal escape sequences
        preprocessed = text.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
        results = []
        start = 0
        decoder = json.JSONDecoder()
        while start < len(preprocessed):
            try:
                obj, end = decoder.raw_decode(preprocessed[start:])
                if isinstance(obj, dict):
                    results.append(obj)
                start += end
            except json.JSONDecodeError:
                start += 1
        
        if results:
            return max(results, key=lambda x: len(json.dumps(x)))
        
        return None


# ============================================================================ #
# Score recalculation tool
# ============================================================================ #

def _recalculate_score(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute score from merits/demerits points.
    
    Model no longer outputs score field; calculated by code.
    Take absolute value of demerit points, compute 50 + sum(merits) - sum(demerits),
    and normalize points to positive. If model still outputs score, keep as original_model_score for reference.
    """
    if not isinstance(result, dict):
        return result
    
    merits = result.get("merits", [])
    demerits = result.get("demerits", [])
    
    # Skip if no merits/demerits structure (e.g. scene_summary)
    if not isinstance(merits, list) or not isinstance(demerits, list):
        return result
    
    # Normalize merits points to positive
    for m in merits:
        if isinstance(m, dict) and "points" in m:
            m["points"] = abs(m["points"])
    
    # Normalize demerits points to positive
    for d in demerits:
        if isinstance(d, dict) and "points" in d:
            d["points"] = abs(d["points"])
    
    sum_merits = sum(m.get("points", 0) for m in merits if isinstance(m, dict))
    sum_demerits = sum(d.get("points", 0) for d in demerits if isinstance(d, dict))
    
    recalculated = min(100, max(0, 50 + sum_merits - sum_demerits))
    
    # If model still outputs score field, keep for reference
    original_score = result.pop("score", None)
    if original_score is not None:
        logger.debug(
            "Model output score ignored: model=%s, calculated=%s (merits=%d, demerits=%d)",
            original_score, recalculated, sum_merits, sum_demerits
        )
        result["original_model_score"] = original_score
    
    result["score"] = recalculated
    return result


# ============================================================================ #
# Per-Scene evaluation (one call per metric)
# ============================================================================ #

def evaluate_scene_summary(
    judge: JudgeClient,
    scene_slice: Dict[str, Any],
) -> str:
    """Generate scene summary (independent call, no scoring).
    
    Parsing strategy (by priority):
    1. Exact match "scene_summary" key
    2. Take first string-type value in JSON
    3. If JSON parsing fails completely, use raw response text as summary
    
    Returns:
        scene summary string; returns empty string on failure
    """
    from prompts import SCENE_SUMMARY_FULL_SYSTEM, build_scene_summary_user_prompt
    
    user_prompt = build_scene_summary_user_prompt(scene_slice)
    
    # First attempt normal JSON parsing path
    result = judge.call(SCENE_SUMMARY_FULL_SYSTEM, user_prompt, max_tokens=4096, metric_name="scene_summary", suppress_parse_warning=True)
    
    if result is not None:
        # Prefer exact key match
        if "scene_summary" in result:
            return result["scene_summary"]
        # fallback: take first string-type value
        for v in result.values():
            if isinstance(v, str) and len(v) > 20:
                logger.info(
                    "Scene %d: 'scene_summary' key not found, using first string value as summary",
                    scene_slice["scene_index"]
                )
                return v
    
    # JSON parsing completely failed; attempt to use raw response text as summary
    # Call again, but manually process response this time
    logger.info(
        "Scene %d: JSON parse failed or no valid string in result, "
        "attempting raw response fallback...",
        scene_slice["scene_index"]
    )
    try:
        messages = [
            {"role": "system", "content": SCENE_SUMMARY_FULL_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        use_completion_tokens = judge.model.startswith('o') or judge.model.startswith('gpt-5')
        token_param = 'max_completion_tokens' if use_completion_tokens else 'max_tokens'
        api_params = {
            "model": judge.model,
            "messages": messages,
            token_param: 4096,
        }
        if judge._supports_temperature:
            api_params["temperature"] = 0.0
        completion = judge.client.chat.completions.create(**api_params)
        raw_text = (completion.choices[0].message.content or "").strip()
        finish_reason = completion.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning(
                "Scene %d: raw response fallback also truncated (finish_reason=length, %d chars)",
                scene_slice["scene_index"], len(raw_text)
            )
        if len(raw_text) > 20:
            logger.info(
                "Scene %d: using raw response text as summary (%d chars)",
                scene_slice["scene_index"], len(raw_text)
            )
            return raw_text
    except Exception as e:
        logger.warning("Scene %d: raw response fallback also failed: %s", scene_slice["scene_index"], e)
    
    logger.warning("Scene summary generation failed for scene %d", scene_slice["scene_index"])
    return ""


def evaluate_per_scene_metric(
    judge: JudgeClient,
    metric: str,
    scene_slice: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate a single metric for a single scene.
    
    Args:
        judge: JudgeClient instance.
        metric: metric name (e.g. "PF", "SSF", etc.)
        scene_slice: Scene data slice.
    
    Returns:
        Dict with merits, demerits, score, reasoning; returns None on failure
    """
    from prompts import build_per_scene_system_prompt, build_per_scene_user_prompt
    
    try:
        system_prompt = build_per_scene_system_prompt(metric)
        user_prompt = build_per_scene_user_prompt(metric, scene_slice)
    except Exception as e:
        logger.error(
            "Per-scene metric %s prompt build failed for scene %d: %s",
            metric, scene_slice.get("scene_index", "?"), e
        )
        return None
    
    result = judge.call(system_prompt, user_prompt, metric_name=metric)
    
    if result is None:
        logger.error(
            "Per-scene metric %s evaluation failed for scene %d",
            metric, scene_slice["scene_index"]
        )
        return None
    
    # Validate required fields (merits/demerits used for scoring)
    if not isinstance(result.get("merits"), list) or not isinstance(result.get("demerits"), list):
        logger.error(
            "Per-scene metric %s result missing 'merits' or 'demerits' for scene %d",
            metric, scene_slice["scene_index"]
        )
        return None
    
    return _recalculate_score(result)


def evaluate_per_scene_paired_metrics(
    judge: JudgeClient,
    metric_a: str,
    metric_b: str,
    scene_slice: Dict[str, Any],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Merge evaluation of two metrics with same input data (one judge call produces both scores).
    
    If paired call fails, auto fallback to individual calls.
    
    Returns:
        {metric_a: result_or_None, metric_b: result_or_None}
    """
    from prompts import build_per_scene_paired_system_prompt, build_per_scene_paired_user_prompt
    
    paired_name = f"{metric_a}+{metric_b}"
    try:
        system_prompt = build_per_scene_paired_system_prompt(metric_a, metric_b)
        user_prompt = build_per_scene_paired_user_prompt(metric_a, metric_b, scene_slice)
    except Exception as e:
        logger.error(
            "Paired metric %s prompt build failed for scene %d: %s",
            paired_name, scene_slice.get("scene_index", "?"), e
        )
        # Fallback: individual calls
        return {
            metric_a: evaluate_per_scene_metric(judge, metric_a, scene_slice),
            metric_b: evaluate_per_scene_metric(judge, metric_b, scene_slice),
        }
    result = judge.call(system_prompt, user_prompt, metric_name=paired_name)
    
    # Attempt to extract independent results for both metrics from paired result
    if result is not None:
        result_a = result.get(metric_a)
        result_b = result.get(metric_b)
        
        # Both parsed successfully with merits/demerits
        if (isinstance(result_a, dict) and isinstance(result_a.get("merits"), list) and
                isinstance(result_b, dict) and isinstance(result_b.get("merits"), list)):
            logger.info(
                "  Paired evaluation %s succeeded for scene %d",
                paired_name, scene_slice["scene_index"]
            )
            return {metric_a: _recalculate_score(result_a), metric_b: _recalculate_score(result_b)}
        
        logger.warning(
            "  Paired result for %s missing expected keys for scene %d, falling back to individual calls",
            paired_name, scene_slice["scene_index"]
        )
    else:
        logger.warning(
            "  Paired evaluation %s failed for scene %d, falling back to individual calls",
            paired_name, scene_slice["scene_index"]
        )
    
    # Fallback: individual calls
    return {
        metric_a: evaluate_per_scene_metric(judge, metric_a, scene_slice),
        metric_b: evaluate_per_scene_metric(judge, metric_b, scene_slice),
    }


def evaluate_per_scene_all_metrics(
    judge: JudgeClient,
    scene_slice: Dict[str, Any],
    metrics_to_evaluate: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Evaluate all (or a specified subset of) metrics for a single scene.
    
    Metric pairs with identical input data are merged into one judge call (paired evaluation),
    remaining metrics use individual calls.
    
    Args:
        judge: JudgeClient instance.
        scene_slice: Scene data slice.
        metrics_to_evaluate: Optional set of metric names to evaluate for this scene.
                             When None, evaluate all metrics.
    
    Returns:
        {
            "scene_summary": str,
            "metrics": {metric_name: {merits, demerits, score, reasoning}, ...}
        }
    """
    from prompts import get_per_scene_metrics, PAIRED_METRICS
    
    metrics_result = {}
    
    # Build paired metric lookup: metric -> (metric_a, metric_b)
    paired_lookup = {}
    for ma, mb in PAIRED_METRICS:
        paired_lookup[ma] = (ma, mb)
        paired_lookup[mb] = (ma, mb)
    
    already_evaluated = set()
    
    for metric in get_per_scene_metrics():
        if metric in already_evaluated:
            continue
        
        # If metrics_to_evaluate is specified, skip metrics not in it
        if metrics_to_evaluate is not None and metric not in metrics_to_evaluate:
            already_evaluated.add(metric)
            continue
        
        if metric in paired_lookup:
            metric_a, metric_b = paired_lookup[metric]
            if metric_a in already_evaluated or metric_b in already_evaluated:
                # The paired counterpart was already evaluated separately (should not happen, defensive)
                logger.info("  Evaluating metric %s for scene %d...", metric, scene_slice["scene_index"])
                result = evaluate_per_scene_metric(judge, metric, scene_slice)
                if result is not None:
                    metrics_result[metric] = result
                else:
                    logger.warning("  Metric %s failed for scene %d", metric, scene_slice["scene_index"])
                already_evaluated.add(metric)
            else:
                # Check if the paired counterpart also needs evaluation
                both_needed = (metrics_to_evaluate is None or metric_b in metrics_to_evaluate) if metric == metric_a else (metrics_to_evaluate is None or metric_a in metrics_to_evaluate)
                
                if both_needed:
                    # Both needed, evaluate together
                    logger.info(
                        "  Evaluating paired metrics %s+%s for scene %d...",
                        metric_a, metric_b, scene_slice["scene_index"]
                    )
                    paired_results = evaluate_per_scene_paired_metrics(
                        judge, metric_a, metric_b, scene_slice
                    )
                    for m, r in paired_results.items():
                        if r is not None:
                            metrics_result[m] = r
                        else:
                            logger.warning("  Metric %s failed for scene %d", m, scene_slice["scene_index"])
                    already_evaluated.add(metric_a)
                    already_evaluated.add(metric_b)
                else:
                    # Only this one needed, evaluate alone
                    logger.info("  Evaluating metric %s for scene %d...", metric, scene_slice["scene_index"])
                    result = evaluate_per_scene_metric(judge, metric, scene_slice)
                    if result is not None:
                        metrics_result[metric] = result
                    else:
                        logger.warning("  Metric %s failed for scene %d", metric, scene_slice["scene_index"])
                    already_evaluated.add(metric)
        else:
            # Non-paired metric, evaluate alone
            logger.info("  Evaluating metric %s for scene %d...", metric, scene_slice["scene_index"])
            result = evaluate_per_scene_metric(judge, metric, scene_slice)
            if result is not None:
                metrics_result[metric] = result
            else:
                logger.warning("  Metric %s failed for scene %d", metric, scene_slice["scene_index"])
            already_evaluated.add(metric)
    
    # Generate scene summary (independent call, always generated regardless of metrics_to_evaluate)
    logger.info("  Generating scene summary for scene %d...", scene_slice["scene_index"])
    summary = evaluate_scene_summary(judge, scene_slice)
    
    return {
        "scene_summary": summary,
        "metrics": metrics_result,
    }


# ============================================================================ #
# Cross-Scene Per-Character evaluation (one call per metric)
# ============================================================================ #

def evaluate_cross_scene_character_metric(
    judge: JudgeClient,
    metric: str,
    char_slice: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate a single cross-scene metric for one character."""
    from prompts import build_cross_scene_character_system_prompt, build_cross_scene_character_user_prompt
    
    try:
        system_prompt = build_cross_scene_character_system_prompt(metric)
        user_prompt = build_cross_scene_character_user_prompt(metric, char_slice)
    except Exception as e:
        logger.error(
            "Cross-scene character metric %s prompt build failed for %s: %s",
            metric, char_slice.get("char_name", "?"), e
        )
        return None
    
    result = judge.call(system_prompt, user_prompt, metric_name=metric)
    
    if result is None:
        logger.error("Cross-scene character metric %s failed for %s", metric, char_slice["char_name"])
        return None
    
    # Validate required fields (merits/demerits used for scoring)
    if not isinstance(result.get("merits"), list) or not isinstance(result.get("demerits"), list):
        logger.error("Cross-scene character metric %s missing 'merits' or 'demerits' for %s", metric, char_slice["char_name"])
        return None
    
    return _recalculate_score(result)


def evaluate_cross_scene_character_all_metrics(
    judge: JudgeClient,
    char_slice: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate all cross-scene metrics for one character."""
    from prompts import get_cross_scene_character_metrics
    
    metrics_result = {}
    for metric in get_cross_scene_character_metrics():
        logger.info("  Evaluating metric %s for character %s...", metric, char_slice["char_name"])
        result = evaluate_cross_scene_character_metric(judge, metric, char_slice)
        if result is not None:
            metrics_result[metric] = result
    
    if not metrics_result:
        return None
    
    return {"metrics": metrics_result}


# ============================================================================ #
# Cross-Scene Global evaluation (one call per metric)
# ============================================================================ #

def evaluate_cross_scene_global_metric(
    judge: JudgeClient,
    metric: str,
    global_slice: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate a single cross-scene global metric."""
    from prompts import build_cross_scene_global_system_prompt, build_cross_scene_global_user_prompt
    
    try:
        system_prompt = build_cross_scene_global_system_prompt(metric)
        user_prompt = build_cross_scene_global_user_prompt(metric, global_slice)
    except Exception as e:
        logger.error("Cross-scene global metric %s prompt build failed: %s", metric, e)
        return None
    
    result = judge.call(system_prompt, user_prompt, metric_name=metric)
    
    if result is None:
        logger.error("Cross-scene global metric %s failed", metric)
        return None
    
    # Validate required fields (merits/demerits used for scoring)
    if not isinstance(result.get("merits"), list) or not isinstance(result.get("demerits"), list):
        logger.error("Cross-scene global metric %s missing 'merits' or 'demerits'", metric)
        return None
    
    return _recalculate_score(result)


def evaluate_cross_scene_global_all_metrics(
    judge: JudgeClient,
    global_slice: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate all cross-scene global metrics."""
    from prompts import get_cross_scene_global_metrics
    
    metrics_result = {}
    for metric in get_cross_scene_global_metrics():
        logger.info("  Evaluating global metric %s...", metric)
        result = evaluate_cross_scene_global_metric(judge, metric, global_slice)
        if result is not None:
            metrics_result[metric] = result
    
    if not metrics_result:
        return None
    
    return {"metrics": metrics_result}
