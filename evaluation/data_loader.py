"""
Data loading and preprocessing module

Loads data from simulation outputs and builds data slices for evaluation.

Data format:
- all_scenes.json: full interaction records for all scenes
  - scenes[i].scene_index: scene index
  - scenes[i].location: location of the scene
  - scenes[i].scenario: scene scenario description
  - scenes[i].involved_characters: list of participating characters
  - scenes[i].interactions: interaction records, each with characters and content

- character_dynamic.json: character dynamic data
  - characters[name].profile_history: profile change history, each with scene_index and profile
  - characters[name].scene_descriptions: per-scene character descriptions (enhanced_motivation, description, hidden_tracker)

- world_dynamic.json: world dynamic data
  - global_card_history: global state change history (scene_index, interaction_index, global_state)
  - location_cards[loc_name]: per-location state change history (scene_index, interaction_index, location_state)
  - final_global_state: final global state
  - final_location_states: final per-location states

Timing notes:
- Entries in global_card_history / location_cards are state snapshots AFTER a given interaction
- profile_history scene_index in character_dynamic is the profile AFTER the scene completes
- scene_descriptions scene_index is the character description AFTER the scene completes
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("evaluation")


def load_sample_data(sample_dir: str, snapshot: Optional[Dict] = None) -> Dict[str, Any]:
    """Load all output data for a single sample.
    
    Args:
        sample_dir: sample directory path, e.g. simulation/outputs/example_run/sample_000000
        snapshot: optional, original snapshot data from test_all.json,
                  containing initial states before simulation (world_state, character_states).
                  If provided, extracts initial states into the initial_states field.
    
    Returns:
        dict with meta, all_scenes, character_dynamic, world_dynamic, initial_states
    """
    meta_path = os.path.join(sample_dir, "meta.json")
    scenes_path = os.path.join(sample_dir, "all_scenes.json")
    char_path = os.path.join(sample_dir, "character_dynamic.json")
    world_path = os.path.join(sample_dir, "world_dynamic.json")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(scenes_path, 'r', encoding='utf-8') as f:
        all_scenes = json.load(f)
    with open(char_path, 'r', encoding='utf-8') as f:
        char_dynamic = json.load(f)
    with open(world_path, 'r', encoding='utf-8') as f:
        world_dynamic = json.load(f)
    
    # Extract initial states from snapshot (pre-simulation)
    initial_states = None
    if snapshot is not None:
        ws = snapshot.get("world_state", {})
        cs = snapshot.get("character_states", {})
        initial_states = {
            "global_state": ws.get("global_state", ""),
            "location_states": ws.get("location_states", {}),
            "character_profiles": {
                name: char.get("profile", "") for name, char in cs.items()
            },
            "character_short_descriptions": {
                name: char.get("short_description", "") for name, char in cs.items()
            },
        }
    
    return {
        "meta": meta,
        "all_scenes": all_scenes,
        "character_dynamic": char_dynamic,
        "world_dynamic": world_dynamic,
        "initial_states": initial_states,
    }


def sanitize_error_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean data from abnormally terminated samples.
    
    When simulation ends abnormally (error like parse failure, or in_progress from external interrupt),
    the last scene may be incomplete (partial interactions, missing character_reflections,
    or partially updated world state).
    
    This function:
    1. Removes the last scene from all_scenes
    2. Removes the last scene profile_history & scene_descriptions from character_dynamic
    3. Removes the last scene global_card_history & location_cards updates from world_dynamic
    4. Rolls back final_global_state & final_location_states to end of previous scene
    
    If stop_reason is not "error" or "in_progress", return data unchanged.
    If no scenes remain after removal, return None (sample unusable).
    
    Args:
        data: dict from load_sample_data
    
    Returns:
        cleaned data dict, or None if no usable scenes
    """
    meta = data["meta"]
    stop_reason = meta.get("stop_reason", "")
    
    # error: parse failure etc.; in_progress: externally interrupted
    if stop_reason not in ("error", "in_progress"):
        return data
    
    all_scenes = data["all_scenes"]
    char_dynamic = data["character_dynamic"]
    world_dynamic = data["world_dynamic"]
    
    scenes = all_scenes.get("scenes", [])
    if not scenes:
        return None
    
    # Get index of the last scene to remove
    last_scene = scenes[-1]
    last_scene_index = last_scene.get("scene_index")
    
    logger.info(
        "Sanitizing error sample: removing last scene (scene_index=%s) with %d interactions",
        last_scene_index, len(last_scene.get("interactions", []))
    )
    
    # 1. Remove last scene
    scenes = scenes[:-1]
    all_scenes["scenes"] = scenes
    
    if not scenes:
        logger.warning("No scenes remaining after removing error scene")
        return None
    
    # 2. Remove last scene records from character_dynamic
    characters = char_dynamic.get("characters", {})
    for char_name, char_data in characters.items():
        # Remove profile_history entries with scene_index == last_scene_index
        char_data["profile_history"] = [
            h for h in char_data.get("profile_history", [])
            if h.get("scene_index") != last_scene_index
        ]
        # Remove scene_descriptions entries with scene_index == last_scene_index
        char_data["scene_descriptions"] = [
            d for d in char_data.get("scene_descriptions", [])
            if d.get("scene_index") != last_scene_index
        ]
    
    # 3. Remove last scene update records from world_dynamic
    # 3a. global_card_history
    world_dynamic["global_card_history"] = [
        h for h in world_dynamic.get("global_card_history", [])
        if h.get("scene_index") != last_scene_index
    ]
    
    # 3b. location_cards
    location_cards = world_dynamic.get("location_cards", {})
    for loc_name in list(location_cards.keys()):
        location_cards[loc_name] = [
            h for h in location_cards[loc_name]
            if h.get("scene_index") != last_scene_index
        ]
        # If location has no records, keep empty list (key may have initial state)
    
    # 4. Roll back final states to penultimate scene end
    prev_last_scene_index = scenes[-1].get("scene_index")
    
    # 4a. Roll back final_global_state
    global_history = world_dynamic.get("global_card_history", [])
    if global_history:
        # Use last record global_state as final
        world_dynamic["final_global_state"] = global_history[-1].get("global_state", "")
    # If global_history empty, keep original (may be initial state)
    
    # 4b. Roll back final_location_states
    final_loc_states = {}
    for loc_name, loc_history in location_cards.items():
        if loc_history:
            final_loc_states[loc_name] = loc_history[-1].get("location_state", {})
        else:
            # Keep original final state (may be initial state)
            final_loc_states[loc_name] = world_dynamic.get("final_location_states", {}).get(loc_name, {})
    world_dynamic["final_location_states"] = final_loc_states
    
    # Update meta markers
    meta["original_stop_reason"] = stop_reason
    meta["stop_reason"] = f"{stop_reason}_sanitized"
    meta["sanitized_removed_scene_index"] = last_scene_index
    
    logger.info(
        "Sanitization complete: %d scenes remaining (removed scene_index=%s)",
        len(scenes), last_scene_index
    )
    
    return data


def dedup_world_state_history(world_dynamic: Dict[str, Any],
                              initial_states: Optional[Dict] = None) -> Dict[str, Any]:
    """Remove duplicate world state history entries identical to previous one.

    Iterate global_card_history and each location location_cards,
    comparing from initial_states (snapshot initial); if two adjacent states
    are identical, remove the latter (reduces prompt redundancy).

    Args:
        world_dynamic: dict from world_dynamic.json (modified in place)
        initial_states: initial states from snapshot，containing global_state and location_states

    Returns:
        {"global_removed": int, "location_removed": {loc_name: int, ...}}
    """
    stats: Dict[str, Any] = {"global_removed": 0, "location_removed": {}}

    # --- De-duplicate global_card_history ---
    global_history = world_dynamic.get("global_card_history", [])
    if global_history:
        prev_state = (initial_states or {}).get("global_state", {})
        deduped = []
        for entry in global_history:
            cur_state = entry.get("global_state", {})
            if cur_state != prev_state:
                deduped.append(entry)
                prev_state = cur_state
        removed = len(global_history) - len(deduped)
        if removed > 0:
            world_dynamic["global_card_history"] = deduped
            stats["global_removed"] = removed
            logger.info("dedup global_card_history: removed %d/%d duplicate entries",
                        removed, len(global_history))

    # --- De-duplicate location_cards ---
    location_cards = world_dynamic.get("location_cards", {})
    initial_loc_states = (initial_states or {}).get("location_states", {})
    for loc_name, loc_history in location_cards.items():
        if not loc_history:
            continue
        prev_state = initial_loc_states.get(loc_name, {})
        deduped = []
        for entry in loc_history:
            cur_state = entry.get("location_state", {})
            if cur_state != prev_state:
                deduped.append(entry)
                prev_state = cur_state
        removed = len(loc_history) - len(deduped)
        if removed > 0:
            location_cards[loc_name] = deduped
            stats["location_removed"][loc_name] = removed
            logger.info("dedup location_cards[%s]: removed %d/%d duplicate entries",
                        loc_name, removed, len(loc_history))

    return stats


# ============================================================================ #
# Error attribution & IC penalty computation
# ============================================================================ #

# task name -> model mapping
WORLD_MODEL_TASKS = {"scene_cast", "location_scenario", "next_character", "world_update"}
CHARACTER_AGENT_TASKS = {"interaction_gen", "character_update", "motivation_update"}


def classify_error_attribution(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract task_name from meta error field to determine error attribution.
    
    Returns:
        {
            "has_error": bool,
            "error_source": "world_model" | "character_agent" | "infra" | None,
            "task_name": str | None,
        }
    """
    import re
    
    stop_reason = meta.get("stop_reason", "")
    # Supports sanitized stop_reason (e.g. "error_sanitized")
    original_stop_reason = meta.get("original_stop_reason", stop_reason)
    
    if original_stop_reason != "error":
        return {"has_error": False, "error_source": None, "task_name": None}
    
    error_msg = meta.get("error", "")
    
    # Matches "{task_name} failed to parse JSON..." format
    m = re.match(r'^(\w+) failed to parse', error_msg)
    if m:
        task_name = m.group(1)
        if task_name in WORLD_MODEL_TASKS:
            return {"has_error": True, "error_source": "world_model", "task_name": task_name}
        elif task_name in CHARACTER_AGENT_TASKS:
            return {"has_error": True, "error_source": "character_agent", "task_name": task_name}
        else:
            # Unknown task, conservatively treat as infra
            logger.warning("Unknown task_name in error: %s", task_name)
            return {"has_error": True, "error_source": "infra", "task_name": task_name}
    
    # Other formats (e.g. "Error code: 400/429/503...") treated as infra error
    return {"has_error": True, "error_source": "infra", "task_name": None}


def compute_error_ic_penalty(
    all_scenes_original: Dict[str, Any],
    error_attribution: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Compute IC_world and IC_char penalty values based on error attribution and call count.
    
    Penalty formula (logarithmic decay): penalty = min(50, 50 / ln(total_calls + 1))
    Final IC score = max(0, judge_score - penalty)
    
    Logarithmic decay is smooth; even with hundreds of calls the penalty stays around 8 points,
    ensuring the fact that "model crashed simulation by not following instructions" is always penalized.
    
    Examples:
        total_calls=1 -> penalty=50 (cap), IC drops to zero
        total_calls=5  → penalty≈27.9
        total_calls=10 → penalty≈20.8
        total_calls=50 → penalty≈12.7
        total_calls=100→ penalty≈10.8
        total_calls=431→ penalty≈8.2
    
    Args:
        all_scenes_original: original all_scenes data before sanitize
        error_attribution: return value of classify_error_attribution
    
    Returns:
        {"IC_world_penalty": float | None, "IC_char_penalty": float | None}
        None means no penalty.
    """
    import math
    result = {"IC_world_penalty": None, "IC_char_penalty": None}
    
    if not error_attribution.get("has_error"):
        return result
    
    error_source = error_attribution.get("error_source")
    task_name = error_attribution.get("task_name")
    
    # No penalty for infra errors
    if error_source == "infra" or error_source is None:
        return result
    
    scenes = all_scenes_original.get("scenes", [])
    if not scenes:
        # No scenes at all, apply max penalty
        if error_source == "world_model":
            result["IC_world_penalty"] = 50.0
        else:
            result["IC_char_penalty"] = 50.0
        return result
    
    # StatisticsTotal call count of the failed model across all scenes (including the failed scene)
    # Execution order of a scene:
    #   scene_cast(WM) → location_scenario(WM) → motivation_update(CA × N)
    #     -> [next_character(WM) -> interaction_gen(CA) -> world_update(WM)] x M rounds
    #   → character_update(CA × N)
    
    last_scene = scenes[-1]
    
    wm_total = 0  # total world model call count
    ca_total = 0  # total character agent call count
    
    for i, scene in enumerate(scenes):
        n_chars = len(scene.get("involved_characters", []))
        n_interactions = len(scene.get("interactions", []))
        is_last = (i == len(scenes) - 1)
        
        if not is_last:
            # Normally completed scene: all tasks executed
            # WM: scene_cast(1) + location_scenario(1) + next_character(M) + world_update(M)
            wm_total += 2 + n_interactions * 2
            # CA: motivation_update(N) + interaction_gen(M) + character_update(N)
            ca_total += n_chars + n_interactions + n_chars
        else:
            # Last scene: infer based on which task the error occurred in
            # scene_cast and location_scenario always completed (otherwise no scene record)
            wm_total += 2  # scene_cast + location_scenario
            
            if task_name == "motivation_update":
                # motivation_update error -> scene failed right at the start
                # WM: scene_cast + location_scenario completed
                # CA: motivation_update partially complete, conservative estimate 0
                ca_total += 0
                
            elif task_name == "next_character":
                # next_character error -> inside interaction loop
                # Completed interactions count = n_interactions (recorded in all_scenes)
                # Per completed interaction: next_character(WM) + interaction_gen(CA) + world_update(WM)
                # The current failed next_character also counts as one WM call
                wm_total += n_interactions * 2 + 1  # completed + failed next_character
                ca_total += n_chars + n_interactions  # motivation_update(N) + interaction_gen(M)
                
            elif task_name == "interaction_gen":
                # interaction_gen error -> current turn next_character completed
                wm_total += n_interactions * 2 + 1  # completed + current turn next_character
                ca_total += n_chars + n_interactions + 1  # motivation(N) + completed interaction_gen + failed
                
            elif task_name == "world_update":
                # world_update error -> current turn next_character + interaction_gen completed
                # Note: interactions in all_scenes include the current turn interaction
                # (because interaction is appended before world_update)
                wm_total += n_interactions * 2  # completed turn (next_char+world_update) + current turn next_char + failed world_update
                # More precise: completed M-1 full turns + current turn next_char(1) + interaction_gen(1) + failed world_update(1)
                # But n_interactions already includes current turn, so:
                # Completed full turns = n_interactions - 1 (if n_interactions > 0)
                # WM: 2(scene_cast+loc) + (n_interactions-1)*2(full turn next_char+world_update) + 1(current next_char) + 1(failed world_update)
                # = 2 + n_interactions*2 - 2 + 2 = 2 + n_interactions*2
                # Already added 2 above, so adding n_interactions*2 here is correct
                ca_total += n_chars + n_interactions  # motivation(N) + interaction_gen(M, includes current turn)
                
            elif task_name == "character_update":
                # character_update error -> entire interaction loop completed
                wm_total += n_interactions * 2  # next_character(M) + world_update(M)
                ca_total += n_chars + n_interactions  # motivation(N) + interaction_gen(M)
                # character_update partially complete, conservative estimate 0
                
            elif task_name == "scene_cast":
                # scene_cast error -> almost no calls
                # wm_total already added 2, but scene_cast is first; failure means only 1 call
                wm_total -= 1  # Fix: only the failed scene_cast call
                
            elif task_name == "location_scenario":
                # location_scenario error -> scene_cast completed
                # wm_total already added 2: scene_cast(1) + failed location_scenario(1)
                pass
                
            else:
                # Unknown task, conservative handling
                wm_total += n_interactions * 2
                ca_total += n_chars + n_interactions + n_chars
    
    # Ensure minimum of 1, avoid division by zero
    wm_total = max(wm_total, 1)
    ca_total = max(ca_total, 1)
    
    # Compute penalty (logarithmic decay)
    if error_source == "world_model":
        penalty = min(50.0, 50.0 / math.log(wm_total + 1))
        result["IC_world_penalty"] = round(penalty, 2)
        logger.info(
            "IC_world penalty: %.2f (total WM calls: %d, ln(n+1)=%.2f, error task: %s)",
            penalty, wm_total, math.log(wm_total + 1), task_name
        )
    elif error_source == "character_agent":
        penalty = min(50.0, 50.0 / math.log(ca_total + 1))
        result["IC_char_penalty"] = round(penalty, 2)
        logger.info(
            "IC_char penalty: %.2f (total CA calls: %d, ln(n+1)=%.2f, error task: %s)",
            penalty, ca_total, math.log(ca_total + 1), task_name
        )
    
    return result


# ============================================================================ #
# State query helpers
# ============================================================================ #

def get_global_state_at_scene_start(world_dynamic: Dict, scene_index: int,
                                     initial_states: Optional[Dict] = None) -> Any:
    """Get global state at the start of a scene.
    
    Logic: find the last global_card_history entry with scene_index strictly less than current scene.
    For the first scene (no earlier records), use initial global_state from initial_states.
    """
    history = world_dynamic.get("global_card_history", [])
    
    # Find last record with scene_index strictly less than current scene
    candidates = [h for h in history if h["scene_index"] < scene_index]
    if candidates:
        return candidates[-1].get("global_state", {})
    
    # No earlier records -> use initial state before simulation
    if initial_states is not None:
        return initial_states.get("global_state", {})
    
    # fallback: if no initial_states, return first history entry (inaccurate, compatible with old logic)
    if history:
        return history[0].get("global_state", {})
    return {}


def get_global_state_at_scene_end(world_dynamic: Dict, scene_index: int,
                                  initial_states: Optional[Dict] = None) -> Any:
    """Get global state at the end of a scene.
    
    Logic: find the last global_card_history record with scene_index equal to current scene.
    If no global state update in this scene, return the state at scene start.
    """
    history = world_dynamic.get("global_card_history", [])
    candidates = [h for h in history if h["scene_index"] == scene_index]
    if candidates:
        return candidates[-1].get("global_state", {})
    return get_global_state_at_scene_start(world_dynamic, scene_index, initial_states)


def get_location_state_at_scene_start(world_dynamic: Dict, location_name: str, scene_index: int,
                                       initial_states: Optional[Dict] = None) -> Dict:
    """Get the state of a location at the start of a scene."""
    location_cards = world_dynamic.get("location_cards", {})
    loc_history = location_cards.get(location_name, [])
    
    candidates = [h for h in loc_history if h["scene_index"] < scene_index]
    if candidates:
        return candidates[-1].get("location_state", {})
    
    # No earlier records -> use initial state before simulation
    if initial_states is not None:
        return initial_states.get("location_states", {}).get(location_name, {})
    
    # fallback
    if loc_history:
        return loc_history[0].get("location_state", {})
    return {}


def get_location_state_at_scene_end(world_dynamic: Dict, location_name: str, scene_index: int,
                                    initial_states: Optional[Dict] = None) -> Dict:
    """Get the state of a location at the end of a scene."""
    location_cards = world_dynamic.get("location_cards", {})
    loc_history = location_cards.get(location_name, [])
    
    candidates = [h for h in loc_history if h["scene_index"] == scene_index]
    if candidates:
        return candidates[-1].get("location_state", {})
    return get_location_state_at_scene_start(world_dynamic, location_name, scene_index, initial_states)


def get_character_profile_at_scene_start(char_dynamic: Dict, char_name: str, scene_index: int,
                                          initial_states: Optional[Dict] = None) -> str:
    """Get a character profile at the start of a scene.
    
    scene_index in profile_history is the profile AFTER the scene completes.
    Thus scene-start profile = previous scene end profile.
    """
    characters = char_dynamic.get("characters", char_dynamic)
    char_data = characters.get(char_name, {})
    profile_history = char_data.get("profile_history", [])
    
    # Find last record with scene_index strictly less than current scene
    candidates = [h for h in profile_history if h["scene_index"] < scene_index]
    if candidates:
        return candidates[-1].get("profile", "")
    
    # No earlier records -> use initial profile from before simulation
    if initial_states is not None:
        return initial_states.get("character_profiles", {}).get(char_name, "")
    
    # fallback
    if profile_history:
        return profile_history[0].get("profile", "")
    return ""


def get_character_profile_at_scene_end(char_dynamic: Dict, char_name: str, scene_index: int,
                                       initial_states: Optional[Dict] = None) -> str:
    """Get a character profile at the end of a scene."""
    characters = char_dynamic.get("characters", char_dynamic)
    char_data = characters.get(char_name, {})
    profile_history = char_data.get("profile_history", [])
    
    candidates = [h for h in profile_history if h["scene_index"] == scene_index]
    if candidates:
        return candidates[-1].get("profile", "")
    return get_character_profile_at_scene_start(char_dynamic, char_name, scene_index, initial_states)


def get_character_description_at_scene_start(
    char_dynamic: Dict,
    char_name: str,
    scene_index: int,
    initial_states: Optional[Dict] = None,
) -> str:
    """Get short description of a character at the start of a scene.
    
    Short description preferred from earlier scene scene_descriptions.
    If this is the first scene, prefer initial short_description from snapshot input.
    """
    characters = char_dynamic.get("characters", char_dynamic)
    char_data = characters.get(char_name, {})
    
    # Prefer from scene_descriptions (last entry with scene_index strictly less than current)
    scene_descs = char_data.get("scene_descriptions", [])
    result = ""
    for sd in scene_descs:
        if sd.get("scene_index", 0) < scene_index:
            desc = sd.get("description", "")
            if desc:
                result = desc
        else:
            break
    
    # Fallback 1: use initial short description from pre-simulation snapshot
    if not result and initial_states is not None:
        result = initial_states.get("character_short_descriptions", {}).get(char_name, "")
    
    # Fallback 2: old logic for compatibility (may contain future info, but safe when snapshot is absent)
    if not result:
        profile_history = char_data.get("profile_history", [])
        if profile_history:
            result = profile_history[0].get("description", "")
    
    return result


def get_scene_description_for_character(char_dynamic: Dict, char_name: str, scene_index: int) -> Optional[Dict]:
    """Get scene_description for a character in a scene (includes enhanced_motivation, description, hidden_tracker)."""
    characters = char_dynamic.get("characters", char_dynamic)
    char_data = characters.get(char_name, {})
    scene_descs = char_data.get("scene_descriptions", [])
    
    for sd in scene_descs:
        if sd.get("scene_index") == scene_index:
            return sd
    return None


# ============================================================================ #
# Active object identification
# ============================================================================ #

def get_active_characters(char_dynamic: Dict) -> List[str]:
    """Get all active characters (participated in at least one scene)."""
    characters = char_dynamic.get("characters", char_dynamic)
    active = []
    for name, data in characters.items():
        if data.get("profile_history") or data.get("scene_descriptions"):
            active.append(name)
    return active


def get_active_locations(world_dynamic: Dict) -> List[str]:
    """Get all active locations (updated at least once)."""
    location_cards = world_dynamic.get("location_cards", {})
    active = []
    for loc_name, loc_history in location_cards.items():
        if loc_history:  # Consider active if has history records
            active.append(loc_name)
    return active


def get_all_character_names(char_dynamic: Dict) -> List[str]:
    """Get all character names."""
    characters = char_dynamic.get("characters", char_dynamic)
    return list(characters.keys())


# ============================================================================ #
# Evaluation slice construction
# ============================================================================ #

def build_per_scene_slice(
    scene: Dict,
    all_scenes: Dict,
    char_dynamic: Dict,
    world_dynamic: Dict,
    prev_scene_summary: str = "",
    initial_states: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build evaluation slice for a single scene.
    
    Contains all information needed by the judge for this scene.
    
    Args:
        initial_states: initial states from snapshot，for obtaining true initial state at scene_index=0
    """
    scene_index = scene["scene_index"]
    location = scene["location"]
    involved_chars = scene.get("involved_characters", [])
    
    # 1. Participating character profiles (scene-start version)
    char_profiles = {}
    for char_name in involved_chars:
        char_profiles[char_name] = get_character_profile_at_scene_start(
            char_dynamic, char_name, scene_index, initial_states
        )
    
    # 2. Participating character scene motivation
    char_motivations = {}
    for char_name in involved_chars:
        sd = get_scene_description_for_character(char_dynamic, char_name, scene_index)
        if sd:
            char_motivations[char_name] = sd.get("enhanced_motivation", "")
    
    # 3. World state (scene start)
    global_state_start = get_global_state_at_scene_start(world_dynamic, scene_index, initial_states)
    location_state_start = get_location_state_at_scene_start(world_dynamic, location, scene_index, initial_states)
    
    # 4. World state (scene end)
    global_state_end = get_global_state_at_scene_end(world_dynamic, scene_index, initial_states)
    location_state_end = get_location_state_at_scene_end(world_dynamic, location, scene_index, initial_states)
    
    # 5. Determine if global/location was updated in this scene
    global_card_history = world_dynamic.get("global_card_history", [])
    global_updated_in_scene = any(h["scene_index"] == scene_index for h in global_card_history)
    
    location_cards = world_dynamic.get("location_cards", {})
    loc_history = location_cards.get(location, [])
    location_updated_in_scene = any(h["scene_index"] == scene_index for h in loc_history)
    
    # 6. Character profile updates and hidden tracker (scene end)
    char_reflections = {}
    for char_name in involved_chars:
        profile_before = get_character_profile_at_scene_start(char_dynamic, char_name, scene_index, initial_states)
        profile_after = get_character_profile_at_scene_end(char_dynamic, char_name, scene_index, initial_states)
        sd = get_scene_description_for_character(char_dynamic, char_name, scene_index)
        
        char_reflections[char_name] = {
            "profile_updated": profile_before != profile_after,
            "profile_before": profile_before,
            "profile_after": profile_after,
            "description": sd.get("description", "") if sd else "",
            "hidden_tracker": sd.get("hidden_tracker", "") if sd else "",
        }
    
    # 7. World state intermediate update history within scene (for GUS/GSA/LUS/LSA)
    # global state intermediate updates
    global_updates_in_scene = [
        h for h in global_card_history if h["scene_index"] == scene_index
    ]
    # location state intermediate updates
    location_updates_in_scene = [
        h for h in loc_history if h["scene_index"] == scene_index
    ]
    
    # 8. Character pool summary (for CSR evaluation)
    # Use per-character short description instead of truncated profile
    all_chars = get_all_character_names(char_dynamic)
    character_pool_summary = []
    for char_name in all_chars:
        if char_name not in involved_chars:
            desc = get_character_description_at_scene_start(char_dynamic, char_name, scene_index, initial_states)
            if desc:
                character_pool_summary.append({
                    "name": char_name,
                    "short_description": desc,
                })
    
    # 9. Short descriptions of participating characters (for TSO: check if speech ratio matches identity)
    char_short_descriptions = {}
    for char_name in involved_chars:
        char_short_descriptions[char_name] = get_character_description_at_scene_start(
            char_dynamic, char_name, scene_index, initial_states
        )
    
    # 10. Optional location summary (for LSR evaluation)
    # List all locations with brief state summaries, letting the judge determine if the chosen location is most appropriate
    available_locations = []
    for loc_name in location_cards.keys():
        loc_state = get_location_state_at_scene_start(world_dynamic, loc_name, scene_index, initial_states)
        # Extract location Detailed Description as brief summary
        short_desc = ""
        if isinstance(loc_state, dict):
            short_desc = loc_state.get("Detailed Description", "")
            if not short_desc:
                # fallback: use first 200 chars of format_location_state
                full = format_location_state(loc_state)
                short_desc = full[:200] + ("..." if len(full) > 200 else "")
        available_locations.append({
            "name": loc_name,
            "short_description": short_desc,
            "is_current": loc_name == location,
        })
    
    return {
        "scene_index": scene_index,
        "book_name": all_scenes.get("book_name", ""),
        "location": location,
        "scenario": scene.get("scenario", ""),
        "involved_characters": involved_chars,
        "interactions": scene.get("interactions", []),
        "char_profiles": char_profiles,
        "char_motivations": char_motivations,
        "global_state_start": global_state_start,
        "location_state_start": location_state_start,
        "global_state_end": global_state_end,
        "location_state_end": location_state_end,
        "global_updated": global_updated_in_scene,
        "location_updated": location_updated_in_scene,
        "global_updates_in_scene": global_updates_in_scene,
        "location_updates_in_scene": location_updates_in_scene,
        "char_reflections": char_reflections,
        "character_pool_summary": character_pool_summary,
        "available_locations": available_locations,
        "char_short_descriptions": char_short_descriptions,
        "prev_scene_summary": prev_scene_summary,
    }


def build_cross_scene_character_slice(
    char_name: str,
    char_dynamic: Dict,
    all_scenes: Dict,
    scene_summaries: Dict[int, str],
    initial_states: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build cross-scene evaluation slice for a single character.
    
    Args:
        char_name: character name
        char_dynamic: character_dynamic.json data
        all_scenes: all_scenes.json data
        scene_summaries: {scene_index: summary} mapping from Per-Scene evaluation output
        initial_states: initial states from snapshot
    """
    characters = char_dynamic.get("characters", char_dynamic)
    char_data = characters.get(char_name, {})
    
    profile_history = char_data.get("profile_history", [])
    scene_descriptions = char_data.get("scene_descriptions", [])
    
    # Get true initial profile (from snapshot)
    if initial_states is not None:
        initial_profile = initial_states.get("character_profiles", {}).get(char_name, "")
    elif profile_history:
        initial_profile = profile_history[0].get("profile", "")
    else:
        initial_profile = ""
    
    # Find all scenes this character participated in
    scenes = all_scenes.get("scenes", [])
    participated_scenes = []
    for scene in scenes:
        if char_name in scene.get("involved_characters", []):
            participated_scenes.append({
                "scene_index": scene["scene_index"],
                "location": scene["location"],
                "scenario": scene.get("scenario", ""),
                "summary": scene_summaries.get(scene["scene_index"], ""),
            })
    
    return {
        "char_name": char_name,
        "book_name": all_scenes.get("book_name", ""),
        "initial_profile": initial_profile,
        "profile_history": profile_history,
        "scene_descriptions": scene_descriptions,
        "participated_scenes": participated_scenes,
        "num_scenes_participated": len(participated_scenes),
    }


def build_cross_scene_global_slice(
    all_scenes: Dict,
    scene_summaries: Dict[int, str],
) -> Dict[str, Any]:
    """Build cross-scene narrative coherence evaluation slice.
    
    SCC only needs all scene summaries to evaluate cross-scene narrative coherence,
    no global state needed (its accuracy is already evaluated in per-scene GUS/GSA).
    """
    scenes = all_scenes.get("scenes", [])
    all_scene_summaries = []
    for scene in scenes:
        si = scene["scene_index"]
        all_scene_summaries.append({
            "scene_index": si,
            "location": scene["location"],
            "scenario": scene.get("scenario", ""),
            "involved_characters": scene.get("involved_characters", []),
            "summary": scene_summaries.get(si, ""),
        })
    
    return {
        "book_name": all_scenes.get("book_name", ""),
        "all_scene_summaries": all_scene_summaries,
    }


# ============================================================================ #
# Global state summary (for token control)
# ============================================================================ #

def format_global_state_full(global_state: Any) -> str:
    """Format global state as full readable text (no truncation).
    
    Compatible with both formats in simulation / snapshot:
    - str: already Markdown/plain text, return as-is
    - dict: expand by category to readable text
    """
    if not global_state:
        return "(No global state)"
    
    if isinstance(global_state, str):
        return global_state
    
    if not isinstance(global_state, dict):
        return str(global_state)
    
    try:
        parts = []
        for category, rules in global_state.items():
            if category in ("Title", "Author"):
                parts.append(f"{category}: {rules}")
                continue
            
            if isinstance(rules, dict):
                rule_texts = list(rules.keys())
            elif isinstance(rules, str):
                rule_texts = [rules]
            else:
                rule_texts = [str(rules)]
            
            category_text = f"\n[{category}]\n"
            for rule in rule_texts:
                category_text += f"- {rule}\n"
            parts.append(category_text)
        
        return "\n".join(parts)
    except Exception:
        import json
        try:
            return json.dumps(global_state, ensure_ascii=False, indent=2)
        except Exception:
            return str(global_state)


def summarize_global_state(global_state: Any, max_chars: int = 2000) -> str:
    """Compress global state into a summary string (for Cross-Scene token control)."""
    if not global_state:
        return "(No global state)"
    
    if isinstance(global_state, str):
        return global_state[:max_chars] + ("..." if len(global_state) > max_chars else "")
    
    if not isinstance(global_state, dict):
        return str(global_state)[:max_chars]
    
    try:
        parts = []
        remaining = max_chars
        
        for category, rules in global_state.items():
            if category in ("Title", "Author"):
                parts.append(f"{category}: {rules}")
                continue
            
            if isinstance(rules, dict):
                rule_texts = list(rules.keys())
            elif isinstance(rules, str):
                rule_texts = [rules]
            else:
                rule_texts = [str(rules)]
            
            category_text = f"\n[{category}]\n"
            for rule in rule_texts:
                line = f"- {rule[:150]}{'...' if len(rule) > 150 else ''}\n"
                if remaining - len(line) - len(category_text) < 0:
                    break
                category_text += line
                remaining -= len(line)
            
            parts.append(category_text)
            if remaining <= 0:
                break
        
        return "\n".join(parts)
    except Exception:
        import json
        try:
            return json.dumps(global_state, ensure_ascii=False, indent=2)[:max_chars]
        except Exception:
            return str(global_state)[:max_chars]


def format_interactions(interactions: List[Dict]) -> str:
    """Format interactions as readable text.
    
    Args:
        interactions: list of interaction records
    """
    if not interactions:
        return "(No interactions)"
    
    # Guard: simulation may return non-list type
    if isinstance(interactions, str):
        return interactions
    if not isinstance(interactions, list):
        return str(interactions)
    
    try:
        lines = []
        for i, interaction in enumerate(interactions):
            if isinstance(interaction, str):
                lines.append(f"[Interaction {i}]:\n{interaction}")
                continue
            if not isinstance(interaction, dict):
                lines.append(f"[Interaction {i}]:\n{interaction}")
                continue
            chars = interaction.get("characters", [])
            content = interaction.get("content", "")
            char_str = ", ".join(chars) if chars else "Unknown"
            # Use 0-based interaction_index, matching world state update timeline
            lines.append(f"[Interaction {i}] ({char_str}):\n{content}")
        
        return "\n\n".join(lines)
    except Exception:
        import json
        try:
            return json.dumps(interactions, ensure_ascii=False, indent=2)
        except Exception:
            return str(interactions)


def format_location_state(location_state: Dict) -> str:
    """Format location state as readable text.
    
    Supports two formats:
    1. With Sub Locations: Description + Sub Locations (each with Important Entities)
    2. Without Sub Locations: Description + top-level Important Entities
    """
    if not location_state:
        return "(No location state)"
    
    # Guard: simulation may return string, not dict (e.g. world model output parse failure)
    if isinstance(location_state, str):
        return location_state
    
    if not isinstance(location_state, dict):
        return str(location_state)
    
    try:
        parts = []
        desc = location_state.get("Detailed Description", "")
        if desc:
            parts.append(f"Description: {desc}")
        
        sub_locs = location_state.get("Sub Locations", [])
        if sub_locs:
            for sub in sub_locs:
                if isinstance(sub, str):
                    parts.append(f"\n  [{sub}]")
                    continue
                if not isinstance(sub, dict):
                    parts.append(f"\n  [{sub}]")
                    continue
                sub_name = sub.get("name", "")
                sub_desc = sub.get("description", "")
                parts.append(f"\n  [{sub_name}]: {sub_desc}")
                
                entities = sub.get("Important Entities", [])
                for entity in entities:
                    if isinstance(entity, str):
                        parts.append(f"    - {entity}")
                    elif isinstance(entity, dict):
                        parts.append(f"    - {entity.get('name', '')}: {entity.get('state', '')}")
                    else:
                        parts.append(f"    - {entity}")
        else:
            # When no Sub Locations, Important Entities at top level
            entities = location_state.get("Important Entities", [])
            for entity in entities:
                if isinstance(entity, str):
                    parts.append(f"  - {entity}")
                elif isinstance(entity, dict):
                    parts.append(f"  - {entity.get('name', '')}: {entity.get('state', '')}")
                else:
                    parts.append(f"  - {entity}")
        
        return "\n".join(parts)
    except Exception:
        # Parse failed, return entire location_state as raw string
        import json
        try:
            return json.dumps(location_state, ensure_ascii=False, indent=2)
        except Exception:
            return str(location_state)
