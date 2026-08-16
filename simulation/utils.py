from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any


def setup_logger(name: str, log_file: Path, console_level: int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Optionally mirror records at or above *console_level* to stderr so
    # progress and failures are visible in the terminal (full detail with
    # timestamps stays in the file).
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)
    return logger


class _ConsoleFormatter(logging.Formatter):
    """Compact terminal format: bare message for INFO, level prefix above that."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def register_character(snapshot: dict[str, Any], definition: dict[str, Any]) -> str:
    """Add a new character to a snapshot's character_states and return its name.

    ``definition``: {name, short_description, profile, relationships?,
    motivation?, hidden_tracker?}. Relationships are folded into the profile
    text so the world model and the other character agents can see them.
    """
    name = (definition.get("name") or "").strip()
    if not name:
        raise ValueError("character definition must include a 'name'")
    profile = definition.get("profile", "")
    relationships = definition.get("relationships") or {}
    if relationships:
        rel_lines = "\n".join(f"- {other}: {desc}" for other, desc in relationships.items())
        profile = f"{profile}\n\nRelationships:\n{rel_lines}".strip()
    snapshot["character_states"][name] = {
        "profile": profile,
        "short_description": definition.get("short_description", ""),
        "motivation": definition.get("motivation", ""),
        "hidden_tracker": definition.get("hidden_tracker", ""),
    }
    return name


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_chatcompletions_suffix(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    suffixes = [
        "/chat/completions",
        "/v1/chat/completions",
    ]
    for suffix in suffixes:
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def normalize_content(content: Any) -> str:
    """Normalize interaction content to a string.

    Most records store content as a string, but some dirty cases may store it
    as a list. Mirror training-side robustness by flattening list values into a
    single string before downstream processing.
    """
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return content if isinstance(content, str) else str(content)


def remove_other_character_thoughts(text: str) -> str:
    text = normalize_content(text)
    text = re.sub(r"\[.*?\]", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_interaction_characters(item: dict[str, Any]) -> list[str]:
    """Return the list of acting character names from an interaction record.

    Supports both the new ``characters`` (list) field and the legacy
    ``character`` (str) field.
    """
    chars = item.get("characters")
    if isinstance(chars, list):
        return chars
    char = item.get("character")
    if char:
        return [char]
    return ["Environment"]


def _normalize_perspective_characters(
    perspective_character: str | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if perspective_character is None:
        return set()
    if isinstance(perspective_character, str):
        return {perspective_character}
    return {str(name) for name in perspective_character if str(name).strip()}


def mask_interactions_for_character(
    interactions: list[dict[str, Any]],
    character_name: str | list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, Any]]:
    visible_perspective = _normalize_perspective_characters(character_name)
    masked: list[dict[str, Any]] = []
    for item in interactions:
        content = normalize_content(item.get("content", ""))
        chars = _get_interaction_characters(item)
        if visible_perspective and not visible_perspective.intersection(chars):
            content = remove_other_character_thoughts(content)
        masked.append(
            {
                "characters": chars,
                "content": content,
            }
        )
    return masked


def summarize_scene_for_history(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_index": scene.get("scene_index"),
        "location": scene.get("location"),
        "involved_characters": scene.get("involved_characters", []),
        "scenario": scene.get("scenario", ""),
        "num_interactions": len(scene.get("interactions", [])),
    }


def _strip_markdown_codeblock(text: str) -> str:
    """Remove markdown code-block wrappers (```json ... ``` or ``` ... ```)."""
    stripped = text.strip()
    pattern = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)```\s*$", re.S)
    m = pattern.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _try_fix_truncated_json(text: str) -> str | None:
    """Attempt to close a truncated JSON object/array by appending missing brackets."""
    text = text.rstrip().rstrip(",")
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    if open_braces <= 0 and open_brackets <= 0:
        return None  # not a truncation issue

    # Remove any trailing incomplete string value (e.g. `"key": "some text...`)
    # by stripping back to the last complete key-value or element boundary.
    candidate = text.rstrip()
    # Strip trailing incomplete string (not closed with a quote)
    candidate = re.sub(r',\s*"[^"]*$', '', candidate)
    candidate = re.sub(r',\s*$', '', candidate)

    # Re-count after cleanup
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # If we're inside an unclosed string, bail out
    if in_string:
        candidate = candidate[:candidate.rfind('"')] + '"'

    closing = ']' * max(open_brackets, 0) + '}' * max(open_braces, 0)
    return candidate + closing


def extract_json_fragment(text: str) -> Any:
    """Extract a JSON object or array from *text*, handling markdown code blocks
    and truncated responses."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    # Step 1: strip markdown code-block wrappers
    text = _strip_markdown_codeblock(text)

    # Step 2: try direct parse and brace/bracket extraction
    candidates = [
        text,
    ]
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidates.append(text[bracket_start : bracket_end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Step 3: try raw_decode to find the longest valid JSON fragment
    decoder = json.JSONDecoder()
    results = []
    start = 0
    while start < len(text):
        try:
            obj, end = decoder.raw_decode(text, start)
            results.append(obj)
            start += end
        except json.JSONDecodeError:
            start += 1
    if results:
        return max(results, key=lambda x: len(json.dumps(x)))

    # Step 4: attempt to fix truncated JSON
    for prefix in (text, text[text.find("{"):] if "{" in text else "", text[text.find("["):] if "[" in text else ""):
        if not prefix:
            continue
        fixed = _try_fix_truncated_json(prefix)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue

    # Step 5: Try ast.literal_eval for Python-style literals (e.g. single-quoted lists like ['name']）
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, (dict, list, str, int, float, bool)):
                return result
        except (ValueError, SyntaxError):
            continue

    raise ValueError(f"Could not parse JSON from response: {text[:300]}")


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _is_empty_value(v: Any) -> bool:
    """Check if a value is "empty" (None, empty str, empty dict, empty list)."""
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (dict, list)) and len(v) == 0:
        return True
    return False


def dict_to_markdown_text(data: Any) -> str:
    """Convert model output dict back to the Markdown plain-text format used in training data.

    Handles two main cases:
    1. global_state dict:
       - top-level simple key-value (e.g. Title, Author) -> **Key:** Value
       - top-level key with number (e.g. "1. Social Order") -> ### 1. Social Order
         when value is a dict, each sub-key becomes paragraph text (empty vals ignored)
    2. profile dict:
       - top-level key -> **Key**
         when value is a string -> follows heading as a paragraph
         when value is a dict (e.g. Key Relationships) -> each sub-key becomes  *   **SubKey:** SubValue

    If input is not a dict, return str(data) directly.
    """
    if not isinstance(data, dict):
        return str(data) if data is not None else ""

    lines: list[str] = []

    for key, value in data.items():
        if _is_empty_value(value):
            # value is empty: output key itself
            # check if numbered heading (e.g. "1. Social Order & Class")
            if re.match(r'^\d+\.', key.strip()):
                lines.append(f"### {key}")
            else:
                lines.append(f"**{key}**")
        elif isinstance(value, str):
            # value is a string
            if re.match(r'^\d+\.', key.strip()):
                # numbered heading + string content
                lines.append(f"### {key}")
                lines.append(value)
            elif key.lower() in ("title", "author"):
                # special metadata fields
                lines.append(f"**{key}:** {value}")
            else:
                # regular profile field: **Key** followed by content
                lines.append(f"**{key}**")
                lines.append(value)
        elif isinstance(value, dict):
            # value is a dict
            if re.match(r'^\d+\.', key.strip()):
                # global_state style: numbered heading, sub-keys are entry text
                lines.append(f"### {key}")
                for sub_key, sub_value in value.items():
                    if _is_empty_value(sub_value):
                        # sub-value empty (e.g. {}), sub_key itself is the entry content
                        lines.append(sub_key)
                    elif isinstance(sub_value, str):
                        lines.append(f"{sub_key}: {sub_value}")
                    elif isinstance(sub_value, dict):
                        # deeper nesting: recursively expand
                        lines.append(sub_key)
                        for ss_key, ss_value in sub_value.items():
                            if _is_empty_value(ss_value):
                                lines.append(f"  - {ss_key}")
                            else:
                                lines.append(f"  - {ss_key}: {ss_value}")
                    else:
                        lines.append(f"{sub_key}: {sub_value}")
            else:
                # profile style: e.g. Key Relationships, sub-keys are person/relation names
                lines.append(f"**{key}**")
                for sub_key, sub_value in value.items():
                    if _is_empty_value(sub_value):
                        lines.append(f"*   **{sub_key}**")
                    elif isinstance(sub_value, str):
                        lines.append(f"*   **{sub_key}:** {sub_value}")
                    elif isinstance(sub_value, dict):
                        # deeper nesting
                        lines.append(f"*   **{sub_key}:**")
                        for ss_key, ss_value in sub_value.items():
                            if _is_empty_value(ss_value):
                                lines.append(f"    - {ss_key}")
                            else:
                                lines.append(f"    - {ss_key}: {ss_value}")
                    else:
                        lines.append(f"*   **{sub_key}:** {sub_value}")
        elif isinstance(value, list):
            # value is a list
            if re.match(r'^\d+\.', key.strip()):
                lines.append(f"### {key}")
            else:
                lines.append(f"**{key}**")
            for item in value:
                if isinstance(item, str):
                    lines.append(f"- {item}")
                elif isinstance(item, dict):
                    lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"**{key}:** {value}")

    return "\n".join(lines)


def format_prompt_block(value: Any, default: str = "(None)") -> str:
    """Format a value for inclusion in a prompt, matching training data's
    ``_format_prompt_block`` in ``data_construction/utils.py``.

    - ``None`` or empty strings become ``default``.
    - dicts and lists are serialized via ``json.dumps`` with ``indent=2``.
    - Everything else is converted to ``str``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value if value.strip() else default
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
        return text if text.strip() else default
    text = str(value)
    return text if text.strip() else default


def format_interaction_history(
    interactions: list[dict[str, Any]],
    perspective_character: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    """Build a text interaction history matching training data's
    ``build_interaction_history`` in ``data_construction/transform.py``.

    Each interaction is formatted as:  ``["CharA", "CharB"]: content``

    If *perspective_character* is given, ``[thought]`` blocks are kept whenever
    the acting side overlaps with that perspective set; otherwise they are
    removed.
    """
    if not interactions:
        return "(None)"
    visible_perspective = _normalize_perspective_characters(perspective_character)
    lines: list[str] = []
    for item in interactions:
        chars = _get_interaction_characters(item)
        content = normalize_content(item.get("content", ""))
        if visible_perspective and not visible_perspective.intersection(chars):
            content = remove_other_character_thoughts(content)
        actor_label = ', '.join('"' + c + '"' for c in chars)
        lines.append(f'[{actor_label}]: {content}')
    return "\n\n".join(lines)


def split_result_views(result: dict[str, Any]) -> dict[str, Any]:
    """Split a simulation result into three separate views matching data_construction format.

    Returns a dict with keys:
        - 'all_scenes': scene structure skeleton (scenes + interactions + characters + locations)
        - 'character_dynamic': character profile evolution history per character
        - 'world_dynamic': global state update history + location state update history
    """
    scenes = result.get("scenes", [])
    final_world_state = result.get("final_world_state", {})
    final_character_states = result.get("final_character_states", {})

    # --- 1. all_scenes: scene structure with interactions (no world_updates/reflections) ---
    all_scenes_list = []
    for scene in scenes:
        scene_entry = {
            "scene_index": scene.get("scene_index"),
            "location": scene.get("location"),
            "scenario": scene.get("scenario", ""),
            "involved_characters": scene.get("involved_characters", []),
            "interactions": [
                {
                    "characters": item.get("characters", []),
                    "content": item.get("content", ""),
                }
                for item in scene.get("interactions", [])
            ],
        }
        all_scenes_list.append(scene_entry)

    all_scenes = {
        "book_name": result.get("book_name"),
        "source_scene_index": result.get("source_scene_index"),
        "scenes": all_scenes_list,
    }

    # --- 2. character_dynamic: profile evolution history per character ---
    # Collect profile snapshots from motivation_updates and character_reflections
    char_dynamic: dict[str, dict[str, Any]] = {}

    # Initialize with final character states
    for char_name, char_state in final_character_states.items():
        char_dynamic[char_name] = {
            "profile_history": [],
            "scene_descriptions": [],
        }

    # Build a lookup: scene_index -> {char_name -> motivation} from motivation_updates
    scene_motivations: dict[int, dict[str, str]] = {}
    for scene in scenes:
        scene_idx = scene.get("scene_index")
        for mu in scene.get("motivation_updates", []):
            char_name = mu.get("character")
            if char_name:
                scene_motivations.setdefault(scene_idx, {})[char_name] = mu.get("motivation", "")

    for scene in scenes:
        scene_idx = scene.get("scene_index")

        # Collect character reflections (scene end)
        # - profile_history: only when profile was actually updated (matches data format)
        # - scene_descriptions: always recorded for every scene (enhanced_motivation + description + hidden_tracker)
        for cr in scene.get("character_reflections", []):
            char_name = cr.get("character")
            if char_name and char_name in char_dynamic:
                # profile_history: only record when profile truly changed
                if cr.get("profile_updated", True):
                    char_dynamic[char_name]["profile_history"].append({
                        "scene_index": scene_idx,
                        "profile": cr.get("profile", ""),
                        "description": cr.get("short_description", ""),
                    })

                # scene_descriptions: always recorded (matches dataset/extracted_data/character_dynamic format)
                char_dynamic[char_name]["scene_descriptions"].append({
                    "scene_index": scene_idx,
                    "enhanced_motivation": scene_motivations.get(scene_idx, {}).get(char_name, ""),
                    "description": cr.get("short_description", ""),
                    "hidden_tracker": cr.get("hidden_tracker", ""),
                })

    character_dynamic = {
        "book_name": result.get("book_name"),
        "source_scene_index": result.get("source_scene_index"),
        "characters": char_dynamic,
    }

    # --- 3. world_dynamic: global state + location state update history ---
    global_card_history: list[dict[str, Any]] = []
    location_cards: dict[str, list[dict[str, Any]]] = {}

    for scene in scenes:
        scene_idx = scene.get("scene_index")
        interactions = scene.get("interactions", [])
        for turn_idx, wu in enumerate(scene.get("world_updates", [])):
            interaction_index = turn_idx  # relative to scene

            if wu.get("update_global") and wu.get("global_state"):
                global_card_history.append({
                    "scene_index": scene_idx,
                    "interaction_index": interaction_index,
                    "global_state": wu["global_state"],
                })

            if wu.get("update_location") and wu.get("location_state"):
                loc_name = wu.get("location", scene.get("location"))
                if loc_name not in location_cards:
                    location_cards[loc_name] = []
                location_cards[loc_name].append({
                    "scene_index": scene_idx,
                    "interaction_index": interaction_index,
                    "location_state": wu["location_state"],
                })

    # Also include the final world state as the latest snapshot
    world_dynamic = {
        "book_name": result.get("book_name"),
        "source_scene_index": result.get("source_scene_index"),
        "global_card_history": global_card_history,
        "location_cards": location_cards,
        "final_global_state": final_world_state.get("global_state", ""),
        "final_location_states": final_world_state.get("location_states", {}),
    }

    return {
        "all_scenes": all_scenes,
        "character_dynamic": character_dynamic,
        "world_dynamic": world_dynamic,
    }
