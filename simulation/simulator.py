from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("simulation")

if __package__ is None or __package__ == "":
    from inference import OpenAICompatibleClient
    from utils import (
        coerce_bool,
        dict_to_markdown_text,
        extract_json_fragment,
        format_prompt_block,
        format_interaction_history,
        mask_interactions_for_character,
        remove_other_character_thoughts,
    )
else:
    from .inference import OpenAICompatibleClient
    from .utils import (
        coerce_bool,
        dict_to_markdown_text,
        extract_json_fragment,
        format_prompt_block,
        format_interaction_history,
        mask_interactions_for_character,
        remove_other_character_thoughts,
    )


WORLD_MODEL_TASKS = {"scene_cast", "location_scenario", "next_character", "world_update"}
CHARACTER_AGENT_TASKS = {"interaction_gen", "character_update", "motivation_update"}


def _normalize_interaction_text_for_repeat_check(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def _split_interaction_units_for_repeat_check(text: str) -> list[str]:
    raw_units = re.split(r'(?:\n\s*\n|(?<=[.!?。！？])\s+)', text or "")
    units: list[str] = []
    for unit in raw_units:
        normalized = _normalize_interaction_text_for_repeat_check(unit)
        if normalized:
            units.append(normalized)
    return units


def _looks_repetitive_interaction(candidate: str, recent_interactions: list[dict[str, Any]]) -> bool:
    candidate_norm = _normalize_interaction_text_for_repeat_check(candidate)
    if not candidate_norm:
        return False
    for interaction in recent_interactions[-2:]:
        prior_norm = _normalize_interaction_text_for_repeat_check(interaction.get("content", ""))
        if prior_norm and candidate_norm == prior_norm:
            return True
    return False


def _looks_self_repetitive_interaction(candidate: str) -> bool:
    candidate_norm = _normalize_interaction_text_for_repeat_check(candidate)
    if not candidate_norm:
        return False

    unit_counts: dict[str, int] = {}
    for unit in _split_interaction_units_for_repeat_check(candidate):
        if len(unit) < 24:
            continue
        unit_counts[unit] = unit_counts.get(unit, 0) + 1
        if unit_counts[unit] >= 2:
            return True

    tokens = re.findall(r"[a-z0-9']+|[^\w\s]", candidate_norm)
    for window_size, min_repeats in ((12, 3), (8, 4)):
        if len(tokens) < window_size * min_repeats:
            continue
        window_counts: dict[str, int] = {}
        for idx in range(len(tokens) - window_size + 1):
            window = " ".join(tokens[idx: idx + window_size])
            if len(window) < 32:
                continue
            window_counts[window] = window_counts.get(window, 0) + 1
            if window_counts[window] >= min_repeats:
                return True

    return False


def _get_interaction_retry_reason(candidate: str, recent_interactions: list[dict[str, Any]]) -> str | None:
    if _looks_repetitive_interaction(candidate, recent_interactions):
        return "recent_repeat"
    if _looks_self_repetitive_interaction(candidate):
        return "self_repeat"
    return None


@dataclass
class SimulationConfig:
    max_scenes: int
    max_turns_per_scene: int
    log_model_io: bool = False
    max_tokens: int = 16384


class StorySimulator:
    def __init__(
        self,
        world_model_client: OpenAICompatibleClient,
        character_agent_client: OpenAICompatibleClient,
        config: SimulationConfig,
    ):
        self.world_model_client = world_model_client
        self.character_agent_client = character_agent_client
        self.config = config
        # Prompt variant choices; initialised per snapshot by _init_prompt_variants()
        self._pv: dict[str, Any] = {}

    def _get_local_interaction_sampling_kwargs(self) -> dict[str, Any]:
        if getattr(self.character_agent_client.config, "mode", None) != "local":
            return {}
        return {
            "extra_body": {
                "repetition_penalty": 1.1,
            },
        }

    def _init_prompt_variants(self) -> None:
        """Randomly select one prompt variant suite for the entire snapshot.

        This mirrors training data's diversity: each book/snapshot gets a
        randomly chosen style (natural / = / # / *) and randomly chosen
        wording variants for every task.  Within a single snapshot run,
        these choices remain fixed for stability.
        """
        styles_pool = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
        style = random.choice(styles_pool)

        decorator_pools = {
            '=': ["==={}===", "=={}==", "={}="],
            '#': ["#{}", "# {}", "## {}", "### {}"],
            '*': ["**{}**", "*{}*", "***{}***"],
        }
        decorator = random.choice(decorator_pools[style]) if style in decorator_pools else None

        # ------------------------------------------------------------------
        # Scene cast (build_diverse_task_prompt style)
        # ------------------------------------------------------------------
        sc_role_variants = [
            "You are the scene-cast planning module for a story simulation. Your job is to decide whether another scene should happen and, if so, which characters should be in that scene's cast.",
            "You are the director-level scene casting planner. Decide whether the story continues and who belongs in the full cast of the next scene.",
            "You are responsible for planning the cast lineup of the next scene based on the current world and character states.",
            "You are the first-stage scene planner. Before the next scene begins, decide whether it should exist and which characters should be included in its cast.",
            "You are the world-planning module that determines whether the story should continue into another scene and, if so, which characters should make up that next scene's cast.",
        ]
        sc_task_variants = [
            "Determine whether a next scene exists, and if it does, choose the full set of characters who should participate in that scene.",
            "Use the current story state to decide whether the story continues and which characters belong in the next scene's cast.",
            "Judge whether the narrative naturally leads to another scene, and if so, select its cast before the scene starts.",
            "Plan the immediate next scene by choosing the most plausible cast based on continuity and the characters' current visible states.",
            "From the current world state and character descriptions, decide the next scene's cast, or conclude that the story ends here.",
        ]

        # ------------------------------------------------------------------
        # Location scenario (build_diverse_task_prompt style)
        # ------------------------------------------------------------------
        ls_role_variants = [
            "You are the second-stage scene planner for a story simulation. Your job is to place the already-selected characters into a concrete next scene.",
            "You are the location-and-scenario planning module. Given the selected cast, decide where the next scene should happen and what its dramatic setup is.",
            "You are the director-level scene setup planner. You receive the chosen characters and must decide the next location and scenario.",
            "You are responsible for turning a selected group of characters into a concrete next scene by choosing the location and writing the scenario.",
            "You are the world-planning module that finalizes the next scene after the participating characters have already been chosen.",
        ]
        ls_task_variants = [
            "Choose the most plausible location for the selected characters and generate the next scene's scenario.",
            "Use the current world state, location overviews, and the selected characters' current descriptions to decide where the next scene happens and what unfolds there.",
            "Finalize the next scene by selecting a location and writing a concise but usable scenario.",
            "Given the selected cast for the next scene, decide where they should meet and describe the dramatic setup.",
            "Plan the concrete setup of the next scene by choosing one location and generating the scenario for the selected characters.",
        ]

        # ------------------------------------------------------------------
        # Next character (_build_rich style)
        # ------------------------------------------------------------------
        nc_begin_variants = [
            "You are the turn-allocation module for a story simulation. Your task is to select which character or entity should act next in the current scene. Read the prompt in display order, but keep the timeline clear: the scenario and character states were established when the scene began; the prior interactions happened after that; the current world state is the result of those prior interactions; now you predict who acts next.",
            "You are the director responsible for deciding who speaks or acts next in an ongoing scene. The prompt may show the current world state first, but the chronology is still: scene start scenario and character states → prior interactions → current world state. Use that timeline to decide the next actor.",
            "You are the next-actor predictor for a simulated dramatic scene. The scene scenario and character profiles/motivations were set at scene start. Prior interactions occurred after that and produced the current world state shown below. Based on all of this, decide who should act next.",
            "Your role is to determine the next acting character in a scene based on dramatic flow and character motivations. Even if the current world state is shown before earlier sections, remember that prior interactions happened BEFORE that state and explain the path to the present moment.",
            "You decide turn order in a story simulation. The scene began with a scenario and character states; prior interactions then unfolded; the current world state reflects the result. Use this ordering to select the next character to act, or end the scene.",
            "You are the turn-taking controller. The prompt is organized for readability, but the chronological order remains: scenario (set at scene start) → character states (at scene start) → prior interactions (already happened) → current world state (result of those interactions). Predict who acts next or signal scene end.",
        ]
        nc_natural_sections = {
            'world_state': [
                "The current world state (this is the state AFTER the prior interactions in this scene have already taken place):\n{world_state}",
                "Current world state. Treat this as the latest state, already updated after the prior interactions listed later in the prompt:\n{world_state}",
                "World state as of now. This is the result of earlier scene interactions that happened BEFORE the current moment:\n{world_state}",
            ],
            'characters': [
                "The characters in the scene and their states at scene start (profile and motivation as of when the scene began):\n{characters}",
                "Here are the present characters with their descriptions and motivations (captured at scene start, before any interactions):\n{characters}",
                "Character state snapshot from scene start (these reflect their condition when the scene began):\n{characters}",
            ],
            'scenario': [
                "The scene scenario (established when this scene began):\n{scenario}",
                "Scene scenario (set at scene start):\n{scenario}",
                "The dramatic setup of the scene (determined at scene start):\n{scenario}",
            ],
            'prior_interactions': [
                "These prior interactions happened earlier in this scene, BEFORE the current world state shown above was reached:\n{prior}",
                "Prior interactions that happened since the scene started. They occurred BEFORE the current world state shown earlier in the prompt and explain how that state was produced:\n{prior}",
                "What has already happened in this scene so far. These interactions came BEFORE the current world state shown above and led to it:\n{prior}",
            ],
            'requirements': [
                "{output_format}",
                "Choose only from the following allowed outputs: {allowed_names}. {output_format}",
                "Base the decision on scene continuity, character motivations, the current world state, and the observed interaction flow. Remember that the prior interactions occurred before the current world state. Output only the JSON list.",
            ],
        }
        nc_pieces = {
            'world_state': ["Current World State (After Prior Interactions)", "World State (Result of Prior Interactions)", "Updated World State"],
            'characters': ["Characters at Scene Start", "Character States (At Scene Start)", "Cast Snapshot (Scene Start)"],
            'scenario': ["Scene Scenario (Set at Scene Start)", "Scenario (Established at Scene Start)", "Scene Setup (From Scene Start)"],
            'prior_interactions': ["Prior Interactions (Happened Before Current World State)", "Interactions So Far (Produced Current World State)", "What Already Happened (Before Current World State)"],
            'requirements': ["Requirements", "Prediction Rules", "Output Format"],
        }

        # ------------------------------------------------------------------
        # World update (build_diverse_task_prompt style)
        # ------------------------------------------------------------------
        wu_role_variants = [
            "You are the world-state update judge for a story simulation. Your job is to decide whether an interaction causes a lasting change to the persistent world state.",
            "You are the world-state controller. After each interaction, determine whether the global or location state needs a persistent update.",
            "You are the persistent-state maintenance module. Decide whether the latest interaction warrants updating the stored world state.",
            "You are the module that tracks lasting changes to the story world. Judge whether the newest interaction should trigger a world-state revision.",
            "You are responsible for deciding when the persistent world representation must be revised based on the latest interaction.",
        ]
        wu_task_variants = [
            "Given the scene context, prior interactions, current world state, and the latest interaction, decide whether the global world state and/or the current location state must be updated.",
            "Judge whether the latest interaction causes a meaningful persistent update to world state, considering the scene background and prior context.",
            "Review the latest interaction against the scene scenario and prior history, then update the relevant world state only when lasting change occurs.",
            "Determine whether the newest interaction creates a durable change that should now be reflected in persistent world memory.",
            "Inspect the latest interaction in light of the scene context and current world state, and revise world state only when something meaningfully and persistently changes.",
        ]

        # ------------------------------------------------------------------
        # Interaction generation (_build_rich style)
        # ------------------------------------------------------------------
        ig_begin_variants = [
            'You are now roleplaying as "{actor_label}" inside an ongoing story simulation. Read the context below as a live scene that is already underway, then continue from the newest interaction cue.',
            'Portray "{actor_label}" in a multi-agent story simulation. The context below is not a summary task; it is the live setup you must continue from in character.',
            'Act as "{actor_label}" in this active scene. Use the world snapshot, the cast information, and the interaction flow to produce the next turn naturally.',
            'You are responsible for the next in-character turn of "{actor_label}". Treat the material below as staged context for a scene already in progress.',
            'Continue this dramatic scene as "{actor_label}". The prompt below gives the current world snapshot, the surrounding cast, and the earlier scene history you need for continuity.',
            'Roleplay "{actor_label}" from "{book_name}" in an ongoing multi-turn simulation. Stay inside the character and respond to the live interaction flow that follows.',
        ]
        ig_natural_sections = {
            'world_state': [
                "Current world snapshot for this segment:\n{world_state}",
                "World state you should treat as CURRENT before the live segment continues:\n{world_state}",
                "Use this up-to-date world and location snapshot as your grounding context:\n{world_state}",
            ],
            'scenario': [
                "Scene setup established at the beginning of this scene:\n{scenario}",
                "Scene scenario from when this scene began:\n{scenario}",
                "Dramatic setup for the current scene:\n{scenario}",
            ],
            'other_characters': [
                "The other characters around you in this scene (public description plus scene-entry motivation):\n{other_characters}",
                "People you are interacting with in this scene. Treat these as the other characters' visible profiles and motivations:\n{other_characters}",
                "Other present characters you should react to (description and current drive):\n{other_characters}",
            ],
            'actor_state': [
                "Your private role sheet for this segment — detailed profile, hidden tracker, and current motivation:\n{actor_state}",
                "Your own character state. This is the most detailed information about who you are, what you carry internally, and what is driving you now:\n{actor_state}",
                "Detailed state for the role you must play — profile, hidden tracker, and motivation:\n{actor_state}",
            ],
            'prior_interactions': [
                "Earlier interactions from this same scene that happened BEFORE the current world snapshot above:\n{prior}",
                "What had already happened in the scene before the world state above became current:\n{prior}",
                "Interaction history from earlier in the scene, preceding the current world snapshot:\n{prior}",
            ],
            'requirements': [
                "{output_format}",
                "Stay faithful to {char_ref}'s profile, hidden tracker, and motivation. Do not use knowledge outside the provided state and visible interaction history.",
                "Prioritize local continuity: the latest visible turn is your immediate cue, so your response should connect tightly to it in emotion, logic, and dramatic momentum.",
            ],
        }
        ig_pieces = {
            'world_state': ["Current World Snapshot", "World State You Should Continue From", "Live World State"],
            'scenario': ["Scene Setup", "Scene Scenario", "Dramatic Situation"],
            'other_characters': ["Other Characters Around You", "Other Present Characters", "Who You Are Interacting With"],
            'actor_state': ["Your Detailed Character State", "Your Private Role Sheet", "Who You Are and What Drives You"],
            'prior_interactions': ["Earlier Scene Interactions", "Interaction History Before This Snapshot", "What Already Happened Earlier"],
            'requirements': ["Roleplay Instructions", "Continuation Rules", "Output Constraints"],
        }

        # Environment interaction generation
        ig_env_begin_variants = [
            "You are the environment layer of an ongoing story simulation. Your job is to continue the current scene by describing how the setting reacts around the characters.",
            "You are generating the environment's next turn in a multi-agent story simulation. Treat the context below as a live scene that is already in progress.",
            "You provide the environmental continuation for an active dramatic scene. Read the snapshot below carefully, then react to what the characters are doing now.",
            "You are responsible for the non-character side of the scene: atmosphere, background movement, physical setting changes, and ambient response.",
            "Continue this scene as the environment narrator. Use the current world snapshot plus the visible interaction flow to produce the next environmental beat.",
        ]

        # ------------------------------------------------------------------
        # Character update (build_diverse_task_prompt style)
        # Character update (build_diverse_task_prompt style)
        # ------------------------------------------------------------------
        cu_role_variants = [
            'You are tracking how "{character_name}" evolves throughout a story, scene by scene. Your goal is to maintain a comprehensive, up-to-date internal state that captures who this character is and how they change over time.',
            'You are the post-scene state manager for "{character_name}". After each completed scene, you analyze what happened and determine how the character\'s internal state \u2014 profile, hidden tracker, and description \u2014 should be updated.',
            'You are responsible for updating "{character_name}"\'s persistent internal state after a completed scene. You must reason carefully about what changed and what accumulated.',
            'You are the character evolution module for "{character_name}". After each scene, you decide which aspects of the character\'s state need updating and produce a structured state snapshot for use in future scenes.',
            'You are revising "{character_name}"\'s internal state after a scene concludes. Your updates will serve as the foundation for this character\'s behavior in all subsequent scenes.',
        ]
        cu_task_variants = [
            (
                "Follow these steps in order:\n"
                "1. **Reason about dimensions**: Identify which profile dimensions are STABLE (e.g., physical description, backstory) vs. DYNAMIC (e.g., relationships, goals, mental state) for this character.\n"
                "2. **Update the Hidden Tracker**: Record events, unresolved tensions, accumulated emotional pressure, or signals from this scene that may lead to future profile changes \u2014 even if the profile itself doesn't change yet.\n"
                "3. **Decide whether to update the profile**: Only update if a meaningful, lasting change occurred (e.g., a relationship shift, a major decision, a trauma). Do NOT update for minor or transient reactions.\n"
                "4. **Write a short description**: Summarize who this character is NOW in 50\u201380 words, third person. Focus on identity, current situation, key relationships, and immediate condition at the end of the scene."
            ),
            (
                "Perform the following tasks sequentially:\n"
                "1. **Dimension reasoning**: Briefly reason about which of this character's profile dimensions are stable vs. dynamic.\n"
                "2. **Hidden tracker update**: Update the tracker with accumulated events/signals from this scene that may eventually trigger a profile change.\n"
                "3. **Profile update decision**: Determine whether the scene justifies a persistent profile change. If yes, produce the full updated profile; if no, leave it null.\n"
                "4. **Short description**: Write a concise description (50\u201380 words) of the character's current identity, situation, and immediate condition after this scene."
            ),
            (
                "Process the completed scene step by step:\n"
                "1. Reason about which character dimensions are likely stable vs. dynamic across the story.\n"
                "2. Update the hidden tracker: record events, emotional pressure, or unresolved tensions that may lead to future profile changes.\n"
                "3. Decide if the profile should be updated NOW \u2014 only when the scene causes a meaningful, observable, lasting change. Accumulated tracker signals combined with this scene may cross the threshold.\n"
                "4. Write a short description (50\u201380 words, third person) covering who the character is right now and what condition they are left in at the end of the scene."
            ),
            (
                "Analyze the scene and update state in this order:\n"
                "1. **Stable vs. dynamic dimensions**: Which aspects of this character's profile are unlikely to change vs. which can evolve with events?\n"
                "2. **Hidden tracker**: Accumulate signals \u2014 experiences, interactions, emotional pressure, unresolved decisions \u2014 that haven't yet caused a visible profile change but may in the future.\n"
                "3. **Profile update**: Only update when the hidden tracker plus this scene's events cross a threshold for genuine, lasting character change. Be selective.\n"
                "4. **Description**: A brief snapshot (50\u201380 words) of the character's current identity, situation, and immediate condition after the scene."
            ),
            (
                "Your job is to translate the completed scene into character-state updates:\n"
                "1. First, reason about which profile dimensions are stable (backstory, physical traits) vs. dynamic (relationships, goals, emotional state).\n"
                "2. Then, update the hidden tracker with events and signals that may accumulate toward future profile changes.\n"
                "3. Next, decide whether to update the profile \u2014 require meaningful, lasting change, not transient reactions.\n"
                "4. Write a short description (50\u201380 words) of the character as of now."
            ),
        ]

        # ------------------------------------------------------------------
        # Motivation update (build_diverse_task_prompt style)
        # ------------------------------------------------------------------
        mu_role_variants = [
            'You are the pre-scene motivation planner for "{character_name}". After the next scene cast, location, and scenario are fixed, determine "{character_name}"\'s motivation entering that scene.',
            'You are responsible for updating "{character_name}"\'s scene-entry motivation once the next scene has been planned.',
            'You infer "{character_name}"\'s motivation for an upcoming scene using their current state plus the newly planned scene setup.',
            'You are the module that determines how "{character_name}" enters the next scene mentally and emotionally after the scene cast and setup are already decided.',
            'You are the next-scene motivation planner for "{character_name}". Given the fixed next-scene setup, produce the motivation that should drive this character into that scene.',
        ]
        mu_task_variants = [
            "Use the character's current profile, hidden tracker, short description, the previous scene context, and the fixed next-scene setup to generate the motivation they carry into that scene.",
            "Given the previous scene plus the planned next scene, infer the character's emotional state, immediate objectives, and action intentions when entering it.",
            "Translate the character's current internal state, the previous scene context, and the chosen next-scene setup into a concrete scene-entry motivation.",
            "Determine what this character feels, wants, and intends to do at the start of the already-planned next scene, using what just happened previously for continuity.",
            "Generate the character's complete motivation for the next scene after cast, location, and scenario are already fixed, while preserving continuity with the prior scene.",
        ]

        # ------------------------------------------------------------------
        # Pre-select one variant index for each task
        # ------------------------------------------------------------------
        self._pv = {
            'style': style,
            'decorator': decorator,
            # Scene cast
            'sc_role': random.choice(sc_role_variants),
            'sc_task': random.choice(sc_task_variants),
            # Location scenario
            'ls_role': random.choice(ls_role_variants),
            'ls_task': random.choice(ls_task_variants),
            # Next character
            'nc_begin': random.choice(nc_begin_variants),
            'nc_natural': {k: random.choice(v) for k, v in nc_natural_sections.items()},
            'nc_pieces': {k: random.choice(v) for k, v in nc_pieces.items()},
            # World update
            'wu_role': random.choice(wu_role_variants),
            'wu_task': random.choice(wu_task_variants),
            # Interaction generation
            'ig_begin': random.choice(ig_begin_variants),
            'ig_env_begin': random.choice(ig_env_begin_variants),
            'ig_natural': {k: random.choice(v) for k, v in ig_natural_sections.items()},
            'ig_pieces': {k: random.choice(v) for k, v in ig_pieces.items()},
            # Character update
            'cu_role': random.choice(cu_role_variants),
            'cu_task': random.choice(cu_task_variants),
            # Motivation update
            'mu_role': random.choice(mu_role_variants),
            'mu_task': random.choice(mu_task_variants),
        }

    def _decorate_section(self, title: str) -> str:
        """Apply the selected decorator style to a section title."""
        if self._pv['decorator']:
            return self._pv['decorator'].format(title)
        return title

    def run_snapshot(
        self,
        snapshot: dict[str, Any],
        on_progress: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a full simulation from a snapshot.

        Args:
            snapshot: The input test snapshot.
            on_progress: Optional callback ``(result_so_far, trace_so_far) -> None``
                invoked after each interaction turn completes so callers can
                persist intermediate progress.
        """
        # Select prompt variants for this snapshot (one style for the whole run)
        self._init_prompt_variants()

        state = {
            "book_name": snapshot["book_name"],
            "source_scene_index": snapshot["scene_index"],
            "world_state": snapshot["world_state"],
            "character_states": snapshot["character_states"],
            "scene_history": [],
            "previous_scene_context": snapshot.get("previous_scene"),
            "story_finished": False,
            "stop_reason": None,
        }

        traces: dict[str, Any] = {
            "book_name": snapshot["book_name"],
            "source_scene_index": snapshot["scene_index"],
            "model_routing": {
                "world_model": {
                    "base_url": self.world_model_client.config.base_url,
                    "model_name": self.world_model_client.config.model_name,
                },
                "character_agent": {
                    "base_url": self.character_agent_client.config.base_url,
                    "model_name": self.character_agent_client.config.model_name,
                },
            },
            "scenes": [],
        }

        for scene_offset in range(self.config.max_scenes):
            logger.info("=== Scene %d/%d | book=%s ===", scene_offset + 1, self.config.max_scenes, state["book_name"])
            scene_record = None  # reset each loop iteration, ensure safe judgment in except
            try:
                scene_cast, scene_cast_trace = self.plan_next_scene_cast(state)
                scene_trace = {"scene_index": scene_offset, "scene_cast_raw": scene_cast_trace}

                if not coerce_bool(scene_cast.get("has_next_scene"), False):
                    state["story_finished"] = True
                    state["stop_reason"] = "world_model_declared_story_end"
                    traces["scenes"].append(scene_trace)
                    break

                involved_characters = scene_cast.get("involved_characters", []) or []
                location_scenario, location_scenario_trace = self.plan_next_scene_location_scenario(
                    state,
                    involved_characters,
                )
                scene_trace["location_scenario_raw"] = location_scenario_trace

                scene_record = {
                    "scene_index": scene_offset,
                    "location": location_scenario.get("location"),
                    "scenario": location_scenario.get("scenario", ""),
                    "involved_characters": involved_characters,
                    "interactions": [],
                    "world_updates": [],
                    "motivation_updates": [],
                    "character_reflections": [],
                }
                motivation_trace = self.prepare_scene_motivations(state, scene_record)
                scene_trace["motivation_updates"] = motivation_trace
                execution_trace = self.run_scene_execution(state, scene_record, traces, on_progress)
                reflection_trace = self.run_scene_reflection(state, scene_record)

                scene_trace["execution"] = execution_trace
                scene_trace["reflection"] = reflection_trace
                traces["scenes"].append(scene_trace)
                state["scene_history"].append(scene_record)
            except Exception as exc:
                logger.error("Scene %d failed: %s", scene_offset, exc, exc_info=True)
                state["stop_reason"] = "error"
                state["error"] = str(exc)
                traces["scenes"].append({"scene_index": scene_offset, "error": str(exc)})
                # save to scene_history even if scene fails, for easier debugging
                if scene_record is not None:
                    scene_record["error"] = str(exc)
                    state["scene_history"].append(scene_record)
                break

        else:
            state["stop_reason"] = "max_scenes_reached"

        if state["stop_reason"] is None:
            state["stop_reason"] = "story_finished" if state["story_finished"] else "completed"

        result = {
            "book_name": state["book_name"],
            "source_scene_index": state["source_scene_index"],
            "stop_reason": state["stop_reason"],
            "story_finished": state["story_finished"],
            "scenes": state["scene_history"],
            "final_world_state": state["world_state"],
            "final_character_states": state["character_states"],
        }
        if "error" in state:
            result["error"] = state["error"]
        return result, traces

    def call_model(
        self,
        task_name: str,
        messages: list[dict[str, str]],
        parse_json: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        max_parse_retries: int = 5,
        expect_type: type | tuple[type, ...] | None = dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        if task_name in WORLD_MODEL_TASKS:
            client = self.world_model_client
            model_label = "world model"
        elif task_name in CHARACTER_AGENT_TASKS:
            client = self.character_agent_client
            model_label = "character agent"
        else:
            raise ValueError(f"Unknown task name: {task_name}")

        last_response_text = ""
        last_parse_error = None

        for attempt in range(max_parse_retries):
            # Use a higher temperature on retries to get different output.
            current_temperature = temperature if attempt == 0 else max(temperature, 0.8)

            t0 = time.time()
            response_text = client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=current_temperature,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            last_response_text = response_text

            separator = "=" * 80
            retry_note = f" (attempt {attempt + 1}/{max_parse_retries})" if attempt > 0 else ""
            # always output short task summary line
            retry_suffix = f" (attempt {attempt + 1})" if attempt > 0 else ""
            logger.info("[%s] %s | %d chars | %.1fs%s", task_name, model_label, len(response_text), elapsed, retry_suffix)

            if self.config.log_model_io:
                logger.info("\n%s", separator)
                logger.info("[%s] ===== Task call log (owner=%s) | %.1fs%s =====", task_name, model_label, elapsed, retry_note)

                # Full input prompt (messages sent to model)
                logger.info("---------- Input prompt / messages ( %d  entries) ----------", len(messages))
                for msg_idx, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    logger.info("  [msg %d] role=%s | length=%d chars\n%s", msg_idx, role, len(content), content)

                # Full output response
                logger.info("---------- Output response ( %d chars) ----------", len(response_text))
                logger.info("%s", response_text)

            parsed_output: Any = response_text
            parse_error = None
            if parse_json:
                try:
                    parsed_output = extract_json_fragment(response_text)
                    # Validate parsed result type; treat mismatch as parse failure to trigger retry
                    if expect_type is not None and not isinstance(parsed_output, expect_type):
                        raise ValueError(
                            f"Expected {expect_type}, got {type(parsed_output).__name__}: {parsed_output!r}"
                        )
                except Exception as exc:
                    parse_error = str(exc)
                    last_parse_error = parse_error
                    logger.warning(
                        "[%s] JSON parse failed (attempt  %d/%d): %s",
                        task_name, attempt + 1, max_parse_retries, parse_error,
                    )
                    if self.config.log_model_io:
                        logger.info("%s\n", separator)
                    # Retry if we haven't exhausted attempts
                    if attempt < max_parse_retries - 1:
                        continue

            # If there is a parsed JSON result, print it too
            if self.config.log_model_io and parse_json and parse_error is None:
                logger.info("---------- Parsed JSON output ----------")
                logger.info("%s", json.dumps(parsed_output, ensure_ascii=False, indent=2) if isinstance(parsed_output, (dict, list)) else str(parsed_output))

            if self.config.log_model_io:
                logger.info("%s\n", separator)

            trace = {
                "task_name": task_name,
                "task_owner": model_label,
                "messages": messages,
                "response_text": response_text,
                "parsed_output": parsed_output if parse_error is None else None,
                "parse_error": parse_error,
                "attempts": attempt + 1,
            }
            if parse_json and parse_error:
                raise ValueError(f"{task_name} failed to parse JSON after {max_parse_retries} attempts: {parse_error}")
            return parsed_output, trace

        # Should not reach here, but just in case
        trace = {
            "task_name": task_name,
            "task_owner": model_label,
            "messages": messages,
            "response_text": last_response_text,
            "parsed_output": None,
            "parse_error": last_parse_error,
            "attempts": max_parse_retries,
        }
        raise ValueError(f"{task_name} failed to parse JSON after {max_parse_retries} attempts: {last_parse_error}")

    def plan_next_scene_cast(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        all_character_descriptions = {
            name: char_state.get("short_description", "")
            for name, char_state in state["character_states"].items()
            if char_state.get("short_description", "")
        }
        # Get previous scene scenario and interactions for continuity
        prev_scenario = "(None)"
        prev_interactions = "(None)"
        previous_scene = state["scene_history"][-1] if state["scene_history"] else state.get("previous_scene_context")
        if previous_scene is not None:
            prev_scenario = previous_scene.get("scenario", "") or "(None)"
            last_interactions = previous_scene.get("interactions", [])
            if last_interactions:
                prev_interactions = format_interaction_history(last_interactions)
        messages = self.build_scene_cast_messages(
            global_state=state["world_state"].get("global_state", ""),
            character_descriptions=all_character_descriptions,
            previous_scenario=prev_scenario,
            previous_interactions=prev_interactions,
        )
        parsed, trace = self.call_model("scene_cast", messages, parse_json=True)
        return parsed, trace

    def plan_next_scene_location_scenario(
        self,
        state: dict[str, Any],
        involved_characters: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_character_descriptions = {
            name: state["character_states"].get(name, {}).get("short_description", "")
            for name in involved_characters
        }
        # Get previous scene scenario and interactions for continuity
        prev_scenario = "(None)"
        prev_interactions = "(None)"
        previous_scene = state["scene_history"][-1] if state["scene_history"] else state.get("previous_scene_context")
        if previous_scene is not None:
            prev_scenario = previous_scene.get("scenario", "") or "(None)"
            last_interactions = previous_scene.get("interactions", [])
            if last_interactions:
                prev_interactions = format_interaction_history(last_interactions)
        messages = self.build_location_scenario_messages(
            global_state=state["world_state"].get("global_state", ""),
            location_descriptions=state["world_state"].get("location_descriptions", {}),
            selected_character_descriptions=selected_character_descriptions,
            previous_scenario=prev_scenario,
            previous_interactions=prev_interactions,
        )
        parsed, trace = self.call_model("location_scenario", messages, parse_json=True)
        return parsed, trace

    def prepare_scene_motivations(
        self,
        state: dict[str, Any],
        scene_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        motivation_trace: list[dict[str, Any]] = []
        involved_characters = scene_record.get("involved_characters", [])
        location_name = scene_record.get("location")
        location_description = state["world_state"].get("location_descriptions", {}).get(location_name, "")
        previous_scene = state["scene_history"][-1] if state["scene_history"] else state.get("previous_scene_context")
        previous_scenario = (previous_scene or {}).get("scenario", "") or "(None)"
        for character_name in involved_characters:
            char_state = state["character_states"].get(character_name)
            if not char_state:
                continue
            previous_interactions = (
                format_interaction_history(
                    previous_scene.get("interactions", []),
                    perspective_character=character_name,
                )
                if previous_scene and previous_scene.get("interactions")
                else "(None)"
            )
            other_characters = {
                other_name: state["character_states"].get(other_name, {}).get("short_description", "")
                for other_name in involved_characters
                if other_name != character_name
            }
            messages = self.build_motivation_update_messages(
                character_name=character_name,
                current_profile=char_state.get("profile", ""),
                hidden_tracker=char_state.get("hidden_tracker", ""),
                current_description=char_state.get("short_description", ""),
                global_state=state["world_state"].get("global_state", ""),
                other_characters=other_characters,
                previous_scene_scenario=previous_scenario,
                previous_scene_interactions=previous_interactions,
                next_scene_location=location_name,
                next_scene_location_description=location_description,
                next_scene_scenario=scene_record.get("scenario", ""),
            )
            update, trace = self.call_model("motivation_update", messages, parse_json=True)
            motivation = update.get("motivation")
            if motivation:
                char_state["motivation"] = motivation
            scene_record["motivation_updates"].append(
                {
                    "character": character_name,
                    "motivation": char_state.get("motivation", ""),
                }
            )
            motivation_trace.append({"character": character_name, "motivation_update_raw": trace})
        return motivation_trace

    def _notify_progress(
        self,
        state: dict[str, Any],
        traces: dict[str, Any],
        on_progress: Any | None,
        current_scene: dict[str, Any] | None = None,
    ) -> None:
        """Build an intermediate result snapshot and invoke *on_progress*."""
        if on_progress is None:
            return
        # Include completed scenes plus the current in-progress scene
        scenes = list(state["scene_history"])
        if current_scene is not None:
            scenes.append(current_scene)
        intermediate_result = {
            "book_name": state["book_name"],
            "source_scene_index": state["source_scene_index"],
            "stop_reason": "in_progress",
            "story_finished": False,
            "scenes": scenes,
            "final_world_state": state["world_state"],
            "final_character_states": state["character_states"],
        }
        try:
            on_progress(intermediate_result, traces)
        except Exception:
            logger.warning("on_progress callback failed", exc_info=True)

    def run_scene_execution(self, state: dict[str, Any], scene_record: dict[str, Any], traces: dict[str, Any] | None = None, on_progress: Any | None = None) -> list[dict[str, Any]]:
        execution_trace: list[dict[str, Any]] = []

        # Track the current segment boundary.  A new segment starts whenever
        # the world-state (global or location) is updated after an interaction,
        # matching the training data's ``build_world_state_segments`` logic.
        #
        # ``seg_start_idx`` is the interaction index at which the current
        # segment began.  When a new segment starts we rebuild the system
        # prompts with the latest world state and treat all previous
        # interactions as "prior interactions" baked into the system prompt.

        seg_start_idx: int = 0  # interaction index where current segment began

        # --- helpers to (re)build multi-turn contexts for a segment ----------
        def _build_nc_context() -> list[dict[str, str]]:
            """Build the next-character multi-turn message list for the current segment."""
            nc_char_descs = {
                name: {
                    "description": state["character_states"].get(name, {}).get("short_description", ""),
                    "motivation": state["character_states"].get(name, {}).get("motivation", ""),
                }
                for name in scene_record["involved_characters"]
            }
            prior = format_interaction_history(scene_record["interactions"][:seg_start_idx]) if seg_start_idx > 0 else None
            nc_sys = self._build_next_character_system_prompt(
                global_state=state["world_state"].get("global_state", ""),
                location_name=scene_record["location"],
                location_state=state["world_state"].get("location_states", {}).get(scene_record["location"], ""),
                scenario=scene_record["scenario"],
                char_descs_in_scene=nc_char_descs,
                prior_interactions=prior,
            )
            return [
                {"role": "system", "content": nc_sys},
                {"role": "user", "content": "===Segment Start===\n\n"},
            ]

        def _build_ig_context(acting_characters: list[str]) -> list[dict[str, str]]:
            """Build a fresh interaction-gen multi-turn context for *acting_characters*.

            When multiple characters act together, all their full states are
            included and the perspective is set to the first character (matching
            training data behaviour).
            """
            actor_states_dict = {
                name: {
                    "profile": state["character_states"].get(name, {}).get("profile", ""),
                    "hidden_tracker": state["character_states"].get(name, {}).get("hidden_tracker", ""),
                    "motivation": state["character_states"].get(name, {}).get("motivation", ""),
                }
                for name in acting_characters
            }
            acting_set = set(acting_characters)
            other_descs = {
                name: {
                    "description": state["character_states"].get(name, {}).get("short_description", ""),
                    "motivation": state["character_states"].get(name, {}).get("motivation", ""),
                }
                for name in scene_record["involved_characters"]
                if name not in acting_set
            }
            # Perspective from the whole acting group, matching training data
            perspective_character = acting_characters
            prior = (
                format_interaction_history(
                    scene_record["interactions"][:seg_start_idx],
                    perspective_character=perspective_character,
                )
                if seg_start_idx > 0
                else None
            )
            ig_sys = self._build_interaction_generation_system_prompt(
                book_name=state["book_name"],
                acting_characters=acting_characters,
                location_name=scene_record["location"],
                global_state=state["world_state"].get("global_state", ""),
                location_state=state["world_state"].get("location_states", {}).get(scene_record["location"], ""),
                scenario=scene_record["scenario"],
                actor_states=actor_states_dict,
                other_char_descs=other_descs,
                prior_interactions=prior,
            )
            return [
                {"role": "system", "content": ig_sys},
                {"role": "user", "content": "===Segment Start===\n\n"},
            ]

        def _append_ig_turn(
            messages: list[dict[str, str]],
            role: str,
            content: str,
        ) -> None:
            """Append a turn while preserving ShareGPT-style role alternation.

            Training data merges consecutive turns with the same role; do the
            same here so lazily rebuilt simulation contexts stay aligned.
            """
            if not content:
                return
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})

        def _replay_segment_interactions_for_actor(
            messages: list[dict[str, str]],
            acting_characters: list[str],
        ) -> list[dict[str, str]]:
            """Replay current-segment interactions into a freshly created actor context.

            The system prompt only contains interactions before ``seg_start_idx``.
            If a new actor appears later in the same segment, we rebuild the
            already-happened in-segment interaction flow as user messages so the
            actor does not miss turns that occurred after the snapshot.
            """
            acting_set = set(acting_characters)
            for interaction in scene_record["interactions"][seg_start_idx:]:
                interaction_characters = interaction.get("characters", [])
                interaction_content = interaction.get("content", "")
                if not interaction_content:
                    continue
                shared_visibility = acting_set.intersection(interaction_characters)
                visible_content = interaction_content if shared_visibility else remove_other_character_thoughts(interaction_content)
                actor_label = ', '.join('"' + a + '"' for a in interaction_characters)
                _append_ig_turn(messages, "user", f'[{actor_label}]: {visible_content}')
            return messages

        # Initialise segment-0 contexts
        nc_messages: list[dict[str, str]] = _build_nc_context()
        # ig_contexts keyed by tuple of acting characters to support multi-character joint actions
        ig_contexts: dict[tuple[str, ...], list[dict[str, str]]] = {}

        for turn_idx in range(self.config.max_turns_per_scene):
            # Step 1: Ask next-character model
            next_actor_parsed, next_actor_trace = self._call_next_character(nc_messages)
            execution_step: dict[str, Any] = {"next_actor_raw": next_actor_trace}
            acting_characters = self.normalize_next_actor(next_actor_parsed)
            execution_step["selected_actor"] = acting_characters

            if acting_characters == ["<SCENE_END>"]:
                logger.info("  Scene ended after %d turns", turn_idx)
                nc_messages.append({"role": "assistant", "content": json.dumps(["<SCENE_END>"], ensure_ascii=False)})
                execution_trace.append(execution_step)
                return execution_trace

            logger.info("  Turn %d | actor=%s", turn_idx + 1, acting_characters)
            # Append the next-actor prediction to nc_messages as assistant turn
            nc_messages.append({"role": "assistant", "content": json.dumps(acting_characters, ensure_ascii=False)})

            # Step 2: Generate interaction using multi-turn context
            # Lazily initialise ig_context for this actor combination (may have been reset by a segment split)
            actor_key = tuple(acting_characters)
            if actor_key not in ig_contexts:
                ig_contexts[actor_key] = _build_ig_context(acting_characters)
                ig_contexts[actor_key] = _replay_segment_interactions_for_actor(
                    ig_contexts[actor_key],
                    acting_characters,
                )

            messages = ig_contexts[actor_key]
            # If the last message is from assistant (same actor acting consecutively), 
            # append a user hint to avoid errors from models without assistant prefill support.
            if messages and messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": "Continue."})
            interaction_sampling_kwargs = self._get_local_interaction_sampling_kwargs()
            parsed, trace = self.call_model(
                "interaction_gen",
                messages,
                parse_json=False,
                temperature=0.6,
                **interaction_sampling_kwargs,
            )
            interaction_text = str(parsed).strip()
            retry_reason = _get_interaction_retry_reason(interaction_text, scene_record["interactions"])
            if retry_reason is not None:
                if retry_reason == "recent_repeat":
                    logger.warning("[interaction_gen] Detected near-exact repetition for actor=%s; retrying once with an anti-repetition reminder.", acting_characters)
                    retry_reminder = (
                        "Reminder: continue from the latest cue and write a NEW next beat. "
                        "Do not repeat or paraphrase the most recent interaction(s). "
                        "Advance the scene with a fresh response that is still logically continuous."
                    )
                else:
                    logger.warning("[interaction_gen] Detected looping self-repetition for actor=%s; retrying once with a de-looping reminder.", acting_characters)
                    retry_reminder = (
                        "Reminder: your previous draft got stuck in self-repetition. "
                        "Write one concise, forward-moving next beat and avoid repeating the same sentence, phrase, action, or thought pattern. "
                        "Keep only the strongest version of each idea."
                    )
                retry_messages = list(messages)
                retry_messages.append({
                    "role": "user",
                    "content": retry_reminder,
                })
                parsed, trace = self.call_model(
                    "interaction_gen",
                    retry_messages,
                    parse_json=False,
                    temperature=0.8,
                    max_parse_retries=1,
                    **interaction_sampling_kwargs,
                )
                interaction_text = str(parsed).strip()
            messages.append({"role": "assistant", "content": interaction_text})

            # Format the actor label for nc_messages and broadcast, matching training format
            actor_label = ', '.join('"' + a + '"' for a in acting_characters)
            interaction_record = {
                "characters": acting_characters,
                "content": interaction_text,
            }
            scene_record["interactions"].append(interaction_record)
            execution_step["interaction_gen_raw"] = trace

            # Append the interaction content to nc_messages as human turn
            nc_messages.append({"role": "user", "content": f'[{actor_label}]: {interaction_text}'})

            # Broadcast the interaction to all other actor ig_contexts as human turn.
            for other_key, other_msgs in ig_contexts.items():
                if other_key != actor_key:
                    shared_visibility = set(other_key).intersection(actor_key)
                    masked_content = interaction_text if shared_visibility else remove_other_character_thoughts(interaction_text)
                    other_turn = f'[{actor_label}]: {masked_content}'
                    _append_ig_turn(other_msgs, "user", other_turn)

            # Step 3: World update (single-turn)
            world_update, world_update_trace = self.update_world_state(state, scene_record, interaction_record)
            scene_record["world_updates"].append(world_update)
            execution_step["world_update_raw"] = world_update_trace
            execution_trace.append(execution_step)

            # Persist intermediate progress after each interaction turn
            self._notify_progress(state, traces, on_progress, current_scene=scene_record)

            # Broadcast to ig_contexts of individual members when a multi-char
            # key exists.  E.g. if ("Alice", "Bob") just acted, we should also
            # push the interaction as a human turn into any existing single-char
            # contexts for "Alice" or "Bob" (they might also act individually
            # later in the scene).
            if len(actor_key) > 1:
                for member in acting_characters:
                    single_key = (member,)
                    if single_key in ig_contexts:
                        # For the member's own single-actor context, the joint
                        # interaction is treated as an assistant-like turn but
                        # since the actor set differs, it shows as a user turn.
                        turn_text = f'[{actor_label}]: {interaction_text}'
                        msgs = ig_contexts[single_key]
                        _append_ig_turn(msgs, "user", turn_text)

            # ---- Segment split check ----
            # If the world state was actually updated, start a new segment.
            # This mirrors training data's build_world_state_segments which cuts
            # at every global_card or location_card update boundary.
            if world_update.get("update_global") or world_update.get("update_location"):
                seg_start_idx = len(scene_record["interactions"])
                # Rebuild all multi-turn contexts with the new world state.
                # Prior interactions (everything up to seg_start_idx) are now
                # baked into the system prompt; the conversation history resets.
                nc_messages = _build_nc_context()
                ig_contexts.clear()  # will be lazily rebuilt on next use

        return execution_trace

    def _call_next_character(self, nc_messages: list[dict[str, str]]) -> tuple[Any, dict[str, Any]]:
        """Call the next-character model using the accumulated multi-turn messages."""
        parsed, trace = self.call_model("next_character", nc_messages, parse_json=True, expect_type=(dict, list))
        return parsed, trace

    def run_scene_reflection(self, state: dict[str, Any], scene_record: dict[str, Any]) -> list[dict[str, Any]]:
        reflection_trace: list[dict[str, Any]] = []
        for character_name in scene_record["involved_characters"]:
            if character_name not in state["character_states"]:
                continue
            char_state = state["character_states"][character_name]
            visible_history = mask_interactions_for_character(scene_record["interactions"], [character_name])
            messages = self.build_character_update_messages(
                character_name=character_name,
                current_profile=char_state.get("profile", ""),
                hidden_tracker=char_state.get("hidden_tracker", ""),
                current_description=char_state.get("short_description", ""),
                current_motivation=char_state.get("motivation", ""),
                scene_scenario=scene_record["scenario"],
                scene_interactions=visible_history,
            )
            update, trace = self.call_model("character_update", messages, parse_json=True)
            reflection_trace.append({"character": character_name, "character_update_raw": trace})

            char_state["hidden_tracker"] = update.get("hidden_tracker", char_state.get("hidden_tracker", ""))
            should_update_profile = coerce_bool(
                update.get("should_update_profile", update.get("profile_updated")),
                False,
            )
            if should_update_profile and update.get("updated_profile"):
                # profile must be plain text; if model outputs dict, convert back to Markdown
                prof = update["updated_profile"]
                if isinstance(prof, (dict, list)):
                    prof = dict_to_markdown_text(prof)
                char_state["profile"] = prof
                update["updated_profile"] = prof
            updated_description = update.get("short_description", update.get("description"))
            if updated_description:
                char_state["short_description"] = updated_description

            scene_record["character_reflections"].append(
                {
                    "character": character_name,
                    "profile_updated": should_update_profile,
                    "hidden_tracker": char_state.get("hidden_tracker", ""),
                    "profile": char_state.get("profile", ""),
                    "short_description": char_state.get("short_description", ""),
                }
            )
        return reflection_trace

    # --- Legacy single-turn methods removed; multi-turn replacements above ---

    def update_world_state(
        self,
        state: dict[str, Any],
        scene_record: dict[str, Any],
        latest_interaction: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prior_interactions = scene_record["interactions"][:-1]
        messages = self.build_world_update_messages(
            scene_scenario=scene_record["scenario"],
            global_state=state["world_state"].get("global_state", ""),
            location=scene_record["location"],
            location_state=state["world_state"].get("location_states", {}).get(scene_record["location"], ""),
            prior_interactions=prior_interactions,
            latest_interaction=latest_interaction,
        )
        parsed, trace = self.call_model("world_update", messages, parse_json=True)

        update_record = {
            "update_global": coerce_bool(parsed.get("update_global"), False),
            "update_location": coerce_bool(parsed.get("update_location"), False),
            "global_state": parsed.get("global_state"),
            "location": parsed.get("location", scene_record["location"]),
            "location_state": parsed.get("location_state"),
            "reason": parsed.get("reason", ""),
        }
        if update_record["update_global"] and update_record["global_state"]:
            # global_state must be plain text; if model outputs dict, convert back to Markdown
            gs = update_record["global_state"]
            if isinstance(gs, (dict, list)):
                gs = dict_to_markdown_text(gs)
            state["world_state"]["global_state"] = gs
            update_record["global_state"] = gs
        if update_record["update_location"] and update_record["location_state"]:
            # location_state must be a dict; if model outputs plain text, try parsing back to dict
            ls = update_record["location_state"]
            if isinstance(ls, str):
                try:
                    ls = json.loads(ls)
                except (json.JSONDecodeError, ValueError):
                    pass  # if parsing fails, keep the original string
            update_record["location_state"] = ls
            state["world_state"].setdefault("location_states", {})[update_record["location"]] = ls
        return update_record, trace

    def normalize_next_actor(self, parsed: Any) -> list[str]:
        """Return a list of actor names, or ["<SCENE_END>"] to signal scene end."""
        _END = ["<SCENE_END>"]
        _END_TOKENS = {"<SCENE_END>", "SCENE_END", "<END_SCENE>", "END_SCENE", "end_scene", "<END CHAT>"}

        if isinstance(parsed, list) and parsed:
            # Check if any element signals scene end
            for item in parsed:
                if str(item).strip() in _END_TOKENS:
                    return _END
            return [str(x).strip() for x in parsed]
        if isinstance(parsed, dict):
            for key in ("next_character", "character", "next_actor"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, list):
                        return [str(x).strip() for x in val] if val else _END
                    return [str(val).strip()]
            if coerce_bool(parsed.get("end_scene"), False):
                return _END
        value = str(parsed).strip()
        if value in _END_TOKENS:
            return _END
        return [value]

    def build_scene_cast_messages(
        self,
        global_state: str,
        character_descriptions: dict[str, str],
        previous_scenario: str = "(None)",
        previous_interactions: str = "(None)",
    ) -> list[dict[str, str]]:
        pv = self._pv

        rules = [
            "Base the decision on continuity and plausibility rather than novelty alone.",
            "Choose the cast that most naturally follows from the latest world state and the previous scene.",
            "Focus on unresolved goals, conflicts, and immediate narrative momentum.",
            "This is a scene-level casting decision before the scene starts, not a turn-level next-actor choice.",
            "If has_next_scene is false, do not include involved_characters.",
            "Return raw JSON only. No markdown. No explanation.",
        ]
        bullets = lambda items: '\n'.join(f'- {i}' for i in items)
        output_fields = [
            'has_next_scene: Boolean. Output false only if there is no subsequent scene.',
            'involved_characters: List of the characters participating in the next scene. Include this only when has_next_scene is true.',
        ]

        system = (
            f"{pv['sc_role']}\n\n"
            f"Task\n{pv['sc_task']}\n\n"
            f"Output\n{bullets(output_fields)}\n\n"
            f"Rules\n{bullets(rules)}"
        )

        user = (
            f"## Global World State\n{global_state}\n\n"
            f"## All Characters (Short Description Only)\n{json.dumps(character_descriptions, ensure_ascii=False, indent=2)}\n\n"
            f"## Previous Scene Scenario\n{previous_scenario}\n\n"
            f"## Previous Scene Interactions\n{previous_interactions}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_location_scenario_messages(
        self,
        global_state: str,
        location_descriptions: dict[str, str],
        selected_character_descriptions: dict[str, str],
        previous_scenario: str = "(None)",
        previous_interactions: str = "(None)",
    ) -> list[dict[str, str]]:
        pv = self._pv

        rules = [
            "Follow naturally from the latest world state and the previous scene.",
            "Choose the location and scenario that best continue the selected characters' immediate goals and conflicts.",
            "The scenario should set up the situation, not script the dialogue.",
            "Keep the scenario concise but usable for downstream acting.",
            "If there is no valid next scene, output location=null and scenario=null.",
            "Return raw JSON only. No markdown. No explanation.",
        ]
        bullets = lambda items: '\n'.join(f'- {i}' for i in items)
        output_fields = [
            'location: The next scene location. Output null only if there is no valid next-scene location to assign.',
            'scenario: A concise but usable dramatic setup for the next scene.',
        ]

        system = (
            f"{pv['ls_role']}\n\n"
            f"Task\n{pv['ls_task']}\n\n"
            f"Output\n{bullets(output_fields)}\n\n"
            f"Rules\n{bullets(rules)}"
        )

        user = (
            f"## Global World State\n{global_state}\n\n"
            f"## All Location Descriptions\n{json.dumps(location_descriptions, ensure_ascii=False, indent=2)}\n\n"
            f"## Characters Who Will Appear In The Next Scene (Short Description Only)\n{json.dumps(selected_character_descriptions, ensure_ascii=False, indent=2)}\n\n"
            f"## Previous Scene Scenario\n{previous_scenario}\n\n"
            f"## Previous Scene Interactions\n{previous_interactions}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_next_character_system_prompt(
        self,
        global_state: str,
        location_name: str,
        location_state: str,
        scenario: str,
        char_descs_in_scene: dict[str, dict[str, str]],
        prior_interactions: str | None = None,
    ) -> str:
        all_characters = list(char_descs_in_scene.keys())
        allowed_names = all_characters + ['Environment', '<SCENE_END>']

        world_state_text = (
            f'Current location: "{location_name}"\n\n'
            f"Global world state:\n{format_prompt_block(global_state)}\n\n"
            f"Location state:\n{format_prompt_block(location_state)}"
        )
        characters_text = format_prompt_block(char_descs_in_scene)
        prior_text = format_prompt_block(prior_interactions)

        return (
            "Task: choose who acts next RIGHT NOW in the current scene.\n"
            "Output only one JSON list and nothing else.\n\n"
            f"Allowed outputs: {allowed_names}.\n"
            "- Use one present character name for a normal turn.\n"
            "- Use ['Environment'] only for a non-character environmental beat.\n"
            "- Use ['<SCENE_END>'] only if the scene should end now.\n"
            "- Output multiple characters together only when the very next beat is one indivisible shared interaction that must be realized by those characters together right now.\n"
            "- Do not output a character group just because multiple characters are present, aligned, nearby, or likely to act one after another.\n"
            "- If one character can naturally act first and the others can respond afterward, choose only that one character.\n"
            "- A multi-character output should include only the characters who must jointly produce the same immediate beat, with no extra passengers.\n\n"
            "Decision priorities:\n"
            "1. Continue directly from the latest visible interaction cue.\n"
            "2. Keep the scene moving forward; do not restart the scene or repeat an already completed beat.\n"
            "3. Use character motivations and the current world state as support for what most naturally happens next.\n"
            "4. When unsure, prefer a single-character turn over a group turn unless the next beat genuinely requires simultaneous or jointly authored participation.\n"
            "5. Choose a multi-character list only for cases like a joint physical action, a jointly delivered line, or a tightly coupled shared reaction that belongs in one beat rather than split turns.\n"
            "6. Do not choose the same role as the current last speaker; avoid consecutive turns by the same character or actor group unless no other continuation is plausible.\n\n"
            "Timeline:\n"
            "- The scene scenario and character states are from scene start.\n"
            "- Prior interactions happened after scene start.\n"
            "- The current world state is the result of those prior interactions.\n"
            "- The live conversation after this prompt continues immediately after that point.\n\n"
            "Conversation protocol:\n"
            "- Each user message is an interaction in the format [\"CharA\", \"CharB\", ...]: content.\n"
            "- The content may contain [...] for inner thoughts, plain text for speech, and (...) for visible actions.\n"
            "- A multi-character list means one shared interaction beat, not separate consecutive turns.\n"
            "- Return raw JSON only. No markdown. No explanation.\n\n"
            f"## Characters At Scene Start\n{characters_text}\n\n"
            f"## Scene Scenario\n{format_prompt_block(scenario)}\n\n"
            f"## Current World State\n{world_state_text}\n\n"
            f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
            "Final reminder before the live segment starts: choose the next actor only, and identify the current last speaker from the full interaction history available at this moment, including the prior interactions above and any live conversation after segment start; do not select the same role as that last speaker for the next turn."
        ).strip()

    def build_world_update_messages(
        self,
        scene_scenario: str,
        global_state: str,
        location: str,
        location_state: str,
        prior_interactions: list[dict[str, Any]],
        latest_interaction: dict[str, Any],
    ) -> list[dict[str, str]]:
        pv = self._pv
        rules = [
            "Update only for meaningful persistent change.",
            "Be conservative: most interactions should not change persistent world state.",
            "Update global state only for broad lasting changes.",
            "Update location state only for lasting local changes to environment, atmosphere, or important non-human entities.",
            "When updating, output the full new state, not a diff.",
            "Return raw JSON only. No markdown. No explanation.",
        ]
        bullets = lambda items: '\n'.join(f'- {i}' for i in items)
        output_fields = [
            'update_global: Boolean. True only when the latest interaction changes the persistent global world state in a meaningful way.',
            'global_state: The complete updated global state as a single plain-text string (NOT a JSON object/dict) when update_global is true, otherwise null.',
            'update_location: Boolean. True only when the latest interaction changes the persistent state of the current location in a meaningful way.',
            'location_state: The complete updated location state as a JSON object/dict (NOT a plain-text string) when update_location is true, otherwise null.',
        ]

        system = (
            f"{pv['wu_role']}\n\n"
            f"Task\n{pv['wu_task']}\n\n"
            f"Output\n{bullets(output_fields)}\n\n"
            f"Rules\n{bullets(rules)}"
        )

        prior_history = format_interaction_history(prior_interactions)
        latest_history = format_interaction_history([latest_interaction]) if latest_interaction else '(None)'
        loc_state_str = json.dumps(location_state, ensure_ascii=False, indent=2) if isinstance(location_state, dict) else str(location_state)

        user = (
            f"## Scene Scenario\n{scene_scenario}\n\n"
            f"## Prior Interactions\n{prior_history if prior_history else '(None)'}\n\n"
            f"## Global World State\n{global_state}\n\n"
            f'## Current Location: "{location}"\n## Location State\n{loc_state_str}\n\n'
            f"## Latest Interaction\n{latest_history}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_interaction_generation_system_prompt(
        self,
        book_name: str,
        acting_characters: list[str],
        location_name: str,
        global_state: str,
        location_state: str,
        scenario: str,
        actor_states: dict[str, dict[str, str]] | None = None,
        other_char_descs: dict[str, dict[str, str]] | None = None,
        prior_interactions: str | None = None,
    ) -> str:
        actor_label = ', '.join(acting_characters)
        world_state_text = (
            f'Book: "{book_name}"\n\n'
            f'Current location: "{location_name}"\n\n'
            f"Global world state:\n{format_prompt_block(global_state)}\n\n"
            f"Location state:\n{format_prompt_block(location_state)}"
        )
        prior_text = format_prompt_block(prior_interactions)
        is_environment = acting_characters == ['Environment']

        if is_environment:
            major_characters = [c for c in (other_char_descs or {}).keys() if c != 'Environment']
            return (
                "Task: write exactly one next environmental beat RIGHT NOW.\n"
                "Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
                "Requirements:\n"
                "- Treat this beat as happening immediately after the prior interactions below.\n"
                "- Write exactly one short environmental turn.\n"
                "- Focus on atmosphere, background movement, physical changes, crowd reaction, sound, weather, or setting consequences.\n"
                f"- Do not take over the deliberate dialogue or private thoughts of the main characters ({major_characters}).\n"
                "- Use the current world state only as support; follow the interaction flow most closely.\n\n"
                "Conversation protocol:\n"
                "- The first user message is only '===Segment Start===' and should not be answered literally.\n"
                "- Every later user message is a new interaction that happens after the snapshot below.\n\n"
                f"## Scene Scenario\n{format_prompt_block(scenario)}\n\n"
                f"## Current World State\n{world_state_text}\n\n"
                f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
                "Final reminder before the live segment starts: you are roleplaying the environment layer. Do not repeat any interaction beat already covered in the prior interactions above or in the live conversation after segment start, and do not turn this beat into a main-character dialogue turn."
            ).strip()

        actor_state_text = format_prompt_block(actor_states)
        other_characters_text = format_prompt_block(other_char_descs)
        multi_char = len(acting_characters) > 1

        if multi_char:
            role_block = (
                f"Task: roleplay {actor_label} and write exactly one next shared interaction beat RIGHT NOW.\n"
                f"You are {actor_label} now. Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
                "Shared-turn rules:\n"
                "- Treat the turn as happening immediately after the prior interactions below.\n"
                "- Only write one truly shared beat.\n"
                "- This shared beat is one local moment jointly realized by the acting group, not a bundle of separate back-to-back turns.\n"
                "- The group should do or say one thing together: for example a joint action, a jointly delivered line, or one tightly coupled shared reaction unfolding in the same moment.\n"
                "- Do not split into separate mini-turns for each character, do not serialize the group into first X then Y then Z, and do not give unrelated contributions from different members in the same output.\n"
                "- Include only material that belongs to this one shared moment. If a member's contribution would naturally happen later as a follow-up, leave it out.\n"
                "- Thoughts from members of the acting group may be used inside this shared turn, but only when they support the same shared moment.\n\n"
            )
        else:
            role_block = (
                f"Task: roleplay {actor_label} and write exactly one next interaction turn RIGHT NOW.\n"
                f"You are {actor_label} now. Continue directly from the latest visible interaction cue. Do not restart the scene. Do not repeat or paraphrase an already completed beat.\n\n"
            )

        return (
            role_block
            + "Priority rules:\n"
            + "1. Stay in character.\n"
            + "2. React or act from the latest visible cue, not from the beginning of the scene.\n"
            + "3. Keep the whole scene logically continuous. The new turn must advance or react within the same ongoing scene.\n"
            + "4. Do not restate, replay, or slightly reword an earlier beat.\n"
            + "5. Use world state and character state only as support for continuity; follow the interaction flow most closely.\n"
            + "6. Keep the turn short and local. No summary. No jump ahead.\n\n"
            + "Output format:\n"
            + "- Write exactly one full turn.\n"
            + "- Start with a real inner-thought block in square brackets.\n"
            + "- Put spoken dialogue in plain text with no speaker label.\n"
            + "- Do NOT put spoken dialogue inside square brackets, and do NOT wrap speech-only text in parentheses such as [Who is that?] (I say).\n"
            + "- Put visible physical actions in parentheses.\n"
            + "- These elements can be interleaved naturally.\n\n"
            + "Conversation protocol:\n"
            + "- The first user message is only '===Segment Start===' and should not be answered literally.\n"
            + "- Every later user message is a new visible interaction that happens after the snapshot below.\n"
            + "- Each user message uses the format [\"CharA\", \"CharB\", ...]: content.\n"
            + "- Other characters' private thoughts are already removed unless the acting group overlaps with them.\n\n"
            + f"## Your Character State\n{actor_state_text}\n\n"
            + (f"## Other Characters\n{other_characters_text}\n\n" if other_characters_text != '(None)' else "")
            + f"## Scene Scenario\n{format_prompt_block(scenario)}\n\n"
            + f"## Current World State\n{world_state_text}\n\n"
            + f"## Prior Interactions Before Current World State\n{prior_text}\n\n"
            + f"Final reminder before the live segment starts: you are roleplaying {actor_label}. Do not repeat any interaction beat already covered in the prior interactions above or in the live conversation after segment start, including earlier turns already produced by {actor_label}."
        ).strip()

    def build_character_update_messages(
        self,
        character_name: str,
        current_profile: str,
        hidden_tracker: str,
        current_description: str,
        current_motivation: str,
        scene_scenario: str,
        scene_interactions: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        pv = self._pv

        role_text = pv['cu_role'].format(character_name=character_name)
        task_text = pv['cu_task']
        rules = [
            "All outputs must describe the character at the END of the current scene.",
            "Update hidden_tracker with accumulated internal signals from this scene.",
            "Update the full profile only for meaningful lasting change, not transient reactions.",
            "Write a concise short_description for the scene end state.",
            "Return raw JSON only. No markdown. No explanation.",
        ]
        bullets = lambda items: '\n'.join(f'- {i}' for i in items)
        output_fields = [
            'hidden_tracker: Updated tracker of accumulated events/signals that may lead to future profile changes. Null if there are no signals worth tracking.',
            'profile_updated: Boolean. True only if the scene caused a meaningful lasting change to the character profile.',
            'updated_profile: The full updated profile as a single plain-text string (NOT a JSON object/dict) when profile_updated is true, otherwise null.',
            'short_description: A brief description (50–80 words, third person) of the character at the end of this scene.',
        ]
        system = (
            f"{role_text}\n\n"
            f"Task\n{task_text}\n\n"
            f"Output\n{bullets(output_fields)}\n\n"
            f"Rules\n{bullets(rules)}"
        )

        scene_history = format_interaction_history(scene_interactions)
        user = (
            f"## Scene Scenario\n{scene_scenario}\n\n"
            f"## Scene Interactions\n{scene_history}\n\n"
            f"## Current Profile (State At Scene Start)\n{current_profile}\n\n"
            f"## Hidden Tracker (State At Scene Start)\n{hidden_tracker if hidden_tracker else '(Empty)'}\n\n"
            f"## Current Motivation (State At Scene Start)\n{current_motivation if current_motivation else '(None)'}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_motivation_update_messages(
        self,
        character_name: str,
        current_profile: str,
        hidden_tracker: str,
        current_description: str,
        global_state: str,
        other_characters: dict[str, str],
        previous_scene_scenario: str,
        previous_scene_interactions: str,
        next_scene_location: Any,
        next_scene_location_description: str,
        next_scene_scenario: str,
    ) -> list[dict[str, str]]:
        pv = self._pv

        role_text = pv['mu_role'].format(character_name=character_name)
        task_text = pv['mu_task']
        rules = [
            "This is a pre-scene motivation step after the next scene has already been planned.",
            "Use the fixed next-scene cast, location, and scenario as constraints.",
            "Keep continuity with the previous scene from this character's perspective.",
            "Keep the motivation specific to this character's own perspective and immediate goals.",
            "Return raw JSON only. No markdown. No explanation.",
        ]
        bullets = lambda items: '\n'.join(f'- {i}' for i in items)
        output_fields = [
            'motivation: The character\'s complete inner drive entering the next scene (1–3 sentences): emotional state, immediate objectives, action intentions, who they intend to seek out, and what they want to do or discuss.',
        ]
        system = (
            f"{role_text}\n\n"
            f"Task\n{task_text}\n\n"
            f"Output\n{bullets(output_fields)}\n\n"
            f"Rules\n{bullets(rules)}"
        )

        user = (
            f"## Current Profile\n{current_profile}\n\n"
            f"## Hidden Tracker\n{hidden_tracker if hidden_tracker else '(Empty)'}\n\n"
            f"## Current Short Description\n{current_description if current_description else '(None)'}\n\n"
            f"## Global World State\n{global_state}\n\n"
            f"## Previous Scene Scenario\n{previous_scene_scenario}\n\n"
            f"## Previous Scene Interactions\n{previous_scene_interactions}\n\n"
            f"## Next Scene Location\n{next_scene_location}\n\n"
            f"## Next Scene Location Description\n{next_scene_location_description if next_scene_location_description else '(None)'}\n\n"
            f"## Next Scene Scenario\n{next_scene_scenario}\n\n"
            f"## Other Characters In Next Scene (Short Description Only)\n{json.dumps(other_characters, ensure_ascii=False, indent=2)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
