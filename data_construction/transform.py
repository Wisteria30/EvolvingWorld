"""
Transform extracted book data into training and test data for the two-model simulation system.

============================================================
Model A: World Model (director/environment model)
  - Task 1: Scene Cast Planning: decide whether the next scene exists and select its full cast from world and character state.
  - Task 2: Location + Scenario Planning: choose the location and generate the scenario after the cast has been selected.
  - Task 3: Next Character Proposal: choose the next actor in a scene, or end the scene.
  - Task 4: World State Update: decide whether persistent global/location state should be updated.

Model B: Character Agent
  - Task 1: Next Interaction Generation: generate character dialogue, action, and thoughts.
  - Task 2: Character State Update: update internal character state after a scene, excluding next-scene motivation.
  - Task 3: Character Motivation Update: generate motivation for characters in the next scene after cast, location, and scenario are fixed.

============================================================
Data split strategy:
  - 10% of books are used as OOD test books; simulation start points are sampled only from the later 70% of scenes.
  - The remaining 90% of books are split into two groups:
    - Half are train+test books: first 70% for training, later 30% for ID testing.
    - Half are train-only books: all scenes are used for training.

Output format: ShareGPT SFT format for supervised fine-tuning.
"""

import os
import json
import copy
import random
import math
import argparse
import re
from pathlib import Path
from collections import Counter
from utils import (
    get_scene_cast_system_prompt,
    get_scene_location_scenario_system_prompt,
    get_next_character_system_prompt,
    get_world_state_update_system_prompt,
    get_interaction_generation_system_prompt,
    get_character_state_update_system_prompt,
    get_character_motivation_update_system_prompt,
)

# ============================================================================ #
# Command-line argument parsing
# ============================================================================ #
parser = argparse.ArgumentParser(description='Transform extracted data to training/test format')
parser.add_argument('--dir', type=str,
                    default='dataset/extracted_data',
                    help='Root extracted data directory')  # Extracted data root directory
parser.add_argument('--seed', type=int,
                    default=40,
                    help='Random seed for reproducible book ordering and sampling')
args = parser.parse_args()

DATA_DIR = args.dir      # Global extracted-data directory path
SEED = args.seed         # Global random seed

# Set the global random seed for reproducibility.
random.seed(SEED)

# Special actor name for environment narration.
ENVIRONMENT = 'Environment'

# ============================================================================ #
# Reproducible seed-controlled shuffling
# ============================================================================ #
def seeded_shuffle(files, seed):
    """Shuffle filenames reproducibly with the given seed.

    The files are sorted first to remove filesystem ordering instability, then
    shuffled with a local RNG so the same input set and seed produce the same order.

    # Args:
        files: List of filenames to shuffle.
        seed: Random seed.
    # Returns:
        Filenames shuffled according to the seed.
    """
    files = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    return files


# ============================================================================ #
# Helper: remove [thought] blocks from interaction content
# ============================================================================ #
def normalize_content(content):
    """Normalize interaction content, including rare list-valued content.

    Most content values are strings, but a few cases are stored as lists and
    need to be joined into a single string.

    # Args:
        content: Interaction content, either a string or a list.
    # Returns:
        Normalized string content.
    """
    if isinstance(content, list):
        return ' '.join(str(item) for item in content)
    return content if isinstance(content, str) else str(content)


def remove_thoughts(content):
    """Remove [thought] blocks from interaction content.

    During simulation, a character can only observe their own thoughts; other
    characters' thoughts are hidden, leaving only speech and actions.

    # Args:
        content: Full interaction text.
    # Returns:
        Text with bracketed thought blocks removed.
    """
    content = normalize_content(content)
    return re.sub(r'\[.*?\]', '', content).strip()


# ============================================================================ #
# Helper: build an interaction-history string
# ============================================================================ #
def build_interaction_history(interactions, up_to_index, perspective_character=None):
    """Build an interaction-history string for interactions before up_to_index.

    This shared helper concatenates previous scene interactions for multiple
    tasks. If a perspective character or character set is provided, thought
    blocks are kept only for interactions involving that perspective; other
    characters' thoughts are hidden.

    Used by next-character proposal, world-state update, interaction generation,
    and character-state update tasks.

    # Args:
        interactions: Current scene interactions with characters and content.
        up_to_index: Exclusive end index for the history slice.
        perspective_character: Character name or set whose thoughts remain visible.
    # Returns:
        Interaction history string with blank lines between turns.
    """
    if perspective_character is None:
        visible_perspective = set()
    elif isinstance(perspective_character, str):
        visible_perspective = {perspective_character}
    else:
        visible_perspective = {str(name) for name in perspective_character if str(name).strip()}

    lines = []
    for i in range(up_to_index):
        inter = interactions[i]
        chars = inter.get('characters', [])       # Characters involved in this interaction
        content = normalize_content(inter.get('content', ''))  # Interaction text including speech/action/thought
        char_str = ', '.join(f'"{c}"' for c in chars) if chars else f'"{ENVIRONMENT}"'  # Join actor names; use environment narration if absent
        
        # Thought masking: hide thoughts when the actors are outside the visible perspective.
        if visible_perspective and not visible_perspective.intersection(chars):
            content = remove_thoughts(content)
        
        lines.append(f"[{char_str}]: {content}")
    return '\n\n'.join(lines)


def get_character_hidden_tracker_before(dynamic_profile, scene_index):
    """Return the latest hidden tracker before the given scene."""
    hidden_tracker = ''
    for entry in dynamic_profile.get('scene_descriptions', []):
        if entry['scene_index'] < scene_index:
            hidden_tracker = entry.get('hidden_tracker', '') or ''
        else:
            break
    return hidden_tracker



def get_scene_character_contexts(scene_chars, char_dynamic, scene_index):
    """Collect descriptions, motivations, and full state for scene characters."""
    char_descs_in_scene = {}
    actor_states = {}
    for char_name in scene_chars:
        dyn = char_dynamic.get(char_name, {})
        desc = get_character_description_before(dyn, scene_index)
        motivation = get_character_motivation_before(dyn, scene_index)
        profile = get_character_profile_at(dyn, scene_index)
        hidden_tracker = get_character_hidden_tracker_before(dyn, scene_index)
        char_descs_in_scene[char_name] = {
            'description': desc,
            'motivation': motivation,
        }
        actor_states[char_name] = {
            'profile': profile,
            'hidden_tracker': hidden_tracker,
            'motivation': motivation,
        }
    return char_descs_in_scene, actor_states



def build_world_state_segments(interactions, global_card_history, location_card_history, scene_index):
    """Split one scene into continuous segments at world-state update boundaries."""
    boundaries = {0}
    global_update_set = {
        entry['interaction_index'] + 1
        for entry in global_card_history
        if entry['scene_index'] == scene_index and entry['interaction_index'] >= 0
    }
    location_update_set = {
        entry['interaction_index'] + 1
        for entry in location_card_history
        if entry['scene_index'] == scene_index and entry['interaction_index'] >= 0
    }
    boundaries.update(global_update_set)
    boundaries.update(location_update_set)
    boundaries.add(len(interactions))
    ordered = sorted(idx for idx in boundaries if 0 <= idx <= len(interactions))
    segments = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        if start < end:
            segments.append((start, end))
    return segments



def build_multi_turn_next_character_sample(book_name, scene_index, location_name, scenario, scene_chars, char_descs_in_scene, interactions, global_card_history, location_card_data):
    """Build multi-turn ShareGPT samples for Model A next-character proposal."""
    location_history = location_card_data.get(location_name, {}).get('card_history', [])
    segments = build_world_state_segments(interactions, global_card_history, location_history, scene_index)
    samples = []

    for seg_start, seg_end in segments:
        global_card = get_global_card_at(global_card_history, scene_index, seg_start - 1)
        location_card = get_location_card_at(location_card_data, location_name, scene_index, seg_start - 1)
        prior_interactions = build_interaction_history(interactions, seg_start)
        system_prompt = get_next_character_system_prompt(
            global_card=global_card,
            location_name=location_name,
            location_card=location_card,
            scenario=scenario,
            char_descs_in_scene=char_descs_in_scene,
            prior_interactions=prior_interactions,
        )

        chat = [
            {'from': 'system', 'value': system_prompt},
            {'from': 'human', 'value': '===Segment Start===\n\n'},
        ]

        for i_inter in range(seg_start, seg_end):
            inter = interactions[i_inter]
            inter_chars = inter.get('characters', [])
            inter_content = normalize_content(inter.get('content', ''))
            if not inter_content:
                continue
            next_actor = inter_chars if inter_chars else [ENVIRONMENT]
            chat.append({'from': 'assistant', 'value': json.dumps(next_actor, ensure_ascii=False)})
            actor_label = ', '.join('"' + a + '"' for a in next_actor)
            chat.append({'from': 'human', 'value': f"[{actor_label}]: {inter_content}"})

        if seg_end >= len(interactions):
            # Scene ends after this segment: append <SCENE_END> prediction
            chat.append({'from': 'assistant', 'value': json.dumps(['<SCENE_END>'], ensure_ascii=False)})
        else:
            # Not scene end: the next actor prediction belongs to the next segment.
            # Remove the trailing human turn (last interaction's content) since
            # there is no corresponding assistant reply in this segment.
            if chat and chat[-1]['from'] == 'human' and chat[-1]['value'] != '===Segment Start===\n\n':
                chat.pop()

        if len([m for m in chat if m['from'] == 'assistant']) == 0:
            continue

        samples.append({
            'conversations': chat,
            'details': {
                'book_name': book_name,
                'scene_index': scene_index,
                'segment_start': seg_start,
                'segment_end': seg_end,
                'task': 'next_character_proposal',
                'format': 'multi_turn_segment',
            },
        })

    return samples



def build_multi_turn_interaction_samples(book_name, scene_index, location_name, scenario, scene_chars, char_descs_in_scene, actor_states, interactions, global_card_history, location_card_data):
    """Build multi-turn ShareGPT samples for Model B interaction generation."""
    location_history = location_card_data.get(location_name, {}).get('card_history', [])
    segments = build_world_state_segments(interactions, global_card_history, location_history, scene_index)
    samples = []

    for seg_start, seg_end in segments:
        global_card = get_global_card_at(global_card_history, scene_index, seg_start - 1)
        location_card = get_location_card_at(location_card_data, location_name, scene_index, seg_start - 1)
        prior_interactions_env = build_interaction_history(interactions, seg_start)

        segment_actors = []
        for i_inter in range(seg_start, seg_end):
            chars = interactions[i_inter].get('characters', [])
            actor_key = tuple(chars) if chars else (ENVIRONMENT,)
            if actor_key not in segment_actors:
                segment_actors.append(actor_key)

        for actor_key in segment_actors:
            acting_characters = list(actor_key)
            is_environment = acting_characters == [ENVIRONMENT]
            perspective_character = None if is_environment else acting_characters
            system_prompt = get_interaction_generation_system_prompt(
                book_name=book_name,
                acting_characters=acting_characters,
                location_name=location_name,
                global_card=global_card,
                location_card=location_card,
                scenario=scenario,
                actor_states=None if is_environment else {name: actor_states.get(name, {}) for name in acting_characters},
                other_char_descs=None if is_environment else {name: value for name, value in char_descs_in_scene.items() if name not in acting_characters},
                prior_interactions=prior_interactions_env if is_environment else build_interaction_history(interactions, seg_start, perspective_character=perspective_character),
                is_environment=is_environment,
            )

            chat = [
                {'from': 'system', 'value': system_prompt},
                {'from': 'human', 'value': '===Segment Start===\n\n'},
            ]

            has_assistant_turn = False
            for i_inter in range(seg_start, seg_end):
                inter = interactions[i_inter]
                chars = inter.get('characters', [])
                content = normalize_content(inter.get('content', ''))
                if not content:
                    continue
                current_actor = chars if chars else [ENVIRONMENT]
                user_content = build_interaction_history([inter], 1, perspective_character=perspective_character)
                if current_actor == acting_characters:
                    role = 'assistant'
                    new_value = content
                    has_assistant_turn = True
                else:
                    role = 'human'
                    new_value = user_content

                # Merge consecutive turns of the same role to maintain
                # strict human/assistant alternation required by ShareGPT.
                if chat and chat[-1]['from'] == role:
                    chat[-1]['value'] += '\n\n' + new_value
                else:
                    chat.append({'from': role, 'value': new_value})

            if not has_assistant_turn:
                continue

            # Ensure the conversation ends with an assistant turn.
            # If the last interaction in the segment was from a different actor,
            # the trailing human turn has no corresponding assistant reply and
            # should be removed (ShareGPT requires the last turn to be assistant).
            while chat and chat[-1]['from'] == 'human' and chat[-1]['value'] != '===Segment Start===\n\n':
                chat.pop()

            # After trimming, re-check that we still have at least one assistant turn.
            if len([m for m in chat if m['from'] == 'assistant']) == 0:
                continue

            samples.append({
                'conversations': chat,
                'details': {
                    'book_name': book_name,
                    'scene_index': scene_index,
                    'segment_start': seg_start,
                    'segment_end': seg_end,
                    'character': acting_characters[0] if len(acting_characters) == 1 else ', '.join(acting_characters),
                    'task': 'interaction_generation',
                    'format': 'multi_turn_segment',
                },
            })

    return samples


# ============================================================================ #
# Helpers for retrieving world state at a specific time.
# These functions traverse chronological histories ordered by scene_index and
# interaction_index to find the latest active global/location state or location
# description at a given moment.
# ============================================================================ #
def get_global_card_at(global_card_history, scene_index, interaction_index=-1):
    """Return the global world state active at a given scene/interaction time."""
    # Default to the first entry, the initial global state.
    result = global_card_history[0]['global_card'] if global_card_history else ''
    for entry in global_card_history:
        es, ei = entry['scene_index'], entry['interaction_index']
        # Use this version if the update occurs before or at the target time.
        if (es < scene_index) or (es == scene_index and ei <= interaction_index):
            result = entry['global_card']
        else:
            break  # Histories are time-sorted, so stop after passing the target time.
    return result


def get_location_card_at(location_card_data, location_name, scene_index, interaction_index=-1):
    """Return the location state active at a given scene/interaction time."""
    loc_data = location_card_data.get(location_name, {})
    card_history = loc_data.get('card_history', [])
    if not card_history:
        return {}
    # Default to the first entry, the initial location state.
    result = card_history[0].get('location_card', {})
    for entry in card_history:
        es, ei = entry['scene_index'], entry['interaction_index']
        if (es < scene_index) or (es == scene_index and ei <= interaction_index):
            result = entry.get('location_card', result)
        else:
            break
    return result



def get_location_description_at(location_card_data, location_name, scene_index):
    """Return the latest short location description before the given scene."""
    loc_data = location_card_data.get(location_name, {})
    descriptions = loc_data.get('scene_descriptions', [])
    result = ''
    for entry in descriptions:
        if entry['scene_index'] < scene_index:
            result = entry.get('description', result)
        else:
            break  # Stop after passing the target scene.
    return result


# ============================================================================ #
# Helpers for retrieving character state at a specific time.
# Character dynamics are stored in character_dynamic:
#   - profile_history: profile snapshots that may evolve over the story.
#   - scene_descriptions: mixed-time entries for scenes the character appears in.
#     - enhanced_motivation: active before the scene starts.
#     - description: short description generated after the scene ends.
#     - hidden_tracker: tracker updated after the scene for signals that may
#       drive future profile changes.
# ============================================================================ #
def get_character_profile_at(dynamic_profile, scene_index):
    """Return the character profile active at the given scene index."""
    profile_history = dynamic_profile.get('profile_history', [])
    if not profile_history:
        return ''
    # Default to the initial profile stored in history[0].
    result = profile_history[0].get('profile', '')
    for entry in profile_history:
        if entry['scene_index'] < scene_index:
            result = entry.get('profile', result)
        else:
            break
    return result


def get_character_scene_info(dynamic_profile, scene_index):
    """Return the scene_descriptions entry for a character at one scene."""
    for entry in dynamic_profile.get('scene_descriptions', []):
        if entry['scene_index'] == scene_index:
            return entry
    return None


def get_character_motivation_before(dynamic_profile, scene_index):
    """Return the motivation active before the given scene.

    enhanced_motivation is stored on the scene where the character next appears,
    so for gaps between appearances the next available motivation is considered
    active; otherwise the latest past non-empty motivation is used.
    """
    scene_descs = dynamic_profile.get('scene_descriptions', [])
    last_past_motivation = ''

    for entry in scene_descs:
        motivation = entry.get('enhanced_motivation', '') or ''
        if entry['scene_index'] >= scene_index:
            return motivation if motivation else last_past_motivation
        if motivation:
            last_past_motivation = motivation

    return last_past_motivation


def get_character_description_before(dynamic_profile, scene_index):
    """Return the latest short character description before the given scene."""
    # Fallback: initial description stored in profile_history[0]
    # (generated before any scene, based solely on the initial profile)
    profile_history = dynamic_profile.get('profile_history', [])
    result = profile_history[0].get('description', '') if profile_history else ''

    for entry in dynamic_profile.get('scene_descriptions', []):
        if entry['scene_index'] < scene_index:
            result = entry.get('description', result)
        else:
            break
    return result


# ============================================================================ #
# Load all data files for one book
# ============================================================================ #
def load_book_data(book_name):
    """Load all required extracted-data files for one book.

    Dynamic files are used directly because their first history entries store
    the corresponding initialization data.

    Args:
        book_name: Book name without extension.

    Returns:
        A dictionary with scenes, char_dynamic, and world_dynamic data, or None
        if any required file is missing.
    """
    # Define the three required data-source paths.
    paths = {
        'scenes': f'{DATA_DIR}/scenes/{book_name}.json',
        'char_dynamic': f'{DATA_DIR}/character_dynamic/{book_name}.json',
        'world_dynamic': f'{DATA_DIR}/world_dynamic/{book_name}.json',
    }
    
    # Check that all required files exist.
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"  Warning: missing {key} for {book_name}, skipping.")
            return None
    
    # Load each JSON file.
    data = {}
    for key, path in paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            data[key] = json.load(f)
    
    return data


# ============================================================================ #
# Process one book: generate training samples and test points.
# Core conversion function for turning one book into samples for the seven tasks.
# ============================================================================ #
def process_book(book_name, data, train_scene_range, is_ood=False):
    """Generate training samples and test candidates for one book.

    Args:
        book_name: Book name without extension.
        data: Dictionary returned by load_book_data().
        train_scene_range: Inclusive-exclusive scene range used for training.
        is_ood: Whether the book is assigned to OOD testing.

    Returns:
        Nested task samples for Model A and Model B, plus test-point candidates.
    """
    # ------------------------------------------------------------------ #
    # Parse data sources
    # ------------------------------------------------------------------ #
    scenes_data = data['scenes']                   # Scene data loaded from scenes
    scenes = scenes_data.get('scenes', [])          # Scene list; each scene contains interactions, location, scenario, summary, and key_characters
    character_list = scenes_data.get('character_list', [])  # Official character list [{name: ..., ...}, ...]
    location_list = scenes_data.get('location_list', [])    # Official location list [{name: ..., ...}, ...]
    
    char_dynamic = data['char_dynamic']     # Character dynamics {name: {profile_history, scene_descriptions}}; scene_descriptions have mixed temporal semantics
    world_dyn = data['world_dynamic']       # World dynamics {global_card_history, location_cards}
    
    global_card_history = world_dyn.get('global_card_history', [])  # Global world-state update history
    location_card_data = world_dyn.get('location_cards', {})        # Per-location state data
    
    official_names = [e['name'] for e in character_list]       # All official character names
    official_locations = [e['name'] for e in location_list]    # All official location names
    
    # Training-sample collectors for each task
    model_a_scene_cast = []           # Model A Task 1: scene cast planning
    model_a_location_scenario = []    # Model A Task 2: location and scenario planning
    model_a_next_character = []       # Model A Task 3: next-character proposal
    model_a_world_update = []         # Model A Task 4: world-state update
    model_b_interaction_gen = []    # Model B Task 1: interaction generation
    model_b_character_update = []   # Model B Task 2: character-state update
    model_b_motivation_update = []  # Model B Task 3: character-motivation update
    test_points = []                # Test candidates from non-training scenes
    
    train_start, train_end = train_scene_range  # Unpack the training scene range
    
    # ------------------------------------------------------------------ #
    # Pre-count positive world-state updates to compute the negative sampling rate.
    # Goal: make negatives roughly match positives (1:1), instead of using a fixed 5%.
    # ------------------------------------------------------------------ #
    n_wu_positive = 0  # Interactions with updates (positives)
    n_wu_total = 0     # Total interactions in the training range
    for i_s_pre, scene_pre in enumerate(scenes):
        if scene_pre is None:
            continue
        if not (train_start <= i_s_pre < train_end) or is_ood:
            continue
        interactions_pre = scene_pre.get('interactions', [])
        location_pre = scene_pre.get('location', {})
        location_name_pre = location_pre.get('name', '') if isinstance(location_pre, dict) else ''
        loc_card_hist_pre = location_card_data.get(location_name_pre, {}).get('card_history', [])
        # Build global/location update index sets for O(1) lookup.
        global_update_set = {(e['scene_index'], e['interaction_index']) for e in global_card_history}
        loc_update_set = {(e['scene_index'], e['interaction_index']) for e in loc_card_hist_pre}
        for i_inter_pre, inter_pre in enumerate(interactions_pre):
            if not normalize_content(inter_pre.get('content', '')):
                continue
            n_wu_total += 1
            if (i_s_pre, i_inter_pre) in global_update_set or (i_s_pre, i_inter_pre) in loc_update_set:
                n_wu_positive += 1
    # Negative sampling rate chosen so negatives roughly match positives.
    n_wu_negative = n_wu_total - n_wu_positive
    if n_wu_negative > 0 and n_wu_positive > 0:
        wu_neg_sample_rate = min(1.0, n_wu_positive / n_wu_negative)
    else:
        wu_neg_sample_rate = 0.05  # Fallback when there are no positives or negatives.

    # ------------------------------------------------------------------ #
    # Iterate over scenes and generate training samples.
    # ------------------------------------------------------------------ #
    for i_s, scene in enumerate(scenes):
        if scene is None:
            continue
        
        # Check whether the current scene is in the training range.
        is_train = (train_start <= i_s < train_end) and not is_ood
        
        interactions = scene.get('interactions', [])  # All interactions in the current scene
        if not interactions:
            continue
        
        # Extract basic scene information
        location = scene.get('location', {})                         # Scene location
        location_name = location.get('name', '') if isinstance(location, dict) else ''  # Location name
        scenario = scene.get('scenario', '')     # Scene scenario description
        summary = scene.get('summary', '')       # Scene summary
        scene_chars = [c['name'] for c in scene.get('key_characters', [])]  # Key character names in the scene
        
        # ============================================================== #
        # Input: current global world state plus short descriptions of all characters
        # Output: whether a next scene exists plus its participating characters
        #
        # Input: current global state, all location summaries, and selected character summaries
        # Output: next scene location plus scenario
        # ============================================================== #
        if is_train:
            # Build short descriptions for all locations at this time.
            all_location_descs = {}
            for loc_name in official_locations:
                desc = get_location_description_at(location_card_data, loc_name, i_s)
                if desc:
                    all_location_descs[loc_name] = desc
            
            # Build short descriptions for all characters at this time.
            all_char_descs = {}
            for char_name in official_names:
                dyn = char_dynamic.get(char_name, {})
                desc = get_character_description_before(dyn, i_s)  # Latest short description before this scene
                if desc:
                    all_char_descs[char_name] = desc
            
            # Get the global world state at the start of this scene.
            global_card = get_global_card_at(global_card_history, i_s)

            # Get the immediately previous scene scenario and interactions for continuity.
            prev_scene_scenario = '(None)'
            prev_scene_interactions = '(None)'
            if i_s > 0:
                prev_scene = scenes[i_s - 1] if i_s - 1 < len(scenes) else None
                if prev_scene is not None:
                    prev_scene_scenario = prev_scene.get('scenario', '') or '(None)'
                    prev_scene_interactions = build_interaction_history(
                        prev_scene.get('interactions', []), len(prev_scene.get('interactions', []))
                    ) or '(None)'

            # Task 1: select participating characters from all characters.
            scene_cast_input = (
                f"## Global World State\n{global_card}\n\n"
                f"## All Characters (Short Description Only)\n{json.dumps(all_char_descs, ensure_ascii=False, indent=2)}\n\n"
                f"## Previous Scene Scenario\n{prev_scene_scenario}\n\n"
                f"## Previous Scene Interactions\n{prev_scene_interactions}"
            )
            scene_cast_output = json.dumps({
                'has_next_scene': True,
                'involved_characters': scene_chars,
            }, ensure_ascii=False, indent=2)
            model_a_scene_cast.append({
                'conversations': [
                    {'from': 'system', 'value': get_scene_cast_system_prompt()},
                    {'from': 'human', 'value': scene_cast_input},
                    {'from': 'assistant', 'value': scene_cast_output},
                ],
                'details': {
                    'book_name': book_name,
                    'scene_index': i_s,
                    'task': 'scene_cast',
                },
            })

            # Task 2: choose a location and scenario based on the selected characters.
            selected_char_descs = {
                char_name: all_char_descs.get(char_name, '')
                for char_name in scene_chars
            }
            location_scenario_input = (
                f"## Global World State\n{global_card}\n\n"
                f"## All Location Descriptions\n{json.dumps(all_location_descs, ensure_ascii=False, indent=2)}\n\n"
                f"## Selected Characters (Short Description Only)\n{json.dumps(selected_char_descs, ensure_ascii=False, indent=2)}\n\n"
                f"## Previous Scene Scenario\n{prev_scene_scenario}\n\n"
                f"## Previous Scene Interactions\n{prev_scene_interactions}"
            )
            location_scenario_output = json.dumps({
                'location': location_name if location_name else None,
                'scenario': scenario if scenario else None,
            }, ensure_ascii=False, indent=2)
            model_a_location_scenario.append({
                'conversations': [
                    {'from': 'system', 'value': get_scene_location_scenario_system_prompt()},
                    {'from': 'human', 'value': location_scenario_input},
                    {'from': 'assistant', 'value': location_scenario_output},
                ],
                'details': {
                    'book_name': book_name,
                    'scene_index': i_s,
                    'task': 'location_scenario',
                },
            })

            # After scene cast, location, and scenario are fixed, generate motivation for each participating character.
            location_desc = all_location_descs.get(location_name, '')
            previous_scene = scenes[i_s - 1] if i_s > 0 else None
            previous_scene_scenario = (previous_scene or {}).get('scenario', '') or '(None)'
            for char_name in scene_chars:
                dyn = char_dynamic.get(char_name, {})
                profile_before = get_character_profile_at(dyn, i_s)
                hidden_tracker_before = get_character_hidden_tracker_before(dyn, i_s)
                short_description_before = get_character_description_before(dyn, i_s)
                motivation_before = get_character_motivation_before(dyn, i_s)
                previous_scene_interactions = (
                    build_interaction_history(
                        previous_scene.get('interactions', []),
                        len(previous_scene.get('interactions', [])),
                        perspective_character=char_name,
                    )
                    if previous_scene and previous_scene.get('interactions')
                    else '(None)'
                )
                other_scene_chars = {
                    other_name: selected_char_descs.get(other_name, '')
                    for other_name in scene_chars
                    if other_name != char_name
                }

                user_input_mu = (
                    f"## Current Profile\n{profile_before}\n\n"
                    f"## Hidden Tracker\n{hidden_tracker_before if hidden_tracker_before else '(Empty)'}\n\n"
                    f"## Current Short Description\n{short_description_before if short_description_before else '(None)'}\n\n"
                    f"## Global World State\n{global_card}\n\n"
                    f"## Previous Scene Scenario\n{previous_scene_scenario}\n\n"
                    f"## Previous Scene Interactions\n{previous_scene_interactions}\n\n"
                    f"## Next Scene Location\n{location_name}\n\n"
                    f"## Next Scene Location Description\n{location_desc if location_desc else '(None)'}\n\n"
                    f"## Next Scene Scenario\n{scenario}\n\n"
                    f"## Other Characters In Next Scene (Short Description Only)\n{json.dumps(other_scene_chars, ensure_ascii=False, indent=2)}"
                )

                output_mu = {
                    'motivation': motivation_before if motivation_before else None,
                }

                model_b_motivation_update.append({
                    'conversations': [
                        {'from': 'system', 'value': get_character_motivation_update_system_prompt(char_name)},
                        {'from': 'human', 'value': user_input_mu},
                        {'from': 'assistant', 'value': json.dumps(output_mu, ensure_ascii=False, indent=2)},
                    ],
                    'details': {
                        'book_name': book_name,
                        'scene_index': i_s,
                        'character': char_name,
                        'task': 'character_motivation_update',
                    },
                })
        
        # ============================================================== #
        # Build multi-turn context from character states at scene start.
        # - Split by scene first.
        # - Then split into segments at global/location world-state update boundaries.
        # - Keep consecutive multi-turn human/assistant supervision within each segment.
        # ============================================================== #
        if is_train:
            char_descs_in_scene, actor_states = get_scene_character_contexts(
                scene_chars, char_dynamic, i_s
            )

            model_a_next_character.extend(
                build_multi_turn_next_character_sample(
                    book_name=book_name,
                    scene_index=i_s,
                    location_name=location_name,
                    scenario=scenario,
                    scene_chars=scene_chars,
                    char_descs_in_scene=char_descs_in_scene,
                    interactions=interactions,
                    global_card_history=global_card_history,
                    location_card_data=location_card_data,
                )
            )

            model_b_interaction_gen.extend(
                build_multi_turn_interaction_samples(
                    book_name=book_name,
                    scene_index=i_s,
                    location_name=location_name,
                    scenario=scenario,
                    scene_chars=scene_chars,
                    char_descs_in_scene=char_descs_in_scene,
                    actor_states=actor_states,
                    interactions=interactions,
                    global_card_history=global_card_history,
                    location_card_data=location_card_data,
                )
            )

            for i_inter, interaction in enumerate(interactions):
                inter_content = normalize_content(interaction.get('content', ''))
                if not inter_content:
                    continue

                # ========================================================== #
                # Input: global state, location state, scenario, and history including the current interaction.
                # Output: whether to update global/location state plus updated content.
                # Note: no-update samples are dynamically sampled to roughly match positives and avoid imbalance.
                # ========================================================== #
                global_updated = False
                location_updated = False
                new_global_card = None
                new_location_card = None

                for entry in global_card_history:
                    if entry['scene_index'] == i_s and entry['interaction_index'] == i_inter:
                        global_updated = True
                        new_global_card = entry['global_card']
                        break

                loc_card_hist = location_card_data.get(location_name, {}).get('card_history', [])
                for entry in loc_card_hist:
                    if entry['scene_index'] == i_s and entry['interaction_index'] == i_inter:
                        location_updated = True
                        new_location_card = entry.get('location_card', {})
                        break

                global_card_before = get_global_card_at(global_card_history, i_s, i_inter - 1)
                loc_card_before = get_location_card_at(location_card_data, location_name, i_s, i_inter - 1)

                if global_updated and new_global_card == global_card_before:
                    global_updated = False
                    new_global_card = None

                if location_updated and new_location_card == loc_card_before:
                    location_updated = False
                    new_location_card = None

                should_add = global_updated or location_updated or (random.random() < wu_neg_sample_rate)

                if should_add:
                    prior_history = build_interaction_history(interactions, i_inter)
                    latest_interaction = build_interaction_history(interactions[i_inter:i_inter+1], 1)
                    loc_card_str = json.dumps(loc_card_before, ensure_ascii=False, indent=2) if isinstance(loc_card_before, dict) else str(loc_card_before)

                    user_input_wu = (
                        f"## Scene Scenario\n{scenario}\n\n"
                        f"## Prior Interactions\n{prior_history if prior_history else '(None)'}\n\n"
                        f"## Global World State\n{global_card_before}\n\n"
                        f"## Current Location: \"{location_name}\"\n## Location State\n{loc_card_str}\n\n"
                        f"## Latest Interaction\n{latest_interaction}"
                    )

                    output_wu = {
                        'update_global': global_updated,
                        'global_state': new_global_card if global_updated else None,
                        'update_location': location_updated,
                        'location_state': new_location_card if location_updated else None,
                    }

                    model_a_world_update.append({
                        'conversations': [
                            {'from': 'system', 'value': get_world_state_update_system_prompt()},
                            {'from': 'human', 'value': user_input_wu},
                            {'from': 'assistant', 'value': json.dumps(output_wu, ensure_ascii=False, indent=2)},
                        ],
                        'details': {
                            'book_name': book_name,
                            'scene_index': i_s,
                            'interaction_index': i_inter,
                            'task': 'world_state_update',
                            'has_update': global_updated or location_updated,
                        },
                    })
        
        # ============================================================== #
        # After each scene, review each participating character and update internal state.
        # Input: current profile, hidden tracker, motivation, scenario, and all scene interactions.
        # Output: new hidden tracker, updated profile if any, and new short description.
        # ============================================================== #
        if is_train:
            # Iterate over each character in the scene.
            for char_name in scene_chars:
                dyn = char_dynamic.get(char_name, {})
                scene_info = get_character_scene_info(dyn, i_s)
                if scene_info is None:
                    continue  # Skip characters without a dynamic record for this scene.
                
                # Get the character profile active at this scene.
                profile_before = get_character_profile_at(dyn, i_s)
                
                # Get the hidden tracker accumulated before this scene.
                hidden_tracker_before = get_character_hidden_tracker_before(dyn, i_s)
                
                # Character motivation active at the start of this scene.
                motivation_before = scene_info.get('enhanced_motivation', '')
                
                # Build full interaction history from this character perspective with other thoughts masked.
                full_history = build_interaction_history(
                    interactions, len(interactions), perspective_character=char_name
                )
                
                # Build the system prompt.
                system_prompt_cu = get_character_state_update_system_prompt(char_name)
                
                # Build user input with scene context and scene-start character state for end-of-scene updates.
                user_input_cu = (
                    f"## Scene Scenario\n{scenario}\n\n"
                    f"## Scene Interactions\n{full_history}\n\n"
                    f"## Current Profile (State At Scene Start)\n{profile_before}\n\n"
                    f"## Hidden Tracker (State At Scene Start)\n{hidden_tracker_before if hidden_tracker_before else '(Empty)'}\n\n"
                    f"## Current Motivation (State At Scene Start)\n{motivation_before if motivation_before else '(None)'}"
                )
                
                # Extract ground-truth output from dynamic data.
                new_hidden_tracker = scene_info.get('hidden_tracker', '')  # Hidden tracker updated after the current scene
                new_description = scene_info.get('description', '')        # New short description
                
                # Check whether the profile was updated in this scene.
                # Skip history[0], which is initialization rather than an update.
                new_profile = None
                for ph_idx, ph_entry in enumerate(dyn.get('profile_history', [])):
                    if ph_idx == 0:
                        continue  # Skip the initialization entry.
                    if ph_entry['scene_index'] == i_s:
                        new_profile = ph_entry['profile']
                        break
                
                # Assemble output JSON.
                output_cu = {
                    'hidden_tracker': new_hidden_tracker if new_hidden_tracker else None,  # Hidden tracker updated after the scene
                    'profile_updated': new_profile is not None,     # Boolean flag for whether the profile was updated
                    'updated_profile': new_profile,                  # Updated profile, or None if unchanged
                    'short_description': new_description,            # Updated short description
                }
                
                model_b_character_update.append({
                    'conversations': [
                        {'from': 'system', 'value': system_prompt_cu},
                        {'from': 'human', 'value': user_input_cu},
                        {'from': 'assistant', 'value': json.dumps(output_cu, ensure_ascii=False, indent=2)},
                    ],
                    'details': {
                        'book_name': book_name,
                        'scene_index': i_s,
                        'character': char_name,
                        'task': 'character_state_update',
                    },
                })
        
        # ============================================================== #
        # Test candidates: scenes outside the training range with more than five interactions.
        # These scenes are candidates for simulation start points during testing.
        # ============================================================== #
        else:
            if len(interactions) >= 5:
                test_points.append({
                    'book_name': book_name,
                    'scene_index': i_s,
                    'location': location_name,
                    'scenario': scenario,
                    'summary': summary,
                    'key_characters': scene_chars,
                    'n_interactions': len(interactions),
                })
    
    # ============================================================== #
    # After the last valid training scene, add a no-next-scene character-selection sample.
    # ============================================================== #
    # Generate has_next_scene=False only when the training range reaches the end of a train-only book.
    # Train+test books still have test scenes after the first 70%, so that boundary is not the story end.
    if not is_ood and train_end >= len(scenes):
        # Find the last valid scene index in the training range.
        last_train_scene_index = None
        for i_s in range(train_end - 1, train_start - 1, -1):
            if i_s < len(scenes) and scenes[i_s] is not None:
                s = scenes[i_s]
                if s.get('interactions'):
                    last_train_scene_index = i_s
                    break

        if last_train_scene_index is not None:
            i_s = last_train_scene_index
            all_location_descs_end = {}
            for loc_name in official_locations:
                desc = get_location_description_at(location_card_data, loc_name, i_s + 1)
                if desc:
                    all_location_descs_end[loc_name] = desc

            all_char_descs_end = {}
            for char_name in official_names:
                dyn = char_dynamic.get(char_name, {})
                desc = get_character_description_before(dyn, i_s + 1)
                if desc:
                    all_char_descs_end[char_name] = desc

            global_card_end = get_global_card_at(global_card_history, i_s + 1)

            # Use the last scene scenario and interactions as previous-scene context.
            last_scene = scenes[i_s]
            prev_scenario_end = last_scene.get('scenario', '') or '(None)'
            last_interactions = last_scene.get('interactions', [])
            prev_interactions_end = build_interaction_history(
                last_interactions, len(last_interactions)
            ) if last_interactions else '(None)'

            user_input_end = (
                f"## Global World State\n{global_card_end}\n\n"
                f"## All Characters (Short Description Only)\n{json.dumps(all_char_descs_end, ensure_ascii=False, indent=2)}\n\n"
                f"## Previous Scene Scenario\n{prev_scenario_end}\n\n"
                f"## Previous Scene Interactions\n{prev_interactions_end}"
            )

            assistant_output_end = json.dumps({
                'has_next_scene': False,
            }, ensure_ascii=False, indent=2)

            model_a_scene_cast.append({
                'conversations': [
                    {'from': 'system', 'value': get_scene_cast_system_prompt()},
                    {'from': 'human', 'value': user_input_end},
                    {'from': 'assistant', 'value': assistant_output_end},
                ],
                'details': {
                    'book_name': book_name,
                    'scene_index': i_s + 1,  # Indicates the point after the last scene
                    'task': 'scene_cast',
                    'is_story_end': True,
                },
            })

    # Return training samples for seven tasks plus test candidates.
    return {
        'model_a': {
            'scene_cast': model_a_scene_cast,                    # Task 1 samples
            'location_scenario': model_a_location_scenario,      # Task 2 samples
            'next_character': model_a_next_character,            # Task 3 samples, including scene end
            'world_update': model_a_world_update,                # Task 4 samples
        },
        'model_b': {
            'interaction_gen': model_b_interaction_gen,
            'character_update': model_b_character_update,   # Task 2 samples
            'motivation_update': model_b_motivation_update,
        },
        'test_points': test_points,    # Test candidate scene list
    }


# ============================================================================ #
# ============================================================================ #
def build_test_snapshot(book_name, data, scene_index):
    """Build a simulation-start snapshot for testing.

    The snapshot contains world state, character state, previous-scene context,
    and up to five ground-truth scenes for evaluation reference.

    Args:
        book_name: Book name.
        data: Loaded book data from load_book_data().
        scene_index: Scene index where simulation starts.

    Returns:
        Snapshot dictionary, or None if the scene is missing.
    """
    scenes_data = data['scenes']
    scenes = scenes_data.get('scenes', [])
    character_list = scenes_data.get('character_list', [])
    location_list = scenes_data.get('location_list', [])
    
    char_dynamic = data['char_dynamic']
    world_dyn = data['world_dynamic']
    
    global_card_history = world_dyn.get('global_card_history', [])
    location_card_data = world_dyn.get('location_cards', {})
    
    official_names = [e['name'] for e in character_list]
    official_locations = [e['name'] for e in location_list]
    
    for char_name in char_dynamic:
        if char_name not in official_names:
            official_names.append(char_name)
    
    scene = scenes[scene_index]
    if scene is None:
        return None
    
    global_card = get_global_card_at(global_card_history, scene_index)
    
    location_cards_snapshot = {}
    location_descs_snapshot = {}
    for loc_name in official_locations:
        location_cards_snapshot[loc_name] = get_location_card_at(
            location_card_data, loc_name, scene_index
        )
        location_descs_snapshot[loc_name] = get_location_description_at(
            location_card_data, loc_name, scene_index
        )
    
    character_snapshots = {}
    for char_name in official_names:
        dyn = char_dynamic.get(char_name, {})
        profile = get_character_profile_at(dyn, scene_index)
        description = get_character_description_before(dyn, scene_index)
        motivation = get_character_motivation_before(dyn, scene_index)
        hidden_tracker = get_character_hidden_tracker_before(dyn, scene_index)
        
        character_snapshots[char_name] = {
            'profile': profile,
            'short_description': description,
            'motivation': motivation,
            'hidden_tracker': hidden_tracker,
        }
    
    previous_scene = None
    if scene_index > 0:
        prev_s = scenes[scene_index - 1] if scene_index - 1 < len(scenes) else None
        if prev_s is not None:
            previous_scene = {
                'scenario': prev_s.get('scenario', ''),
                'interactions': prev_s.get('interactions', []),
            }

    ground_truth_scenes = []
    for gt_s in range(scene_index, min(scene_index + 5, len(scenes))):
        if scenes[gt_s] is None:
            continue
        gt_scene = copy.deepcopy(scenes[gt_s])
        ground_truth_scenes.append(gt_scene)
    
    return {
        'book_name': book_name,
        'scene_index': scene_index,
        'world_state': {
            'global_state': global_card,
            'location_states': location_cards_snapshot,
            'location_descriptions': location_descs_snapshot,
        },
        'character_states': character_snapshots,
        'previous_scene': previous_scene,
        'ground_truth_scenes': ground_truth_scenes,
    }


# ============================================================================ #
# ============================================================================ #
def to_sharegpt_format(samples):
    """Convert internal samples to ShareGPT SFT conversation format.

    Args:
        samples: Internal task samples.

    Returns:
        ShareGPT-style conversations without task details.
    """
    results = []
    role_map = {
        'system': 'system',
        'human': 'human',
        'user': 'human',
        'assistant': 'assistant',
    }
    for sample in samples:
        conversation = []
        for msg in sample['conversations']:
            conversation.append({
                'from': role_map.get(msg['from'], msg['from']),
                'value': msg['value'],
            })
        results.append({'conversations': conversation})
    return results


# ============================================================================ #
# ============================================================================ #
if __name__ == '__main__':
    
    required_dirs = {
        'scenes': f'{DATA_DIR}/scenes',
        'character_dynamic': f'{DATA_DIR}/character_dynamic',
        'world_dynamic': f'{DATA_DIR}/world_dynamic',
    }
    
    for dir_name, dir_path in required_dirs.items():
        if not os.path.exists(dir_path):
            print(f"Error: {dir_path} does not exist.")
            exit(1)
    
    candidate_files = [f for f in os.listdir(required_dirs['scenes']) if f.endswith('.json')]
    
    valid_files = []
    skipped_books = []
    for f in candidate_files:
        book_name = os.path.splitext(f)[0]
        all_exist = all(
            os.path.exists(os.path.join(dir_path, f'{book_name}.json'))
            for dir_path in required_dirs.values()
        )
        if all_exist:
            valid_files.append(f)
        else:
            skipped_books.append(book_name)
    
    if skipped_books:
        print(f"Warning: {len(skipped_books)} books skipped (missing files in some directories):")
        for sb in skipped_books[:10]:
            missing = [name for name, path in required_dirs.items()
                       if not os.path.exists(os.path.join(path, f'{sb}.json'))]
            print(f"  {sb}: missing {missing}")
        if len(skipped_books) > 10:
            print(f"  ... and {len(skipped_books) - 10} more")
    
    files = seeded_shuffle(valid_files, SEED)
    
    book_names = [os.path.splitext(f)[0] for f in files]
    
    print(f"Total valid books found: {len(book_names)} (out of {len(candidate_files)} candidates)")
    print(f"Seed: {SEED}")
    print(f"First 10: {book_names[:10]}")
    
    n_books = len(book_names)
    
    n_ood = max(1, math.ceil(n_books * 0.1))
    ood_books = book_names[:n_ood]
    non_ood_books = book_names[n_ood:]
    
    n_remaining = len(non_ood_books)
    n_train_only = n_remaining // 2
    n_train_test = n_remaining - n_train_only
    
    train_test_books = non_ood_books[:n_train_test]
    train_only_books = non_ood_books[n_train_test:]
    
    print(f"\nBook split:")
    print(f"  OOD test books: {len(ood_books)}")
    print(f"  Train+Test books (70/30): {len(train_test_books)}")
    print(f"  Train-only books (100%): {len(train_only_books)}")
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    all_model_a = {
        'scene_cast': [],
        'location_scenario': [],
        'next_character': [],
        'world_update': [],
    }
    all_model_b = {'interaction_gen': [], 'character_update': [], 'motivation_update': []}
    
    test_snapshots_id = []
    test_snapshots_ood = []
    
    print(f"\n--- Processing OOD test books ---")
    for book_name in ood_books:
        print(f"  Processing: {book_name}")
        data = load_book_data(book_name)
        if data is None:
            continue
        
        scenes = data['scenes'].get('scenes', [])
        n_scenes = len(scenes)
        print(f"    {n_scenes} scenes found")
        
        result = process_book(book_name, data, train_scene_range=(0, 0), is_ood=True)
        
        if n_scenes > 4:
            print(f"    {len(result['test_points'])} valid scenes found")
            tail_start = max(1, int(n_scenes * 0.3))
            candidates = [
                tp['scene_index'] for tp in result['test_points']
                if tail_start <= tp['scene_index'] < n_scenes
            ]
            random.shuffle(candidates)
            n_samples = min(20, len(candidates))
            print(f"    Sampled {n_samples} test snapshots from {book_name}")
            
            for si in candidates[:n_samples]:
                snapshot = build_test_snapshot(book_name, data, si)
                if snapshot:
                    snapshot['tag'] = 'ood'
                    test_snapshots_ood.append(snapshot)
    
    print(f"\n--- Processing train+test books ---")
    for book_name in train_test_books:
        print(f"  Processing: {book_name}")
        data = load_book_data(book_name)
        if data is None:
            continue
        
        scenes = data['scenes'].get('scenes', [])
        n_scenes = len(scenes)
        print(f"    {n_scenes} scenes found")
        
        split_index = int(n_scenes * 0.7)
        
        result = process_book(book_name, data, train_scene_range=(0, split_index))
        
        for task_key in all_model_a:
            all_model_a[task_key].extend(result['model_a'][task_key])
        for task_key in all_model_b:
            all_model_b[task_key].extend(result['model_b'][task_key])
        
        test_scene_indices = [tp['scene_index'] for tp in result['test_points']]
        if test_scene_indices:
            random.shuffle(test_scene_indices)
            for si in test_scene_indices[:5]:
                snapshot = build_test_snapshot(book_name, data, si)
                if snapshot:
                    snapshot['tag'] = 'id'
                    test_snapshots_id.append(snapshot)
    
    print(f"\n--- Processing train-only books ---")
    for book_name in train_only_books:
        print(f"  Processing: {book_name}")
        data = load_book_data(book_name)
        if data is None:
            continue
        
        scenes = data['scenes'].get('scenes', [])
        n_scenes = len(scenes)
        
        result = process_book(book_name, data, train_scene_range=(0, n_scenes))
        
        for task_key in all_model_a:
            all_model_a[task_key].extend(result['model_a'][task_key])
        for task_key in all_model_b:
            all_model_b[task_key].extend(result['model_b'][task_key])
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print(f"\n=== Training Data Statistics ===")
    print(f"Model A - Scene Cast: {len(all_model_a['scene_cast'])}")
    print(f"Model A - Location Scenario: {len(all_model_a['location_scenario'])}")
    print(f"Model A - Next Character: {len(all_model_a['next_character'])}")
    print(f"Model A - World Update: {len(all_model_a['world_update'])}")
    print(f"Model B - Interaction Gen: {len(all_model_b['interaction_gen'])}")
    print(f"Model B - Character Update: {len(all_model_b['character_update'])}")
    print(f"Model B - Motivation Update: {len(all_model_b['motivation_update'])}")
    print(f"\n=== Test Data Statistics ===")
    print(f"ID test snapshots: {len(test_snapshots_id)}")
    print(f"OOD test snapshots: {len(test_snapshots_ood)}")
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    print(f"\n=== Validating ShareGPT format ===")
    validation_errors = []
    all_task_samples = {
        'model_a/scene_cast': all_model_a['scene_cast'],
        'model_a/location_scenario': all_model_a['location_scenario'],
        'model_a/next_character': all_model_a['next_character'],
        'model_a/world_update': all_model_a['world_update'],
        'model_b/interaction_gen': all_model_b['interaction_gen'],
        'model_b/character_update': all_model_b['character_update'],
        'model_b/motivation_update': all_model_b['motivation_update'],
    }
    for task_name, samples in all_task_samples.items():
        task_errors = 0
        for s_idx, sample in enumerate(samples):
            conv = sample['conversations']
            details = sample.get('details', {})
            errors = []
            
            if len(conv) < 3:
                errors.append(f"too few turns ({len(conv)})")
            else:
                # Rule 1: first turn must be system
                if conv[0]['from'] != 'system':
                    errors.append(f"first turn is '{conv[0]['from']}', expected 'system'")
                
                # Rule 2: second turn must be human
                if len(conv) > 1 and conv[1]['from'] != 'human':
                    errors.append(f"second turn is '{conv[1]['from']}', expected 'human'")
                
                # Rule 3: from index 1 onward, human and assistant must strictly alternate
                for t_idx in range(2, len(conv)):
                    prev_role = conv[t_idx - 1]['from']
                    curr_role = conv[t_idx]['from']
                    if prev_role == 'system':
                        expected = 'human'
                    elif prev_role == 'human':
                        expected = 'assistant'
                    else:
                        expected = 'human'
                    if curr_role != expected:
                        errors.append(f"turn {t_idx}: '{curr_role}' after '{prev_role}' (expected '{expected}')")
                        break  # Only report the first alternation error per sample
                
                # Rule 4: last turn must be assistant
                if conv[-1]['from'] != 'assistant':
                    errors.append(f"last turn is '{conv[-1]['from']}', expected 'assistant'")
            
            if errors:
                task_errors += 1
                if len(validation_errors) < 50:  # Cap detailed error output
                    validation_errors.append(
                        f"  [{task_name}] sample #{s_idx} "
                        f"(book={details.get('book_name','?')}, scene={details.get('scene_index','?')}): "
                        f"{'; '.join(errors)}"
                    )
        
        status = "✓ PASS" if task_errors == 0 else f"✗ FAIL ({task_errors} errors)"
        print(f"  {task_name}: {len(samples)} samples — {status}")
    
    if validation_errors:
        print(f"\nFirst {len(validation_errors)} error details:")
        for err in validation_errors:
            print(err)
        print(f"\n⚠️  ShareGPT format validation FAILED. Please fix the issues above before training.")
    else:
        print(f"\n✅ All {sum(len(s) for s in all_task_samples.values())} samples passed ShareGPT format validation.")
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    final_dir = Path('dataset')
    train_dir = str(final_dir / 'train')
    test_dir = str(final_dir / 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for task_key in all_model_a:
        random.shuffle(all_model_a[task_key])
    for task_key in all_model_b:
        random.shuffle(all_model_b[task_key])
    
    for task_key, samples in all_model_a.items():
        path = f'{train_dir}/model_a_{task_key}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(to_sharegpt_format(samples), f, ensure_ascii=False, indent=2)
        print(f"Saved: {path} ({len(samples)} samples)")
    
    for task_key, samples in all_model_b.items():
        path = f'{train_dir}/model_b_{task_key}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(to_sharegpt_format(samples), f, ensure_ascii=False, indent=2)
        print(f"Saved: {path} ({len(samples)} samples)")
    
    all_train_with_details = []
    for task_key, samples in all_model_a.items():
        all_train_with_details.extend(samples)
    for task_key, samples in all_model_b.items():
        all_train_with_details.extend(samples)
    random.shuffle(all_train_with_details)
    
    with open(f'{train_dir}/all_tasks_with_details.json', 'w', encoding='utf-8') as f:
        json.dump(all_train_with_details, f, ensure_ascii=False, indent=2)
    print(f"Saved: {train_dir}/all_tasks_with_details.json ({len(all_train_with_details)} total samples)")
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    with open(f'{test_dir}/test_snapshots_id.json', 'w', encoding='utf-8') as f:
        json.dump(test_snapshots_id, f, ensure_ascii=False, indent=2)
    print(f"Saved: {test_dir}/test_snapshots_id.json ({len(test_snapshots_id)} snapshots)")
    
    with open(f'{test_dir}/test_snapshots_ood.json', 'w', encoding='utf-8') as f:
        json.dump(test_snapshots_ood, f, ensure_ascii=False, indent=2)
    print(f"Saved: {test_dir}/test_snapshots_ood.json ({len(test_snapshots_ood)} snapshots)")
    
    with open(f'{test_dir}/test_all.json', 'w', encoding='utf-8') as f:
        json.dump(test_snapshots_id + test_snapshots_ood, f, ensure_ascii=False, indent=2)
    print(f"Saved: {test_dir}/test_all.json ({len(test_snapshots_id) + len(test_snapshots_ood)} snapshots)")
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    split_info = {
        'ood_books': ood_books,
        'train_test_books': train_test_books,
        'train_only_books': train_only_books,
    }
    book_split_path = final_dir / 'book_split.json'
    with open(book_split_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"Saved: {book_split_path}")
    
    print(f"\n=== Done! ===")
