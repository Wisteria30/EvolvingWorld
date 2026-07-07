"""
Prompt template module (Per-Metric independent evaluation)

Each metric calls the LLM Judge once, evaluating only one dimension per call.
This keeps each input shorter and evaluation instructions more detailed.

Architecture:
- Each evaluation layer has a generic template (per-scene / cross-scene-char / cross-scene-loc / cross-scene-global)
- Each metric has independent dimension_brief and dimension_criteria
- build_*_user_prompt functions include only the data subset needed for that metric
- Additionally, a dedicated scene_summary prompt generates scene summaries for downstream layers
"""

import json
from typing import Dict, Any, List, Optional


def _format_hidden_tracker(hidden_tracker: Any) -> str:
    """Format hidden tracker as stable, readable text."""
    if hidden_tracker is None:
        return "(No hidden tracker)"
    if isinstance(hidden_tracker, str):
        return hidden_tracker
    if isinstance(hidden_tracker, (list, dict)):
        return json.dumps(hidden_tracker, ensure_ascii=False, indent=2)
    return str(hidden_tracker)

# ============================================================================ #
# General scoring rubric description
# ============================================================================ #

SCORING_INSTRUCTION = """
## Scoring Method

Use the following scoring method:
- Base score: 50
- First, identify **Merits** (excellent aspects): each merit awards +1 to +10 points
- Then, identify **Demerits** (problematic aspects): each demerit penalizes -1 to -10 points
- The final score will be automatically calculated as: min(100, max(0, 50 + sum(merits) - sum(demerits)))

Output:
1. A list of merits with their point values
2. A list of demerits with their point values
3. A brief reasoning explaining your evaluation

Do NOT output a "score" field — the score will be computed automatically from your merits and demerits.
"""

# ============================================================================ #
# General output format (single metric)
# ============================================================================ #

SINGLE_METRIC_OUTPUT_FORMAT = """
## Output Format

You MUST respond with a valid JSON object in the following format (no extra text outside the JSON):

```json
{
  "merits": [{"description": "<what was done well>", "points": <1-10>}, ...],
  "demerits": [{"description": "<what was problematic>", "points": <1-10>}, ...],
  "reasoning": "<brief explanation of your evaluation>"
}
```

Do NOT include a "score" field — the score will be computed automatically from your merits and demerits.
"""

PAIRED_METRIC_OUTPUT_FORMAT = """
## Output Format

You are evaluating TWO dimensions simultaneously on the same scene data.
You MUST respond with a valid JSON object containing results for BOTH dimensions:

```json
{{
  "{metric_a}": {{
    "merits": [{{"description": "<what was done well>", "points": <1-10>}}, ...],
    "demerits": [{{"description": "<what was problematic>", "points": <1-10>}}, ...],
    "reasoning": "<brief explanation of your evaluation>"
  }},
  "{metric_b}": {{
    "merits": [{{"description": "<what was done well>", "points": <1-10>}}, ...],
    "demerits": [{{"description": "<what was problematic>", "points": <1-10>}}, ...],
    "reasoning": "<brief explanation of your evaluation>"
  }}
}}
```

Do NOT include a "score" field — the score will be computed automatically from your merits and demerits.
Evaluate each dimension independently — do NOT let the assessment of one dimension influence the other.
"""

# ============================================================================ #
# Paired Metrics config: metric pairs with identical input data, merged into one judge call
# ============================================================================ #

PAIRED_METRICS = [
    ("PF", "SSF"),    # both use char_profiles + char_hidden_trackers + interactions
    ("EA", "EU"),    # both use global_state_update + location_state_update + interactions
    ("GUS", "GSA"),   # both use interactions + global_state_update
    ("LUS", "LSA"),   # both use interactions + location_state_update
]

# ============================================================================ #
# Scene Summary dedicated prompt (no scoring, only summary generation)
# ============================================================================ #

SCENE_SUMMARY_SYSTEM_PROMPT = """You are an expert summarizer for interactive fiction simulations. Your task is to produce a concise summary of a single scene that captures the key events, character actions, emotional shifts, and narrative developments. This summary will be used as context for evaluating subsequent scenes and cross-scene coherence.

Do NOT retell every interaction turn-by-turn; focus on the overall arc and key turning points.

Aim for a concise paragraph — typically a few hundred words is sufficient."""

SCENE_SUMMARY_OUTPUT_FORMAT = """
## Output Format

You MUST respond with a valid JSON object:

```json
{
  "scene_summary": "<A concise summary capturing key events, character actions, emotional shifts, and narrative developments.>"
}
```
"""

# ============================================================================ #
# Per-Scene evaluation: Metric definitions
# ============================================================================ #

PER_SCENE_METRIC_DETAILS = {
    # ---- CHARACTER metrics ----
    "PF": {
        "dimension_name": "Profile Fidelity (PF)",
        "dimension_brief": "Whether the character's knowledge, skills, and behavior stay within profile boundaries throughout the scene.",
        "dimension_criteria": """### Profile Fidelity (PF)

Evaluate whether each character's behavior remains consistent with their established profile and hidden tracker. Specifically:

**1. Knowledge Boundaries**
- Does the character demonstrate knowledge or skills that are NOT documented in their profile?
- Does the character reference events, technologies, or concepts they should not know about given their background?
- Example flaw: A medieval peasant character discussing quantum physics without any profile basis.

**2. Background Consistency**
- Does the character's behavior match their age, social class, education level, era, and historical background?
- Watch for anachronistic behavior, tone mismatches with the character's class/era, or emotional maturity inconsistent with their age.
- Example flaw: A sheltered noble character showing street-smart survival skills not mentioned in their profile.

**3. Ability Constraints**
- Does the character perform actions requiring abilities (physical, intellectual, social) not documented in their profile?
- Characters should not suddenly display undocumented competencies (e.g., combat skill, medical knowledge, leadership).

**4. Hidden Tracker Alignment**
- Does the character's behavior align with their current psychological state, motivations, and internal conflicts as described in the hidden tracker?
- Contradictions between behavior and hidden tracker state (e.g., acting boldly when the tracker says fearful, confiding in someone they distrust) are profile violations.

**5. Profile Drift**
- Does the character gradually drift from their profile as the scene progresses — starting in-character but slowly becoming more generic, more "helpful", or more emotionally balanced than warranted?
- A pattern of small deviations accumulating over the scene should be penalized even if no single turn is a clear violation.

**Length Neutrality**: Do NOT favor longer or more detailed interactions. A character's response length and level of detail should match the speaking style established in their profile and the original book examples (if provided). A terse character who speaks in short, clipped sentences should not be penalized for brevity, nor should a verbose character be rewarded simply for producing more text. Evaluate fidelity to the character's authentic style, not output volume.""",
        "data_sections": ["char_profiles", "char_hidden_trackers", "interactions"],
    },
    "SSF": {
        "dimension_name": "Speaking Style Fidelity (SSF)",
        "dimension_brief": "Whether each character's speaking style matches their profile and feels natural, not AI-generated.",
        "dimension_criteria": """### Speaking Style Fidelity (SSF)

Evaluate whether each character speaks in a way that matches their profile and sounds natural. Specifically:

**Important Note**: In this simulation system, each character's interaction output (thoughts, actions, speech) is written in the FIRST PERSON from that character's perspective. This is the expected output convention — do NOT penalize the use of first-person pronouns ("I", "my", "myself") in character outputs. This metric evaluates STYLE, not output format — format compliance is evaluated separately under Instruction Compliance (IC).

**Reference Sources**: When evaluating a character's speaking style, you should jointly consider TWO sources:
1. **Character Profile** — the personality traits, background, and style descriptions defined in the profile.
2. **Original Speaking Style Examples** (if provided) — these are reference interaction excerpts taken directly from the original source book, showing how the character speaks in the canonical text. They provide useful evidence for the character's vocabulary, sentence structure, tone, verbal habits, and overall speaking manner. The character's simulated interactions should be consistent with these examples in style.

**1. Style Markers**
- Does the character use the language features defined in their profile AND demonstrated in the original book examples (catchphrases, terminology, dialects, speech patterns)?
- Is their vocabulary level appropriate for their background (e.g., a scholar uses formal language, a street urchin uses slang)?
- Do they maintain consistent verbal tics or habits throughout the scene?
- Example flaw: A character defined as speaking in short, gruff sentences suddenly delivering eloquent monologues.

**2. Emotional Tone**
- Does the character's tone match their personality type (e.g., a cynical character sounds cynical, not cheerful)?
- Is the emotional expression authentic to the character, not a generic "AI assistant" tone?
- Does the character avoid being unnaturally helpful, verbose, didactic, or moralistic unless that's their personality?
- Example flaw: A cold, reserved character suddenly becoming warm and effusive without narrative justification.

**3. Naturalness**
- Does the dialogue sound like something a real person (with this character's background) would say?
- Is the language free of AI artifacts — both obvious ("As an AI...", "I'd be happy to help...") and subtle (overly diplomatic phrasing, unnecessary hedging like "It's worth noting that...", formulaic emotions, unnaturally smooth turn-taking, restating before responding)? Even mildly AI-flavored language should be penalized.
- Are there natural speech imperfections (hesitations, interruptions, incomplete thoughts) where appropriate?
- Example flaw: A street-tough character saying "That's a really valid point, and I understand where you're coming from" — polished language no such character would use, despite lacking explicit AI markers.

**Length Neutrality**: Do NOT favor longer or more detailed interactions. A character's response length, verbosity, and level of detail should match the speaking style established in their profile and the original book examples (if provided). A terse character who speaks in short, clipped sentences is being faithful to their style — do not penalize brevity. Conversely, do not reward a character simply for producing longer, more elaborate text if that does not match their canonical style. Evaluate style fidelity, not output volume.

**Scope Exclusion**: Do NOT evaluate output format compliance (tag usage, structure, etc.) — that belongs to Instruction Compliance (IC). Do NOT evaluate whether the character's behavior matches their profile — that belongs to Profile Fidelity (PF) and Motivation-Driven Behavior (MDB).""",
        "data_sections": ["char_profiles", "char_hidden_trackers", "interactions"],
    },
    "MDB": {
        "dimension_name": "Motivation-Driven Behavior (MDB)",
        "dimension_brief": "Whether each character's core motivation consistently drives their decisions and actions throughout the scene.",
        "dimension_criteria": """### Motivation-Driven Behavior (MDB)

Evaluate whether characters' behaviors are driven by their established motivations. This metric focuses specifically on the MOTIVATION → BEHAVIOR link. Every action, decision, and reaction should be traceable to the character's documented motivations.

**1. Behavioral Attribution**
- Can each major decision or action be traced back to the character's core motivation or scene-specific motivation?
- Are there actions that seem random, unmotivated, or driven by plot convenience rather than character motivation?
- Does the character act in a way that reflects their goals or inner drive? If a character remains purely reactive despite having an established motivation that should influence the scene, treat it as a motivation issue. Passive or restrained behavior is acceptable when it fits the character's profile, situation, or motivation.
- Characters who act "helpfully" or "cooperatively" without motivational basis are exhibiting generic AI behavior, not motivation-driven behavior.

**2. Trinity Coherence (Thought → Action → Speech)**
- Are the character's inner thoughts, physical actions, and spoken words logically consistent with each other AND with their motivations?
- Do thoughts reveal motivations that explain the subsequent actions and speech?
- Is there appropriate tension when a character's public speech differs from private thoughts (e.g., deception, hidden agendas)?
- Incoherence between thought and action is a serious flaw — if a character thinks one thing but does the opposite without justification, this is a motivation failure.

**3. Motivation Persistence & Drift**
- Do core motivations remain active and visible throughout the scene, not just at the beginning?
- If a character's motivation is strong (e.g., revenge, survival, ambition), it should color their behavior consistently — not appear once and then be forgotten.
- Are motivation shifts (if any) caused by significant in-scene events, not arbitrary? Sudden unmotivated changes in goals or priorities are serious violations.

**Length Neutrality**: Do NOT favor longer or more detailed interactions. A character's response length and level of detail should match the speaking style established in their profile and the original book examples (if provided). Motivation-driven behavior can be expressed concisely — a character who acts decisively with few words is not inferior to one who deliberates at length. Evaluate whether motivations drive behavior, not whether the output is verbose or detailed.

**Scope Exclusion**: Do NOT evaluate general conversational continuity or context-following — that belongs to Contextual Responsiveness (CR). Do NOT evaluate whether the character's knowledge or abilities exceed their profile — that belongs to Profile Fidelity (PF). Focus strictly on whether MOTIVATIONS drive BEHAVIOR.""",
        "data_sections": ["char_profiles", "char_motivations", "char_hidden_trackers", "interactions"],
    },
    "PUF": {
        "dimension_name": "Profile Update Fidelity (PUF)",
        "dimension_brief": "Whether the post-scene profile update and hidden tracker update together accurately capture what should persist from the scene, at the right level of significance.",
        "dimension_criteria": """### Profile Update Fidelity (PUF)

Evaluate whether the post-scene profile update and hidden tracker update work together as a faithful persistence mechanism. **The primary focus of this metric is the quality of PROFILE UPDATES.** The hidden tracker is an auxiliary tool that supports profile updates — evaluate it with appropriate leniency.

Specifically:

**1. Causal Chain**
- Does each change in the updated profile have a clear triggering event in the scene's interactions?
- Does each hidden tracker entry also have a concrete basis in the scene? (The tracker entries must be truthful/factual, but minor redundancy is acceptable.)
- Can you trace every persisted item (whether in profile or tracker) back to a specific moment in the scene?
- Example flaw: A profile or hidden tracker entry introduces information that never appeared in the scene.

**2. Growth / Signal Capture**
- Are important, threshold-crossing developments captured in the updated profile? Including:
  - Significant emotional shifts or realizations
  - New or changed relationships
  - Key information the character learned
  - Changes in goals or priorities
- Are meaningful but still sub-threshold signals captured in the hidden tracker? Including:
  - Subtle emotional fluctuations that may accumulate later
  - Unresolved tensions or contradictions
  - Early signs of attitude or relationship change
- Example flaw: A major revelation happens but neither the profile nor the tracker records it in any form.

**3. Threshold Judgment**
- Are major, lasting changes written into the profile rather than left only in the hidden tracker?
- Are minor or still-ambiguous signals kept in the hidden tracker rather than over-promoted into the profile?
- Is the boundary between "lasting profile change" and "sub-threshold signal" judged appropriately?
- Only major events (betrayals, revelations, life-changing decisions) should trigger significant profile modifications.
- Example flaw: A worldview-changing event is recorded only in the tracker, or a momentary irritation is written into the profile as a stable trait.

**4. No Over-Updating / No Under-Updating**
- Does the profile stay concise and focused on lasting changes?
- Do the profile and tracker together avoid both omission and overreaction?
- Example flaw: Both profile and tracker fail to preserve an obviously important signal.

**Hidden Tracker Leniency**: The hidden tracker is a supporting mechanism for profile updates. As long as its entries are truthful (traceable to scene events), minor redundancy or verbosity in the tracker should receive only light penalties (1-2 points). Reserve heavier penalties for the tracker only when it contains fabricated information or completely misses critical signals.

Compare the profile BEFORE the scene with the post-scene reflections, and evaluate whether the profile update and hidden tracker update together correctly preserve what should carry forward from this scene.""",
        "data_sections": ["char_profiles", "interactions", "char_reflections"],
    },
    "EA": {
        "dimension_name": "Environment Awareness (EA)",
        "dimension_brief": "Whether characters are appropriately constrained by and responsive to the current environment (global world state + location world state).",
        "dimension_criteria": """### Environment Awareness (EA)

Evaluate whether characters demonstrate awareness of and respond appropriately to the environment. This metric assesses whether characters truly "live" in the current scene rather than conversing in a vacuum.

**Note**: In this simulation, there are two types of Character Agents:
- **Non-Environment Character Agents**: Regular characters who participate in dialogue and actions.
- **Environment Character Agent**: A special agent that generates environmental descriptions (narration about the setting, atmosphere, sensory details). This agent's output appears as environmental/narrative text rather than character dialogue.

Evaluate each type with the appropriate criteria below.

#### For Non-Environment Character Agents:

**1. Global Awareness**
- Do characters react appropriately to global conditions (e.g., showing tension during wartime, being mindful of conservation during resource scarcity)?
- Do their plans and decisions account for global constraints?
- Example flaw: Characters planning an outdoor festival while the global state indicates a severe storm.

**2. Location Awareness**
- Do characters notice and respond to their current location's features (e.g., a shopkeeper mentioning "last night's storm blew the roof off," a fisherman complaining "the river's been polluted")?
- Do they interact with the environment in ways consistent with the location description?
- Example flaw: A character searching for a book in a location that has no library or bookshelf.

**3. State Change Response**
- When the world state changes between scenes, do characters notice and adjust accordingly?
- Do they acknowledge environmental changes (e.g., if a fire breaks out, do they react)?
- Example flaw: Characters continuing a casual conversation while the location description indicates the building is collapsing.

#### For Environment Character Agent:

**1. Global State Consistency**
- Are environmental descriptions consistent with the current global state (e.g., describing distant beacon fires and fleeing crowds during wartime, describing deserted streets during a plague)?
- Example flaw: Describing a bustling marketplace when the global state indicates the city is under siege.

**2. Location State Accuracy**
- Do environmental descriptions accurately reflect the current state of the location card (e.g., damaged buildings should not be described as intact, streams should not be described as flowing during a drought)?
- Example flaw: Describing a pristine garden when the location state says it was destroyed by fire.

**3. State Change Presentation**
- When the world state changes between scenes, do environmental descriptions reflect these transitions (e.g., post-war scenes adding descriptions of ruins and scorched earth, seasonal changes in natural landscapes)?
- Example flaw: The environment description remains identical despite major world state changes.

**Length Neutrality**: Do NOT favor longer or more detailed interactions. Characters' environmental awareness can be expressed concisely — a brief but accurate reference to the surroundings is just as valid as a lengthy description. The length and detail level of each character's interaction should match their speaking style as established in their profile and the original book examples (if provided). Evaluate awareness quality, not output volume.""",
        "data_sections": ["global_state_update", "location_state_update", "interactions"],
    },
    "EU": {
        "dimension_name": "Environmental Utilization (EU)",
        "dimension_brief": "Whether characters appropriately and meaningfully use environmental elements when relevant to the scene.",
        "dimension_criteria": """### Environmental Utilization (EU)

Evaluate whether characters make good use of the environment. This metric assesses whether characters use environmental elements in ways that are relevant, grounded, and natural for the scene.

**Note**: In this simulation, there are two types of Character Agents:
- **Non-Environment Character Agents**: Regular characters who participate in dialogue and actions.
- **Environment Character Agent**: A special agent that generates environmental descriptions (narration about the setting, atmosphere, sensory details).

Evaluate each type with the appropriate criteria below.

#### For Non-Environment Character Agents:

**1. Environmental Sensory Details**
- Do characters convey their perception of the environment through sensory descriptions (e.g., smelling kitchen grease, hearing distant sirens, feeling the ground shake)?
- Are sensory details specific and character-appropriate, rather than generic visual/auditory descriptions?
- Example merit: A character noting the smell of old books in a library, the creak of floorboards, dust motes in sunlight.

**2. Prop Interaction**
- Do characters interact with items and entities present in the location to advance the plot (e.g., using a streetlight to examine a wound, using cover to hide)?
- Are these interactions natural and serve the narrative (not forced)?
- Example merit: A nervous character fidgeting with a quill on the desk, or using a map on the wall to explain their plan.

**3. Atmosphere Building**
- Is the environmental atmosphere used to enhance immersion, rather than conversing in a "blank room"?
- Do characters' perceptions of the environment shift with the emotional tone?
- Example merit: A tense negotiation scene where a character notices the flickering candlelight and howling wind outside.

#### For Environment Character Agent:

**1. Multi-Sensory Richness**
- Do environmental descriptions engage multiple sensory dimensions (visual light and shadow changes, auditory wind and rain sounds, olfactory earth scents, tactile biting cold)?
- Are descriptions limited to only visual descriptions, or do they create a rich sensory tapestry?
- Example merit: Describing not just what the room looks like, but the musty smell, the cold draft, and the distant sound of thunder.

**2. Scene Element Usage**
- Do environmental descriptions specifically utilize items and entities in the location (e.g., describing candlelight flickering on a table, flags being torn by wind outside the window)?
- Are descriptions grounded in the specific scene rather than generic and detached?
- Example flaw: Generic descriptions like "the room was dark" when the location has specific candles, furniture, and windows to reference.

**3. Atmosphere-Narrative Alignment**
- Does the atmosphere of environmental descriptions match the current narrative pace and emotional tone?
- Example merit: Describing oppressive silence and distant thunder during a tense standoff; describing soft twilight and cooking smoke in a warm scene.
- Example flaw: A cheerful, sunny environmental description during a funeral scene.

**Length Neutrality**: Do NOT favor longer or more detailed interactions. Environmental utilization quality is about relevance and naturalness, not quantity. A single well-chosen sensory detail or prop interaction that fits the character's style can be sufficient; multiple forced or generic descriptions should not be rewarded. The length and detail level of each character's interaction should match their speaking style as established in their profile and the original book examples (if provided). Evaluate utilization quality, not output volume.""",
        "data_sections": ["global_state_update", "location_state_update", "interactions"],
    },
    "CR": {
        "dimension_name": "Contextual Responsiveness (CR)",
        "dimension_brief": "Whether characters' responses closely follow the conversational and narrative context.",
        "dimension_criteria": """### Contextual Responsiveness (CR)

Evaluate whether characters respond appropriately to the immediate conversational and narrative context. This metric focuses on turn-by-turn responsiveness within the scene.

**1. Information Continuity**
- Do characters remember and reference information shared earlier in the conversation?
- Do they avoid ignoring key revelations, questions, or events?
- Do they follow up on important topics rather than letting them drop?
- Example flaw: Character A reveals a shocking secret, but Character B never acknowledges or reacts to it.

**2. Logical Continuity**
- Do characters react logically to others' actions and statements?
- Are cause-and-effect chains maintained (e.g., if someone is insulted, they show some reaction)?
- Are there non-sequiturs or responses that don't connect to what was just said?
- Example flaw: Character A asks "Where is the treasure?" and Character B responds with an unrelated philosophical musing.

**3. Relationship Matching**
- Does the tone and content of interactions match the established relationship between characters?
- Do power dynamics, familiarity levels, and emotional bonds influence how characters speak to each other?
- Do attitudes naturally adjust as the conversation evolves within the scene?
- Example flaw: A servant speaking to their king with casual familiarity when their relationship is formal and hierarchical.

**Scope Exclusion**: Do NOT evaluate output format compliance (tag usage, structure, etc.) — that belongs to Instruction Compliance (IC). Do NOT evaluate whether a character's overall personality is consistent with their profile — that belongs to Character Consistency (PF, SSF, MDB). Focus strictly on whether each response appropriately follows the CONTEXT of the conversation.""",
        "data_sections": ["char_profiles", "interactions"],
    },
    "NP": {
        "dimension_name": "Narrative Progression (NP)",
        "dimension_brief": "Whether the interactions meaningfully advance the narrative rather than stagnating or going in circles.",
        "dimension_criteria": """### Narrative Progression (NP)

Evaluate whether the scene's interactions advance the story. Specifically:

**1. Information Increment**
- Does each turn provide new information, actions, or emotional developments?
- Are there turns that merely repeat what was already said or known?
- Does the conversation avoid circular patterns where the same points are rehashed?
- Example flaw: Three consecutive turns where characters repeat the same argument without any new perspective.

**2. Suspense and Hooks**
- Does the scene create anticipation for future events?
- Are there unresolved tensions, unanswered questions, or promises of future conflict?
- Does the scene end with narrative momentum rather than a flat conclusion?
- Example merit: The scene ends with a character discovering a clue that raises new questions.

**3. Foreshadowing Payoff**
- If earlier scenes or earlier parts of this scene planted foreshadowing, is it followed up on?
- Are narrative threads picked up and advanced rather than abandoned?
- Example flaw: A mysterious letter mentioned at the start of the scene is never referenced again.

**4. Pacing**
- Is the scene's pacing appropriate (not too rushed, not too slow)?
- Do important moments get adequate attention while routine moments are handled efficiently?
- Example flaw: A climactic confrontation resolved in a single turn, while a mundane greeting takes five turns.

**Scope Exclusion**: Do NOT evaluate output format compliance (tag usage, structure, etc.) — that belongs to Instruction Compliance (IC). Do NOT evaluate speaking order or turn management — that belongs to Turn & Scene Orchestration (TSO). Focus strictly on whether the NARRATIVE CONTENT progresses meaningfully.""",
        "data_sections": ["prev_scene_summary", "interactions"],
    },
    "MQ": {
        "dimension_name": "Motivation Quality (MQ)",
        "dimension_brief": "Whether the generated scene motivation for each character is reasonable, well-aligned with their profile, and actionable.",
        "dimension_criteria": """### Motivation Quality (MQ)

Evaluate the quality of the motivations generated for each character in this scene. Specifically:

**1. Profile Alignment**
- Does the motivation align with the character's current personality, goals, and values?
- Is it consistent with the character's recent experiences and development?
- Example flaw: A pacifist character given a motivation to "seek violent revenge" without any profile basis.

**2. Situational Fit**
- Does the motivation consider the current world state, location, and scene scenario?
- Is it responsive to recent events and the current narrative context?
- Example flaw: A character motivated to "enjoy a peaceful day" when the scenario describes an urgent crisis.

**3. Actionability**
- Is the motivation specific enough to guide concrete behavior in the scene?
- Does it suggest clear goals or intentions rather than vague feelings?
- Can the character realistically pursue this motivation given the scene's constraints?
- Example flaw: A motivation like "feel things" that provides no behavioral guidance.

**4. Diversity**
- Are different characters given distinct, non-overlapping motivations?
- Do the motivations create interesting dynamics (complementary, conflicting, or orthogonal goals)?
- Example flaw: All characters in the scene given nearly identical motivations.""",
        "data_sections": ["char_profiles", "char_motivations", "scenario", "global_state"],
    },
    "IC_char": {
        "dimension_name": "Instruction Compliance - Character (IC_char)",
        "dimension_brief": "Whether the Character Agent follows output formatting rules and constraints across all its tasks.",
        "dimension_criteria": """### Instruction Compliance - Character (IC_char)

The Character Agent is responsible for three tasks: (1) generating interaction content each turn, (2) updating character motivations, and (3) updating character profiles after the scene. Evaluate whether ALL outputs follow the expected rules.

Note: The data you see has already been parsed from the model's raw JSON output. If parsing failed entirely, that error is handled separately. Your job is to evaluate the FORMAT and COMPLIANCE of the successfully parsed content shown to you.

#### Area 1: Interactions

You will see the interactions section formatted as:
```
[Interaction 0] (CharacterName):
[inner thoughts] spoken dialogue (visible actions)

[Interaction 1] (CharacterName):
...
```

The expected format within each interaction turn is:
- Inner thoughts in square brackets: `[I wonder if he's telling the truth...]`
- Spoken dialogue in plain text with no speaker label: `Good morning, how are you?`
- Visible physical actions in parentheses: `(picks up the letter and examines it)`
- These elements can be interleaved naturally in any order within a single turn.

**1. No Overstepping**
- Does each character only output their own content (thoughts, actions, speech)?
- Does any character narrate or control another character's behavior?
- Example flaw: Character A's interaction includes "Character B nodded in agreement" — controlling another character.

**2. Format Compliance**
- Are inner thoughts enclosed in square brackets `[...]`?
- Is spoken dialogue written as plain text (not inside brackets or parentheses)?
- Are visible physical actions enclosed in parentheses `(...)`?
- Are these three elements clearly distinguishable and not mixed up?
- Example flaw: `[Who is that?] (I say)` — speech incorrectly placed inside brackets and action parentheses.

#### Area 2: Post-Scene Profile Updates

You will see the profile update section formatted as:
```
### CharacterName
Profile Updated: **YES** (or **NO**)
Profile AFTER scene:
(the updated profile text)
```

The expected profile format is structured Markdown prose organized under section headers (e.g., **Social Standing**, **Core Personality**, **Historical Baggage**, **Key Relationships**, **Core Motivations**, **Moral Code**), with bullet points or paragraphs under each section.

**3. Profile Update Format Compliance**
- Is the updated profile well-structured Markdown prose with clear section headers and organized content?
- Is it free of garbled text, raw JSON artifacts (e.g., `{"name": ...}`), or broken formatting?
- Does the output actually look like a character profile (not a scene summary, world state description, or other unrelated content)?
- Example flaw: The profile text is a raw JSON dict instead of Markdown prose, or the output is clearly not a character profile at all.

#### Area 3: Character Motivations

You will see the motivations section formatted as:
```
### CharacterName
(motivation text)
```

The expected motivation format is 1-3 sentences describing the character's inner drive entering the next scene.

**4. Motivation Format Compliance**
- Does the output actually contain a character motivation (not a scene summary, dialogue, or other unrelated content)?
- Is the format a short prose passage (1-3 sentences)?
- Example flaw: The motivation field contains a full scene script or a copy of the character's profile instead of a concise inner drive statement.""",
        "data_sections": ["interactions", "char_reflections_after", "char_motivations"],
    },
    # ---- WORLD metrics ----
    "CSR": {
        "dimension_name": "Cast Selection Rationality (CSR)",
        "dimension_brief": "Whether the characters selected for this scene are the right ones for the narrative needs.",
        "dimension_criteria": """### Cast Selection Rationality (CSR)

Note: In the simulation pipeline, character selection happens FIRST (before location and scenario are decided). The system selects characters based on the global world state and each character's current short description. Evaluate whether this selection is appropriate.

**1. Narrative-Driven Selection**
- Are the selected characters essential to advancing the current narrative?
- Given the world state and character descriptions, does this cast make sense for what should happen next?
- Example merit: Selecting characters whose unresolved tensions from previous scenes need to be addressed.

**2. Goal Relevance**
- Does each selected character have a clear reason to be involved (plot connection, relationship, unfinished business)?
- Are there characters whose inclusion seems arbitrary or forced?
- Example flaw: Including a character who has no connection to any ongoing narrative threads.

**3. Avoid Redundancy**
- Are there characters who serve the same narrative function (redundant roles)?
- Could the scene work with fewer characters without losing anything?
- Example flaw: Three characters all serving as "the voice of reason" with no differentiation.

**4. Missing Key Characters**
- Are there characters from the available pool who should logically be present but were excluded?
- Would the narrative be significantly improved by including a specific available character?
- Example flaw: A scene about a family crisis that excludes a key family member who is available.

Review both the selected characters AND the available character pool to assess whether the selection is optimal.""",
        "data_sections": ["global_state", "involved_characters", "char_profiles", "character_pool_summary"],
    },
    "LSR": {
        "dimension_name": "Location & Scenario Rationality (LSR)",
        "dimension_brief": "Whether the chosen location and generated scenario are appropriate for the selected cast and current narrative state.",
        "dimension_criteria": """### Location & Scenario Rationality (LSR)

Note: In the simulation pipeline, location and scenario are decided AFTER characters have been selected. The system chooses a location and writes a scenario based on the global world state, available locations, the selected characters' descriptions, and the previous scene's context. Evaluate whether these choices are appropriate.

**1. Location Appropriateness**
- Is the chosen location a plausible and logical place for the selected characters to meet?
- Does the location serve the narrative needs (e.g., a private room for a secret conversation, a marketplace for a public confrontation)?
- Is the location consistent with the characters' current situations and the world state?
- Example flaw: Characters who are supposed to be in hiding meeting in a crowded public square.

**2. Scenario Quality**
- Does the scenario provide a clear dramatic setup that gives characters something to do?
- Is the scenario specific enough to guide the scene without being overly prescriptive?
- Does it establish the right atmosphere and stakes for the selected cast?
- Example flaw: A vague scenario like "characters meet and talk" that provides no dramatic foundation.

**3. Continuity with Previous Scene**
- Does the location/scenario follow naturally from what happened in the previous scene?
- Are there logical transitions (characters don't teleport without explanation)?
- Does the scenario build on unresolved threads from previous events?
- Example flaw: Characters who just had a dramatic confrontation suddenly appearing in a completely unrelated setting with no transition.

**4. Character-Setting Fit**
- Is the setting appropriate for the selected characters' abilities and backgrounds?
- Does the environment create interesting dynamics for the specific cast?
- Example flaw: Placing aquatic characters in a desert with no narrative justification.""",
        "data_sections": ["scenario", "location_state", "global_state", "involved_characters", "char_profiles", "available_locations", "prev_scene_summary"],
    },
    "TSO": {
        "dimension_name": "Turn & Scene Orchestration (TSO)",
        "dimension_brief": "Whether speaker selection, environmental description timing, and scene ending timing are appropriate.",
        "dimension_criteria": """### Turn & Scene Orchestration (TSO)

Evaluate the World Model's orchestration of the scene — specifically its decisions about WHO speaks WHEN and WHEN the scene ends. This metric evaluates the World Model's turn management, NOT the quality of what characters say (that belongs to other metrics).

**1. Speaker Selection**
- At each turn, does the most appropriate character respond?
- Are there moments where a different character should have spoken but didn't?
- Example flaw: A question directed at Character A is answered by Character B for no reason.

**2. Environmental Description Timing**
- Are environmental descriptions introduced at appropriate moments (scene changes, important events occurring)?
- Are they integrated naturally rather than dumped in large blocks?
- Example flaw: A long environmental description interrupting a tense dialogue exchange.

**3. Group Character Actions**
- In multi-character scenes, are character combinations reasonably selected for joint interactions (e.g., a group applauding together, several characters entering a door together, two people simultaneously turning to look somewhere)?
- Does the combination selection fit the current situation and character relationships?

**4. Character Coverage Balance**
- Is each character's participation level consistent with their identity, role, and narrative importance in the scene?
- A character with higher authority or more at stake may reasonably speak more; a minor or observing character may speak less.
- Core characters should receive participation proportional to their narrative importance, avoiding being neglected for extended periods.
- Transient characters (e.g., a passing delivery person, a chance-encountered beggar) should naturally fade out after fulfilling their narrative function rather than being forced into excessive screen time.
- Are there characters who are present but contribute nothing at all (no speech, no action, no reaction)?
- Example flaw: A king summoned to a council meeting who never speaks, while a servant dominates the entire discussion without narrative justification.

**5. Ending Timing**
- Does the scene end at a natural narrative juncture (resolution, cliffhanger, transition point)?
- Is the ending abrupt or does it drag on past its natural conclusion?
- Example flaw: The scene ending mid-sentence or mid-conflict without any narrative reason.

**Scope Exclusion**: Do NOT evaluate the dramatic quality or content of dialogue — that belongs to Narrative Progression (NP) and other character metrics. Do NOT evaluate narrative continuity or repetition in content — that belongs to Narrative Progression (NP). Do NOT evaluate time/space logic of the scenario — that belongs to Location & Scenario Rationality (LSR). Focus strictly on the World Model's ORCHESTRATION decisions: who speaks, when environment is described, and when the scene ends.""",
        "data_sections": ["interactions", "involved_characters"],
    },
    "GUS": {
        "dimension_name": "Global Update Sensitivity (GUS)",
        "dimension_brief": "Whether the timing of global state updates during this scene is appropriate.",
        "dimension_criteria": """### Global Update Sensitivity (GUS)

Evaluate whether the global state was updated at the right times during this scene. This metric focuses ONLY on the TIMING of updates (when to trigger and when not to trigger), NOT on the accuracy of the updated content (that belongs to Global State Accuracy, GSA).

The global state may be updated multiple times within a single scene (after different interactions). You will see the complete update timeline.

**1. No Over-Updating**
- Were routine conversations or minor events incorrectly treated as globally significant? Were trivial interactions triggering unnecessary global state changes?
- Example flaw: Two characters having a private chat triggers a global state update.

**2. No Missing Updates**
- Did globally significant events (wars, political changes, natural disasters, major discoveries) correctly trigger updates? Were there interactions with globally significant events that were not reflected in global updates?
- Example flaw: A king's assassination occurs in the scene but the global state is not updated.

**3. Trigger Scope**
- Was the distinction between "local impact" and "global impact" correctly made? Events that only affect the current location/characters should not trigger global updates; events that affect the broader world should.
- Example flaw: A tavern brawl triggers a global state update about "rising violence."

**4. Update Timing Within Scene**
- Were updates triggered at the right interaction points (not too early, not too late)?
- If multiple updates occurred, was each one justified by new events?

**Scope Exclusion**: Do NOT evaluate the FORMAT of the global state output (JSON structure, field completeness), as that belongs to Instruction Compliance (IC). Do NOT evaluate the ACCURACY of the updated content, as that belongs to Global State Accuracy (GSA). Focus strictly on WHETHER and WHEN updates were triggered.

**Handling Redundant Updates (triggered but content unchanged)**: Determine the root cause — if no globally significant event occurred before that update, the trigger itself was wrong → this is a GUS issue (penalize here); if a globally significant event did occur but the content failed to reflect it (i.e., trigger was correct but content update failed), that is a GSA issue → do NOT penalize here.

**Interaction Count Consideration**: Take the number of interaction turns in this scene into account when scoring. Longer scenes expose the model to more update opportunities and routine-versus-significant event distinctions, increasing the risk of over-triggering, missing updates, or triggering at the wrong time. Therefore, maintaining mostly correct update timing over many turns should be interpreted in that context rather than judged as if the scene were short. Do not penalize length itself; focus on whether observed timing errors are substantial relative to the amount of evidence and number of update opportunities. For very short scenes, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability.
""",
        "data_sections": ["interactions", "global_state_update"],
    },
    "GSA": {
        "dimension_name": "Global State Accuracy (GSA)",
        "dimension_brief": "Whether the updated global state content accurately reflects what happened in the scene.",
        "dimension_criteria": """### Global State Accuracy (GSA)

Evaluate the ACCURACY of the global state content when updates occur. This metric focuses on WHETHER THE CONTENT IS CORRECT, not on whether the update should have been triggered (that belongs to Global Update Sensitivity, GUS).

You will see the complete update timeline (the state may be updated multiple times within a single scene).

**Handling Redundant Updates (triggered but content unchanged)**: If a globally significant event did occur but the content failed to reflect it → this is a GSA issue (penalize here); if no globally significant event occurred (the trigger itself was wrong) → this is a GUS issue (do NOT penalize here). Only evaluate updates where the content actually changed, unless the redundant update falls into the GSA category above.

**1. Factual Accuracy**
- Does each updated global state accurately reflect the events that occurred up to that point? Are there distortions, exaggerations, or fabrications?
- Example flaw: The global state says "war has ended" when the scene only showed a temporary ceasefire.

**2. Timely Retirement**
- Has outdated information been removed or updated? Are there stale entries that no longer reflect the current world state?
- Example flaw: A character has been found in this scene, but the global state still lists them as "missing."

**3. Concise Expression**
- Is the global state expressed concisely without unnecessary verbosity? Does it capture the essence of changes without excessive detail?
- Example flaw: A full paragraph describing a minor political shift that could be summarized in one sentence.

**4. Incremental Accuracy**
- If multiple updates occurred, does each one build correctly on the previous state?
- Are there contradictions between successive updates within the same scene?

**Scope Exclusion**: Do NOT evaluate the FORMAT of the global state output (JSON structure, field completeness), as that belongs to Instruction Compliance (IC). Do NOT evaluate whether the update SHOULD have been triggered, as that belongs to Global Update Sensitivity (GUS). Focus strictly on whether the CONTENT of each update is accurate and well-maintained.

**Interaction Count Consideration**: Take the number of interaction turns in this scene into account when scoring. Longer scenes expose the model to more state changes, incremental updates, and opportunities for factual inconsistency. Therefore, maintaining mostly accurate global-state content over many turns should be interpreted in that context rather than judged as if the scene were short. Do not penalize length itself; focus on whether observed content errors are substantial relative to the amount of evidence and number of state-change opportunities. For very short scenes, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability.
""",
        "data_sections": ["interactions", "global_state_update"],
    },
    "LUS": {
        "dimension_name": "Location Update Sensitivity (LUS)",
        "dimension_brief": "Whether the timing of location state updates during this scene is appropriate.",
        "dimension_criteria": """### Location Update Sensitivity (LUS)

Evaluate whether the location state was updated at the right times during this scene. This metric focuses ONLY on the TIMING of updates (when to trigger and when not to trigger), NOT on the accuracy of the updated content (that belongs to Location State Accuracy, LSA).

The location state may be updated multiple times within a single scene (after different interactions). You will see the complete update timeline.

**1. No Over-Updating**
- Were temporary events (a character sitting down, a brief sound) incorrectly treated as persistent location changes?
- Were minor, reversible actions triggering unnecessary location state updates?
- Example flaw: A character picking up a cup triggers a location state update that removes the cup from entities entirely.

**2. No Missing Updates**
- Were persistent physical changes (structural damage, new objects placed, important entities arriving/leaving) correctly captured?
- Were there persistent location changes that were not reflected in updates?
- Example flaw: A fire destroys part of the building during the scene, but the location state remains unchanged.

**3. Persistence Judgment**
- Was the distinction between temporary and persistent changes correctly made?
- Temporary: character positions, momentary sounds, brief weather
- Persistent: structural changes, added/removed objects, lasting environmental effects
- Example flaw: Recording "Character A is standing by the window" as a permanent location feature.

**4. Update Timing Within Scene**
- Were updates triggered at the right interaction points?
- If multiple updates occurred, was each one justified by new physical changes?

**Scope Exclusion**: Do NOT evaluate the FORMAT of the location state output (JSON structure, field completeness), as that belongs to Instruction Compliance (IC). Do NOT evaluate the ACCURACY of the updated content, as that belongs to Location State Accuracy (LSA). Focus strictly on WHETHER and WHEN updates were triggered.

**Handling Redundant Updates (triggered but content unchanged)**: Determine the root cause — if no persistent physical change occurred before that update, the trigger itself was wrong → this is a LUS issue (penalize here); if a persistent physical change did occur but the content failed to reflect it (i.e., trigger was correct but content update failed), that is an LSA issue → do NOT penalize here.

**Interaction Count Consideration**: Take the number of interaction turns in this scene into account when scoring. Longer scenes expose the model to more update opportunities and temporary-versus-persistent event distinctions, increasing the risk of over-triggering, missing updates, or triggering at the wrong time. Therefore, maintaining mostly correct location-update timing over many turns should be interpreted in that context rather than judged as if the scene were short. Do not penalize length itself; focus on whether observed timing errors are substantial relative to the amount of evidence and number of update opportunities. For very short scenes, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability.
""",
        "data_sections": ["interactions", "location_state_update"],
    },
    "LSA": {
        "dimension_name": "Location State Accuracy (LSA)",
        "dimension_brief": "Whether the updated location state and Important Entities list are accurate.",
        "dimension_criteria": """### Location State Accuracy (LSA)

Evaluate the ACCURACY of the location state content when updates occur. This metric focuses on WHETHER THE CONTENT IS CORRECT, not on whether the update should have been triggered (that belongs to Location Update Sensitivity, LUS).

You will see the complete update timeline (the state may be updated multiple times within a single scene).

**Handling Redundant Updates (triggered but content unchanged)**: If a persistent physical change did occur but the content failed to reflect it → this is an LSA issue (penalize here); if no persistent physical change occurred (the trigger itself was wrong) → this is a LUS issue (do NOT penalize here). Only evaluate updates where the content actually changed, unless the redundant update falls into the LSA category above.

**1. Spatial Consistency**
- Is the updated spatial layout self-consistent (no contradictions in where things are)?
- Do the updated sub-locations and their relationships make physical sense?
- Example flaw: An entity listed as being in two different sub-locations simultaneously.

**2. Entity Accuracy**
- Does the important entities list accurately reflect the important entities currently at the location?
- Have important entities that arrived/departed been correctly added/removed?
- Are entity states (conditions, positions) accurately described?
- Note: Only track important, lasting entities. Trivial or transient details (e.g., character clothing, briefly mentioned objects) do not need to appear in the important entities list; not recording them is normal and should not be penalized.
- Example flaw: A destroyed building still listed as intact in the entities list.

**3. Scene Consistency**
- Is the location description consistent with what happened in the scene?
- Example flaw: After an intense fight, the description still reads "the room is spotless."

**4. Incremental Accuracy**
- If multiple updates occurred, does each one build correctly on the previous state?
- Are there contradictions between successive updates within the same scene?

**Scope Exclusion**: Do NOT evaluate the FORMAT of the location state output (JSON structure, field completeness), as that belongs to Instruction Compliance (IC). Do NOT evaluate whether the update SHOULD have been triggered, as that belongs to Location Update Sensitivity (LUS). Focus strictly on whether the CONTENT of each update is accurate.

**Interaction Count Consideration**: Take the number of interaction turns in this scene into account when scoring. Longer scenes expose the model to more state changes, incremental updates, and opportunities for spatial or entity-level inconsistency. Therefore, maintaining mostly accurate location-state content over many turns should be interpreted in that context rather than judged as if the scene were short. Do not penalize length itself; focus on whether observed content errors are substantial relative to the amount of evidence and number of state-change opportunities. For very short scenes, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability.
""",
        "data_sections": ["interactions", "location_state_update"],
    },
    "IC_world": {
        "dimension_name": "Instruction Compliance - World (IC_world)",
        "dimension_brief": "Whether the World Model follows output formatting rules and constraints across all its tasks.",
        "dimension_criteria": """### Instruction Compliance - World (IC_world)

The World Model is responsible for four tasks: (1) selecting the cast of characters for each scene, (2) choosing a location and generating a scenario, (3) selecting the next speaker each turn, and (4) updating global/location state. Evaluate whether ALL outputs follow the expected rules.

Note: The data you see has already been parsed from the model's raw JSON output. If parsing failed entirely (e.g., invalid JSON), that error is handled separately via an error penalty. Your job is to evaluate the FORMAT and COMPLIANCE of the successfully parsed content shown to you.

#### Area 1: Involved Characters

You will see:
```
**Involved Characters**: CharA, CharB, CharC
```
This is the parsed result of the cast selection task.

**1. Cast Selection Compliance**
- Is the character list non-empty and reasonable in size for a scene?
- Example flaw: An empty character list, or an absurdly large number of characters (e.g., 15+) that would make a scene unmanageable.

#### Area 2: Scene Scenario

You will see:
```
## Scene Scenario
(scenario text)
```
This is the parsed scenario from the location & scenario selection task.

**2. Scenario Format Compliance**
- Is the scenario a readable prose passage (not garbled, not containing raw JSON artifacts)?
- Does the output actually look like a scene scenario (not a character profile, world state, or other unrelated content)?
- Example flaw: The scenario field contains raw JSON syntax or is clearly not a scenario description at all.

#### Area 3: Speaker Selection Sequence

You will see:
```
## Speaker Selection Sequence
Turn 0: CharA
Turn 1: CharB
Turn 2: CharA, CharC
Turn 3: Environment
```
This is the sequence of speaker selections made by the World Model throughout the scene.

**3. Speaker Selection Compliance**
- Is each selected speaker one of the involved characters listed above (or "Environment" for environmental beats)?
- Are there any turns where an invalid or non-existent character name appears?
- Are multi-character turns (e.g., "CharA, CharB") used sparingly and only for genuinely shared beats?
- Example flaw: A speaker name that doesn't match any of the involved characters.

#### Area 4: Global State Update Timeline

The global state is a structured Markdown document describing the world's rules, norms, and conditions, organized under thematic section headers (e.g., `### 1. Social Order & Class` or `**Economy & Trade**`), with bullet points or paragraphs under each section.

You will see one of two cases:

**Case A — Update(s) occurred:**
```
## Global State Update Timeline
Global State BEFORE scene:
(Markdown prose)

Global State Updated: **YES** (N update(s) during this scene)
### Global Update 1 (after interaction X)
(updated Markdown prose)
```

**Case B — No update:**
```
## Global State Update Timeline
Global State BEFORE scene:
(Markdown prose)

Global State Updated: **NO** (unchanged)
```

**4. Global State Format**
- Is the global state presented as readable Markdown prose (not garbled, not containing raw JSON artifacts like `{"key": "value"}`)?
- Does the output actually look like a global world state description (not a character profile, scenario, or other unrelated content)?
- Example flaw: The global state text contains raw JSON syntax or is clearly not a world state description at all.

#### Area 5: Location State Update Timeline

The location state supports two structural variants depending on the location:

**Variant 1 — With Sub Locations:** The state has a "Description" line, followed by "Sub Locations" formatted as `[SubLocationName]: description`, each containing "Important Entities" formatted as `- EntityName: entity state`.

**Variant 2 — Without Sub Locations:** The state has a "Description" line, followed directly by "Important Entities" formatted as `- EntityName: entity state` (no sub-location grouping).

You will see one of two update cases:

**Case A — Update(s) occurred (Variant 1 example):**
```
## Location State Update Timeline: LocationName
Location State BEFORE scene:
Description: ...
  [SubLocationName]: ...
    - EntityName: entity state

Location State Updated: **YES** (N update(s) during this scene)
### Location Update 1 (after interaction X)
Description: ...
  [SubLocationName]: ...
    - EntityName: entity state
```

**Case A — Update(s) occurred (Variant 2 example):**
```
## Location State Update Timeline: LocationName
Location State BEFORE scene:
Description: ...
  - EntityName: entity state

Location State Updated: **YES** (N update(s) during this scene)
### Location Update 1 (after interaction X)
Description: ...
  - EntityName: entity state
```

**Case B — No update:**
```
## Location State Update Timeline: LocationName
Location State BEFORE scene:
Description: ...
(entities and/or sub-locations as applicable)

Location State Updated: **NO** (unchanged)
```

**5. Location State Format**
- Does the location state follow one of the two expected structural variants?
  - Variant 1 (with Sub Locations): Description → Sub Locations (each with Important Entities)
  - Variant 2 (without Sub Locations): Description → Important Entities directly
- If sub-locations are present, are they properly structured with names, descriptions, and entities with states?
- If no sub-locations, are Important Entities listed directly after the Description with names and states?
- Does the output actually look like a location state (not a character profile, global state, or other unrelated content)?
- Example flaw: Entities listed without states, or the output is clearly not a location state at all.

**6. No Overstepping**
- Do world state updates (both global and location) avoid generating character dialogue or controlling character behavior?
- Are updates purely descriptive of the environment, not prescriptive of character actions?
- Example flaw: The location state includes "Character A decides to leave" — overstepping into character territory.""",
        "data_sections": ["involved_characters", "scenario", "interaction_speakers", "global_state_update", "location_state_update"],
    },
}

# ============================================================================ #
# Cross-Scene Per-Character evaluation: Metric definitions
# ============================================================================ #

CROSS_SCENE_CHARACTER_METRIC_DETAILS = {
    "PES": {
        "dimension_name": "Profile Evolution Smoothness (PES)",
        "dimension_brief": "Whether the character's profile evolution and hidden tracker evolution together remain gradual, coherent, and narratively smooth across scenes.",
        "dimension_criteria": """### Profile Evolution Smoothness (PES)

Evaluate the SMOOTHNESS of this character's cross-scene evolution by considering BOTH the profile trajectory and the hidden tracker trajectory together. This metric focuses on whether the evolution TRAJECTORY is smooth and coherent over time, NOT on whether individual scene updates are accurate (that belongs to Profile Update Fidelity, PUF).

Specifically:

**1. Gradualness**
- Do changes in personality, attitude, and relationships go through reasonable transitional stages across scenes?
- Does the hidden tracker preserve intermediate stages before they eventually become profile changes?
- Are there abrupt jumps where a character goes from one extreme to another without intermediate signals in either profile or tracker?
- Example flaw: Scene 3 shows mild doubt, but Scene 4 profile suddenly declares complete distrust with no accumulated intermediate signals.

**2. Magnitude Proportionality Across the Timeline**
- Looking at the full evolution timeline, is the magnitude of each profile change proportional to what happened in the corresponding scene(s)?
- Are there scenes where nothing significant happened but the profile changed dramatically, or vice versa?
- Example flaw: A profile undergoes a massive rewrite after a routine conversation, while remaining unchanged after a life-altering event.

**3. Directional Coherence Across Profile + Tracker**
- Do the profile and hidden tracker point in compatible directions over time?
- When tracker signals accumulate enough to justify a profile update, does that transition feel traceable and well-grounded?
- Are there contradictions where the tracker implies one trajectory while the profile jumps to another without explanation?
- Example flaw: The tracker repeatedly records growing trust, but the next profile revision abruptly claims deepening hostility with no causal basis.

**Scope Exclusion**: Do NOT evaluate whether individual scene updates are accurate or whether the right things were captured — that belongs to Profile Update Fidelity (PUF). Focus strictly on whether the OVERALL TRAJECTORY across all scenes is smooth, gradual, and directionally coherent.

**Scene Count Consideration**: Take the total number of scenes into account when scoring. Longer trajectories expose the model to more profile and hidden-tracker updates, accumulated signals, and opportunities for drift or contradiction. Therefore, maintaining mostly smooth and coherent profile evolution over many scenes should be interpreted in that context rather than judged as if the trajectory were short. Do not penalize length itself; focus on whether observed trajectory errors are substantial relative to the amount of evidence and number of update opportunities. For very short trajectories, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability.

Review the complete profile evolution timeline and hidden tracker timeline together, and assess whether they form one coherent and smooth character development arc.""",
    },
}

# ============================================================================ #
# Cross-Scene Global evaluation: Metric definitions (SCC only)
# ============================================================================ #

CROSS_SCENE_GLOBAL_METRIC_DETAILS = {
    "SCC": {
        "dimension_name": "Scene Continuity & Coherence (SCC)",
        "dimension_brief": "Whether cross-scene planning forms a coherent narrative arc with well-managed pacing and narrative threads.",
        "dimension_criteria": """### Scene Continuity & Coherence (SCC)

Evaluate the overall narrative coherence across all scenes. You are provided with each scene's summary (including location, scenario, involved characters, and what happened) to assess cross-scene coherence.

**1. Narrative Arc**
- Do consecutive scenes form a directional narrative progression?
- Is there a discernible story arc (setup → rising action → climax → resolution)?
- Do scenes build upon each other rather than being disconnected episodes?
- Example flaw: Scenes that feel like random, unconnected vignettes with no overarching story.

**2. Scene Transitions**
- Do location choices and scene descriptions naturally connect?
- Are transitions between scenes logical (characters move to locations that make sense)?
- Is there narrative justification for each scene's setting?
- Example flaw: Characters teleporting between distant locations without travel time or explanation.

**3. Pacing**
- Is the story pacing appropriate across the full simulation?
- Are there sections that feel too rushed or too slow?
- Do important plot points get adequate development time?
- Example flaw: The climactic confrontation happening in Scene 2 of 10, with the remaining 8 scenes being anticlimactic.

**4. Thread Management**
- Are narrative threads introduced, developed, and resolved (or intentionally left open)?
- Are there abandoned plot threads that were set up but never followed through?
- Example flaw: A mystery introduced in Scene 1 that is never mentioned again in any subsequent scene.

**5. Scene Count Consideration**
- Take the total number of scenes into account when scoring. Longer simulations expose the model to more narrative arcs, pacing decisions, scene transitions, and thread-management challenges.
- Maintaining mostly coherent narrative development over many scenes should be interpreted in that context rather than judged as if the simulation were short. Do not penalize length itself; focus on whether observed coherence errors are substantial relative to the amount of evidence and number of long-range narrative dependencies.
- For very short simulations, there may be limited evidence for assessing this metric, so avoid over-interpreting the absence of errors as strong evidence of capability."""    },
}

# ============================================================================ #
# Generic System Prompt templates (by evaluation layer)
# ============================================================================ #

PER_SCENE_SYSTEM_TEMPLATE = """You are an expert evaluator for interactive fiction simulation systems. Your task is to evaluate a single scene from a simulation on ONE specific dimension: **{dimension_name}**.

## Dimension to Evaluate

**{dimension_name}**: {dimension_brief}

{dimension_criteria}

""" + SCORING_INSTRUCTION

PER_SCENE_PAIRED_SYSTEM_TEMPLATE = """You are an expert evaluator for interactive fiction simulation systems. Your task is to evaluate a single scene from a simulation on TWO specific dimensions simultaneously: **{dimension_name_a}** and **{dimension_name_b}**.

Evaluate each dimension independently based on the same scene data. Do NOT let the assessment of one dimension influence the other.

## Dimension 1: {dimension_name_a}

**{dimension_name_a}**: {dimension_brief_a}

{dimension_criteria_a}

## Dimension 2: {dimension_name_b}

**{dimension_name_b}**: {dimension_brief_b}

{dimension_criteria_b}

""" + SCORING_INSTRUCTION

SCENE_SUMMARY_FULL_SYSTEM = SCENE_SUMMARY_SYSTEM_PROMPT

CROSS_SCENE_CHARACTER_SYSTEM_TEMPLATE = """You are an expert evaluator for interactive fiction simulation systems. Your task is to evaluate a single character's evolution trajectory across multiple scenes on ONE specific dimension: **{dimension_name}**.

**Important**: The scene summaries provided are auto-generated auxiliary context. Their quality, formatting, or completeness is NOT within your evaluation scope. Base your evaluation solely on the recorded content of each scene (profile history, scene descriptions, hidden trackers, etc.).

## Dimension to Evaluate

**{dimension_name}**: {dimension_brief}

{dimension_criteria}

""" + SCORING_INSTRUCTION

CROSS_SCENE_GLOBAL_SYSTEM_TEMPLATE = """You are an expert evaluator for interactive fiction simulation systems. Your task is to evaluate the global world state evolution and scene planning coherence across the entire simulation on ONE specific dimension: **{dimension_name}**.

**Important**: The scene summaries provided are auto-generated auxiliary context. Their quality, formatting, or completeness is NOT within your evaluation scope. Base your evaluation solely on the recorded content of each scene (scenarios, involved characters, location information, etc.).

## Dimension to Evaluate

**{dimension_name}**: {dimension_brief}

{dimension_criteria}

""" + SCORING_INSTRUCTION

# ============================================================================ #
# System Prompt builder functions
# ============================================================================ #

def build_per_scene_system_prompt(metric: str) -> str:
    """Build Per-Scene single-metric system prompt."""
    details = PER_SCENE_METRIC_DETAILS[metric]
    return PER_SCENE_SYSTEM_TEMPLATE.format(
        dimension_name=details["dimension_name"],
        dimension_brief=details["dimension_brief"],
        dimension_criteria=details["dimension_criteria"],
    )


def build_per_scene_paired_system_prompt(metric_a: str, metric_b: str) -> str:
    """Build Per-Scene paired-metrics system prompt (two dimensions evaluated together)."""
    da = PER_SCENE_METRIC_DETAILS[metric_a]
    db = PER_SCENE_METRIC_DETAILS[metric_b]
    return PER_SCENE_PAIRED_SYSTEM_TEMPLATE.format(
        dimension_name_a=da["dimension_name"],
        dimension_brief_a=da["dimension_brief"],
        dimension_criteria_a=da["dimension_criteria"],
        dimension_name_b=db["dimension_name"],
        dimension_brief_b=db["dimension_brief"],
        dimension_criteria_b=db["dimension_criteria"],
    )


def build_cross_scene_character_system_prompt(metric: str) -> str:
    """Build Cross-Scene Per-Character single-metric system prompt."""
    details = CROSS_SCENE_CHARACTER_METRIC_DETAILS[metric]
    return CROSS_SCENE_CHARACTER_SYSTEM_TEMPLATE.format(
        dimension_name=details["dimension_name"],
        dimension_brief=details["dimension_brief"],
        dimension_criteria=details["dimension_criteria"],
    )

def build_cross_scene_global_system_prompt(metric: str) -> str:
    """Build Cross-Scene Global single-metric system prompt."""
    details = CROSS_SCENE_GLOBAL_METRIC_DETAILS[metric]
    return CROSS_SCENE_GLOBAL_SYSTEM_TEMPLATE.format(
        dimension_name=details["dimension_name"],
        dimension_brief=details["dimension_brief"],
        dimension_criteria=details["dimension_criteria"],
    )

# ============================================================================ #
# User Prompt builders: Per-Scene (data pruned per metric)
# ============================================================================ #

def build_per_scene_user_prompt(metric: str, scene_slice: Dict[str, Any]) -> str:
    """Build Per-Scene single-metric user prompt.
    
    Includes only data needed by the metric based on its data_sections, reducing token usage.
    """
    from data_loader import format_interactions, format_location_state, format_global_state_full
    
    details = PER_SCENE_METRIC_DETAILS[metric]
    needed = set(details["data_sections"])
    
    parts = []
    
    # Basic info (needed by all metrics)
    parts.append(f"# Scene {scene_slice['scene_index']} Evaluation: {details['dimension_name']}")
    parts.append(f"**Book**: {scene_slice['book_name']}")
    parts.append(f"**Location**: {scene_slice['location']}")
    
    # Scene description
    if "scenario" in needed or metric in ("CSR", "MQ"):
        parts.append(f"\n## Scene Scenario\n{scene_slice['scenario']}")
    
    # Previous scene summary
    if "prev_scene_summary" in needed and scene_slice.get("prev_scene_summary"):
        parts.append(f"\n## Previous Scene Summary\n{scene_slice['prev_scene_summary']}")
    
    # Participating character list
    if "involved_characters" in needed or metric in ("CSR", "TSO"):
        if metric == "TSO" and scene_slice.get("char_short_descriptions"):
            # TSO needs short descriptions to check if speech ratio matches identity
            parts.append("\n## Involved Characters")
            for char_name in scene_slice['involved_characters']:
                desc = scene_slice['char_short_descriptions'].get(char_name, '')
                if desc:
                    parts.append(f"- **{char_name}**: {desc}")
                else:
                    parts.append(f"- **{char_name}**")
        else:
            parts.append(f"\n**Involved Characters**: {', '.join(scene_slice['involved_characters'])}")
    
    # Character profiles
    if "char_profiles" in needed:
        parts.append("\n## Character Profiles (at scene start)")
        for char_name, profile in scene_slice["char_profiles"].items():
            parts.append(f"\n### {char_name}\n{profile}")
    
    # For SSF/MDB/EA/EU, include original book speaking style examples as reference
    _METRICS_WITH_STYLE_EXAMPLES = {"SSF", "MDB", "EA", "EU"}
    if metric in _METRICS_WITH_STYLE_EXAMPLES and scene_slice.get("speaking_style_examples"):
        style_examples = scene_slice["speaking_style_examples"]
        # Only show examples for characters in current scene
        has_examples = False
        for char_name in scene_slice["involved_characters"]:
            if char_name in style_examples and style_examples[char_name]:
                if not has_examples:
                    if metric == "SSF":
                        # SSF: reference excerpts for judging speaking style fidelity
                        parts.append("\n## Original Speaking Style Examples (Reference from the Source Book)")
                        parts.append("*The following are interaction excerpts taken directly from the original source book. They provide reference evidence for each character's vocabulary, sentence structure, tone, and verbal habits. When evaluating the character's simulated interactions, jointly consider both these original book examples AND the character's current profile to determine whether the speaking style is faithful.*")
                    else:
                        # MDB/EA/EU: reference only, helping judge understand original speaking style
                        parts.append("\n## Original Speaking Style Examples (Reference from the Source Book)")
                        parts.append("*The following are interaction excerpts from the original source book, provided as reference to help you understand each character's authentic speaking style — their vocabulary, tone, and verbal habits. Use these to better contextualize the character's behavior in the simulation, but note that this metric's evaluation criteria remain the primary focus.*")
                    has_examples = True
                parts.append(f"\n### {char_name}")
                for i, example in enumerate(style_examples[char_name], 1):
                    parts.append(f"**Example {i}**: {example}")
    
    # Character motivations (at end for MQ, normal position for other metrics)
    if "char_motivations" in needed and scene_slice.get("char_motivations") and metric != "MQ":
        parts.append("\n## Character Motivations (generated for this scene)")
        for char_name, motivation in scene_slice["char_motivations"].items():
            parts.append(f"\n### {char_name}\n{motivation}")
    
    # Character hidden trackers (for PF/SSF/MDB etc. needing latent state awareness)
    if "char_hidden_trackers" in needed and scene_slice.get("char_reflections"):
        parts.append("\n## Character Hidden Trackers (at scene start)")
        parts.append("*The Hidden Tracker is an internal bookkeeping mechanism that records sub-threshold psychological signals for each character — subtle emotional shifts, unresolved tensions, early signs of attitude change, and other latent states that have not yet risen to the level of a profile update. These signals provide important context for understanding why a character may behave in certain ways.*")
        for char_name, reflection in scene_slice["char_reflections"].items():
            ht_str = _format_hidden_tracker(reflection.get("hidden_tracker", ""))
            if ht_str and ht_str != "(No hidden tracker)":
                parts.append(f"\n### {char_name}\n{ht_str}")
    
    # World state
    if "global_state" in needed:
        parts.append(f"\n## Global State (at scene start)\n{format_global_state_full(scene_slice['global_state_start'])}")
    
    if "location_state" in needed:
        parts.append(f"\n## Location State: {scene_slice['location']} (at scene start)\n{format_location_state(scene_slice['location_state_start'])}")
    
    # Full interactions (annotated with interaction index for timeline alignment)
    if "interactions" in needed:
        parts.append(f"\n## Interactions ({len(scene_slice['interactions'])} turns)")
        parts.append(format_interactions(scene_slice["interactions"]))
    
    # Interaction speaker sequence (for IC_world speaker selection compliance)
    if "interaction_speakers" in needed:
        parts.append("\n## Speaker Selection Sequence")
        parts.append("*The World Model selects the next speaker for each turn. Below is the sequence of selected speakers (extracted from interaction keys).*")
        for i, interaction in enumerate(scene_slice.get('interactions', [])):
            if isinstance(interaction, str):
                parts.append(f"Turn {i}: {interaction}")
                continue
            if not isinstance(interaction, dict):
                parts.append(f"Turn {i}: Unknown")
                continue
            chars = interaction.get('characters', [])
            char_str = ', '.join(chars) if chars else 'Unknown'
            parts.append(f"Turn {i}: {char_str}")
    
    # World-state update timeline used by GUS/GSA/LUS/LSA/IC_world.
    if "global_state_update" in needed:
        parts.append("\n## Global State Update Timeline")
        parts.append(f"Global State BEFORE scene:\n{format_global_state_full(scene_slice['global_state_start'])}")
        updates = scene_slice.get("global_updates_in_scene", [])
        if updates:
            parts.append(f"\nGlobal State Updated: **YES** ({len(updates)} update(s) during this scene)")
            for i, upd in enumerate(updates):
                parts.append(f"\n### Global Update {i+1} (after interaction {upd['interaction_index']})")
                parts.append(format_global_state_full(upd.get("global_state", {})))
        else:
            parts.append(f"\nGlobal State Updated: **NO** (unchanged)")
    
    if "location_state_update" in needed:
        parts.append(f"\n## Location State Update Timeline: {scene_slice['location']}")
        parts.append(f"Location State BEFORE scene:\n{format_location_state(scene_slice['location_state_start'])}")
        updates = scene_slice.get("location_updates_in_scene", [])
        if updates:
            parts.append(f"\nLocation State Updated: **YES** ({len(updates)} update(s) during this scene)")
            for i, upd in enumerate(updates):
                parts.append(f"\n### Location Update {i+1} (after interaction {upd['interaction_index']})")
                parts.append(format_location_state(upd.get("location_state", {})))
        else:
            parts.append(f"\nLocation State Updated: **NO** (unchanged)")
    
    # Character reflections — only profile_after (for IC_char profile update compliance)
    if "char_reflections_after" in needed and scene_slice.get("char_reflections"):
        parts.append("\n## Post-Scene Profile Updates (Character Agent output)")
        parts.append("*These are the updated profiles produced by the Character Agent after the scene. Evaluate their format and compliance.*")
        for char_name, reflection in scene_slice["char_reflections"].items():
            parts.append(f"\n### {char_name}")
            profile_updated = reflection.get('profile_updated', False)
            parts.append(f"Profile Updated: {'**YES**' if profile_updated else '**NO**'}")
            if profile_updated:
                parts.append(f"Profile AFTER scene:\n{reflection.get('profile_after', '')}")
            else:
                parts.append("(Profile unchanged)")
    
    # Character reflections — full version (for PUF)
    if "char_reflections" in needed:
        parts.append("\n## Post-Scene Character Reflections")
        parts.append("*The Hidden Tracker is an internal bookkeeping mechanism that records sub-threshold psychological signals — subtle emotional shifts, unresolved tensions, early signs of attitude change, and other latent states that have not yet risen to the level of a profile update. Evaluate whether the hidden tracker captures the right signals from this scene.*")
        for char_name, reflection in scene_slice["char_reflections"].items():
            parts.append(f"\n### {char_name}")
            parts.append(f"Profile BEFORE scene: (see Character Profiles section above)")
            profile_updated = reflection.get('profile_updated', False)
            parts.append(f"Profile Updated: {'**YES**' if profile_updated else '**NO**'}")
            if profile_updated:
                parts.append(f"Profile AFTER scene:\n{reflection.get('profile_after', '')}")
            parts.append(f"Description: {reflection.get('description', '')}")
            ht_str = _format_hidden_tracker(reflection.get("hidden_tracker", ""))
            parts.append(f"Hidden Tracker:\n{ht_str}")
    
    # Character pool summary (for CSR)
    if "character_pool_summary" in needed and scene_slice.get("character_pool_summary"):
        parts.append("\n## Available Character Pool (not selected for this scene)")
        for char_info in scene_slice["character_pool_summary"][:20]:
            parts.append(f"- **{char_info.get('name', 'Unknown')}**: {char_info.get('short_description', '')}")
    
    # Optional location summary (for LSR)
    if "available_locations" in needed and scene_slice.get("available_locations"):
        parts.append("\n## Available Locations")
        for loc_info in scene_slice["available_locations"]:
            marker = " ✅ **(SELECTED)**" if loc_info.get("is_current") else ""
            parts.append(f"- **{loc_info.get('name', 'Unknown')}**{marker}: {loc_info.get('short_description', '')}")
    
    # For MQ, motivation placed last (after all context)
    if "char_motivations" in needed and scene_slice.get("char_motivations") and metric == "MQ":
        parts.append("\n## Character Motivations (generated for this scene — this is the OUTPUT to evaluate)")
        for char_name, motivation in scene_slice["char_motivations"].items():
            parts.append(f"\n### {char_name}\n{motivation}")
    
    parts.append(SINGLE_METRIC_OUTPUT_FORMAT)
    
    return "\n".join(parts)


def build_per_scene_paired_user_prompt(metric_a: str, metric_b: str, scene_slice: Dict[str, Any]) -> str:
    """Build Per-Scene paired-metrics user prompt.
    
    Both metrics have identical data_sections, so reuse metric_as build logic,
    only replacing the output format at the end with the paired version, and fixing the title line.
    """
    # If paired includes a metric needing speaking style examples, prefer it for base_prompt
    # Priority: SSF > MDB > EA > EU (SSF examples description is most detailed)
    _STYLE_PRIORITY = ["SSF", "MDB", "EA", "EU"]
    build_metric = metric_a
    for _m in _STYLE_PRIORITY:
        if _m in (metric_a, metric_b):
            build_metric = _m
            break
    base_prompt = build_per_scene_user_prompt(build_metric, scene_slice)
    # Replace title line: change from single metric name to two metric names
    da = PER_SCENE_METRIC_DETAILS[metric_a]
    db = PER_SCENE_METRIC_DETAILS[metric_b]
    d_build = PER_SCENE_METRIC_DETAILS[build_metric]
    old_title = f"# Scene {scene_slice['scene_index']} Evaluation: {d_build['dimension_name']}"
    new_title = f"# Scene {scene_slice['scene_index']} Evaluation: {da['dimension_name']} & {db['dimension_name']}"
    base_prompt = base_prompt.replace(old_title, new_title, 1)
    # Replace output format: substitute single-metric format with paired format
    paired_format = PAIRED_METRIC_OUTPUT_FORMAT.format(metric_a=metric_a, metric_b=metric_b)
    return base_prompt.replace(SINGLE_METRIC_OUTPUT_FORMAT, paired_format)


def build_scene_summary_user_prompt(scene_slice: Dict[str, Any]) -> str:
    """Build Scene Summary user prompt."""
    from data_loader import format_interactions
    
    parts = []
    parts.append(f"# Scene {scene_slice['scene_index']}")
    parts.append(f"**Location**: {scene_slice['location']}")
    parts.append(f"\n## Scene Scenario\n{scene_slice['scenario']}")
    parts.append(f"\n**Involved Characters**: {', '.join(scene_slice['involved_characters'])}")
    
    if scene_slice.get("prev_scene_summary"):
        parts.append(f"\n## Previous Scene Summary\n{scene_slice['prev_scene_summary']}")
    
    parts.append(f"\n## Interactions ({len(scene_slice['interactions'])} turns)")
    parts.append(format_interactions(scene_slice["interactions"]))
    
    parts.append(SCENE_SUMMARY_OUTPUT_FORMAT)
    
    return "\n".join(parts)

# ============================================================================ #
# User Prompt builders: Cross-Scene Per-Character
# ============================================================================ #

def build_cross_scene_character_user_prompt(metric: str, char_slice: Dict[str, Any]) -> str:
    """Build Cross-Scene Per-Character single-metric user prompt.
    
    Merge profile_history, scene_descriptions, participated_scenes by scene_index into
    a unified timeline, ordered by time/causality within each scene:
    motivation → summary → description → hidden_tracker → profile
    """
    parts = []
    
    parts.append(f"# Character Evolution Evaluation: {char_slice['char_name']}")
    parts.append(f"**Book**: {char_slice['book_name']}")
    parts.append(f"**Scenes Participated**: {char_slice['num_scenes_participated']}")
    
    # Initial profile
    parts.append(f"\n## Initial Profile\n{char_slice['initial_profile']}")
    
    # Build unified data indexed by scene_index
    scene_data = {}  # scene_index -> {motivation, description, hidden_tracker, profile, location, scenario, summary}
    
    for sd in char_slice["scene_descriptions"]:
        si = sd.get("scene_index", -1)
        if si not in scene_data:
            scene_data[si] = {}
        scene_data[si]["motivation"] = sd.get("enhanced_motivation", "N/A")
        scene_data[si]["description"] = sd.get("description", "N/A")
        scene_data[si]["hidden_tracker"] = sd.get("hidden_tracker")
    
    for ph in char_slice["profile_history"]:
        si = ph.get("scene_index", -1)
        if si not in scene_data:
            scene_data[si] = {}
        scene_data[si]["profile"] = ph.get("profile", "")
    
    for ps in char_slice["participated_scenes"]:
        si = ps.get("scene_index", -1)
        if si not in scene_data:
            scene_data[si] = {}
        scene_data[si]["location"] = ps.get("location", "")
        scene_data[si]["scenario"] = ps.get("scenario", "")
        scene_data[si]["summary"] = ps.get("summary", "")
    
    # Output unified timeline sorted by scene_index
    parts.append("\n## Scene-by-Scene Evolution Timeline")
    parts.append("*The Hidden Tracker is an internal bookkeeping mechanism that records sub-threshold psychological signals — subtle emotional shifts, unresolved tensions, early signs of attitude change, and other latent states that have not yet risen to the level of a profile update. Track how these signals accumulate and eventually graduate into profile changes.*")
    
    for si in sorted(scene_data.keys()):
        sd = scene_data[si]
        location = sd.get('location', '')
        header = f"\n### Scene {si}"
        if location:
            header += f" at {location}"
        parts.append(header)
        
        # 1. 1. Motivation (generated before scene starts)
        if sd.get("motivation"):
            parts.append(f"**Motivation**: {sd['motivation']}")
        
        # 2. Scenario
        if sd.get("scenario"):
            parts.append(f"**Scenario**: {sd['scenario']}")
        
        # 3. 3. Summary (what happened in the scene)
        if sd.get("summary"):
            parts.append(f"**Scene Summary**: {sd['summary']}")
        
        # 4. 4. Description (post-scene character description)
        if sd.get("description"):
            parts.append(f"**Description**: {sd['description']}")
        
        # 5. 5. Hidden Tracker (post-scene latent signal records)
        ht_str = _format_hidden_tracker(sd.get("hidden_tracker"))
        if ht_str and ht_str != "(No hidden tracker)":
            parts.append(f"**Hidden Tracker**:\n{ht_str}")
        
        # 6. 6. Profile (post-scene profile snapshot)
        if sd.get("profile"):
            parts.append(f"**Profile After Scene**:\n{sd['profile']}")
    
    parts.append(SINGLE_METRIC_OUTPUT_FORMAT)
    
    return "\n".join(parts)

def build_cross_scene_global_user_prompt(metric: str, global_slice: Dict[str, Any]) -> str:
    """Build Cross-Scene Global single-metric user prompt.
    
    SCC input: all scene summaries, for evaluating cross-scene narrative coherence.
    """
    parts = []
    
    parts.append(f"# Cross-Scene Narrative Coherence Evaluation")
    parts.append(f"**Book**: {global_slice.get('book_name', 'Unknown')}")
    parts.append(f"**Total Scenes**: {len(global_slice.get('all_scene_summaries', []))}")
    
    # All scene summaries
    parts.append("\n## Scene Sequence")
    for scene_info in global_slice.get("all_scene_summaries", []):
        si = scene_info.get("scene_index", "?")
        parts.append(f"\n### Scene {si} at {scene_info.get('location', 'Unknown')}")
        parts.append(f"**Scenario**: {scene_info.get('scenario', 'N/A')}")
        parts.append(f"**Involved Characters**: {', '.join(scene_info.get('involved_characters', []))}")
        if scene_info.get("summary"):
            parts.append(f"**Summary**: {scene_info['summary']}")
    
    parts.append(SINGLE_METRIC_OUTPUT_FORMAT)
    
    return "\n".join(parts)

# ============================================================================ #
# Get metric lists for each layer
# ============================================================================ #

def get_per_scene_metrics() -> List[str]:
    """Return all metric names for the Per-Scene layer."""
    return list(PER_SCENE_METRIC_DETAILS.keys())

def get_per_scene_character_metrics() -> List[str]:
    """Return CHARACTER metric names for the Per-Scene layer."""
    return ["PF", "SSF", "MDB", "PUF", "EA", "EU", "CR", "NP", "MQ", "IC_char"]

def get_per_scene_world_metrics() -> List[str]:
    """Return WORLD metric names for the Per-Scene layer."""
    return ["CSR", "LSR", "TSO", "GUS", "GSA", "LUS", "LSA", "IC_world"]

def get_cross_scene_character_metrics() -> List[str]:
    """Return metric names for the Cross-Scene Per-Character layer."""
    return list(CROSS_SCENE_CHARACTER_METRIC_DETAILS.keys())

def get_cross_scene_global_metrics() -> List[str]:
    """Return metric names for the Cross-Scene Global layer."""
    return list(CROSS_SCENE_GLOBAL_METRIC_DETAILS.keys())
