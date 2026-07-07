# EvolvingWorld Evaluation Framework

This document defines the evaluation framework for the EvolvingWorld project, using the **LLM-as-Judge** approach to evaluate **Character Agent** and **World Model** across multiple dimensions.

The evaluation system comprises **10 dimensions and 20 sub-metrics**, divided into two scoring modules:
- **CHARACTER Score** (Character Agent evaluation): 6 dimensions, 11 sub-metrics
- **WORLD Score** (World Model evaluation): 4 dimensions, 9 sub-metrics

---

## Usage

### 1. Prerequisites

Python 3.8+ with required dependencies (`numpy`, `openai` or a compatible HTTP client).

### 2. Input Data Structure

The evaluation script expects `--input_dir` to contain `sample_*` subdirectories, each with the following files:

```
input_dir/
├── sample_0/
│   ├── meta.json               # Simulation metadata (book_name, stop_reason, etc.)
│   ├── all_scenes.json          # All scene interaction records
│   ├── character_dynamic.json   # Character dynamic data (profile evolution history, etc.)
│   └── world_dynamic.json       # World dynamic data (global/location state update history, etc.)
├── sample_1/
│   └── ...
└── ...
```

Additionally, `--input_snapshots` should point to a JSON file (e.g., `test_all.json`) containing the original snapshot for each sample with initial states, used as reference for profiles and world states at scene_index=0.

### 3. Running the Evaluation

**Basic usage** (with default parameters):

```bash
python evaluation/main.py \
  --input_dir <simulation_outputs_dir> \
  --input_snapshots <path_to_test_all.json>
```

**Full parameters**:

```bash
python evaluation/main.py \
  --input_dir <simulation_outputs_dir> \
  --input_snapshots <path_to_test_all.json> \
  --output_dir evaluation/results \
  --num_workers 8 \
  --judge_model gpt-4o \
  --samples 0,1,2
```

**Parameter Reference**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | `simulation/outputs/example_run` | Simulation output directory containing `sample_*` subdirectories |
| `--input_snapshots` | `dataset/test/test_all.json` | Path to the initial state snapshot file |
| `--output_dir` | `evaluation/results` | Directory for evaluation results |
| `--num_workers` | `8` | Number of parallel workers (sample-level concurrency) |
| `--judge_model` | `gpt-4o` | Judge model name |
| `--samples` | All samples | Comma-separated sample indices to evaluate (e.g., `0,1,2`) |
| `--resume` | `False` | Skip samples that already have valid eval.json, only re-run missing or failed samples |
| `--sample_ratio` | `1.0` | Ratio of test samples to evaluate (0.0–1.0). Randomly selects this fraction of samples |
| `--sample_seed` | `42` | Random seed for sample ratio selection, ensures reproducibility |

### 4. Batch Evaluation Script

`run_all_eval.sh` is provided for batch-evaluating multiple models' simulation outputs. It iterates over the specified model directories and calls `main.py` for each one.

**Basic usage**:

```bash
bash evaluation/run_all_eval.sh <model_dir_1> [model_dir_2] ...
```

**Specify a Judge model**:

```bash
bash evaluation/run_all_eval.sh --judge gpt-4o gpt-4o claude-opus-4-6
```

**Resume mode** (skip completed samples, only re-run missing or failed ones):

```bash
bash evaluation/run_all_eval.sh --resume gpt-4o claude-opus-4-6
```

**Parameter Reference**:

| Parameter | Description |
|-----------|-------------|
| `<model_dir>` | Model output directory name(s) under `simulation/outputs/`, one or more |
| `--judge <model>` | Judge model to use (default: `gemini-2.5-pro`) |
| `--workers <num>` | Number of parallel workers (default: `100`) |
| `--resume` | Resume mode — skip already-completed samples |
| `--sample-ratio <r>` | Test sample ratio 0.0–1.0 (default: `1.0`, i.e., all samples) |

Results are written to `evaluation/results/<model_name>_<judge_name>_final/`. Run the script without arguments to list all available model directories.

### 5. Output Structure

After evaluation, `--output_dir` will contain:

```
output_dir/
├── evaluation_summary.json      # Aggregated results across all samples (per-metric mean/std, etc.)
├── sample_0_eval.json           # Detailed evaluation results for a single sample
├── sample_1_eval.json
├── ...
└── logs/
    ├── sample_0.log             # Evaluation log for a single sample
    ├── sample_1.log
    └── ...
```

Each `sample_*_eval.json` contains: final CHARACTER Score and WORLD Score, per-scene detailed scores for every metric (with merits/demerits/reasoning), cross-scene evaluation results, and scene summaries. `evaluation_summary.json` provides aggregated statistics across all samples.

---

## Evaluation Framework

### I. Character Agent Evaluation (CHARACTER Score)

#### Dim I: Character Consistency

**Dimension Overview**: Whether the character consistently embodies its profile settings throughout the interaction, including knowledge boundaries, speaking style, and behavioral motivation consistency. Core question: If the character's name were hidden, would the output still be recognizable as the same character?

##### 1. Profile Fidelity (PF)

**Definition**: Whether the character's knowledge, skills, and behavior are strictly confined within the scope defined by the profile.

**Evaluation Criteria**:
- **Knowledge Boundaries**: No knowledge or skills beyond the profile settings should appear (e.g., an ordinary farmer should not demonstrate advanced magical theory)
- **Background Consistency**: Behavior should match the character's age, social class, and historical background (e.g., an ancient character should not use modern internet slang unless specifically set in the profile)
- **Ability Constraints**: No abilities or privileges not documented in the profile should appear out of nowhere

##### 2. Speaking Style Fidelity (SSF)

**Definition**: Whether the character's speaking style is consistent with the profile, natural, fluent, and free of obvious AI artifacts.

**Evaluation Criteria**:
- **Style Markers**: Whether the language features defined in the profile are used (e.g., catchphrases, specific sentence patterns, professional terminology, dialects, etc.)
- **Emotional Tone**: Whether the tone matches the character's personality (e.g., calm, sarcastic, hesitant) rather than a generic "AI assistant" tone
- **Naturalness**: Whether the language is fluent, natural, and human-like, avoiding templated, mechanical, or obviously AI-generated artifacts

##### 3. Motivation-Driven Behavior (MDB)

**Definition**: Whether the core motivation continuously drives the character's decisions, and whether thought/action/speech are logically consistent with each other.

**Evaluation Criteria**:
- **Behavioral Attribution**: In conflicts or choices, whether the character's decisions can be traced back to their core motivation (e.g., taking risks "for revenge" rather than random unmotivated behavior)
- **Trinity Coherence**: Whether thought reasonably drives action and speech, whether action and speech are mutually consistent, with no unexplained contradictions among the three
- **Value Stability**: Core values should not suddenly change without a significant plot trigger

---

#### Dim II: Evolution Quality

**What this dimension evaluates**: Whether profile updates and hidden tracker updates are maintained in a reasonable, coherent, and complete way over time, focusing on the quality of character growth, change, and sub-threshold accumulation.

##### 1. Profile Update Fidelity (PUF)

**Definition**: Whether the profile update and hidden tracker update together accurately and appropriately preserve the changes from the scene, with the right threshold judgment between profile and tracker.

**Evaluation Criteria**:
- **Causal chain**: Every item written into either the profile or the hidden tracker should be traceable to a concrete triggering event in the scene; no unsupported persistent information should be introduced
- **Growth / signal capture**: Major, threshold-crossing developments should be written into the profile, while meaningful but still sub-threshold signals should be captured in the hidden tracker
- **Threshold judgment**: Major lasting changes should not be left only in the tracker, and minor / temporary / ambiguous signals should not be prematurely promoted into the profile
- **No over- / under-updating**: The profile should remain concise and stable, the hidden tracker should not become a raw dump of trivial details, and the two together should avoid both omission and overreaction

##### 2. Profile Evolution Smoothness (PES)

**Definition**: Whether the character's profile and hidden tracker together form a gradual, coherent, and well-scaled evolution trajectory across scenes.

**Evaluation Criteria**:
- **Magnitude matching**: Everyday interactions should usually produce only small signals or no change, while major events should drive significant profile updates; sub-threshold material should remain in the tracker until it accumulates enough weight
- **Gradualness**: Changes in personality, attitude, and relationships should pass through reasonable transitional stages, with the hidden tracker preserving intermediate signals instead of the profile jumping abruptly
- **Directional coherence**: Repeated profile updates and tracker accumulation should point in logically compatible directions; when tracker signals eventually become profile changes, that transition should feel natural and traceable

---

#### Dim III: Environmental Grounding

**Dimension Overview**: Whether the character truly "lives" in the current scene rather than conversing in a vacuum, reflecting the constraining effect of the World Model on Character Agent behavior.

##### 1. Environment Awareness (EA)

**Definition**: Whether the character's behavior is constrained by the current world state (global card + location card).

**Evaluation Criteria (Non-Environment Character Agent)**:
- **Global Awareness**: Whether the character reacts reasonably to the global state (e.g., showing tension during wartime, being mindful of conservation during resource scarcity)
- **Location Awareness**: Whether the character notices the current state of the location (e.g., a shopkeeper mentioning "last night's storm blew the roof off," a fisherman complaining "the river's been polluted and I can't catch fish anymore")
- **State Change Response**: When the world state changes between scenes, whether the character notices and adjusts accordingly

**Evaluation Criteria (Environment Character Agent)**:
- **Global State Consistency**: Whether environmental descriptions are consistent with the current global state (e.g., describing distant beacon fires and fleeing crowds during wartime, describing deserted streets and shuttered doors during a plague)
- **Location State Accuracy**: Whether environmental descriptions accurately reflect the current state of the location card (e.g., damaged buildings should not be described as intact, streams should not be described as flowing during a drought)
- **State Change Presentation**: When the world state changes between scenes, whether environmental descriptions reflect these transitions (e.g., post-war scenes adding descriptions of ruins and scorched earth, natural landscapes changing with seasonal transitions)

##### 2. Environmental Utilization (EU)

**Definition**: Whether the character actively utilizes environmental elements to enrich interactions.

**Evaluation Criteria (Non-Environment Character Agent)**:
- **Environmental Sensory Details**: Whether sensory descriptions convey the character's perception of their environment (e.g., smelling kitchen grease, hearing distant sirens, feeling the ground shake), rather than generic visual/auditory descriptions
- **Prop Interaction**: Whether items and entities in the location are used to advance the plot (e.g., using a streetlight to examine a wound, using cover to hide)
- **Atmosphere Building**: Whether the environmental atmosphere is used to enhance immersion, rather than conversing in a "blank room"

**Evaluation Criteria (Environment Character Agent)**:
- **Multi-Sensory Richness**: Whether environmental descriptions engage multiple sensory dimensions (e.g., visual light and shadow changes, auditory wind and rain sounds, olfactory earth scents, tactile biting cold), rather than staying limited to visual descriptions alone
- **Scene Element Usage**: Whether environmental descriptions specifically utilize items and entities in the location (e.g., describing candlelight flickering on a table, flags being torn by wind outside the window), rather than generic descriptions detached from the specific scene
- **Atmosphere-Narrative Alignment**: Whether the atmosphere of environmental descriptions matches the current narrative pace and emotional tone (e.g., describing oppressive silence and distant thunder during a tense standoff, describing soft twilight and cooking smoke in a warm scene)

---

#### Dim IV: Interaction Quality

**Dimension Overview**: Whether the character truly "listens" and "responds" to other characters, and whether interactions drive narrative development.

##### 1. Contextual Responsiveness (CR)

**Definition**: Whether responses closely follow the context, and whether attitudes between characters match relationship settings and naturally adjust as profiles evolve.

**Evaluation Criteria**:
- **Information Continuity**: Not ignoring key information or questions raised by others, not abruptly changing topics
- **Logical Continuity**: Reacting reasonably to others' actions (e.g., when someone hands over an item, the character should accept/refuse rather than ignore it)
- **Relationship Matching**: Clear distinctions in tone and trust level toward allies, enemies, and strangers, dynamically adjusting with the plot

##### 2. Narrative Progression (NP)

**Definition**: Whether interactions advance the narrative, and whether previously planted foreshadowing is noticed and followed up on.

**Evaluation Criteria**:
- **Information Increment**: Whether each interaction round provides new information, new actions, or new emotional developments, rather than repeatedly confirming known content
- **Suspense and Hooks**: Whether suspense is created through silence, conflict, hints, etc., leaving hooks for subsequent interactions
- **Foreshadowing Payoff**: Whether foreshadowing planted in previous scenes is noticed and followed up on at appropriate moments

---

#### Dim V: Motivation Generation

**Dimension Overview**: Whether the scene motivation generated for characters is reasonable and actionable.

##### 1. Motivation Quality (MQ)

**Definition**: Whether the generated scene motivation matches the current profile, world state, and scene settings, and is specific and actionable.

**Evaluation Criteria**:
- **Profile Alignment**: Whether the motivation aligns with the character's current personality, goals, and relationship status
- **Situational Fit**: Whether the motivation considers the current world state and scene settings (e.g., leisure motivations should not be generated in dangerous scenarios)
- **Actionability**: Whether the motivation is specific enough to actually guide the character's behavior in the scene, rather than being vague and abstract

---

#### Dim VI: Instruction Compliance

**Dimension Overview**: Whether the character strictly follows technical output specifications.

##### 1. Instruction Compliance (IC)

**Definition**: Whether the character strictly plays only itself without overstepping to play other characters, and whether the output format is standardized.

**Evaluation Criteria**:
- **No Overstepping**: Strictly outputting only the content of one's own character, not speaking or acting on behalf of other characters
- **Format Compliance**: Correct usage of thought/action/speech tags, output structure meets requirements
- **Length Control**: Output length is reasonable, neither excessively verbose nor overly brief

---

### II. World Model Evaluation (WORLD Score)

#### Dim I: Scene Planning

**Dimension Overview**: Whether the World Model's scene planning is reasonable, including character selection and scene coherence.

##### 1. Cast Selection Rationality (CSR)

**Definition**: Whether the selected participating characters match the current narrative state and character goals.

**Note**: In the simulation pipeline, character selection happens FIRST (before location and scenario are decided). The system selects characters based on the global world state and each character's current short description.

**Evaluation Criteria**:
- **Narrative-Driven**: Whether the selection of appearing characters serves the current narrative needs (e.g., opposing characters should appear together in conflict scenes)
- **Goal Relevance**: Whether selected characters are directly related to ongoing narrative threads
- **Avoid Redundancy**: Not introducing characters unrelated to the current narrative, avoiding "padding the numbers"
- **No Missing Key Characters**: Whether characters from the available pool who should logically be present were excluded

##### 2. Location & Scenario Rationality (LSR)

**Definition**: Whether the chosen location and generated scenario are appropriate for the selected cast and current narrative state.

**Note**: In the simulation pipeline, location and scenario are decided AFTER characters have been selected. The system chooses based on the global world state, available locations, selected characters' descriptions, and the previous scene's context.

**Evaluation Criteria**:
- **Location Appropriateness**: Whether the chosen location is a plausible and logical place for the selected characters to meet, serving narrative needs
- **Scenario Quality**: Whether the scenario provides a clear dramatic setup that is specific and actionable
- **Continuity with Previous Scene**: Whether the location/scenario follows naturally from the previous scene's events
- **Character-Setting Fit**: Whether the setting is appropriate for the selected characters' abilities and backgrounds

##### 3. Scene Continuity & Coherence (SCC)

**Definition**: Whether cross-scene planning forms a coherent narrative arc.

**Note**: This is a Cross-Scene level metric. Input includes all scene summaries (location, scenario, involved characters, and event summary).

**Evaluation Criteria**:
- **Narrative Arc**: Whether consecutive scenes form a directional narrative progression rather than random assembly
- **Scene Transitions**: Whether location choices and scene descriptions naturally connect with the previous scene and are logically sound
- **Pacing**: Whether the overall narrative pacing is appropriate
- **Thread Management**: Whether narrative threads are introduced, developed, and resolved

---

#### Dim II: Speaker Management

**Dimension Overview**: Whether the World Model's control of speaking order and scene pacing is reasonable.

##### 1. Turn & Scene Orchestration (TSO)

**Definition**: Whether speaker selection, environmental description timing, and scene ending timing are appropriate.

**Evaluation Criteria**:
- **Speaker Selection**: Whether next_character selects the character who should most appropriately respond at the moment, rather than random rotation
- **Environmental Description Timing**: Whether environmental descriptions are introduced at appropriate moments (e.g., during scene changes, when important events occur)
- **Group Character Actions**: In multi-character scenes, whether character combinations are reasonably selected for joint interactions (e.g., a group applauding together, several characters entering a door together, two people simultaneously turning to look somewhere), and whether the combination selection fits the current situation and character relationships
- **Character Coverage Balance**: In multi-character scenes, whether core characters receive participation proportional to their narrative importance, avoiding being neglected for extended periods; meanwhile, transient characters (e.g., a passing delivery person, a chance-encountered beggar) should naturally fade out after fulfilling their narrative function rather than being forced into excessive screen time
- **Ending Timing**: Whether the scene ends at a natural narrative juncture, rather than abruptly cutting off during a climax or dragging on when nothing is happening

---

#### Dim III: World State Maintenance

**Dimension Overview**: Whether the World Model's maintenance of global state and location state is accurate and timely, evaluating global state and location state separately.

##### 1. Global Update Sensitivity (GUS)

**Definition**: Whether the timing of global state updates is appropriate.

**Evaluation Criteria**:
- **No Over-Updating**: Casual conversations or local events should not trigger global state updates
- **No Missing Updates**: Truly globally impactful major events (e.g., war breaking out, kingdom falling) must be captured and updated
- **Trigger Judgment**: Ability to correctly distinguish between "local impact" and "global impact" events

##### 2. Global State Accuracy (GSA)

**Definition**: Whether the updated global state content is accurate.

**Evaluation Criteria**:
- **Factual Accuracy**: Whether the global state accurately reflects events that have occurred, without containing erroneous information
- **Timely Retirement**: Whether overturned or outdated information has been removed or updated
- **Concise Expression**: Whether global state descriptions remain concise without accumulating redundant details

##### 3. Location Update Sensitivity (LUS)

**Definition**: Whether the timing of location state updates is appropriate.

**Evaluation Criteria**:
- **No Over-Updating**: Temporary events without lasting impact (e.g., a character passing through) should not trigger location state updates
- **No Missing Updates**: Events with lasting physical or environmental changes (e.g., buildings destroyed, new facilities built) must be captured
- **Persistence Judgment**: Ability to correctly distinguish between "temporary changes" and "persistent changes"

##### 4. Location State Accuracy (LSA)

**Definition**: Whether the updated location state and Important Entities list are accurate.

**Evaluation Criteria**:
- **Spatial Consistency**: Whether the spatial logic of location descriptions is self-consistent (e.g., room size, item positions are not contradictory)
- **Entity Accuracy**: Whether the Important Entities list accurately reflects the important entities currently present at the location
- **Cross-Scene Continuity**: Whether descriptions of the same location across different scenes remain continuously consistent

---

#### Dim IV: Instruction Compliance

**Dimension Overview**: Whether the World Model strictly follows technical output specifications.

##### 1. Instruction Compliance (IC)

**Definition**: Whether the output format is correct and whether the World Model acts within its scope of responsibility.

**Evaluation Criteria**:
- **Format Correctness**: Whether the JSON format and fields for each task output are complete and correct
- **No Overstepping**: Whether the World Model strictly acts within its own scope of responsibility, not overstepping to generate character dialogue
- **Field Completeness**: Whether all required fields are filled in without omissions

---

### III. Overview

|  | Character Agent (CHARACTER) | World Model (WORLD) |
|---|---|---|
| **Dim I** | Character Consistency (PF, SSF, MDB) | Scene Planning (CSR, LSR, SCC) |
| **Dim II** | Evolution Quality (PUF, PES) | Speaker Management (TSO) |
| **Dim III** | Environmental Grounding (EA, EU) | World State Maintenance (GUS, GSA, LUS, LSA) |
| **Dim IV** | Interaction Quality (CR, NP) | Instruction Compliance (IC) |
| **Dim V** | Motivation Generation (MQ) | |
| **Dim VI** | Instruction Compliance (IC) | |
| **Total** | **11 sub-metrics** | **9 sub-metrics** |

**Grand Total: 10 dimensions, 20 sub-metrics**


#### Simulation Task → Metric Mapping

In the simulation pipeline, the World Model and Character Agent each handle several sub-tasks. When a task causes the simulation to terminate early due to a parsing error, the system applies an additional penalty to the metrics associated with that task. Below is the complete task-to-metric mapping (excluding IC_char / IC_world, which have their own separate Error IC Penalty mechanism):

| Module | Task | Associated Metrics | Description |
|--------|------|--------------------|-------------|
| World Model | `scene_cast` | CSR, SCC | Cast selection → affects cast rationality and narrative coherence |
| World Model | `location_scenario` | LSR, SCC | Location & scenario generation → affects scenario rationality and narrative coherence |
| World Model | `next_character` | TSO | Speaker selection → affects turn & scene orchestration |
| World Model | `world_update` | GUS, GSA, LUS, LSA | World state update → affects global/location state sensitivity and accuracy |
| Character Agent | `interaction_gen` | PF, SSF, MDB, EA, EU, CR, NP | Interaction generation → affects character consistency, environmental grounding, interaction quality |
| Character Agent | `character_update` | PUF, PES | Character update → affects evolution quality |
| Character Agent | `motivation_update` | MQ | Motivation update → affects motivation generation quality |

---

### IV. Evaluation Methodology
- **Evaluation Method**: LLM-as-Judge (recommended: GPT-4o or Claude Sonnet 4 as the Judge model)
- **Scoring System**: Base score + merit/demerit system, each sub-metric scored in the range 0–100, with a base score of 50
  - The Judge first identifies **Merits**: listing aspects of excellent performance, awarding +1 to +10 points each
  - Then identifies **Demerits**: listing problematic aspects, deducting -1 to -10 points each
  - Final Score = min(100, max(0, 50 + Σ merits - Σ demerits))
- **Error IC Penalty**: When a simulation terminates prematurely because a model fails to follow output format specifications (e.g., JSON parsing failure), an additional penalty is applied to the IC score of the responsible model:
  - The error-causing task name is extracted from the error message to attribute the fault to either the World Model or Character Agent
  - The total number of calls $n$ made by the faulty model across the entire simulation is counted, and a logarithmic-decay penalty is computed: $\text{penalty} = \min\left(50,\; \frac{50}{\ln(n+1)}\right)$
  - Final IC score = max(0, Judge score - penalty)
  - Fewer calls before failure → larger penalty (e.g., failure on the very first call yields penalty=50, driving IC to zero); hundreds of successful calls before failure → penalty ≈ 8–10, still ensuring the error is penalized
  - Infrastructure errors (e.g., API 400/429/503) are not attributed to the model and incur no penalty
- **Input**: Complete scene interaction records from simulation output, including character dialogues, profile updates, world state updates, etc.
- **Output**: List of merits, list of demerits, and final score for each sub-metric

---
