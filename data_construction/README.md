# EvolvingWorld Data Construction

This folder contains the data construction pipeline used to turn raw books into structured EvolvingWorld data, training samples, and test snapshots.

Run all commands from the `EvolvingWorld/` directory.

## Pipeline Overview

```text
Raw books
  -> structured scene / character / world data
  -> ShareGPT training data + simulation test snapshots
```

The pipeline has two main stages:

1. `main.py` extracts and standardizes structured book data with an LLM.
2. `transform.py` converts the structured data into model training files and test snapshots.

`split.py` and `utils.py` are helper modules used by the two stages.

---

The default raw-book input is:

```text
dataset/original_books_from_gutenberg.jsonl
```

Each JSONL row should contain a book record with fields such as `title`, `author`, and `content`.

---

## Step 1: Extract Structured Book Data

```bash
python data_construction/main.py \
  --input dataset/original_books_from_gutenberg.jsonl \
  --output_dir data \
  --num_workers 8 \
  --model gemini-2.5-pro \
  --candidate_model claude-sonnet-4-5
```

Important arguments:

| Argument | Default | Description |
|---|---|---|
| `--input` | `dataset/original_books_from_gutenberg.jsonl` | Raw book JSONL file |
| `--output_dir` | `data` | Root directory for intermediate structured outputs |
| `--num_workers` | `57` | Number of books processed in parallel |
| `--model` | `gemini-2.5-pro` | Main LLM used for extraction and rewriting |
| `--candidate_model` | `claude-sonnet-4-5` | Fallback/candidate model used when needed |
| `--regenerate` | `false` | Force regeneration even when outputs already exist |

Intermediate outputs under `data/` include:

```text
data/
├── extracted/
├── cleaned/
├── standardized/
├── standardized_merge/
├── standardized_merge_clean/
├── character_profiles_initialization/
├── total_scenes/
├── locations_extracted/
└── world_initialization/
```

The final extracted data used by the next stage is written under `dataset/extracted_data/`:

```text
dataset/extracted_data/
├── scenes/
├── character_dynamic/
└── world_dynamic/
```

---

## Step 2: Build Training and Test Data

```bash
python data_construction/transform.py \
  --dir dataset/extracted_data \
  --seed 40
```

Important arguments:

| Argument | Default | Description |
|---|---|---|
| `--dir` | `dataset/extracted_data` | Root directory containing extracted `scenes`, `character_dynamic`, and `world_dynamic` files |
| `--seed` | `40` | Random seed for reproducible book ordering and sampling |

Outputs are written to `dataset/`:

```text
dataset/
├── train/
│   ├── model_a_scene_cast.json
│   ├── model_a_location_scenario.json
│   ├── model_a_next_character.json
│   ├── model_a_world_update.json
│   ├── model_b_interaction_gen.json
│   ├── model_b_character_update.json
│   ├── model_b_motivation_update.json
│   └── all_tasks_with_details.json
├── test/
│   ├── test_snapshots_id.json
│   ├── test_snapshots_ood.json
│   └── test_all.json
└── book_split.json
```

---

## Data Split

`transform.py` uses a reproducible book-level split:

- 10% of books become OOD test books. They are not used for training; test snapshots are sampled from the later 70% of scenes.
- The remaining books are split into two groups.
- Half of the remaining books are train+test books: the first 70% of scenes are used for training, and the later 30% are used for ID test snapshots.
- The other half are train-only books: all scenes are used for training.

The split is saved to:

```text
dataset/book_split.json
```

---

## Generated Tasks

Model A is the world/director model:

- `scene_cast`: decide whether the next scene exists and select its full cast.
- `location_scenario`: choose the location and write the scene scenario.
- `next_character`: choose the next actor in a scene or end the scene.
- `world_update`: decide and write persistent world-state updates.

Model B is the character-agent model:

- `interaction_gen`: generate the next character/environment interaction.
- `character_update`: update character state after a scene.
- `motivation_update`: generate character motivation for the next scene.

The generated training files use ShareGPT-style conversations and are consumed by the training pipeline.

---

## Notes

- `main.py` is LLM-call heavy and can be expensive on a large book set. Start with a small JSONL subset when debugging.
- Logs are written under `data_construction/`, for example `data_construction/main.log` and `data_construction/utils.log`.
- Intermediate outputs are intentionally kept because later stages depend on them and they are useful for inspection.
