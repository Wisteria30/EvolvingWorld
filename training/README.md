## EvolvingWorld Training

This directory contains the training pipeline for the EvolvingWorld SFT experiments based on LLaMA-Factory.

### Directory Overview

- `train_config.yaml`: main configuration for data preparation and training
- `run.sh`: **unified training launcher**
- `common_training.sh`: shared training workflow invoked by `run.sh`
- `prepare_data.py`: merges task data and optionally mixes in Tulu3
- `download_tulu3.py`: download and convert Tulu3 general-domain data
- `plot_training_log.py`: plot training curves
- `accelerate.yaml`: single-GPU accelerate config
- `ds_z2_config.json` / `ds_z3_config.json`: DeepSpeed configs (multi-GPU)

### 1. Prerequisites

```bash
pip install llamafactory pyyaml datasets matplotlib
```

If you plan to use the base model from Hugging Face, make sure the cluster can access the model or that you have already cached it locally.

Grant execute permission to the training scripts once:

```bash
chmod +x training/*.sh
```

### 2. Required Input Data

Before running training, make sure these files already exist under `dataset/train/`:

- `model_a_scene_cast.json`
- `model_a_location_scenario.json`
- `model_a_next_character.json`
- `model_a_world_update.json`
- `model_b_interaction_gen.json`
- `model_b_character_update.json`
- `model_b_motivation_update.json`

These are the task-level ShareGPT files consumed by `prepare_data.py`.

> **model_a = World Model (director)**: handles scene cast planning, location/scenario planning, next-character proposal, and world state updates.
> **model_b = Character Agent (actor)**: handles interaction generation, character state updates, and motivation updates.

### 3. Pre-training Preparation

#### 3.1 Download Tulu3

To mix Tulu3 general-domain data:

```bash
python3 training/download_tulu3.py --max_samples 100000 --seed 42
```

This generates `dataset/tulu3_sft_sharegpt.json`. Then set in `train_config.yaml`:

```yaml
tulu3_path: dataset/tulu3_sft_sharegpt.json
tulu3_ratio: 1.0
```

#### 3.2 Plot curves / Fix per-task eval loss (optional)

After training, use `plot_training_log.py` to visualize loss curves:

```bash
python3 training/plot_training_log.py --log-file training/outputs/<run_dir>/trainer_log.jsonl
```

This project has multiple eval tasks. LLaMA-Factory's default `on_log(...)` only keeps a few fixed keys, dropping per-task metrics like `eval_evolvingworld_*_loss` from `trainer_log.jsonl` (training itself is unaffected — just use the last checkpoint). To see each task's eval loss in the curves, patch the installed LLaMA-Factory source.

Locate `callbacks.py`:

```bash
python3 -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'train', 'callbacks.py'))"
```

In `LogCallback.on_log(...)`, replace the original `logs = dict(...)` with:

```python
latest = dict(state.log_history[-1]) if state.log_history else {}
logs = dict(
    **latest,
    current_steps=self.cur_steps,
    total_steps=self.max_steps,
    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
    elapsed_time=self.elapsed_time,
    remaining_time=self.remaining_time,
)
```

All `eval_*` metrics will then be preserved in `trainer_log.jsonl`, and `plot_training_log.py` can draw per-task eval loss curves.

### 4. Configuration

Edit `train_config.yaml`. Key fields:

- `model_name_or_path`: base model (overridable via `run.sh --model-name-or-path`)
- `template`: `llama3` or `qwen`
- `cutoff_len`: max sequence length
- `do_eval: true` + `eval_holdout_ratio: 0.1`: auto-split eval set from training data
- `deepspeed: training/ds_z2_config.json` for multi-GPU

### 5. Launch with `run.sh`

All experiments now use a single script. Model, mode, GPU, and other options are passed as arguments:

```bash
bash training/run.sh --model <model_a|model_b> --mode <full|balanced|custom> [--with-tulu3] [--gpu N] [OPTIONS]
```

**Required arguments:**

| Argument | Choices | Description |
|---|---|---|
| `--model` | `model_a`, `model_b` | Which model to train |
| `--mode` | `full`, `balanced`, `custom` | Data mixing mode |

**Optional arguments:**

| Argument | Description |
|---|---|
| `--with-tulu3` | Mix Tulu3 general-domain data (requires `tulu3_path` in config) |
| `--gpu N` | GPU index (sets `CUDA_VISIBLE_DEVICES`) |
| `--model-name-or-path PATH` | Override base model in `train_config.yaml` |
| `--config PATH` | Override config file (default: `training/train_config.yaml`) |
| `--accelerate-config PATH` | Override accelerate config (default: `training/accelerate.yaml`) |

Remaining arguments are forwarded to `llamafactory-cli train`.

**Examples:**

```bash
# Model A, full data, GPU 0
bash training/run.sh --model model_a --mode full --gpu 0

# Model B, balanced data, with Tulu3, GPU 5
bash training/run.sh --model model_b --mode balanced --with-tulu3 --gpu 5

# Override base model
bash training/run.sh --model model_a --mode full --gpu 0 --model-name-or-path Qwen/Qwen2.5-7B-Instruct

# Resume from checkpoint
bash training/run.sh --model model_a --mode full --gpu 0 --resume_from_checkpoint path/to/checkpoint
```

Each run writes prepared data into its own subdirectory under `data_output_dir`. The subdirectory is separated by `model`, `mode`, base-model tag, and optional `tulu3_ratio`, so different training runs do not conflict.

### 6. All Predefined Experiments

```bash
# ===== Full data, no Tulu3 =====
bash training/run.sh --model model_a --mode full --gpu 0
bash training/run.sh --model model_b --mode full --gpu 1

# ===== Balanced data, no Tulu3 =====
bash training/run.sh --model model_a --mode balanced --gpu 2
bash training/run.sh --model model_b --mode balanced --gpu 3

# ===== Full data + Tulu3 (set tulu3_path in train_config.yaml first) =====
# bash training/run.sh --model model_a --mode full --with-tulu3 --gpu 4
# bash training/run.sh --model model_b --mode full --with-tulu3 --gpu 5

# ===== Balanced data + Tulu3 =====
# bash training/run.sh --model model_a --mode balanced --with-tulu3 --gpu 6
# bash training/run.sh --model model_b --mode balanced --with-tulu3 --gpu 7
```

By default, only the first two groups (4 experiments) are active. Tulu3 experiments require `tulu3_path` in `train_config.yaml` before uncommenting.

### 7. Output Directories

- **prepared data**: `training/prepared_data/<model>_<mode>_<base_model>/`
- **checkpoints**: `training/outputs/<model>_<mode>_<base_model>_<timestamp>/`, containing checkpoints, a copy of `train_config.yaml`, and `training_command.txt`

### 8. Multi-GPU / DeepSpeed

For multi-GPU training, set in `train_config.yaml`:

```yaml
deepspeed: training/ds_z2_config.json
```

Paths are resolved relative to the `EvolvingWorld/` project root.
