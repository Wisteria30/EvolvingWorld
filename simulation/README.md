# EvolvingWorld Simulation

Simulation pipeline for EvolvingWorld. Reads test snapshots, rolls the story forward scene-by-scene, and saves structured results.

## Quick Start

Run from the `EvolvingWorld/` directory.

**Remote API mode** (single sample):

```bash
python simulation/main.py \
  --input dataset/test/test_all.json \
  --mode remote \
  --world-model gemini-2.5-pro \
  --character-agent-model gemini-2.5-pro \
  --offset 0 --limit 1 \
  --output-dir simulation/outputs/remote
```

**Local vLLM mode** (single sample):

Start two OpenAI-compatible vLLM servers first, one for the world model and one for the character agent. They can use the same checkpoint or two different checkpoints.

```bash
# Terminal 1: world model
 CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --served-model-name world-model

# Terminal 2: character agent
 CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --served-model-name character-agent
```

After both servers are ready, run:

```bash
python simulation/main.py \
  --input dataset/test/test_all.json \
  --mode local \
  --world-base-url http://127.0.0.1:8000/v1 \
  --character-agent-base-url http://127.0.0.1:8001/v1 \
  --offset 0 --limit 1 \
  --output-dir simulation/outputs/local
```

**Parallel execution** (all samples, 8 concurrent):

```bash
python simulation/main.py \
  --input dataset/test/test_all.json \
  --mode remote \
  --world-model gemini-2.5-pro \
  --character-agent-model gemini-2.5-pro \
  --num-workers 8 \
  --output-dir simulation/outputs/remote
```

---

## `main.py` Parameters

Python script supports both **sequential** and **parallel** execution. Use `--num-workers` to control concurrency, `--offset` and `--limit` to select which samples to run, and `--rerun` to automatically retry failed samples.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--input` | **Yes** | — | Path to snapshot input JSON (e.g. `dataset/test/test_all.json`) |
| `--output-dir` | No | `simulation/outputs/default_run` | Root output directory. Each sample creates a sub-folder `sample_NNNNNN/` inside |
| `--mode` | No | `remote` | `remote` (use a configured API endpoint) or `local` (use local vLLM endpoints) |
| `--config-path` | No | `config.json` | Path to the remote API config file |
| `--world-model` | remote mode | `None` | Model name for the world model. **Required in remote mode** |
| `--character-agent-model` | remote mode | `None` | Model name for the character agent. **Required in remote mode** |
| `--world-base-url` | local mode | `None` | OpenAI-compatible base URL for the world model. **Required in local mode** |
| `--character-agent-base-url` | local mode | `None` | OpenAI-compatible base URL for the character agent. **Required in local mode** |
| `--max-scenes` | No | `5` | Max number of scenes to simulate per sample |
| `--max-turns-per-scene` | No | `12` | Max interaction turns within a single scene |
| `--log-model-io` | No | `false` | Whether to print full model prompts/responses into `simulation.log` (`true` / `false`). Task summary lines (task name, response length, elapsed time) are always logged regardless of this setting |
| `--offset` | No | `0` | Starting index into the snapshot list (0-based) |
| `--limit` | No | `None` (all) | Number of samples to run. `None` means run all from offset to end |
| `--num-workers` | No | `1` | Number of parallel workers. `1` = sequential execution; `>1` = use `ProcessPoolExecutor` for parallel execution |
| `--rerun` | No | `false` | Rerun mode: automatically scan `output-dir` for failed/missing samples and rerun them. Ignores `--offset` and `--limit` when enabled |
| `--rerun-error` | No | `false` | When used with `--rerun`, also rerun samples with `stop_reason='error'` (default: skip errors, only rerun in-progress / missing) |
| `--rerun-server-error` | No | `false` | When used with `--rerun`, only rerun samples whose error is caused by server issues (e.g. `socket hang up`, `ECONNRESET`). Mutually exclusive with `--rerun-error` |
| `--server-error-patterns` | No | see below | Custom server error patterns for substring matching (case-insensitive). Default: `socket hang up`, `ECONNRESET`, `ETIMEDOUT`, `ECONNREFUSED`, `Connection reset`, `Connection refused`, `502`, `503`, `504` |
| `--max-tokens` | No | `16384` | Max output tokens per model call. Lower this for models with smaller context windows (e.g. `8192` for Qwen) |
| `--sample-ratio` | No | `1.0` | Randomly sample a fraction of test snapshots (0.0~1.0). E.g. `0.3` runs only 30% of samples. Sampling is applied after `--offset`/`--limit` slicing. Ignored in rerun mode |
| `--seed` | No | `42` | Random seed for `--sample-ratio`. Same ratio + seed guarantees identical sample selection across runs |

### Examples

```bash
# Run all samples sequentially
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro

# Run only sample #5
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --offset 5 --limit 1

# Run samples 10~14 with 5 parallel workers
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --offset 10 --limit 5 --num-workers 5

# Rerun failed samples with 4 parallel workers
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --rerun --num-workers 4 --output-dir simulation/outputs/remote

# Rerun only server-error samples (e.g. socket hang up)
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --rerun --rerun-server-error --num-workers 4 --output-dir simulation/outputs/remote

# Enable full model I/O logging for debugging
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --log-model-io true --offset 0 --limit 1

# Local vLLM mode with Qwen (lower max-tokens to fit 32K context)
python simulation/main.py --input dataset/test/test_all.json --mode local \
  --world-base-url http://127.0.0.1:8000/v1 \
  --character-agent-base-url http://127.0.0.1:8001/v1 \
  --max-tokens 8192 --output-dir simulation/outputs/local

# Randomly sample 30% of test snapshots (seed=42 for reproducibility)
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --sample-ratio 0.3 --seed 42 --num-workers 8 \
  --output-dir simulation/outputs/remote
```

---

## Output Structure

Each sample produces a folder `sample_NNNNNN/` (index matches the test file order):

```
simulation/outputs/remote/
├── sample_000000/
│   ├── meta.json              # Metadata (book_name, stop_reason, etc.)
│   ├── trace.json             # Debug trace of all model calls
│   ├── all_scenes.json        # Scene structure with interactions
│   ├── character_dynamic.json # Character profile evolution history
│   ├── world_dynamic.json     # World/location state update history
│   └── simulation.log         # Log for this sample
├── sample_000001/
│   └── ...
└── ...
```

**Index semantics**

- `source_scene_index`: the absolute scene index in the original book/test source where simulation starts
- `all_scenes.json` → `scenes[*].scene_index`: the relative scene index inside the current simulation run, starting from `0`
- when aligning generated scenes with `ground_truth_scenes`, compare generated scene `k` with original-book scene `source_scene_index + k`

All files are **updated in real-time** during simulation (after each interaction turn), so partial results are available even if the process crashes.

---

## Simulation Pipeline

```
Scene Planning ──→ Scene Execution ──→ State Reflection ──→ (loop)
```

1. **Scene Planning**: world model decides cast → world model picks location & scenario → character agent generates per-character motivation
2. **Scene Execution**: loop of (world model picks next actor → character agent generates interaction → world model updates world state) until scene ends or max turns reached
3. **State Reflection**: character agent updates each character's profile, short_description, and hidden_tracker

---

## Files

| File | Description |
|---|---|
| `main.py` | Entry point, argument parsing, sequential/parallel sample execution, rerun logic, result writing |
| `simulator.py` | Core simulation loop and prompt construction |
| `inference.py` | OpenAI-compatible API client with retry logic |
| `utils.py` | JSON I/O, logging, thought masking, JSON parsing helpers |
