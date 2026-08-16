# Repository Guidelines

## Project Structure & Module Organization

EvolvingWorld is organized as four pipeline stages. `data_construction/` converts raw book JSONL into structured scenes, character/world state, and ShareGPT training data. `training/` contains LLaMA-Factory preparation scripts, YAML configuration, and launchers. `simulation/` runs the world-model and character-agent loop, while `evaluation/` scores generated trajectories with an LLM judge. Documentation artwork lives in `figure/`; downloaded `dataset/` content and generated outputs are intentionally ignored by Git. Run commands from the repository root unless a module README says otherwise.

## Build, Test, and Development Commands

Create the Python 3.10 runtime described in `README.md`, then install the relevant dependency set. Common entry points are:

```bash
python data_construction/transform.py --dir dataset/extracted_data --seed 40
python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model <model> --character-agent-model <model> --offset 0 --limit 1
bash evaluation/run_all_eval.sh --judge <model> <simulation-run-name>
bash training/run.sh --model model_a --mode full --gpu 0
python -m compileall data_construction simulation evaluation training
```

Use `--limit 1` for simulation smoke tests; full construction, simulation, training, and judging can be slow or costly.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, variables, and JSON fields; use `PascalCase` for classes and `UPPER_CASE` for constants. Add type hints to new interfaces and keep CLI parsing in each module's `main.py`. Preserve existing file formats and command-line option spelling. No formatter or linter is configured, so keep imports grouped, lines readable, and changes consistent with adjacent code.

## Testing Guidelines

There is currently no automated test directory or coverage threshold. At minimum, run `compileall` and exercise the changed entry point on the smallest representative input. For pipeline changes, inspect generated JSON for schema stability and deterministic behavior when a seed is provided. Do not commit `simulation/outputs/`, `evaluation/results/`, checkpoints, logs, or downloaded datasets.

## Commit & Pull Request Guidelines

The short project history uses concise, imperative subjects such as `Move pipeline figure to overview`. Keep commits focused on one pipeline concern. Pull requests should explain the affected stage, data/config assumptions, exact validation commands, and any API or model dependencies. Link related issues and include screenshots only for figure, documentation, or plot changes. Never commit real credentials in `config.json`; verify the diff before pushing.
