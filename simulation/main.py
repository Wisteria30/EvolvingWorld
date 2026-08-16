from __future__ import annotations

import argparse
import logging
import math
import os
import random
import signal
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    if "simulation" not in sys.path:
        sys.path.insert(0, "simulation")

    from agents import HumanCliController
    from inference import ClientConfig, OpenAICompatibleClient, discover_single_model
    from simulator import SimulationConfig, StorySimulator
    from translator import StreamingTranslator
    from utils import dump_json, load_json, register_character, setup_logger, split_result_views
else:
    from .agents import HumanCliController
    from .inference import ClientConfig, OpenAICompatibleClient, discover_single_model
    from .simulator import SimulationConfig, StorySimulator
    from .translator import StreamingTranslator
    from .utils import dump_json, load_json, register_character, setup_logger, split_result_views


def _resolve_log_model_io(cli_value: str | None) -> bool:
    if cli_value is None:
        return False
    lowered = cli_value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid --log-model-io value: {cli_value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvolvingWorld simulation from test snapshots.")
    parser.add_argument("--input", required=True, help="Path to snapshot input JSON")
    parser.add_argument("--output-dir", default="simulation/outputs/default_run", help="Directory to save results and traces")
    parser.add_argument("--mode", choices=["remote", "local"], default="remote")
    parser.add_argument("--config-path", default="config.json")
    parser.add_argument("--world-model", default=None, help="World model name (required in remote mode)")
    parser.add_argument("--character-agent-model", default=None, help="Character agent model name (required in remote mode)")
    parser.add_argument("--world-base-url", default=None, help="World model OpenAI-compatible base URL in local mode")
    parser.add_argument("--character-agent-base-url", default=None, help="Character agent OpenAI-compatible base URL in local mode")
    parser.add_argument("--max-scenes", type=int, default=5, help="Maximum number of simulated scenes per sample")
    parser.add_argument("--max-turns-per-scene", type=int, default=12, help="Maximum interaction turns per scene")
    parser.add_argument("--log-model-io", default=None, help="Whether to print full model input/output into simulation.log (true/false). Default: false")
    parser.add_argument("--offset", type=int, default=0, help="Starting index into the snapshot list (used with --limit for per-sample runs)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of input snapshots")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--rerun", action="store_true", help="Rerun mode: automatically scan output-dir for failed/missing samples and rerun them")
    parser.add_argument("--rerun-error", action="store_true", help="When used with --rerun, also rerun samples with stop_reason='error' (default: skip errors)")
    parser.add_argument("--rerun-server-error", action="store_true", help="When used with --rerun, only rerun samples whose error is caused by server issues (e.g. 'socket hang up', 'ECONNRESET', 'timeout'). Mutually exclusive with --rerun-error.")
    parser.add_argument("--server-error-patterns", nargs="+", default=None, help="Custom server error patterns to match against the 'error' field in meta.json (case-insensitive substring match). Default: 'socket hang up', 'ECONNRESET', 'ETIMEDOUT', 'ECONNREFUSED', 'Connection reset', 'Connection refused', 'read ECONNRESET', '502', '503', '504')")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per model call (default: 16384, lower for models with smaller context window e.g. Qwen)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Randomly sample a fraction of test snapshots to run (0.0~1.0, default: 1.0 = all)")
    parser.add_argument("--translate-model", default=None, help="Stream a live translated view of the story to the terminal using this model (e.g. openai/gpt-5.6-terra). Display only: saved outputs stay in the source language. Best with --num-workers 1 and a single sample.")
    parser.add_argument("--play", default=None, help="Play a character interactively via the terminal. Pass an existing character name (e.g. 'Sherlock Holmes') or a path to a JSON file defining a new character ({name, short_description, profile, relationships?, motivation?, hidden_tracker?}) to register into the snapshot. Requires --num-workers 1 and a single sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample-ratio sampling (default: 42)")
    return parser.parse_args()


def build_clients(args: argparse.Namespace, simulation_config: dict[str, Any]) -> tuple[OpenAICompatibleClient, OpenAICompatibleClient, dict[str, Any]]:
    if args.mode == "remote":
        api_key = simulation_config["api_key"]
        base_url = simulation_config["base_url"]
        extra_headers = simulation_config.get("extra_headers")
        if not args.world_model or not args.character_agent_model:
            raise ValueError("Remote mode requires --world-model and --character-agent-model.")

        world_model_config = ClientConfig(
            label="world model",
            base_url=base_url,
            api_key=api_key,
            model_name=args.world_model,
            mode="remote",
            extra_headers=extra_headers,
        )
        character_agent_config = ClientConfig(
            label="character agent",
            base_url=base_url,
            api_key=api_key,
            model_name=args.character_agent_model,
            mode="remote",
            extra_headers=extra_headers,
        )
        trace_meta = {
            "mode": "remote",
            "world_model": {"base_url": base_url, "model_name": args.world_model},
            "character_agent": {"base_url": base_url, "model_name": args.character_agent_model},
        }
    else:
        if not args.world_base_url or not args.character_agent_base_url:
            raise ValueError("Local mode requires --world-base-url and --character-agent-base-url.")
        discovered_world_model = discover_single_model(args.world_base_url)
        discovered_character_agent_model = discover_single_model(args.character_agent_base_url)
        world_model_config = ClientConfig(
            label="world model",
            base_url=args.world_base_url,
            api_key=None,
            model_name=discovered_world_model,
            mode="local",
        )
        character_agent_config = ClientConfig(
            label="character agent",
            base_url=args.character_agent_base_url,
            api_key=None,
            model_name=discovered_character_agent_model,
            mode="local",
        )
        trace_meta = {
            "mode": "local",
            "world_model": {"base_url": args.world_base_url, "model_name": discovered_world_model},
            "character_agent": {"base_url": args.character_agent_base_url, "model_name": discovered_character_agent_model},
        }

    return (
        OpenAICompatibleClient(world_model_config),
        OpenAICompatibleClient(character_agent_config),
        trace_meta,
    )


def _extract_meta(result: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields from a simulation result (excluding scenes/states
    that are already covered by the split view files)."""
    meta = {
        "sample_index": result.get("sample_index"),
        "book_name": result.get("book_name"),
        "source_scene_index": result.get("source_scene_index"),
        "stop_reason": result.get("stop_reason"),
        "story_finished": result.get("story_finished"),
    }
    if "error" in result:
        meta["error"] = result["error"]
    return meta


def _write_split_views(result: dict[str, Any], output_dir: Path) -> None:
    """Write the three split view files alongside the main result."""
    views = split_result_views(result)
    dump_json(views["all_scenes"], output_dir / "all_scenes.json")
    dump_json(views["character_dynamic"], output_dir / "character_dynamic.json")
    dump_json(views["world_dynamic"], output_dir / "world_dynamic.json")


# Default server error matching patterns (case-insensitive substring match)
DEFAULT_SERVER_ERROR_PATTERNS = [
    "socket hang up",
    "ECONNRESET",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "Connection reset",
    "Connection refused",
    "read ECONNRESET",
    "502",
    "503",
    "504",
]


def _is_server_error(error_msg: str, patterns: list[str]) -> bool:
    """Check if error message is a server error (case-insensitive substring match)."""
    error_lower = error_msg.lower()
    return any(p.lower() in error_lower for p in patterns)


def _scan_failed_samples(
    output_dir: Path,
    total_samples: int,
    rerun_error: bool = True,
    rerun_server_error: bool = False,
    server_error_patterns: list[str] | None = None,
) -> list[int]:
    """Scan output_dir for sample indices that need rerunning.

    Criteria:
      1. sample directory does not exist
      2. meta.json does not exist
      3. stop_reason in meta.json is "in_progress"
      4. (only when rerun_error=True) stop_reason in meta.json is "error"
      5. (only when rerun_server_error=True) stop_reason in meta.json is "error"
         and the error field matches server error patterns
    """
    if server_error_patterns is None:
        server_error_patterns = DEFAULT_SERVER_ERROR_PATTERNS

    failed: list[int] = []
    count_no_dir = 0
    count_no_meta = 0
    count_in_progress = 0
    count_error = 0
    count_server_error = 0
    count_error_skipped = 0

    for i in range(total_samples):
        sample_dir = output_dir / f"sample_{i:06d}"

        if not sample_dir.is_dir():
            failed.append(i)
            count_no_dir += 1
            continue

        meta_file = sample_dir / "meta.json"
        if not meta_file.is_file():
            failed.append(i)
            count_no_meta += 1
            continue

        try:
            meta = load_json(meta_file)
            stop_reason = meta.get("stop_reason", "")
        except Exception:
            failed.append(i)
            count_no_meta += 1
            continue

        if stop_reason == "in_progress":
            failed.append(i)
            count_in_progress += 1
        elif stop_reason == "error":
            error_msg = meta.get("error", "")
            if rerun_error:
                # --rerun-error: rerun all errors
                failed.append(i)
                count_error += 1
            elif rerun_server_error and _is_server_error(error_msg, server_error_patterns):
                # --rerun-server-error: only rerun server errors
                failed.append(i)
                count_server_error += 1
            else:
                count_error_skipped += 1

    print(f">>> Scan complete, found  {len(failed)}  samples to rerun:")
    print(f"    - no directory:              {count_no_dir}")
    print(f"    - no meta.json:            {count_no_meta}")
    print(f"    - stop_reason=in_progress: {count_in_progress}")
    if rerun_error:
        print(f"    - stop_reason=error:       {count_error}")
    elif rerun_server_error:
        print(f"    - server error: {count_server_error}")
        print(f"    - other error (skipped):        {count_error_skipped}")
    else:
        print(f"    - stop_reason=error (skipped): {count_error_skipped}")

    return failed


def _resolve_played_character(play_arg: str, snapshot: dict[str, Any]) -> str:
    """Resolve --play into a character name, registering a new character if needed.

    Pattern 1: *play_arg* is the name of a character already in the snapshot.
    Pattern 2: *play_arg* is a path to a JSON file defining a new character
    ({name, short_description, profile, relationships?, motivation?,
    hidden_tracker?}); the character is injected into the snapshot's
    character_states, with relationships folded into the profile so the world
    model and the other character agents can see them.
    """
    path = Path(play_arg)
    if path.suffix.lower() == ".json":
        if not path.exists():
            raise ValueError(f"--play character file not found: {play_arg}")
        return register_character(snapshot, load_json(path))
    if play_arg not in snapshot["character_states"]:
        available = ", ".join(sorted(snapshot["character_states"]))
        raise ValueError(f"--play character '{play_arg}' not found in this snapshot. Available: {available}")
    return play_arg


def run_single_sample(
    sample_index: int,
    snapshot: dict[str, Any],
    args: argparse.Namespace,
    simulation_config: dict[str, Any],
    meta_path: Path | None = None,
    split_views_dir: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    world_model_client, character_agent_client, trace_meta = build_clients(args, simulation_config)

    played_name = None
    if args.play:
        played_name = _resolve_played_character(args.play, snapshot)

    translator = None
    if args.translate_model:
        translator = StreamingTranslator(
            ClientConfig(
                label="display translator",
                base_url=simulation_config["base_url"],
                api_key=simulation_config["api_key"],
                model_name=args.translate_model,
                mode="remote",
                extra_headers=simulation_config.get("extra_headers"),
            ),
            perspective_character=played_name,
        )
        # Cached per-book display data: name map, relation tags, character
        # and location cards (includes a newly registered player character).
        translator.prepare_book_context(snapshot, played_name)
        translator.set_sample_index(sample_index)

    controllers = None
    if played_name:
        # Introspection (?質問) thinks with the character-agent model — the
        # same brain that would play the character.
        controllers = {played_name: HumanCliController(
            played_name,
            translator=translator,
            introspection_client=character_agent_client,
        )}

    simulator = StorySimulator(
        world_model_client=world_model_client,
        character_agent_client=character_agent_client,
        config=SimulationConfig(
            max_scenes=args.max_scenes,
            max_turns_per_scene=args.max_turns_per_scene,
            log_model_io=_resolve_log_model_io(args.log_model_io),
            max_tokens=args.max_tokens,
        ),
        controllers=controllers,
    )

    def _on_progress(intermediate_result: dict[str, Any], intermediate_trace: dict[str, Any]) -> None:
        intermediate_result["sample_index"] = sample_index
        if meta_path is not None:
            dump_json(_extract_meta(intermediate_result), meta_path)
        if split_views_dir is not None:
            _write_split_views(intermediate_result, split_views_dir)
        if translator is not None:
            translator.on_progress(intermediate_result)

    result, _trace = simulator.run_snapshot(snapshot, on_progress=_on_progress)
    result["sample_index"] = sample_index
    return sample_index, result


def _worker(worker_args: tuple) -> int:
    """Top-level worker function for ProcessPoolExecutor.
    
    Completes a single sample in a child process:
    create logger -> run simulation -> handle exceptions -> write result files.
    
    Returns sample_index for the main process to track progress.
    """
    sample_index, snapshot, args_dict, simulation_config, output_dir_str = worker_args

    # Rebuild argparse.Namespace in child process (dict serializes cross-process, Namespace may not)
    args = argparse.Namespace(**args_dict)
    output_dir = Path(output_dir_str)

    sample_dir = output_dir / f"sample_{sample_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    log_file = sample_dir / "simulation.log"
    # Must use "simulation" logger name, since simulator.py and inference.py both use 
    # logging.getLogger("simulation") for output. Each child process has independent logging state.
    logger = setup_logger("simulation", log_file)

    logger.info("Starting sample %s (book: %s)", sample_index, snapshot.get("book_name"))

    m_path = sample_dir / "meta.json"

    try:
        _, result = run_single_sample(
            sample_index, snapshot, args, simulation_config,
            meta_path=m_path, split_views_dir=sample_dir,
        )
    except Exception as exc:
        has_intermediate = m_path.exists() and (sample_dir / "all_scenes.json").exists()
        if has_intermediate:
            try:
                meta = load_json(m_path)
                meta["stop_reason"] = "error"
                meta["error"] = str(exc)
                dump_json(meta, m_path)
            except Exception:
                pass
            logger.exception("Failed sample %s (intermediate progress preserved)", sample_index)
            return sample_index

        result = {
            "sample_index": sample_index,
            "book_name": snapshot.get("book_name"),
            "source_scene_index": snapshot.get("scene_index"),
            "stop_reason": "error",
            "story_finished": False,
            "scenes": [],
            "final_world_state": snapshot.get("world_state"),
            "final_character_states": snapshot.get("character_states"),
            "error": str(exc),
        }
        logger.exception("Failed sample %s", sample_index)

    dump_json(_extract_meta(result), m_path)
    _write_split_views(result, sample_dir)

    logger.info("Finished sample %s: %s", sample_index, result.get("stop_reason", "unknown"))
    return sample_index


def _kill_child_processes(parent_pid: int) -> None:
    """Force kill all child processes of the given parent process via SIGKILL."""
    try:
        import subprocess
        # Use pgrep to find all child processes
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True, text=True, timeout=5,
        )
        child_pids = [int(pid) for pid in result.stdout.strip().split() if pid.strip()]
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # child process has exited
        if child_pids:
            print(f"    killed  {len(child_pids)}  child processes: {child_pids}")
    except Exception as e:
        # If pgrep is unavailable, fall back to pkill
        try:
            os.system(f"pkill -KILL -P {parent_pid}")
        except Exception:
            print(f"    Warning: unable to auto-kill, run manually: pkill -f  'multiprocessing-fork'")


def main() -> None:
    args = parse_args()
    if args.play and args.num_workers != 1:
        raise SystemExit("--play is an interactive session and requires --num-workers 1.")
    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    all_snapshots = load_json(input_path)

    simulation_config = load_json(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Random sampling ──
    sample_ratio = getattr(args, 'sample_ratio', 1.0)
    seed = getattr(args, 'seed', 42)

    # ── Rerun mode: auto-scan failed samples and rerun ──
    if args.rerun:
        total_samples = len(all_snapshots)
        print(f">>> [Rerun mode] input file has  {total_samples}  samples")
        print(f">>> Scanning  {output_dir}  for samples to rerun...")

        failed_offsets = _scan_failed_samples(
            output_dir, total_samples,
            rerun_error=args.rerun_error,
            rerun_server_error=args.rerun_server_error,
            server_error_patterns=args.server_error_patterns,
        )

        if not failed_offsets:
            print(">>> All samples completed successfully, nothing to rerun!")
            return

        print(f">>> Sample indices to rerun: {failed_offsets}")

        # Cleaning old output directories
        print(">>> Cleaning old output directories...")
        for offset in failed_offsets:
            sample_dir = output_dir / f"sample_{offset:06d}"
            if sample_dir.is_dir():
                shutil.rmtree(sample_dir)
                print(f"    Removed: {sample_dir}")

        # Build snapshot list with only failed samples, inject _original_offset
        snapshots = []
        for offset in failed_offsets:
            snap = dict(all_snapshots[offset])  # shallow copy, avoid modifying original data
            snap["_original_offset"] = offset
            snapshots.append(snap)

        # ignore --offset / --limit in rerun mode
        global_offset = 0
    else:
        # ── Normal mode ──
        snapshots = all_snapshots[args.offset:]
        if args.limit is not None:
            snapshots = snapshots[: args.limit]
        global_offset = args.offset

        # Random sampling: after offset/limit slicing, sample by ratio
        if 0.0 < sample_ratio < 1.0:
            total_count = len(snapshots)
            sample_count = max(1, math.ceil(total_count * sample_ratio))
            rng = random.Random(seed)
            # Randomly select indices, keep original order
            sampled_local_indices = sorted(rng.sample(range(total_count), sample_count))
            # Inject _original_offset to preserve original sample number
            sampled_snapshots = []
            for li in sampled_local_indices:
                snap = dict(snapshots[li])
                snap["_original_offset"] = global_offset + li
                sampled_snapshots.append(snap)
            print(f">>> Random sampling: selected {sample_count} from {total_count} sample(s) (ratio={sample_ratio}, seed={seed})")
            snapshots = sampled_snapshots

    num_workers = args.num_workers

    if num_workers > 1:
        # ── Concurrent mode: using ProcessPoolExecutor ──
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Convert args to dict for cross-process serialization
        args_dict = vars(args)

        worker_args_list = []
        for local_idx, snapshot in enumerate(snapshots):
            # Prefer _original_offset from snapshot (rerun), else compute as offset + local_idx
            sample_index = snapshot.pop('_original_offset', global_offset + local_idx)
            worker_args_list.append((
                sample_index, snapshot, args_dict, simulation_config, str(output_dir),
            ))

        print(f"Starting parallel simulation with {num_workers} workers for {len(worker_args_list)} sample(s)...")
        print(f"Main process PID: {os.getpid()}  |  Press Ctrl+C to kill all child processes")

        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = {
            executor.submit(_worker, wa): wa[0]  # wa[0] = sample_index
            for wa in worker_args_list
        }

        def _shutdown_handler(signum, frame):
            """On SIGINT/SIGTERM, cancel all pending futures and force kill child processes."""
            sig_name = signal.Signals(signum).name
            print(f"\n>>> Received  {sig_name}, terminating all child processes...")
            # Cancel all futures that have not started
            for fut in futures:
                fut.cancel()
            # Force shutdown executor (do not wait for child processes)
            executor.shutdown(wait=False, cancel_futures=True)
            # Kill all remaining child processes
            _kill_child_processes(os.getpid())
            print(">>> All child processes terminated.")
            sys.exit(1)

        # Register signal handlers
        old_sigint = signal.signal(signal.SIGINT, _shutdown_handler)
        old_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)

        try:
            completed = 0
            for future in as_completed(futures):
                sample_index = futures[future]
                completed += 1
                try:
                    future.result()
                    print(f"  [{completed}/{len(worker_args_list)}] Sample {sample_index} done.")
                except Exception as exc:
                    print(f"  [{completed}/{len(worker_args_list)}] Sample {sample_index} failed: {exc}")
        except KeyboardInterrupt:
            _shutdown_handler(signal.SIGINT, None)
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            executor.shutdown(wait=False)

        print(f"All {len(snapshots)} sample(s) complete.")

    else:
        # ── Sequential mode: process one by one ──
        for local_idx, snapshot in enumerate(snapshots):
            # Prefer _original_offset from snapshot (rerun), else compute as offset + local_idx
            sample_index = snapshot.pop('_original_offset', global_offset + local_idx)
            sample_dir = output_dir / f"sample_{sample_index:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            log_file = sample_dir / "simulation.log"
            # Sequential runs are watched interactively; mirror progress logs
            # to the terminal too (full detail with timestamps stays in
            # simulation.log).
            logger = setup_logger("simulation", log_file, console_level=logging.INFO)

            logger.info("Starting sample %s (book: %s)", sample_index, snapshot.get("book_name"))

            m_path = sample_dir / "meta.json"

            try:
                _, result = run_single_sample(
                    sample_index, snapshot, args, simulation_config,
                    meta_path=m_path, split_views_dir=sample_dir,
                )
            except Exception as exc:
                has_intermediate = m_path.exists() and (sample_dir / "all_scenes.json").exists()
                if has_intermediate:
                    try:
                        meta = load_json(m_path)
                        meta["stop_reason"] = "error"
                        meta["error"] = str(exc)
                        dump_json(meta, m_path)
                    except Exception:
                        pass
                    logger.exception("Failed sample %s (intermediate progress preserved)", sample_index)
                    continue

                result = {
                    "sample_index": sample_index,
                    "book_name": snapshot.get("book_name"),
                    "source_scene_index": snapshot.get("scene_index"),
                    "stop_reason": "error",
                    "story_finished": False,
                    "scenes": [],
                    "final_world_state": snapshot.get("world_state"),
                    "final_character_states": snapshot.get("character_states"),
                    "error": str(exc),
                }
                logger.exception("Failed sample %s", sample_index)

            dump_json(_extract_meta(result), m_path)
            _write_split_views(result, sample_dir)

            logger.info("Finished sample %s: %s", sample_index, result.get("stop_reason", "unknown"))

        print(f"All {len(snapshots)} sample(s) complete.")


if __name__ == "__main__":
    main()
