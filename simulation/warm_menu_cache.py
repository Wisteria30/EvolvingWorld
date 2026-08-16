"""Pre-generate the launcher's Japanese menu preset for the whole dataset.

Fills simulation/menu_cache.json (a committed preset) with, per book:
  - the character name map (English -> Japanese renderings)
  - translated character-encyclopedia descriptions
  - one-line episode teasers
and optionally (--synopses) the full "これまでのあらすじ" translations.

With the preset in place, the launcher's "準備中..." spinners resolve
instantly; runtime translation only fires for entries missing here.
Run from the repo root:

    python simulation/warm_menu_cache.py            # or: task warmup
    python simulation/warm_menu_cache.py --synopses  # also pre-translate synopses
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

if "simulation" not in sys.path:
    sys.path.insert(0, "simulation")
from launcher import INPUT_PATH, MenuTranslator  # noqa: E402
from translator import (  # noqa: E402
    MENU_CACHE_PATH,
    get_location_cards,
    get_location_names,
    get_relation_tags,
)
from utils import load_json  # noqa: E402


def warm_book(
    translator: MenuTranslator,
    book: str,
    entries: list[tuple[int, dict]],
    include_synopses: bool,
) -> str:
    # Character roster is shared across a book's snapshots; source it from the
    # earliest one (the same maps are keyed per book, exactly as the launcher
    # and the runtime display look them up).
    entries = sorted(entries, key=lambda item: item[1].get("scene_index", 0))
    first_snap = entries[0][1]
    char_states = first_snap.get("character_states", {})
    names = list(char_states)

    descs = {n: char_states[n].get("short_description", "") for n in names}
    translator.name_map(book, names)
    translator.translate_map(f"{book}|characters", descs)
    get_relation_tags(translator.client, book, descs)
    loc_descs = first_snap.get("world_state", {}).get("location_descriptions", {})
    get_location_names(translator.client, book, list(loc_descs))
    get_location_cards(translator.client, book, loc_descs)
    synopses = {
        str(global_idx): (snap.get("previous_scene") or {}).get("scenario", "")
        for global_idx, snap in entries
    }
    translator.summarize_map(
        f"{book}|episode_teasers", {k: v for k, v in synopses.items() if v},
    )
    if include_synopses:
        for global_idx, text in synopses.items():
            if text:
                translator.translate_text(f"{book}|{global_idx}|synopsis", text)
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate the launcher menu preset.")
    parser.add_argument("--synopses", action="store_true", help="Also pre-translate full episode synopses (more calls)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel books (default: 6)")
    args = parser.parse_args()

    snapshots = load_json(INPUT_PATH)
    by_book: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, snap in enumerate(snapshots):
        by_book[snap["book_name"]].append((idx, snap))

    translator = MenuTranslator()
    print(f"{len(by_book)} books -> {MENU_CACHE_PATH}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(warm_book, translator, book, entries, args.synopses): book
            for book, entries in by_book.items()
        }
        for future in as_completed(futures):
            done += 1
            try:
                print(f"[{done}/{len(by_book)}] {future.result()}")
            except Exception as exc:  # keep warming the rest
                print(f"[{done}/{len(by_book)}] FAILED {futures[future]}: {exc}")
    print("done.")


if __name__ == "__main__":
    main()
