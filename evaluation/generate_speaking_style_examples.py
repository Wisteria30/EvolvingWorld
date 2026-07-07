"""
Randomly sample up to 5 interactions per test-set character from extracted scenes.
Generates speaking_style_examples.json for original book style reference in SSF eval.

Usage:
    python evaluation/generate_speaking_style_examples.py

Output file saved at dataset/test/speaking_style_examples.json
"""

import json
import os
import random
from collections import defaultdict

# Path configuration, relative to project root
TEST_ALL_PATH = "dataset/test/test_all.json"
SCENES_DIR = "dataset/extracted_data/scenes"
OUTPUT_PATH = "dataset/test/speaking_style_examples.json"

MAX_EXAMPLES = 5
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    # 1. Collect all (book_name, character_name) pairs from test_all.json
    print("Loading test_all.json ...")
    with open(TEST_ALL_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    book_chars = defaultdict(set)
    for sample in test_data:
        book = sample["book_name"]
        for char_name in sample.get("character_states", {}).keys():
            book_chars[book].add(char_name)

    total_pairs = sum(len(chars) for chars in book_chars.values())
    print(f"Found {len(book_chars)} books, {total_pairs} unique (book, character) pairs")

    # 2. Iterate books, extract character interactions from extracted scenes
    result = {}  # book_name -> {char_name -> [interaction_content, ...]}

    for book_name, char_names in sorted(book_chars.items()):
        scenes_path = os.path.join(SCENES_DIR, f"{book_name}.json")
        if not os.path.exists(scenes_path):
            print(f"  WARNING: {scenes_path} not found, skipping {book_name}")
            continue

        with open(scenes_path, "r", encoding="utf-8") as f:
            book_data = json.load(f)

        scenes = book_data.get("scenes", [])

        # Collect all interactions per character
        char_interactions = defaultdict(list)
        for scene in scenes:
            for interaction in scene.get("interactions", []):
                if not isinstance(interaction, dict):
                    continue
                characters = interaction.get("characters", [])
                content = interaction.get("content", "")
                if not content:
                    continue
                # Skip Environment monologues
                if characters == ["Environment"]:
                    continue
                # Assign interaction to each participating character
                for char in characters:
                    if char != "Environment" and char in char_names:
                        char_interactions[char].append(content)

        # Randomly sample up to MAX_EXAMPLES per character
        book_result = {}
        for char_name in sorted(char_names):
            all_inters = char_interactions.get(char_name, [])
            if not all_inters:
                continue
            n = min(MAX_EXAMPLES, len(all_inters))
            sampled = random.sample(all_inters, n)
            book_result[char_name] = sampled

        result[book_name] = book_result
        found = len(book_result)
        total = len(char_names)
        print(f"  {book_name}: {found}/{total} characters have speaking examples")

    # 3. Save result
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Statistics
    total_chars_with_examples = sum(len(v) for v in result.values())
    total_examples = sum(
        len(examples)
        for book in result.values()
        for examples in book.values()
    )
    print(f"\nDone! Saved to {OUTPUT_PATH}")
    print(f"  Characters with examples: {total_chars_with_examples}/{total_pairs}")
    print(f"  Total interaction examples: {total_examples}")


if __name__ == "__main__":
    main()
