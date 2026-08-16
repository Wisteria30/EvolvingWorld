from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    from inference import ClientConfig, OpenAICompatibleClient
    from utils import extract_json_fragment, remove_other_character_thoughts
else:
    from .inference import ClientConfig, OpenAICompatibleClient
    from .utils import extract_json_fragment, remove_other_character_thoughts

logger = logging.getLogger("simulation")

# Display-only translation layer. The simulation state, saved outputs and
# evaluation inputs always stay in the source language; this module only
# affects what is shown in the terminal while a run is in progress.

TARGET_LANGUAGE = "Japanese"

# Number of recent (source, translation) pairs replayed into each request so
# names and terminology stay consistent across turns.
_MAX_CONTEXT_PAIRS = 6

# Japanese titles of the dataset books (also used by the launcher menus).
JP_TITLES = {
    "A Doll's House": "人形の家",
    "Alice’s Adventures in Wonderland - Through the Looking-Glass": "不思議の国のアリス/鏡の国のアリス",
    "Anthem": "アンセム",
    "Around the World in Eighty Days": "八十日間世界一周",
    "Far From the Madding Crowd": "遥か群衆を離れて",
    "Jude the Obscure": "日陰者ジュード",
    "Middlemarch": "ミドルマーチ",
    "My Ántonia": "マイ・アントニーア",
    "Notes from Underground": "地下室の手記",
    "Oliver Twist": "オリバー・ツイスト",
    "Othello": "オセロー",
    "Pride and Prejudice": "高慢と偏見",
    "Sense and Sensibility": "分別と多感",
    "The Adventures of Huckleberry Finn": "ハックルベリー・フィンの冒険",
    "The Adventures of Sherlock Holmes (Sherlock Holmes, #3)": "シャーロック・ホームズの冒険",
    "The Adventures of Tom Sawyer": "トム・ソーヤーの冒険",
    "The Call of the Wild": "野性の呼び声",
    "The Hound of the Baskervilles (Sherlock Holmes, #5)": "バスカヴィル家の犬",
    "The House of Mirth": "歓楽の家",
    "The Jungle": "ジャングル",
    "The Phantom of the Opera": "オペラ座の怪人",
    "The Pilgrim's Progress": "天路歴程",
    "The Portrait of a Lady": "ある婦人の肖像",
    "The Scarlet Letter": "緋文字",
    "The Sorrows of Young Werther": "若きウェルテルの悩み",
    "The Sun Also Rises": "日はまた昇る",
    "The Tempest": "テンペスト",
    "The Turn of the Screw": "ねじの回転",
    "The Wind in the Willows": "たのしい川べ",
    "Treasure Island": "宝島",
    "Uncle Tom’s Cabin": "アンクル・トムの小屋",
}

# ---------------------------------------------------------------------- #
# Terminal typography
# ---------------------------------------------------------------------- #

_WIDTH = 64  # display columns for wrapped prose
_DIM = "\x1b[90m"
_BOLD = "\x1b[1m"
_HEADER = "\x1b[33m"
_RESET = "\x1b[0m"

try:
    import unicodedata

    def display_width(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
except Exception:  # pragma: no cover
    def display_width(text: str) -> int:
        return len(text)


def _wrap_line(text: str, width: int) -> list[str]:
    """CJK-aware hard wrap of a single line to *width* display columns."""
    lines: list[str] = []
    current, current_w = "", 0
    for ch in text:
        w = display_width(ch)
        if current and current_w + w > width:
            lines.append(current)
            current, current_w = "", 0
        current += ch
        current_w += w
    if current:
        lines.append(current)
    return lines or [""]


_SENTENCE_END = re.compile(r"(?<=[。!?!?])")


def format_paragraphs(text: str, indent: str = "  ", width: int = _WIDTH) -> str:
    """Wrap prose into indented paragraphs.

    Paragraphs are taken from blank lines; a paragraph that is still a wall of
    text is additionally split every 3 sentences, so readability does not
    depend on the translation model inserting the breaks.
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        flat = " ".join(line.strip() for line in block.splitlines() if line.strip())
        sentences = [s for s in _SENTENCE_END.split(flat) if s.strip()]
        if len(sentences) > 4:
            for i in range(0, len(sentences), 3):
                paragraphs.append("".join(sentences[i:i + 3]))
        else:
            paragraphs.append(flat)
    wrapped = [
        "\n".join(indent + line for line in _wrap_line(p, width))
        for p in paragraphs if p
    ]
    return "\n\n".join(wrapped)


# Notation segments inside an interaction: [...] thought, (...) action
# (full-width variants included: ［］ = [], （） = ()),
# everything else speech/prose.
_SEGMENT_RE = re.compile(
    r"\[[^\]]*\]"
    r"|［[^］]*］"
    r"|\([^)]*\)"
    r"|（[^）]*）"
)


def format_interaction_body(text: str) -> str:
    """Render an interaction's notation as novel/TRPG-style lines.

    [thought] -> dim 💭 lines, (action) -> dim ▷ lines, plain text -> 「speech」.
    """
    out: list[str] = []

    def emit(kind: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        if kind == "thought":
            prefix, style = "💭 ", _DIM
        elif kind == "action":
            prefix, style = "▷ ", _DIM
        else:
            if not (content.startswith("「") or content.startswith("『")):
                content = f"「{content}」"
            prefix, style = "", ""
        pad = " " * display_width(prefix)
        lines = _wrap_line(content, _WIDTH - display_width(prefix))
        reset = _RESET if style else ""
        out.append(f"  {style}{prefix}{lines[0]}{reset}")
        for line in lines[1:]:
            out.append(f"  {style}{pad}{line}{reset}")

    pos = 0
    for match in _SEGMENT_RE.finditer(text):
        emit("speech", text[pos:match.start()])
        token = match.group(0)
        kind = "thought" if token[0] in ("[", "［") else "action"
        emit(kind, token[1:-1])
        pos = match.end()
    emit("speech", text[pos:])
    return "\n".join(out)


# ---------------------------------------------------------------------- #
# Shared per-book menu cache (committed preset, see `task warmup`)
# ---------------------------------------------------------------------- #

MENU_CACHE_PATH = Path("simulation/menu_cache.json")
_menu_cache: dict[str, Any] | None = None


def _get_menu_cache() -> dict[str, Any]:
    global _menu_cache
    if _menu_cache is None:
        if MENU_CACHE_PATH.exists():
            try:
                _menu_cache = json.loads(MENU_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("メニューキャッシュ %s の読み込みに失敗しました(破損の可能性)", MENU_CACHE_PATH, exc_info=True)
                _menu_cache = {}
        else:
            _menu_cache = {}
    return _menu_cache


def _save_menu_cache() -> None:
    try:
        MENU_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MENU_CACHE_PATH.write_text(
            json.dumps(_get_menu_cache(), ensure_ascii=False, indent=1), encoding="utf-8",
        )
    except Exception:
        logger.warning("メニューキャッシュ %s の書き込みに失敗しました(生成結果は保存されません)", MENU_CACHE_PATH, exc_info=True)


def translate_json_map(
    client: OpenAICompatibleClient | None,
    cache_key: str,
    mapping: dict[str, str],
    instruction: str,
) -> dict[str, str]:
    """Translate the values of *mapping* in one batch call, cached under *cache_key*.

    Returns {key: translated_value}; on failure (or with no client) falls back
    to whatever is cached, else {}.
    """
    cache = _get_menu_cache()
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and set(cached) >= set(mapping):
        return cached
    fallback = cached if isinstance(cached, dict) else {}
    if client is None or not mapping:
        return fallback
    translated = _translate_values(client, mapping, instruction)
    if not translated:
        return fallback
    result = {**fallback, **translated}
    cache[cache_key] = result
    _save_menu_cache()
    return result


def _translate_values(
    client: OpenAICompatibleClient,
    mapping: dict[str, str],
    instruction: str,
) -> dict[str, str]:
    """One batch translation call. On failure the entries stay missing.

    No splitting/degradation: a failed batch is logged loudly and simply not
    cached, so the gap is visible and retried on the next generation run.
    Note: very large batches on reasoning models can exhaust the token budget
    on thinking (finish_reason='length' with empty content) — if that happens,
    the caller should reduce the batch size deliberately, not this function.
    """
    prompt = (
        f"{instruction} "
        "Keep the keys exactly as they are (do not translate keys). "
        "Return ONLY the JSON object, no commentary.\n\n"
        + json.dumps(mapping, ensure_ascii=False)
    )
    try:
        raw = client.chat([{"role": "user", "content": prompt}], max_tokens=16384, max_retries=2)
        parsed = extract_json_fragment(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not a dict")
        return {k: str(parsed.get(k) or v) for k, v in mapping.items()}
    except Exception:
        logger.warning("Batch translation failed for %d entries; leaving them ungenerated", len(mapping), exc_info=True)
        return {}


def get_character_name_map(
    client: OpenAICompatibleClient | None,
    book_name: str,
    names: list[str],
) -> dict[str, str]:
    """Return {english_name: japanese_rendering} for a book's cast, cached.

    Generated once per book with a single batch call (not per turn): standard
    katakana transliterations for personal names, natural Japanese for
    descriptive names ("The Postmaster" -> 郵便局長).
    """
    cache = _get_menu_cache()
    key = f"{book_name}|name_map"
    cached = cache.get(key)
    if isinstance(cached, dict) and set(cached) >= set(names):
        return cached
    fallback = cached if isinstance(cached, dict) else {}
    if client is None or not names:
        return fallback
    prompt = (
        f'These are character names from the novel "{book_name}". '
        "For each name, give the Japanese rendering used in published Japanese "
        "translations of the novel: katakana transliteration for personal "
        "names, natural Japanese for descriptive names (e.g. \"The Postmaster\" "
        "-> \"郵便局長\"). Return ONLY a JSON object mapping each original "
        "name to its Japanese rendering.\n\n"
        + json.dumps(names, ensure_ascii=False)
    )
    try:
        raw = client.chat([{"role": "user", "content": prompt}], max_tokens=16384, max_retries=2)
        parsed = extract_json_fragment(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not a dict")
        result = {**fallback, **{name: str(parsed.get(name) or name) for name in names}}
        cache[key] = result
        _save_menu_cache()
        return result
    except Exception:
        logger.warning("Character name map generation failed for %s", book_name, exc_info=True)
        return fallback


def get_relation_tags(
    client: OpenAICompatibleClient | None,
    book_name: str,
    descriptions: dict[str, str],
) -> dict[str, str]:
    """Short Japanese role/relation tags per character, cached per book.

    e.g. "Miranda" -> "プロスペローの娘", "Barrymore" -> "バスカヴィル館の執事".
    Shown next to character names, TRPG-scenario style, so the reader never
    has to remember who is who.
    """
    return translate_json_map(
        client, f"{book_name}|relation_tags", descriptions,
        instruction=(
            f'Each value describes a character of the novel "{book_name}". '
            "Replace each value with a very short Japanese role/relationship "
            "tag of 4-12 characters, like a dramatis personae entry (e.g. "
            "ミランダの父, ナポリの王子, バスカヴィル館の執事). No sentences."
        ),
    )


def get_location_names(
    client: OpenAICompatibleClient | None,
    book_name: str,
    names: list[str],
) -> dict[str, str]:
    """Japanese renderings of location names, cached per book."""
    return translate_json_map(
        client, f"{book_name}|location_names", {n: n for n in names},
        instruction=(
            f'The keys are location names from the novel "{book_name}". '
            "Set each value to a natural short Japanese rendering of the "
            "location name (e.g. \"Prospero's Cave\" -> プロスペローの洞窟)."
        ),
    )


def get_location_cards(
    client: OpenAICompatibleClient | None,
    book_name: str,
    descriptions: dict[str, str],
) -> dict[str, str]:
    """1-2 sentence Japanese location introductions, cached per book."""
    return translate_json_map(
        client, f"{book_name}|location_cards", descriptions,
        instruction=(
            f'Each value describes a location of the novel "{book_name}". '
            "Replace each value with a compact Japanese introduction of the "
            "place in 1-2 sentences (what it is and its atmosphere), with no "
            "story spoilers."
        ),
    )


_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional literary translator. Translate story text from the "
    'interactive simulation of the book "{book_name}" into {target_language}.\n'
    "Rules:\n"
    "- Output ONLY the translation, no commentary, no source text.\n"
    "- Preserve the line structure of the input (thoughts, speech, actions stay on their own lines).\n"
    "- Keep the [ ] and ( ) notation markers exactly where the source uses them.\n"
    "- Render character names consistently, using their established {target_language} renderings when they exist.\n"
    "- Keep the literary tone of the original novel."
)


class StreamingTranslator:
    """Renders simulation events in the terminal as a readable Japanese story.

    Consumes the same ``intermediate_result`` snapshots that ``on_progress``
    receives, diffs them against what has already been displayed, and prints
    newly appeared scene headers and interactions in a novel/TRPG-style
    layout: dramatis personae, slug-line chapter headers anchored on
    locations, and dialogue with thoughts/actions typographically separated.
    Any translation failure falls back to printing the original text, so the
    simulation itself is never interrupted.
    """

    def __init__(self, client_config: ClientConfig, perspective_character: str | None = None):
        self._client = OpenAICompatibleClient(client_config)
        self._context_pairs: list[tuple[str, str]] = []
        self._book_name: str | None = None
        # Player mode: when set, interactions the player did not take part in
        # are shown with other characters' private thoughts masked out, and
        # the player's own (self-authored) interactions are not re-rendered.
        self._perspective_character = perspective_character
        # Per-book display context, filled by prepare_book_context()
        self._name_map: dict[str, str] = {}
        self._ja_descs: dict[str, str] = {}
        self._relation_tags: dict[str, str] = {}
        self._location_names: dict[str, str] = {}
        self._location_cards: dict[str, str] = {}
        self._sample_index: int | None = None
        # Display cursor / continuity anchors
        self._printed_interactions: dict[int, int] = {}
        self._cast_card_printed = False
        self._introduced: set[str] = set()
        self._seen_locations: set[str] = set()
        self._last_location: str | None = None
        # Player mode: scenes the player's character is not cast in. Their
        # content is never rendered (the character wouldn't know it) — only a
        # chapter marker is shown. The simulation itself still runs fully.
        self._hidden_scenes: set[int] = set()

    def prepare_book_context(self, snapshot: dict[str, Any], played_name: str | None = None) -> None:
        """Load (or lazily generate) all per-book display data.

        With the committed preset (`task warmup`) in place this is pure cache
        reads; generation only fires for entries missing from the preset.
        """
        book = snapshot.get("book_name", "")
        self._book_name = book
        char_states = snapshot.get("character_states", {})
        names = list(char_states)
        descs = {n: char_states[n].get("short_description", "") for n in names}
        self._name_map = get_character_name_map(self._client, book, names)
        self._ja_descs = translate_json_map(
            self._client, f"{book}|characters", descs,
            instruction="Translate each value of this JSON object into natural, concise Japanese.",
        )
        self._relation_tags = get_relation_tags(self._client, book, descs)
        loc_descs = snapshot.get("world_state", {}).get("location_descriptions", {})
        self._location_names = get_location_names(self._client, book, list(loc_descs))
        self._location_cards = get_location_cards(self._client, book, loc_descs)
        if played_name is not None:
            self._perspective_character = played_name

    def set_sample_index(self, sample_index: int) -> None:
        self._sample_index = sample_index

    def display_name(self, name: str) -> str:
        """Japanese rendering of a character name (falls back to the original)."""
        return self._name_map.get(name) or name

    def display_location(self, name: str) -> str:
        """Japanese rendering of a location name (falls back to the original)."""
        return self._location_names.get(name) or name

    def relation_tag(self, name: str) -> str:
        return self._relation_tags.get(name, "")

    def format_block(self, text: str) -> str:
        """Public paragraph formatter (used by the player-turn prompt)."""
        return format_paragraphs(text)

    # ------------------------------------------------------------------ #
    # Translation core
    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT_TEMPLATE.format(
            book_name=self._book_name or "unknown",
            target_language=TARGET_LANGUAGE,
        )

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        for source, translation in self._context_pairs:
            messages.append({"role": "user", "content": source})
            messages.append({"role": "assistant", "content": translation})
        messages.append({"role": "user", "content": text})
        return messages

    def _translate_story(self, text: str) -> str | None:
        """Translate story text with rolling context.

        Returns None on failure. Callers skip rendering the block: the display
        never substitutes untranslated text — the source is always available
        in the saved run files.
        """
        try:
            translation = self._client.chat(
                self._build_messages(text), max_tokens=8192, temperature=0.3, max_retries=2,
            ).strip()
            if not translation:
                raise ValueError("empty translation")
            self._context_pairs.append((text, translation))
            del self._context_pairs[:-_MAX_CONTEXT_PAIRS]
            return translation
        except Exception:
            logger.warning("Display translation failed; skipping this block", exc_info=True)
            sys.stdout.write(f"{_DIM}  (翻訳に失敗したため表示をスキップします — 原文は保存ファイルにあります){_RESET}\n")
            sys.stdout.flush()
            return None

    def _translate_once(self, system_prompt: str, text: str) -> str | None:
        """One quiet contextless translation call; None on failure."""
        try:
            result = self._client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=4096,
                temperature=0.3,
                max_retries=2,
            )
            return result.strip() or None
        except Exception:
            logger.warning("Quiet translation failed", exc_info=True)
            return None

    def translate_quiet(self, text: str) -> str | None:
        """Translate *text* into the target language without printing it."""
        return self._translate_once(self._system_prompt(), text)

    def translate_input(self, text: str) -> str | None:
        """Translate player input into the simulation's source language (English).

        Returns None on translation failure. Callers must NOT fall back to the
        raw input — the simulation's canonical record stays in the source
        language, so untranslated text must never be injected.
        """
        system_prompt = (
            "You are a professional literary translator working on the "
            f'interactive simulation of the book "{self._book_name or "unknown"}". '
            f"The player writes their character's turn in {TARGET_LANGUAGE}; "
            "translate it into English in the literary style of the book.\n"
            "Rules:\n"
            "- Output ONLY the translation, no commentary.\n"
            "- If the input is already entirely in English, output it unchanged.\n"
            "- Preserve the notation: [...] private thoughts, (...) physical "
            "actions, plain text spoken dialogue, each kept on its own line.\n"
            "- Render character names by their canonical English names."
        )
        return self._translate_once(system_prompt, text)

    def _time_of_day(self, scenario: str) -> str | None:
        try:
            raw = self._client.chat(
                [{
                    "role": "user",
                    "content": (
                        "Read this scene description and answer ONLY the time of "
                        "day in Japanese, 2-5 characters (例: 早朝/真昼/午後/夕暮れ/夜/深夜). "
                        "If unclear, answer 不明.\n\n" + scenario
                    ),
                }],
                max_tokens=4096,
                max_retries=1,
            ).strip()
            if not raw or "不明" in raw or len(raw) > 8:
                # The scene genuinely doesn't state a time of day — omit it.
                return None
            return raw
        except Exception:
            # API failure is not "no data": surface it (console mirrors warnings).
            logger.warning("時刻の抽出に失敗したため章見出しの時刻を省略します", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Progress hook / rendering
    # ------------------------------------------------------------------ #

    def on_progress(self, intermediate_result: dict[str, Any]) -> None:
        """Diff *intermediate_result* against the display cursor and print what's new."""
        if self._book_name is None:
            self._book_name = intermediate_result.get("book_name")
        if self._sample_index is None and isinstance(intermediate_result.get("sample_index"), int):
            self._sample_index = intermediate_result["sample_index"]
        for scene in intermediate_result.get("scenes", []):
            scene_index = scene.get("scene_index")
            if scene_index is None:
                continue
            if scene_index not in self._printed_interactions:
                hidden = (
                    self._perspective_character is not None
                    and self._perspective_character not in scene.get("involved_characters", [])
                )
                if hidden:
                    self._hidden_scenes.add(scene_index)
                    self._print_skipped_scene_header(scene)
                else:
                    if not self._cast_card_printed:
                        self._print_title_and_cast(scene)
                    self._print_scene_header(scene)
                self._printed_interactions[scene_index] = 0
            interactions = scene.get("interactions", [])
            printed = self._printed_interactions[scene_index]
            if scene_index not in self._hidden_scenes:
                for interaction in interactions[printed:]:
                    self._print_interaction(interaction)
            self._printed_interactions[scene_index] = len(interactions)

    def _print_skipped_scene_header(self, scene: dict[str, Any]) -> None:
        """Marker for a scene the player is not part of.

        Deliberately reveals nothing — not the location, time, cast or
        content: the player's character isn't there and wouldn't know. The
        full record is in the saved run files for after-play reading.
        """
        chapter = scene.get("scene_index", 0) + 1
        self._out()
        self._out(
            f"{_HEADER}{_BOLD}▌第{chapter}章{_RESET}"
            f"{_DIM} (あなたのいない場面のため省略 — 記録は保存ファイルにあります){_RESET}"
        )
        # The player wasn't tracking the world here: reset the location
        # anchor so the next visible scene introduces its place afresh.
        self._last_location = None

    def _out(self, text: str = "") -> None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def _name_with_tag(self, name: str) -> str:
        tag = self.relation_tag(name)
        return f"{self.display_name(name)}({tag})" if tag else self.display_name(name)

    def _print_title_and_cast(self, first_scene: dict[str, Any]) -> None:
        self._cast_card_printed = True
        book = self._book_name or ""
        title = JP_TITLES.get(book, book)
        teaser = ""
        if self._sample_index is not None:
            teasers = _get_menu_cache().get(f"{book}|episode_teasers", {})
            teaser = teasers.get(str(self._sample_index), "")
        rule = "━" * (_WIDTH // 2)
        self._out()
        self._out(f"{_HEADER}{rule}{_RESET}")
        heading = f"  {title}" + (f" 「{teaser}」" if teaser else "")
        self._out(f"{_HEADER}{_BOLD}{heading}{_RESET}")
        self._out(f"{_HEADER}{rule}{_RESET}")
        cast = first_scene.get("involved_characters", [])
        if not cast:
            return
        self._out()
        self._out(f"  {_HEADER}◆ 登場人物{_RESET}")
        for name in cast:
            marker = "(あなた)" if name == self._perspective_character else ""
            tag = self.relation_tag(name)
            headline = f"  ● {_BOLD}{self.display_name(name)}{_RESET}{marker}"
            if tag and not marker:
                headline += f" —— {tag}"
            self._out(headline)
            desc = self._ja_descs.get(name, "")
            if desc:
                self._out(f"{_DIM}{format_paragraphs(desc, indent='      ', width=_WIDTH - 4)}{_RESET}")
        # The cast card already introduced these characters; first-speech
        # intro lines are only for characters joining in later scenes.
        self._introduced.update(cast)

    def _print_scene_header(self, scene: dict[str, Any]) -> None:
        location = scene.get("location") or ""
        scenario = scene.get("scenario") or ""
        chapter = scene.get("scene_index", 0) + 1
        loc_ja = self._location_names.get(location, location)
        time_label = self._time_of_day(scenario) if scenario else None

        slug = f"第{chapter}章  {loc_ja}"
        if time_label:
            slug += f" —— {time_label}"
        self._out()
        self._out(f"{_HEADER}{_BOLD}▌{slug}{_RESET}")
        if self._last_location is not None:
            if location != self._last_location:
                prev_ja = self._location_names.get(self._last_location, self._last_location)
                self._out(f"{_HEADER}▌{_RESET}{_DIM} (前章: {prev_ja} から移動){_RESET}")
            else:
                self._out(f"{_HEADER}▌{_RESET}{_DIM} (引き続き: {loc_ja}){_RESET}")
        self._last_location = location

        if location and location not in self._seen_locations:
            self._seen_locations.add(location)
            card = self._location_cards.get(location, "")
            if card:
                self._out()
                self._out(f"  {_HEADER}✦ この場所について{_RESET}")
                self._out(f"{_DIM}{format_paragraphs(card)}{_RESET}")

        if scenario:
            translated = self._translate_story(scenario)
            if translated is not None:
                self._out()
                self._out(f"  {_HEADER}◆ 場面{_RESET}")
                self._out(format_paragraphs(translated))

        new_cast = [n for n in scene.get("involved_characters", []) if n not in self._introduced]
        if self._cast_card_printed and new_cast and self._printed_interactions:
            # Scenes after the first: point out who is on stage, tags included
            self._out()
            self._out(f"{_DIM}  登場: " + " / ".join(self._name_with_tag(n) for n in scene.get("involved_characters", [])) + _RESET)
        self._out()

    def _print_interaction(self, interaction: dict[str, Any]) -> None:
        characters = interaction.get("characters", [])
        content = interaction.get("content", "")
        if not content:
            return
        if self._perspective_character is not None:
            if self._perspective_character in characters:
                # The player authored this interaction and already saw it at
                # input time; don't re-render it.
                return
            content = remove_other_character_thoughts(content)
            if not content:
                return

        if characters == ["Environment"]:
            self._out(f"  {_HEADER}✦ 情景{_RESET}")
            translated = self._translate_story(content)
            if translated is not None:
                self._out(f"{_DIM}{format_paragraphs(translated)}{_RESET}")
            self._out()
            return

        # Speaker line first (immediate feedback while translation runs)
        label = " / ".join(f"{_BOLD}{self._name_with_tag(n)}{_RESET}" for n in characters)
        self._out(f"● {label}")
        for name in characters:
            if name not in self._introduced:
                self._introduced.add(name)
                desc = self._ja_descs.get(name, "")
                if desc:
                    self._out(f"{_DIM}{format_paragraphs(desc, indent='    ', width=_WIDTH - 2)}{_RESET}")
        translated = self._translate_story(content)
        if translated is not None:
            self._out(format_interaction_body(translated))
        self._out()
