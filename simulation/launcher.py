"""Interactive game launcher for EvolvingWorld.

A game-style CLI (questionary + rich) that walks the player through
choosing a story, an episode, a character to control, models and run
length, then hands off to simulation/main.py. Every screen offers
"← 戻る"; the flow is a small state machine. Run from the repo root:

    python simulation/launcher.py      # or: task game
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import questionary
from questionary import Choice, Separator, Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if "simulation" not in sys.path:
    sys.path.insert(0, "simulation")
from inference import ClientConfig, OpenAICompatibleClient  # noqa: E402
from translator import JP_TITLES, display_width, get_character_name_map, translate_json_map  # noqa: E402
from utils import load_json  # noqa: E402

INPUT_PATH = Path("dataset/test/test_all.json")
CONFIG_PATH = Path("config.json")
MENU_TRANSLATE_MODEL = "openai/gpt-5.6-terra"

console = Console()

# Sentinel returned by a step when the player chose "← 戻る".
BACK = object()

STYLE = Style([
    ("qmark", "fg:#f5a623 bold"),
    ("question", "bold"),
    ("pointer", "fg:#f5a623 bold"),
    ("highlighted", "fg:#f5a623 bold"),
    ("answer", "fg:#61afef bold"),
    ("separator", "fg:#f5a623 bold"),
    ("count", "fg:#7a7a7a"),
])


SERIES_JP = {"Sherlock Holmes": "シャーロック・ホームズ"}

# Genre sections for the story menu (every test-set book appears exactly once).
GENRES = [
    ("🕵️  ミステリー・怪奇", [
        "The Adventures of Sherlock Holmes (Sherlock Holmes, #3)",
        "The Hound of the Baskervilles (Sherlock Holmes, #5)",
        "The Phantom of the Opera",
        "The Turn of the Screw",
    ]),
    ("🗺️  冒険", [
        "Around the World in Eighty Days",
        "Treasure Island",
        "The Call of the Wild",
        "The Adventures of Huckleberry Finn",
        "The Adventures of Tom Sawyer",
        "The Wind in the Willows",
    ]),
    ("💐 恋愛・結婚", [
        "Pride and Prejudice",
        "Sense and Sensibility",
        "Far From the Madding Crowd",
        "Middlemarch",
        "The Portrait of a Lady",
        "The House of Mirth",
        "Jude the Obscure",
    ]),
    ("🏙  社会・人間ドラマ", [
        "Uncle Tom’s Cabin",
        "The Jungle",
        "Oliver Twist",
        "The Scarlet Letter",
        "Notes from Underground",
        "The Sorrows of Young Werther",
        "The Sun Also Rises",
        "My Ántonia",
    ]),
    ("🧚 幻想・寓話", [
        "Alice’s Adventures in Wonderland - Through the Looking-Glass",
        "Anthem",
        "The Pilgrim's Progress",
    ]),
    ("🎭 戯曲", [
        "A Doll's House",
        "Othello",
        "The Tempest",
    ]),
]

_SERIES_RE = re.compile(r"\((.+?), #\d+\)\s*$")

MODEL_PRESETS = [
    ("anthropic/claude-opus-4-6", "最高品質(コスト高)"),
    ("anthropic/claude-sonnet-4-6", "バランス型"),
    ("openai/gpt-5.6-sol", "OpenAI最上位"),
    ("openai/gpt-5.6-terra", "OpenAI中位(低コスト)"),
]

LENGTH_PRESETS = [
    ("短編", 2, 8, "目安: Opusで$3〜6 / 15〜25分"),
    ("標準", 5, 12, "目安: Opusで$10〜20 / 40〜70分"),
    ("長編", 8, 12, "目安: Opusで$18〜35 / 1.5〜2時間"),
]


def _ask(question: Any) -> Any:
    """Run a questionary prompt; exit gracefully on Ctrl-C/EOF."""
    answer = question.ask()
    if answer is None:
        console.print("\n[dim]また遊びに来てください。[/dim]")
        sys.exit(0)
    return answer


def _select(message: str, choices: list[Any], with_back: bool = True) -> Any:
    if with_back:
        choices = [*choices, Separator(" "), Choice(title="← 戻る", value=BACK)]
    return _ask(questionary.select(message, choices=choices, style=STYLE))


def _pad_display(text: str, width: int) -> str:
    """Pad *text* to a display width (CJK characters count as 2 columns)."""
    return text + " " * max(1, width - display_width(text))


def _ellipsize(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


# Display-width column where the dim episode counts line up.
_TITLE_COL = 36


def _entry_title(name: str, count: str) -> list[tuple[str, str]]:
    """Two-column choice title: work name + right-aligned dim count."""
    return [("class:text", _pad_display(name, _TITLE_COL)), ("class:count", count)]


def _genre_header(label: str) -> Separator:
    line = f"━━ {label} "
    return Separator(line + "━" * max(2, 44 - display_width(line)))


def _series_of(book: str) -> str | None:
    match = _SERIES_RE.search(book)
    return match.group(1) if match else None


def _short_title(book: str) -> str:
    """Japanese title (falls back to the English one, series suffix removed)."""
    return JP_TITLES.get(book, _SERIES_RE.sub("", book).strip())


# ---------------------------------------------------------------------- #
# Menu translation (Japanese labels via the shared cached batch translator)
# ---------------------------------------------------------------------- #

class MenuTranslator:
    """Menu-text translation via the shared batch translator.

    Requires a working gateway config: the launcher refuses to start without
    one (no silent English degradation). Individual entries that fail to
    generate stay visibly missing/untranslated instead of being papered over.
    """

    def __init__(self) -> None:
        cfg = load_json(CONFIG_PATH)
        self._client = OpenAICompatibleClient(ClientConfig(
            label="menu translator",
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key"),
            model_name=MENU_TRANSLATE_MODEL,
            mode="remote",
            extra_headers=cfg.get("extra_headers"),
        ))

    def translate_map(self, cache_key: str, mapping: dict[str, str]) -> dict[str, str]:
        """Translate {key: english} -> {key: japanese} in one cached batch call.

        Returns only the successfully translated keys — callers render missing
        entries as visibly missing.
        """
        return translate_json_map(
            self._client, cache_key, mapping,
            instruction="Translate each value of this JSON object into natural, concise Japanese.",
        )

    def translate_text(self, cache_key: str, text: str) -> str | None:
        """Translate one string; None (not the English source) on failure."""
        return self.translate_map(cache_key, {"text": text}).get("text")

    def summarize_map(self, cache_key: str, mapping: dict[str, str]) -> dict[str, str]:
        """One-line spoiler-free Japanese teasers for episode labels, cached."""
        return translate_json_map(
            self._client, cache_key, mapping,
            instruction=(
                "Each value of this JSON object is a scene description from a novel. "
                "Replace each value with a one-line Japanese teaser of at most 35 "
                "characters, phrased like a game episode subtitle. Do not spoil "
                "later outcomes."
            ),
        )

    def name_map(self, book_name: str, names: list[str]) -> dict[str, str]:
        """Per-book Japanese character-name map (shared cache with the runtime display)."""
        return get_character_name_map(self._client, book_name, names)

    @property
    def client(self) -> OpenAICompatibleClient | None:
        return self._client


# ---------------------------------------------------------------------- #
# Steps (each returns a value, or BACK)
# ---------------------------------------------------------------------- #

def banner() -> None:
    console.print(Panel.fit(
        "[bold yellow]📖  E V O L V I N G   W O R L D[/bold yellow]\n"
        "[dim]名作文学の世界に入り込み、物語を生きるインタラクティブ・ゲーム[/dim]\n"
        "[dim]矢印キーで選択、Enterで決定。Ctrl-C でいつでも終了できます。[/dim]",
        border_style="yellow",
    ))


def _confirm_story(book: str, episode_count: int) -> bool:
    card = Table.grid(padding=(0, 1))
    card.add_row("[bold]題名[/bold]", _short_title(book))
    card.add_row("[bold]原題[/bold]", _SERIES_RE.sub("", book).strip())
    card.add_row("[bold]エピソード[/bold]", f"全{episode_count}話")
    console.print(Panel(card, title="📖 作品情報", border_style="cyan"))
    return _ask(questionary.confirm("この物語をはじめますか?", default=True, style=STYLE))


def pick_story(episodes_by_book: dict[str, int]) -> str:
    """One flat story menu, visually sectioned by genre headers."""
    series_books: dict[str, list[str]] = defaultdict(list)
    for book in episodes_by_book:
        series = _series_of(book)
        if series:
            series_books[series].append(book)
    multi_series = {s: sorted(books) for s, books in series_books.items() if len(books) > 1}
    book_to_series = {b: s for s, books in multi_series.items() for b in books}

    choices: list[Any] = []
    for genre_label, genre_books in GENRES:
        entries: list[Choice] = []
        seen_series: set[str] = set()
        for book in genre_books:
            if book not in episodes_by_book:
                continue
            series = book_to_series.get(book)
            if series:
                if series in seen_series:
                    continue
                seen_series.add(series)
                jp = SERIES_JP.get(series, series)
                entries.append(Choice(
                    title=_entry_title(f"📚 {jp} シリーズ", f"{len(multi_series[series])}作品"),
                    value=("series", series),
                ))
            else:
                entries.append(Choice(
                    title=_entry_title(_short_title(book), f"全{episodes_by_book[book]}話"),
                    value=("book", book),
                ))
        if entries:
            choices.append(Separator(" "))
            choices.append(_genre_header(genre_label))
            choices.extend(entries)

    while True:
        kind, value = _select("物語を選んでください:", choices, with_back=False)

        if kind == "series":
            picked = _select(
                f"{SERIES_JP.get(value, value)} シリーズのどの作品にしますか:",
                [
                    Choice(title=_entry_title(_short_title(b), f"全{episodes_by_book[b]}話"), value=b)
                    for b in multi_series[value]
                ],
            )
            if picked is BACK:
                continue
            book = picked
        else:
            book = value

        if _confirm_story(book, episodes_by_book[book]):
            return book


def pick_episode(snapshots: list[dict[str, Any]], book: str, translator: MenuTranslator) -> Any:
    entries = [(i, s) for i, s in enumerate(snapshots) if s["book_name"] == book]
    entries.sort(key=lambda item: item[1].get("scene_index", 0))

    synopses = {
        str(global_idx): (snap.get("previous_scene") or {}).get("scenario", "")
        for global_idx, snap in entries
    }
    with console.status("[dim]エピソード一覧を準備中...[/dim]"):
        teasers = translator.summarize_map(f"{book}|episode_teasers", {
            k: v for k, v in synopses.items() if v
        })

    choices = []
    for number, (global_idx, snap) in enumerate(entries, start=1):
        teaser = teasers.get(str(global_idx))
        if teaser:
            label = f"「{teaser}」"
        elif not synopses.get(str(global_idx)):
            label = "(前日譚なし — このエピソードから物語が始まる)"
        else:
            label = "(あらすじ未生成 — task warmup で生成可)"
        choices.append(Choice(
            title=[("class:count", f"第{number:>2}話  "), ("class:text", label)],
            value=global_idx,
        ))
    picked = _select("どのエピソードを遊びますか:", choices)
    if picked is BACK:
        return BACK

    synopsis = synopses.get(str(picked), "")
    if synopsis:
        with console.status("[dim]あらすじを翻訳中...[/dim]"):
            synopsis_ja = translator.translate_text(f"{book}|{picked}|synopsis", synopsis)
        if synopsis_ja is not None:
            console.print(Panel(synopsis_ja, title="📜 これまでのあらすじ", border_style="cyan"))
        else:
            console.print("[dim](あらすじの翻訳に失敗したため表示をスキップします)[/dim]")
    return picked


def pick_style() -> Any:
    return _select(
        "遊びかたを選んでください:",
        [
            Choice("🎭 キャラクターになりきってプレイ", value="play"),
            Choice("🍿 観劇モード(すべてAIにまかせて眺める)", value="watch"),
        ],
    )


def pick_character(snap: dict[str, Any], translator: MenuTranslator) -> Any:
    """Return the --play value (name or JSON path), or BACK."""
    char_states = snap.get("character_states", {})
    # Longer profiles ≈ more central characters; surface protagonists first.
    names = sorted(char_states, key=lambda n: -len(char_states[n].get("profile", "")))
    descs = {n: char_states[n].get("short_description", "") for n in names}
    with console.status("[dim]キャラクター図鑑を準備中...[/dim]"), ThreadPoolExecutor(max_workers=2) as pool:
        descs_future = pool.submit(translator.translate_map, f"{snap['book_name']}|characters", descs)
        names_future = pool.submit(translator.name_map, snap["book_name"], names)
        ja_descs, name_map = descs_future.result(), names_future.result()

    def ja_name(name: str) -> str:
        ja = name_map.get(name)
        return f"{ja}({name})" if ja and ja != name else name

    while True:
        choices: list[Any] = [
            Choice(
                title=[
                    ("class:text", _pad_display(ja_name(name), 40)),
                    ("class:count", _ellipsize(ja_descs.get(name, ""), 36)),
                ],
                value=name,
            )
            for name in names
        ]
        choices.append(Choice(title="✍ 新しいキャラクターを作って参加する(JSONファイル)", value="__new__"))
        picked = _select("誰になりきりますか:", choices)
        if picked is BACK:
            return BACK
        if picked == "__new__":
            path = _ask(questionary.path("キャラクター定義JSONのパス:", style=STYLE))
            if Path(path).exists():
                return path
            console.print("[red]ファイルが見つかりません。[/red]")
            continue
        detail = Table.grid(padding=(0, 1))
        detail.add_row("[bold]人物[/bold]", ja_descs.get(picked, descs.get(picked, "")))
        motivation = char_states[picked].get("motivation", "")
        if motivation:
            with console.status("[dim]いまの想いを翻訳中...[/dim]"):
                motivation_ja = translator.translate_text(f"{snap['book_name']}|{picked}|motivation", motivation)
            if motivation_ja is not None:
                detail.add_row("[bold]いまの想い[/bold]", motivation_ja)
            else:
                detail.add_row("[dim]いまの想い[/dim]", "[dim](翻訳に失敗したため表示をスキップします)[/dim]")
        console.print(Panel(detail, title=f"🎭 {ja_name(picked)}", border_style="magenta"))
        if _ask(questionary.confirm("このキャラクターで冒険しますか?", default=True, style=STYLE)):
            return picked


def pick_models() -> Any:
    choices = [Choice(title=f"{mid} — {desc}", value=mid) for mid, desc in MODEL_PRESETS]
    choices.append(Choice(title="カスタム(手入力)", value="__custom__"))
    model = _select("物語を紡ぐAIを選んでください:", choices)
    if model is BACK:
        return BACK
    if model == "__custom__":
        world = _ask(questionary.text("世界モデル(director):", default="anthropic/claude-opus-4-6", style=STYLE))
        char = _ask(questionary.text("キャラクターモデル(actor):", default=world, style=STYLE))
        return world, char
    return model, model


def pick_length() -> Any:
    choices = [
        Choice(title=f"{name}(全{scenes}章・1章あたり最大{turns}ターン)| {cost}", value=(scenes, turns))
        for name, scenes, turns, cost in LENGTH_PRESETS
    ]
    choices.append(Choice(title="カスタム", value="__custom__"))
    picked = _select("物語の長さを選んでください:", choices)
    if picked is BACK:
        return BACK
    if picked == "__custom__":
        scenes = int(_ask(questionary.text("章の数:", default="5", style=STYLE)))
        turns = int(_ask(questionary.text("1章あたり最大ターン数:", default="12", style=STYLE)))
        return scenes, turns
    return picked


def pick_translate_model() -> Any:
    return _select(
        "日本語表示(リアルタイム翻訳)の設定:",
        [
            Choice("openai/gpt-5.6-terra — 標準", value="openai/gpt-5.6-terra"),
            Choice("openai/gpt-5.6-luna — 高速・低コスト", value="openai/gpt-5.6-luna"),
            Choice("翻訳なし(英語のまま)", value="__off__"),
        ],
    )


# ---------------------------------------------------------------------- #
# Launch
# ---------------------------------------------------------------------- #

def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("launcher.py は対話専用です。TTYのある端末から実行してください(スクリプトからは simulation/main.py を直接使用)。")
    if not INPUT_PATH.exists():
        sys.exit("dataset/test/test_all.json が見つかりません。先に `task dataset` を実行してください。")

    banner()
    snapshots = load_json(INPUT_PATH)
    try:
        translator = MenuTranslator()
    except Exception as exc:
        sys.exit(
            f"config.json のゲートウェイ接続設定を読み込めないため起動できません({exc})。\n"
            "設定を修正するか、翻訳なしで遊ぶ場合は simulation/main.py を直接使用してください。"
        )
    episodes_by_book: dict[str, int] = defaultdict(int)
    for snap in snapshots:
        episodes_by_book[snap["book_name"]] += 1

    while True:  # outer loop: "やり直す" from the final confirmation restarts here
        answers: dict[str, Any] = {}
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("book", lambda: pick_story(episodes_by_book)),
            ("offset", lambda: pick_episode(snapshots, answers["book"], translator)),
            ("style", pick_style),
            ("play_arg", lambda: (
                pick_character(snapshots[answers["offset"]], translator)
                if answers["style"] == "play" else None
            )),
            ("models", pick_models),
            ("length", pick_length),
            ("translate", pick_translate_model),
        ]
        index = 0
        while index < len(steps):
            name, step = steps[index]
            result = step()
            if result is BACK:
                index = max(0, index - 1)
                # Stepping back over the character step in watch mode lands on style
                if steps[index][0] == "play_arg" and answers.get("style") != "play":
                    index = max(0, index - 1)
                continue
            answers[name] = result
            index += 1

        book = answers["book"]
        offset = answers["offset"]
        play_arg = answers["play_arg"]
        world_model, char_model = answers["models"]
        scenes, turns = answers["length"]
        translate_model = None if answers["translate"] == "__off__" else answers["translate"]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"simulation/outputs/game_{offset}_{stamp}"

        summary = Table.grid(padding=(0, 2))
        summary.add_row("物語", _short_title(book))
        summary.add_row("遊びかた", f"なりきりプレイ: {play_arg}" if play_arg else "観劇モード")
        summary.add_row("AI", f"world={world_model} / character={char_model}")
        summary.add_row("長さ", f"全{scenes}章(1章あたり最大{turns}ターン)")
        summary.add_row("日本語表示", translate_model or "なし(英語)")
        summary.add_row("冒険の記録", output_dir)
        console.print(Panel(summary, title="⚔  出撃準備", border_style="yellow"))

        decision = _ask(questionary.select(
            "準備はいいですか?",
            choices=[
                Choice("▶ 物語をはじめる", value="start"),
                Choice("↺ 最初から選び直す", value="restart"),
                Choice("✖ やめる", value="quit"),
            ],
            style=STYLE,
        ))
        if decision == "quit":
            console.print("[dim]また遊びに来てください。[/dim]")
            return
        if decision == "restart":
            continue
        break

    cmd = [
        sys.executable, "simulation/main.py",
        "--input", str(INPUT_PATH), "--mode", "remote",
        "--world-model", world_model,
        "--character-agent-model", char_model,
        "--max-scenes", str(scenes), "--max-turns-per-scene", str(turns),
        "--offset", str(offset), "--limit", "1", "--num-workers", "1",
        "--output-dir", output_dir,
    ]
    if translate_model:
        cmd += ["--translate-model", translate_model]
    if play_arg:
        cmd += ["--play", play_arg]

    console.print("\n[bold yellow]物語が始まります……(Ctrl-Cでいつでも中断できます。途中経過は自動保存されます)[/bold yellow]\n")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        console.print(Panel.fit(
            f"[bold]物語はここまで。[/bold]\n冒険の記録: {output_dir}\n"
            f"AIの演技を採点する: task eval RUN={Path(output_dir).name}",
            border_style="yellow",
        ))


if __name__ == "__main__":
    main()
