from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

if __package__ is None or __package__ == "":
    from introspection import IntrospectionSession
    from translator import format_paragraphs
else:
    from .introspection import IntrospectionSession
    from .translator import format_paragraphs

# RL-style agent/environment split for playable simulations.
#
# The environment is StorySimulator: it owns the world model, the other
# character agents, and all hidden state. A CharacterController is the policy
# for one character. Each time the world model selects a controlled character
# to act, the environment builds a masked observation (see
# StorySimulator.build_controller_observation) and calls ``act`` on the
# controller, which returns the action as free-form interaction text in the
# simulation's source language (English), using the corpus conventions:
# ``[...]`` private thought, ``(...)`` physical action, plain text speech.
#
# HumanCliController (a terminal-driven human) is one policy implementation;
# a learned game agent implements the same interface. Rewards are external to
# the environment: the evaluation pipeline scores finished runs, so a
# training loop can wrap StorySimulator + evaluation into a classic
# observation/action/reward cycle without touching the simulator.

_DIM = "\x1b[90m"
_RESET = "\x1b[0m"


class CharacterController(ABC):
    """Policy interface: one character's controller inside the simulation."""

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> str:
        """Return the interaction text for this turn.

        ``observation`` contains only what the character can legitimately
        see: book/scene framing, the character's own full state, other
        characters' public descriptions, and the scene's interaction history
        with other characters' private thoughts masked out.
        """

    def decide_scene_entry(self, observation: dict[str, Any]) -> str | None:
        """Decide whether to barge into a scene this character was not cast in.

        Called after the director has planned the scene (cast + location are
        in ``observation``; the scenario stays hidden). Return None to stay
        out (the scene runs without the character), or the entry reason as
        interaction-language text — it is injected as the character's
        motivation for the scene, so the world reacts to why they came.
        Default: stay out.
        """
        return None


class HumanCliController(CharacterController):
    """A human player acting through the terminal.

    The surrounding story context is already streamed to stdout by the
    display layer (translator / progress logs); this controller renders the
    decision prompts, collects multi-line input (finished with an empty
    line), and optionally translates it into the simulation's source
    language via a StreamingTranslator.

    Introspection substream: at any decision prompt, a line starting with
    ``?`` opens a read-only Q&A with the character's inner voice
    (IntrospectionSession), bounded by the current observation. It is fully
    separate from the action stream — nothing asked there enters the
    simulation or its records.
    """

    def __init__(
        self,
        character_name: str,
        translator: Any | None = None,
        introspection_client: Any | None = None,
    ):
        self.character_name = character_name
        self.translator = translator
        self.introspection_client = introspection_client
        self._last_motivation: str | None = None

    # ------------------------------------------------------------------ #
    # Introspection substream
    # ------------------------------------------------------------------ #

    def _handle_introspection(
        self,
        line: str,
        session: IntrospectionSession | None,
        observation: dict[str, Any],
    ) -> IntrospectionSession | None:
        """Process a ``?`` command line; returns the (possibly new) session."""
        write = sys.stdout.write
        question = line.lstrip("?").lstrip("?").strip()
        if not question:
            write("  (使い方: ?質問文 — あなたのキャラクターが知る範囲で答えるQ&Aです)\n")
            return session
        if self.introspection_client is None:
            write("  (内省は利用できません: introspection用クライアント未設定)\n")
            return session
        if session is None:
            session = IntrospectionSession(
                self.introspection_client,
                self.character_name,
                observation.get("book_name"),
                observation,
            )
        answer = session.ask(question)
        if answer is None:
            write(f"{_DIM}  ┊ (内省に失敗しました — もう一度どうぞ){_RESET}\n")
        else:
            write(f"{_DIM}  ┊ 💭 内省{_RESET}\n")
            body = format_paragraphs(answer, indent="  ┊ ", width=60)
            write(f"{_DIM}{body}{_RESET}\n")
        sys.stdout.flush()
        return session

    # ------------------------------------------------------------------ #
    # Decision prompts
    # ------------------------------------------------------------------ #

    def act(self, observation: dict[str, Any]) -> str:
        self._print_turn_prompt(observation)
        write = sys.stdout.write
        session: IntrospectionSession | None = None
        while True:
            try:
                first = input("> ")
            except EOFError:
                first = ""
            stripped = first.strip()
            if stripped.startswith(("?", "?")):
                session = self._handle_introspection(stripped, session, observation)
                continue
            if not stripped:
                write("(入力が空です。もう一度どうぞ。?質問 で内省もできます)\n")
                continue
            lines = [first, *self._read_continuation()]
            text = "\n".join(lines).strip()
            if self.translator is None:
                # No translation layer configured: the player writes in the
                # simulation's source language directly.
                return text
            translated = self.translator.translate_input(text)
            if translated is None:
                # Never inject untranslated text into the simulation state —
                # the canonical record must stay in the source language.
                write("(翻訳に失敗しました。もう一度入力してください)\n")
                continue
            if translated != text:
                write(f"  ↳ {translated}\n")
                sys.stdout.flush()
            return translated

    def decide_scene_entry(self, observation: dict[str, Any]) -> str | None:
        write = sys.stdout.write
        chapter = observation.get("scene_index", 0) + 1
        cast = observation.get("involved_characters", [])
        location = observation.get("location") or ""
        if self.translator is not None:
            cast_label = " / ".join(self.translator.display_name(n) for n in cast)
            location_label = self.translator.display_location(location)
        else:
            cast_label = " / ".join(cast)
            location_label = location
        write("\n" + "─" * 60 + "\n")
        write(f"▌第{chapter}章 に あなた({self._display_name(self.character_name)})は配役されていません\n")
        write(f"▌ 場所: {location_label} / 出演者: {cast_label}\n")
        write("─" * 60 + "\n")
        write("  介入しますか? 乗り込む場合は y、見送る場合はそのままEnter。?質問 で内省できます。\n")
        session: IntrospectionSession | None = None
        while True:
            try:
                choice = input("> ").strip()
            except EOFError:
                return None
            if choice.startswith(("?", "?")):
                session = self._handle_introspection(choice, session, observation)
                continue
            lowered = choice.lower()
            if lowered in ("", "n", "no"):
                return None
            if lowered in ("y", "yes"):
                break
            write("  y か Enter で答えてください(?質問 で内省)。\n")
        write("  なぜ・どうやってこの場面に現れますか?(この章のあなたの動機になります。空行で送信)\n")
        while True:
            try:
                first = input("> ")
            except EOFError:
                return None
            stripped = first.strip()
            if stripped.startswith(("?", "?")):
                session = self._handle_introspection(stripped, session, observation)
                continue
            if stripped.lower() == "n":
                return None
            if not stripped:
                write("(入力が空です。見送る場合は n、内省は ?質問)\n")
                continue
            reason = "\n".join([first, *self._read_continuation()]).strip()
            if self.translator is None:
                return reason
            translated = self.translator.translate_input(reason)
            if translated is None:
                write("(翻訳に失敗しました。もう一度入力してください)\n")
                continue
            if translated != reason:
                write(f"  ↳ {translated}\n")
                sys.stdout.flush()
            return translated

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_continuation() -> list[str]:
        """Collect further input lines until an empty line (or EOF)."""
        lines: list[str] = []
        while True:
            try:
                line = input("  ")
            except EOFError:
                break
            if not line.strip():
                break
            lines.append(line)
        return lines

    def _display_name(self, name: str) -> str:
        if self.translator is not None:
            return self.translator.display_name(name)
        return name

    def _print_turn_prompt(self, observation: dict[str, Any]) -> None:
        write = sys.stdout.write
        write("\n" + "─" * 60 + "\n")
        co_actors = [n for n in observation.get("acting_characters", []) if n != self.character_name]
        header = f"▶ あなたの番 —— {self._display_name(self.character_name)}"
        if co_actors:
            header += f"(共同行動: {', '.join(self._display_name(n) for n in co_actors)})"
        write(header + "\n" + "─" * 60 + "\n")
        motivation = observation.get("self_state", {}).get("motivation", "")
        if motivation and motivation != self._last_motivation:
            self._last_motivation = motivation
            if self.translator is not None:
                translated = self.translator.translate_quiet(motivation)
                if translated is None:
                    write("  (翻訳に失敗したため動機の表示をスキップします — 原文は保存ファイルにあります)\n")
                else:
                    write("  ◆ いまの想い\n" + self.translator.format_block(translated) + "\n\n")
            else:
                write("  ◆ いまの想い\n  " + motivation + "\n\n")
        write("  記法: [内心] / (動作) / 素の文=セリフ ※空行で送信。?質問 で内省。\n")
        sys.stdout.flush()
