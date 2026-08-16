from __future__ import annotations

import json
import logging
from typing import Any

if __package__ is None or __package__ == "":
    from inference import OpenAICompatibleClient
else:
    from .inference import OpenAICompatibleClient

logger = logging.getLogger("simulation")

# Introspection substream: a read-only inner-monologue Q&A the player can use
# to sharpen their understanding before acting. It is deliberately a separate
# API from the action stream — nothing asked or answered here touches the
# simulation state, the canonical record, or any other character's context.

_SYSTEM_PROMPT = (
    'You are the inner voice of {name}, a character in "{book}".\n'
    "The OBSERVATION below is everything the character currently knows, sees "
    "and thinks: their own full state and the visible scene (other "
    "characters' private thoughts are absent because the character cannot "
    "know them).\n"
    "The player controlling {name} asks you questions to sharpen their "
    "understanding before deciding how to act.\n"
    "Rules:\n"
    "- Answer as the character's introspection, in first person, strictly "
    "within the OBSERVATION.\n"
    "- If the character could not know the answer, say so plainly.\n"
    "- Conjecture is allowed but must be clearly framed as the character's "
    "own speculation.\n"
    "- Never invent facts about the world beyond the OBSERVATION.\n"
    "- Answer in Japanese, concisely (a few sentences).\n\n"
    "OBSERVATION:\n{observation}"
)


class IntrospectionSession:
    """Inner-monologue Q&A over a single controller observation.

    One session spans one decision prompt (the observation is a snapshot of
    that moment); questions within the session share context so follow-ups
    work. Failures return None — nothing is ever substituted or recorded.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        character_name: str,
        book_name: str | None,
        observation: dict[str, Any],
    ):
        self._client = client
        self._messages: list[dict[str, str]] = [{
            "role": "system",
            "content": _SYSTEM_PROMPT.format(
                name=character_name,
                book=book_name or "unknown",
                observation=json.dumps(observation, ensure_ascii=False, indent=1),
            ),
        }]

    def ask(self, question: str) -> str | None:
        """Ask one question; returns the Japanese answer, or None on failure."""
        self._messages.append({"role": "user", "content": question})
        try:
            answer = self._client.chat(
                self._messages, max_tokens=4096, temperature=0.6, max_retries=2,
            ).strip()
            if not answer:
                raise ValueError("empty answer")
            self._messages.append({"role": "assistant", "content": answer})
            return answer
        except Exception:
            logger.warning("Introspection query failed", exc_info=True)
            self._messages.pop()  # keep history consistent for the next ask
            return None
