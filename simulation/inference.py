from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI

if __package__ is None or __package__ == "":
    from utils import strip_chatcompletions_suffix
else:
    from .utils import strip_chatcompletions_suffix

logger = logging.getLogger("simulation")


@dataclass
class ClientConfig:
    label: str
    base_url: str
    api_key: str | None
    model_name: str
    mode: str


class OpenAICompatibleClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=strip_chatcompletions_suffix(config.base_url),
            timeout=600,
        )
        # Determine parameter format based on model name (no probing to avoid rate limit issues)
        self._use_max_completion_tokens, self._skip_temperature = self._infer_param_style()

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        """Check if this is a reasoning model (requires max_completion_tokens and does not support custom temperature).

        Known reasoning models:
        - OpenAI o-series: o1, o1-mini, o1-pro, o3, o3-mini, o4-mini, etc.
        - GPT-5.x series: gpt-5.1, gpt-5.1-chat, gpt-5.3-chat, etc. (all require max_completion_tokens)
        - Kimi-K2.5 series: Kimi-K2.5, etc. (Moonshot reasoning models)
        """
        name = model_name.lower()
        # o-series reasoning models: o1, o1-mini, o3, o3-mini, o4-mini, etc.
        if re.match(r'^o[1-9]', name):
            return True
        # GPT-5.x series (gpt-5.1, gpt-5.3-chat, etc., all use max_completion_tokens and do not support custom temperature)
        if re.match(r'^gpt-5\.\d', name):
            return True
        # Kimi-K2.5 series (Moonshot reasoning models, use max_completion_tokens, do not support custom temperature)
        if name.startswith('kimi-k2'):
            return True
        return False

    @staticmethod
    def _is_qwen3_series(model_name: str) -> bool:
        """Check if this is a Qwen3 series model (including Qwen3 and Qwen3.5).

        Qwen3 series unified handling:
        - content in messages must be wrapped in [{"type": "text", "text": "..."}] format
        - must pass enable_thinking=False to disable thinking mode
        - hyperparameters use temperature=0.7, top_p=0.8
        Matching examples: Qwen/Qwen3-32B, qwen3-235b-a22b, qwen3.5-plus, Qwen3.5-cgame, etc.
        """
        name = model_name.lower()
        return bool(re.search(r'qwen3', name))

    def _infer_param_style(self) -> tuple[bool, bool]:
        """Infer parameter format based on model name.

        Returns (use_max_completion_tokens, skip_temperature).
        """
        model = self.config.model_name
        self._is_qwen3 = self._is_qwen3_series(model)
        if self._is_qwen3:
            logger.info(
                "[params] model=%s recognized as Qwen3 series -> using max_tokens + multimodal content format + enable_thinking=False + temperature=0.7 + top_p=0.8",
                model,
            )
            return False, False
        if self._is_reasoning_model(model):
            logger.info(
                "[params] model=%s recognized as reasoning model -> using max_completion_tokens, skipping temperature",
                model,
            )
            return True, True
        else:
            logger.info(
                "[params] model=%s recognized as standard model -> using max_tokens + temperature",
                model,
            )
            return False, False

    @staticmethod
    def _wrap_qwen3_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert the content of each message from a plain string to the multimodal list format required by the Qwen3 series.

        Before: {"role": "user", "content": "Hello"}
        After: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        """
        wrapped: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            # If content is already in list format, do not re-wrap
            if isinstance(content, list):
                wrapped.append(msg)
            else:
                wrapped.append({
                    **msg,
                    "content": [{"type": "text", "text": content}],
                })
        return wrapped

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.6,
        max_retries: int = 5,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                # Qwen3 series requires content to be wrapped in multimodal format
                actual_messages = self._wrap_qwen3_messages(messages) if self._is_qwen3 else messages

                request_kwargs: dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": actual_messages,
                }
                # Choose correct token limit based on detection
                if self._use_max_completion_tokens:
                    request_kwargs["max_completion_tokens"] = max_tokens
                else:
                    request_kwargs["max_tokens"] = max_tokens
                # Qwen3 series uses fixed hyperparameters: temperature=0.7, top_p=0.8, enable_thinking=False
                if self._is_qwen3:
                    request_kwargs["temperature"] = 0.7
                    request_kwargs["top_p"] = 0.8
                    request_kwargs["extra_body"] = {
                        **(extra_body or {}),
                        "enable_thinking": False,
                    }
                else:
                    # Decide whether to pass temperature based on detection
                    if not self._skip_temperature:
                        request_kwargs["temperature"] = temperature
                    if extra_body:
                        request_kwargs["extra_body"] = extra_body

                completion = self.client.chat.completions.create(**request_kwargs)
                choice = completion.choices[0]
                if choice.finish_reason == "length":
                    logger.warning(
                        "Response truncated (finish_reason='length') | model=%s, max_tokens=%d",
                        self.config.model_name, max_tokens,
                    )
                return choice.message.content or ""

            except openai.RateLimitError as e:
                error_msg = str(e)
                response_body = getattr(e, 'body', None) or getattr(e, 'response', None)
                retry_after_match = re.search(r'retry_after[\'"]?\s*:\s*(\d+)', error_msg)
                wait_time = int(retry_after_match.group(1)) if retry_after_match else 60
                logger.warning(
                    "Rate limit exceeded (attempt %d/%d) for model=%s. Waiting %ds before retry...\nError: %s\nResponse body: %s",
                    attempt, max_retries, self.config.model_name, wait_time, error_msg, response_body,
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit exceeded after %d attempts for model=%s", max_retries, self.config.model_name)
                    raise

            except openai.InternalServerError as e:
                error_msg = str(e)
                response_body = getattr(e, 'body', None) or getattr(e, 'response', None)
                logger.warning(
                    "Internal server error (attempt %d/%d) for model=%s.\nError: %s\nResponse body: %s",
                    attempt, max_retries, self.config.model_name, error_msg, response_body,
                )
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                else:
                    logger.error("Internal server error after %d attempts for model=%s", max_retries, self.config.model_name)
                    raise

            except Exception as e:
                error_msg = str(e)
                response_body = getattr(e, 'body', None) or getattr(e, 'response', None)
                logger.warning(
                    "Unexpected error (attempt %d/%d) for model=%s.\nError: %s\nResponse body: %s",
                    attempt, max_retries, self.config.model_name, error_msg, response_body,
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    logger.error("All %d attempts failed for model=%s", max_retries, self.config.model_name)
                    raise

        raise RuntimeError(f"All {max_retries} API call attempts failed for model={self.config.model_name}")


def discover_single_model(base_url: str) -> str:
    client = OpenAI(api_key="EMPTY", base_url=strip_chatcompletions_suffix(base_url), timeout=120)
    models = client.models.list()
    ids = [item.id for item in models.data]
    if len(ids) == 1:
        return ids[0]
    elif len(ids) == 2:
        logger.info(
            "Found 2 models at %s: %s, using the second one (LoRA): %s",
            base_url, ids, ids[1],
        )
        return ids[1]
    else:
        raise ValueError(
            f"Expected 1 or 2 served models at {base_url}, but found {len(ids)}: {ids}"
        )
