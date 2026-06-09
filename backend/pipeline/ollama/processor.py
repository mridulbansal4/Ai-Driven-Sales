"""Async client for a local Ollama chat completion endpoint.

The processor holds a long-lived :class:`httpx.AsyncClient`. Call
:meth:`aclose` on shutdown to release the connection pool.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from pipeline.core.config import settings

logger = logging.getLogger(__name__)

_OUT_OF_CHARACTER = re.compile(
    r"language model|as an ai|i'?m an ai|openai|assistant here to help|"
    r"i cannot|i can't help|how can i assist",
    re.IGNORECASE,
)

_FALLBACK = "I'm not sure I follow — could you explain that again in simpler terms?"


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama itself is unreachable (network/process down)."""


class OllamaModelMissingError(RuntimeError):
    """Raised when the configured model has not been pulled."""


class LLMProcessor:
    """Thin async wrapper around Ollama's ``/api/chat`` endpoint."""

    def __init__(self, model: str | None = None) -> None:
        self.ollama_url: str = settings.ollama.url
        self.model: str = (model or settings.ollama.model).strip() or settings.ollama.model
        self._timeout = httpx.Timeout(
            settings.ollama.request_timeout_s,
            connect=min(10.0, settings.ollama.request_timeout_s),
        )
        self._client = httpx.AsyncClient(base_url=self.ollama_url, timeout=self._timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "user")
            if role == "developer":
                role = "system"
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                conversation.append({"role": role, "content": content})

        if not system_parts:
            system_parts.append(
                "You are a banking customer. The user is the bank officer. "
                "Stay in character. Reply in 2-3 sentences."
            )

        out: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        out.extend(conversation)
        return out

    @staticmethod
    def _clean_reply(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if _OUT_OF_CHARACTER.search(text):
            return ""
        return text

    async def _post_chat(self, body: dict[str, Any]) -> str:
        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.ollama_url}. "
                "Start it with: `ollama serve`."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaUnavailableError(
                f"Ollama at {self.ollama_url} timed out after "
                f"{settings.ollama.request_timeout_s:.0f}s."
            ) from e

        if resp.status_code == 404:
            text = resp.text.lower()
            if "model" in text and ("not found" in text or "no such" in text):
                raise OllamaModelMissingError(
                    f"Ollama model {self.model!r} is not pulled. "
                    f"Run: `ollama pull {self.model}`."
                )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP %s: %s", e.response.status_code, e.response.text)
            raise

        payload = resp.json()
        return self._clean_reply(payload["message"]["content"])

    async def generate_messages(self, messages: list[dict[str, str]]) -> str:
        body_messages = self._build_messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": body_messages,
            "stream": False,
            "options": {
                "temperature": settings.ollama.temperature,
                "num_predict": settings.ollama.max_tokens,
            },
        }

        try:
            text = await self._post_chat(body)
            if text:
                return text

            # One retry with a stricter reminder when the first reply broke character.
            retry_messages = body_messages + [
                {
                    "role": "user",
                    "content": (
                        "[Stay in character as the banking customer only. "
                        "Reply in 2-3 sentences to what the officer just said.]"
                    ),
                }
            ]
            text2 = await self._post_chat({**body, "messages": retry_messages})
            return text2 or _FALLBACK

        except OllamaUnavailableError as e:
            logger.error("%s", e)
        except OllamaModelMissingError as e:
            logger.error("%s", e)
        except httpx.HTTPError as e:
            logger.error("Ollama HTTP error: %s", e)
        return _FALLBACK
