"""LiveKit LLM plugin backed by local Ollama."""

from __future__ import annotations

import logging
import time

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChatMessage, ChoiceDelta
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import shortuuid

from pipeline.ollama.processor import LLMProcessor

logger = logging.getLogger(__name__)

_OLLAMA_ROLES = {'developer': 'system', 'system': 'system', 'user': 'user', 'assistant': 'assistant'}


class LocalOllamaLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        messages: list[dict[str, str]] = []
        for item in self._chat_ctx.items:
            if isinstance(item, ChatMessage) and item.text_content:
                role = _OLLAMA_ROLES.get(item.role, 'user')
                messages.append({'role': role, 'content': item.text_content})

        if not messages:
            logger.warning('Ollama: empty chat context, skipping')
            return

        processor: LLMProcessor = self._llm._processor  # type: ignore[attr-defined]
        logger.info('Ollama thinking (%d messages)...', len(messages))
        t0 = time.perf_counter()
        text = await processor.generate_messages(messages)
        logger.info('Ollama done in %.1fs: %r', time.perf_counter() - t0, text[:80])
        if text:
            self._event_ch.send_nowait(
                ChatChunk(id=shortuuid(), delta=ChoiceDelta(content=text))
            )


class LocalOllamaLLM(llm.LLM):
    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._processor = LLMProcessor(model=model)

    @property
    def model(self) -> str:
        return self._processor.model

    @property
    def provider(self) -> str:
        return 'ollama'

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return LocalOllamaLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._processor.aclose()
