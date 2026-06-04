"""LiveKit plugins: Whisper STT, Ollama LLM, Kokoro TTS."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time

import numpy as np
import soundfile as sf
from livekit.agents import llm, stt, tts
from livekit.agents.llm import ChatChunk, ChatMessage, ChoiceDelta
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.tts import TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, combine_frames, shortuuid

from pipeline.llm import LLMProcessor
from pipeline.stt import STTProcessor
from pipeline.tts import TTSProcessor

logger = logging.getLogger(__name__)

_OLLAMA_ROLES = {'developer': 'system', 'system': 'system', 'user': 'user', 'assistant': 'assistant'}


class LocalWhisperSTT(stt.STT):
    def __init__(self) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._processor: STTProcessor | None = None

    @property
    def model(self) -> str:
        return os.getenv('WHISPER_MODEL', 'large-v3-turbo')

    @property
    def provider(self) -> str:
        return 'faster-whisper'

    def _processor_or_load(self) -> STTProcessor:
        if self._processor is None:
            self._processor = STTProcessor()
        return self._processor

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frame = combine_frames(buffer)
        processor = self._processor_or_load()
        text = await asyncio.to_thread(
            processor.transcribe, frame.data.tobytes()
        )
        logger.info('Heard (%s): %r', processor.last_language, text)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language=processor.last_language, text=text or '')],
        )


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
    def __init__(self) -> None:
        super().__init__()
        self._processor = LLMProcessor()

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


class KokoroChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        logger.info('Kokoro speaking: %r', self._input_text[:80])
        t0 = time.perf_counter()
        processor: TTSProcessor = self._tts._processor  # type: ignore[attr-defined]
        wav_bytes, sample_rate = await asyncio.to_thread(
            processor.synthesize, self._input_text
        )

        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type='audio/pcm',
            stream=False,
        )

        data, _ = sf.read(io.BytesIO(wav_bytes), dtype='int16')
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)

        output_emitter.push(data.tobytes())
        output_emitter.flush()
        logger.info('Kokoro done in %.1fs (%d bytes)', time.perf_counter() - t0, len(wav_bytes))


class LocalKokoroTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._processor = TTSProcessor()

    @property
    def model(self) -> str:
        return 'kokoro'

    @property
    def provider(self) -> str:
        return 'kokoro'

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )
