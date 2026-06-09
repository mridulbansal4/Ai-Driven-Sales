"""LiveKit STT plugin backed by local faster-whisper."""

from __future__ import annotations

import asyncio
import logging

from livekit.agents import stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, combine_frames

from pipeline.core.config import settings
from pipeline.whisper.processor import STTProcessor

logger = logging.getLogger(__name__)


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
        return settings.whisper.model

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
