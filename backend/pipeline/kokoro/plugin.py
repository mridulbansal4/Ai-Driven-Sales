"""LiveKit TTS plugin backed by local Kokoro."""

from __future__ import annotations

import asyncio
import io
import logging
import time

import numpy as np
import soundfile as sf
from livekit.agents import tts
from livekit.agents.tts import TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from pipeline.a2f import get_client as get_a2f_client
from pipeline.core.config import settings
from pipeline.kokoro.processor import TTSProcessor

logger = logging.getLogger(__name__)


class KokoroChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        logger.info('Kokoro speaking: %r', self._input_text[:80])
        t0 = time.perf_counter()
        processor: TTSProcessor = self._tts._processor  # type: ignore[attr-defined]
        wav_bytes, sample_rate = await asyncio.to_thread(
            processor.synthesize, self._input_text
        )

        # Fan out the same utterance to NVIDIA ACE Audio2Face. Fire-and-forget
        # so a slow / unreachable A2F server can never stall LiveKit playback.
        if settings.a2f.enabled and wav_bytes:
            asyncio.create_task(get_a2f_client().animate(wav_bytes))

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
