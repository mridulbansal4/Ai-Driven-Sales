"""Async gRPC client for ``nvidia_ace.services.a2f.v1.A2FService``.

The Kokoro plugin calls :meth:`Audio2FaceClient.animate` with the WAV bytes it
just produced. We decode -> mono int16 -> resample to ACE's expected rate,
then drive the ``PushAudioStream`` client-streaming RPC: one
``AudioStreamHeader`` followed by N ``AudioWithEmotion`` chunks; ACE returns
a single ``Status``.

The client is a process-wide singleton (``get_client()``) — one persistent
HTTP/2 channel reused across utterances. Fan-out is fire-and-forget from the
TTS plugin's perspective so a slow / unreachable A2F server never stalls
LiveKit audio.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf

from pipeline.core.config import A2FSettings, settings

logger = logging.getLogger(__name__)


# Lazy proto imports — these only resolve once ``protoc`` has generated the
# stubs into ``pipeline/a2f/proto/``. See ``pipeline/a2f/proto/README.md``
# (or the docstring in ``proto/__init__.py``) for the exact command.
def _load_proto():
    from pipeline.a2f.proto import nvidia_ace_a2f_v1_pb2 as a2f_pb2
    from pipeline.a2f.proto import nvidia_ace_animation_id_v1_pb2 as anim_id_pb2
    from pipeline.a2f.proto import nvidia_ace_audio_v1_pb2 as audio_pb2
    from pipeline.a2f.proto import nvidia_ace_services_a2f_v1_pb2_grpc as svc_grpc

    return a2f_pb2, anim_id_pb2, audio_pb2, svc_grpc


class Audio2FaceClient:
    """Async client that streams a single utterance per ``animate()`` call."""

    def __init__(self, cfg: Optional[A2FSettings] = None) -> None:
        self.cfg = cfg or settings.a2f
        self._channel = None  # type: ignore[assignment]
        self._stub = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()  # serialize utterances per avatar

        # One stream_id per process lifetime (== "this avatar session"). Each
        # utterance gets a fresh request_id underneath it. ACE uses these to
        # route animation through the rest of its pipeline.
        self._stream_id = str(uuid.uuid4())

        # Cooldown: when A2F is unreachable, skip the next N utterances
        # instead of paying the gRPC failure cost on every reply.
        self._skip_until_monotonic: float = 0.0
        self._cooldown_s: float = 10.0

    # ---------- lifecycle ----------
    async def connect(self) -> None:
        if self._stub is not None:
            return
        async with self._connect_lock:
            if self._stub is not None:
                return
            import grpc  # imported lazily so tests / lint don't require it

            a2f_pb2, anim_id_pb2, audio_pb2, svc_grpc = _load_proto()
            target = f"{self.cfg.host}:{self.cfg.port}"
            options = [
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10_000),
            ]
            self._channel = grpc.aio.insecure_channel(target, options=options)
            try:
                await asyncio.wait_for(
                    self._channel.channel_ready(),
                    timeout=self.cfg.connect_timeout_s,
                )
                logger.info("A2F channel ready at %s (stream_id=%s)",
                            target, self._stream_id)
            except asyncio.TimeoutError:
                logger.warning(
                    "A2F channel not ready within %.1fs at %s "
                    "(will retry on next utterance)",
                    self.cfg.connect_timeout_s, target,
                )
            self._stub = svc_grpc.A2FServiceStub(self._channel)
            self._a2f_pb2 = a2f_pb2
            self._anim_id_pb2 = anim_id_pb2
            self._audio_pb2 = audio_pb2

    async def close(self) -> None:
        if self._channel is not None:
            try:
                await self._channel.close()
            finally:
                self._channel = None
                self._stub = None

    # ---------- audio prep ----------
    def _wav_to_pcm16(self, wav_bytes: bytes) -> bytes:
        """Decode Kokoro's WAV blob and resample to ACE's expected rate."""
        data, src_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        target_rate = self.cfg.sample_rate
        if src_rate != target_rate:
            n_out = int(round(len(data) * target_rate / src_rate))
            if n_out <= 0:
                return b""
            x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)

        data = np.clip(data, -1.0, 1.0)
        return (data * 32767.0).astype(np.int16).tobytes()

    def _chunk_bytes(self) -> int:
        # samples_per_chunk = rate * (ms/1000); 2 bytes per int16 sample.
        return max(2, int(self.cfg.sample_rate * self.cfg.chunk_ms / 1000) * 2)

    async def _chunks(self, pcm16: bytes) -> AsyncIterator[bytes]:
        # Send chunks as fast as gRPC will accept them; HTTP/2 flow control
        # handles backpressure. Pacing on the asyncio loop just steals time
        # from LiveKit audio I/O without any benefit to A2F.
        cs = self._chunk_bytes()
        for i in range(0, len(pcm16), cs):
            yield pcm16[i : i + cs]

    # ---------- public API ----------
    async def animate(self, wav_bytes: bytes) -> None:
        """Stream one Kokoro utterance to ACE Audio2Face.

        Safe to call from any coroutine; wrap in ``asyncio.create_task`` to
        keep the TTS path non-blocking.
        """
        if not self.cfg.enabled or not wav_bytes:
            return

        loop = asyncio.get_event_loop()
        if loop.time() < self._skip_until_monotonic:
            return  # in cooldown — A2F was just unreachable

        try:
            await self.connect()
        except Exception as exc:  # noqa: BLE001 — never kill the agent
            logger.warning("A2F connect failed: %s", exc)
            return

        try:
            # Decode + resample is CPU-bound numpy work. Run it in a worker
            # thread so the asyncio loop (and LiveKit audio I/O) stays snappy.
            pcm16 = await asyncio.to_thread(self._wav_to_pcm16, wav_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("A2F audio prep failed: %s", exc)
            return
        if not pcm16:
            return

        a2f_pb2 = self._a2f_pb2
        anim_id_pb2 = self._anim_id_pb2
        audio_pb2 = self._audio_pb2

        request_id = str(uuid.uuid4())

        async def request_iter():
            # First message: AudioStreamHeader (animation_ids + audio_header).
            header = a2f_pb2.AudioStreamHeader(
                animation_ids=anim_id_pb2.AnimationIds(
                    request_id=request_id,
                    stream_id=self._stream_id,
                    target_object_id="",   # fill in if you target a specific avatar
                ),
                audio_header=audio_pb2.AudioHeader(
                    audio_format=audio_pb2.AudioHeader.AUDIO_FORMAT_PCM,
                    channel_count=1,
                    samples_per_second=self.cfg.sample_rate,
                    bits_per_sample=16,
                ),
            )
            yield a2f_pb2.AudioStream(audio_stream_header=header)

            # Subsequent messages: AudioWithEmotion carrying raw PCM chunks.
            # We send no emotion timecodes — ACE will use defaults.
            async for chunk in self._chunks(pcm16):
                yield a2f_pb2.AudioStream(
                    audio_with_emotion=a2f_pb2.AudioWithEmotion(
                        audio_buffer=chunk,
                    )
                )

        import grpc

        async with self._send_lock:
            try:
                # PushAudioStream is client-streaming -> unary Status.
                status = await self._stub.PushAudioStream(request_iter())
                logger.debug("A2F PushAudioStream status: code=%s message=%r",
                             status.code, status.message)
            except grpc.aio.AioRpcError as e:
                logger.warning("A2F gRPC error %s: %s — pausing A2F for %.0fs",
                               e.code(), e.details(), self._cooldown_s)
                self._skip_until_monotonic = loop.time() + self._cooldown_s
            except Exception as exc:  # noqa: BLE001
                logger.warning("A2F stream failed: %s — pausing A2F for %.0fs",
                               exc, self._cooldown_s)
                self._skip_until_monotonic = loop.time() + self._cooldown_s


# ---------- singleton accessor ----------
_singleton: Optional[Audio2FaceClient] = None


def get_client() -> Audio2FaceClient:
    """Return the process-wide Audio2Face client (lazily constructed)."""
    global _singleton
    if _singleton is None:
        _singleton = Audio2FaceClient()
    return _singleton
