"""faster-whisper STT processor.

Loads the Whisper model once per process and reuses it. The first call to
:class:`STTProcessor` instantiates and caches the model; subsequent calls
return the same instance.
"""

from __future__ import annotations

import logging
import os
import string
import time

if os.name == "nt":
    # Side effect only — prepend CUDA / espeak DLL dirs to PATH before torch loads.
    from pipeline.core.cuda_env import setup_cuda_dlls  # noqa: F401

import numpy as np

from pipeline.core.config import settings

logger = logging.getLogger(__name__)

# Below this many samples the audio is almost certainly silence/click;
# faster-whisper handles short clips, but skipping saves a model call.
_MIN_SAMPLES = 1600  # 0.1s at 16kHz


def _resolve_device(requested: str) -> tuple[str, str]:
    """Return ``(device, compute_type)`` honoring CUDA availability."""
    requested = (requested or "cuda").lower()
    if requested != "cuda":
        return requested, settings.whisper.compute_type

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", settings.whisper.compute_type
        logger.warning(
            "WHISPER_DEVICE=cuda requested but torch.cuda.is_available() is False; "
            "falling back to CPU with compute_type=int8."
        )
    except ImportError:
        logger.warning("torch not importable while validating CUDA; falling back to CPU.")
    return "cpu", "int8"


class STTProcessor:
    """Singleton wrapping a faster-whisper ``WhisperModel``."""

    _instance: "STTProcessor | None" = None

    def __new__(cls) -> "STTProcessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device(settings.whisper.device)
        logger.info(
            "Loading Whisper model=%s device=%s compute=%s",
            settings.whisper.model, device, compute_type,
        )
        self.model = WhisperModel(
            settings.whisper.model, device=device, compute_type=compute_type
        )
        self.device = device
        self.compute_type = compute_type
        self.last_language = "en"
        self._initialized = True

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe a chunk of int16 PCM at 16 kHz. Returns text or ``None``."""
        if not audio_bytes:
            return None

        start = time.perf_counter()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if audio_array.size < _MIN_SAMPLES:
            return None
        audio_array /= 32768.0

        env_lang = settings.whisper.language
        language = None if env_lang == "auto" else env_lang

        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            suppress_blank=True,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        transcript = "".join(seg.text for seg in segments).strip()
        self.last_language = info.language

        logger.info(
            "STT lang=%s prob=%.3f took=%.2fs",
            info.language, info.language_probability, time.perf_counter() - start,
        )

        if not transcript or all(c in string.punctuation + " " for c in transcript):
            return None
        return transcript
