"""faster-whisper STT processor.

Loads the Whisper model once per process and reuses it. The first call to
:class:`STTProcessor` instantiates and caches the model; subsequent calls
return the same instance.
"""

from __future__ import annotations

import logging
import os
import re
import string
import time

if os.name == "nt":
    # Side effect only — prepend CUDA / espeak DLL dirs to PATH before torch loads.
    from pipeline.core.cuda_env import setup_cuda_dlls  # noqa: F401

import numpy as np

from pipeline.core.config import settings

logger = logging.getLogger(__name__)

# Below this many samples the audio is almost certainly silence/click;
# faster-whisper handles short clips but reliably hallucinates on <~0.4s of
# audio. LiveKit's Silero VAD upstream already gates speech, so anything
# shorter than this is noise, a lip-smack, or a cut-off frame.
_MIN_SAMPLES = 6400  # 0.4s at 16kHz

# Phrases Whisper is notorious for hallucinating on silence/music/noise —
# they're embedded in its training data (YouTube captions). Match the
# normalized (lowercase, punctuation-stripped) transcript exactly.
_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thanks for watching!",
        "thank you for watching",
        "thank you so much",
        "thank you very much",
        "thanks",
        "bye",
        "goodbye",
        "you",
        "okay",
        "ok",
        "uh",
        "um",
        "hmm",
        "mm",
        "mhm",
        "ah",
        "oh",
        ".",
        "..",
        "...",
        "please subscribe",
        "subscribe",
        "like and subscribe",
        "music",
        "[music]",
        "(music)",
        "applause",
        "[applause]",
        "silence",
        "[silence]",
        "शुक्रिया",  # Hindi "thanks" — another common hallucination
        "धन्यवाद",
    }
)

_PUNCT_STRIP_RE = re.compile(r"[^\w\sऀ-ॿ]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _PUNCT_STRIP_RE.sub("", text).strip().lower()


def _resolve_device(requested: str) -> tuple[str, str]:
    """Return ``(device, compute_type)`` honoring CUDA availability.

    On CUDA we override the configured compute_type to ``float16`` — int8 on
    GPU gives noticeably worse transcription quality (more hallucinations on
    noisy/accented speech) while barely saving VRAM on the turbo model.
    """
    requested = (requested or "cuda").lower()
    configured = settings.whisper.compute_type

    if requested != "cuda":
        # CTranslate2's CPU backend doesn't support float16 — coerce to int8.
        if configured in ("float16", "int8_float16"):
            logger.info("WHISPER_COMPUTE=%s on CPU coerced to int8", configured)
            return requested, "int8"
        return requested, configured

    try:
        import torch

        if torch.cuda.is_available():
            # int8 on CUDA is a known accuracy footgun; promote to float16.
            compute = configured
            if compute in ("int8", "int8_float16"):
                logger.info(
                    "WHISPER_COMPUTE=%s on CUDA promoted to float16 for accuracy",
                    compute,
                )
                compute = "float16"
            return "cuda", compute
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

        # Energy-based pre-gate: VAD upstream sometimes lets through chunks
        # that are mostly background hum. RMS < ~0.005 on float32 normalized
        # audio is well below intelligible speech.
        rms = float(np.sqrt(np.mean(audio_array * audio_array)))
        if rms < 0.005:
            logger.debug("STT skip: rms=%.4f below threshold", rms)
            return None

        env_lang = settings.whisper.language
        language = None if env_lang == "auto" else env_lang

        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            beam_size=5,
            # Temperature fallback: start greedy, escalate only if the decode
            # tripped a hallucination guard. A single temperature=0.0 disables
            # those guards entirely, which is why noisy frames produce
            # confident-sounding garbage.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # Reject segments where no_speech_prob > this OR avg_logprob < the
            # threshold — i.e. low confidence => empty output instead of guess.
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            suppress_blank=True,
            condition_on_previous_text=False,
            without_timestamps=True,
            # Internal Silero VAD on top of LiveKit's — trims residual leading/
            # trailing silence inside the frame, which is the single biggest
            # source of the "Thank you." hallucination.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "threshold": 0.5},
        )
        transcript = "".join(seg.text for seg in segments).strip()
        self.last_language = info.language

        logger.info(
            "STT lang=%s prob=%.3f rms=%.4f took=%.2fs text=%r",
            info.language, info.language_probability, rms,
            time.perf_counter() - start, transcript,
        )

        if not transcript:
            return None
        if all(c in string.punctuation + " " for c in transcript):
            return None

        normalized = _normalize(transcript)
        if not normalized:
            return None
        if normalized in _HALLUCINATION_PHRASES:
            logger.info("STT drop: hallucination phrase %r", transcript)
            return None

        return transcript
