"""Kokoro TTS processor (singleton, language-pipeline cached)."""

from __future__ import annotations

import io
import logging
import re
import sys

if sys.platform == 'win32':
    from pipeline.core.cuda_env import setup_cuda_dlls  # noqa: F401

from pipeline.core.config import settings

logger = logging.getLogger(__name__)

# Kokoro repo (passed explicitly to suppress its default-repo warning).
_KOKORO_REPO = 'hexgrad/Kokoro-82M'

# Devanagari Unicode block — its presence means the reply is Hindi.
_DEVANAGARI = re.compile(r'[ऀ-ॿ]')

# Kokoro lang_code -> default voice (override via env).
_DEFAULT_VOICES = {'a': 'af_heart', 'h': 'hf_alpha'}


class TTSProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from pipeline.core.cuda_env import ensure_espeak

        ensure_espeak()

        self._voices = {
            'a': settings.kokoro.voice_en,
            'h': settings.kokoro.voice_hi,
        }
        self._pipelines: dict[str, object] = {}
        self._model = None  # shared KModel across language pipelines
        # Warm up English so the first reply isn't slow.
        self._get_pipeline('a')
        self._initialized = True

    def _get_pipeline(self, lang_code: str):
        """Lazily build (and cache) a Kokoro pipeline for a language.

        The 82M model is created once and shared across languages. On failure
        (e.g. missing Hindi voice/deps) we fall back to English so the agent
        keeps talking instead of crashing.
        """
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]

        from kokoro import KPipeline

        try:
            pipeline = KPipeline(
                lang_code=lang_code,
                repo_id=_KOKORO_REPO,
                model=self._model if self._model is not None else True,
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.warning(
                "Kokoro lang_code=%r failed to load (%s); using English instead.",
                lang_code, exc,
            )
            if lang_code != 'a':
                return self._get_pipeline('a')
            raise

        if self._model is None:
            self._model = pipeline.model
        self._pipelines[lang_code] = pipeline
        return pipeline

    @staticmethod
    def _lang_for_text(text: str) -> str:
        """Pick Kokoro language from the reply's script."""
        return 'h' if _DEVANAGARI.search(text) else 'a'

    def synthesize(self, text: str) -> tuple[bytes, int]:
        import numpy as np
        import soundfile as sf

        lang_code = self._lang_for_text(text)
        pipeline = self._get_pipeline(lang_code)
        voice = self._voices.get(lang_code, _DEFAULT_VOICES['a'])
        logger.info("Kokoro synth lang=%s voice=%s", lang_code, voice)

        audio_chunks = []
        sample_rate = 24000

        try:
            for _, _, audio in pipeline(text, voice=voice):
                if audio is not None:
                    audio_chunks.append(audio)
                    if hasattr(audio, 'sample_rate'):
                        sample_rate = int(audio.sample_rate)
        except Exception as exc:  # noqa: BLE001 — never kill the agent on TTS error
            logger.warning("Kokoro synth failed for lang=%s (%s)", lang_code, exc)
            return b'', sample_rate

        if not audio_chunks:
            return b'', sample_rate

        audio_array = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        return buf.getvalue(), sample_rate
