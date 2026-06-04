import os

if os.name == 'nt':
    from pipeline.cuda_env import setup_cuda_dlls  # noqa: F401 — side effect: PATH/DLL dirs

import time
import logging
import string
import numpy as np

logger = logging.getLogger(__name__)


class STTProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from faster_whisper import WhisperModel

        model_size = os.getenv('WHISPER_MODEL', 'large-v3-turbo')
        device = os.getenv('WHISPER_DEVICE', 'cuda')
        compute_type = os.getenv('WHISPER_COMPUTE', 'int8')
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self._initialized = True

    def transcribe(self, audio_bytes: bytes) -> str:
        start = time.perf_counter()
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0

        segments, _ = self.model.transcribe(
            audio_array, beam_size=5, language=None
        )
        transcript = ''.join(seg.text for seg in segments).strip()

        if not transcript or all(c in string.punctuation + ' ' for c in transcript):
            transcript = ''

        elapsed = time.perf_counter() - start
        logger.info(f"STT transcribe took {elapsed:.2f}s")
        return transcript
