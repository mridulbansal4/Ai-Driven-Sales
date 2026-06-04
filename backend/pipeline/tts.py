import io
import logging
import sys

if sys.platform == 'win32':
    from pipeline.cuda_env import setup_cuda_dlls  # noqa: F401

logger = logging.getLogger(__name__)


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
        from pipeline.cuda_env import ensure_espeak

        ensure_espeak()
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code='a')
        self.voice = 'af_heart'
        self._initialized = True

    def synthesize(self, text: str) -> tuple[bytes, int]:
        import numpy as np
        import soundfile as sf

        audio_chunks = []
        sample_rate = 24000

        for _, _, audio in self.pipeline(text, voice=self.voice):
            if audio is not None:
                audio_chunks.append(audio)
                if hasattr(audio, 'sample_rate'):
                    sample_rate = int(audio.sample_rate)

        if not audio_chunks:
            return b'', sample_rate

        audio_array = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        return buf.getvalue(), sample_rate
