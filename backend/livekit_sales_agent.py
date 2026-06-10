"""
LiveKit Sales Trainer - Local Voice Agent
==========================================
Mic -> Whisper -> Qwen (Ollama) -> Kokoro -> Speakers

Run: uv run python livekit_sales_agent.py console
"""

import pipeline.core.bootstrap  # noqa: F401, I001 — MUST run first: loads .env, configures DLLs

import logging

from livekit import agents
from livekit.agents import AgentSession
from livekit.plugins import silero

from pipeline import LocalKokoroTTS, LocalOllamaLLM, LocalWhisperSTT
from pipeline.kokoro.processor import TTSProcessor
from pipeline.livekit.agent import CustomerAgent

logger = logging.getLogger(__name__)


async def entrypoint(ctx: agents.JobContext) -> None:
    """Bring up the TTS + STT model singletons (Kokoro before Whisper for
    Windows CUDA DLL load order), then wire up the AgentSession."""
    TTSProcessor()
    from pipeline.whisper.processor import STTProcessor
    STTProcessor()

    session = AgentSession(
        stt=LocalWhisperSTT(),
        llm=LocalOllamaLLM(),
        tts=LocalKokoroTTS(),
        vad=silero.VAD.load(),
    )

    agent = CustomerAgent()
    await session.start(room=ctx.room, agent=agent)
    # on_enter fires automatically after session.start — no extra calls needed here

    # Warm the A2F gRPC channel in the background. Never block startup — if
    # the A2F server is down or slow, we want the user's first reply now.
    import asyncio as _asyncio
    from pipeline.a2f import get_client as get_a2f_client
    from pipeline.core.config import settings as _settings
    if _settings.a2f.enabled:
        _asyncio.create_task(get_a2f_client().connect())


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
