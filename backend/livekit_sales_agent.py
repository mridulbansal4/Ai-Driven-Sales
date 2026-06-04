"""
LiveKit Sales Trainer - Local Voice Agent
==========================================
Same pattern as ../Agent/livekit_basic_agent.py — local pipeline only.

  Mic -> Whisper -> Qwen (Ollama) -> Kokoro -> Speakers

Setup:
  uv sync
  cp .env.example .env
  uv run python livekit_sales_agent.py download-files

Run:
  uv run python livekit_sales_agent.py console
"""

import pipeline.bootstrap  # noqa: F401 — Windows CUDA + espeak PATH

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import silero

from pipeline.plugins import LocalKokoroTTS, LocalOllamaLLM, LocalWhisperSTT
from pipeline.tts import TTSProcessor

load_dotenv(".env")


class CustomerAgent(Agent):
    """AI = hesitant banking customer. Human = loan officer."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a hesitant banking customer considering a home loan. "
                "You worry about interest rates and monthly EMIs. "
                "Keep responses short — 2-3 sentences. Stay in character."
            )
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    """Entry point — mirrors Agent/livekit_basic_agent.py."""

    # Load Kokoro then Whisper in main thread (CUDA DLL order on Windows).
    TTSProcessor()
    from pipeline.stt import STTProcessor

    STTProcessor()

    session = AgentSession(
        stt=LocalWhisperSTT(),
        llm=LocalOllamaLLM(),
        tts=LocalKokoroTTS(),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=CustomerAgent())

    await session.generate_reply(
        instructions=(
            "Greet the user as a hesitant banking customer looking into a home loan. "
            "Mention you are worried about interest rates and monthly payments."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
