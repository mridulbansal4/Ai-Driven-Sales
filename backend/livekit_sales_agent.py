"""
LiveKit Sales Trainer - Local Voice Agent
==========================================
Mic -> Whisper -> Qwen (Ollama) -> Kokoro -> Speakers

Run: uv run python livekit_sales_agent.py console
"""

import logging

import pipeline.bootstrap  # noqa: F401

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.agents.voice.turn import TurnHandlingOptions
from livekit.plugins import silero

from pipeline.plugins import LocalKokoroTTS, LocalOllamaLLM, LocalWhisperSTT
from pipeline.scenarios import Scenario, build_instructions, random_scenario
from pipeline.tts import TTSProcessor

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerAgent(Agent):
    def __init__(self, scenario: Scenario) -> None:
        super().__init__(instructions=build_instructions(scenario))
        self._scenario = scenario

    async def on_enter(self) -> None:
        """Tell the trainee the scenario, then wait for them to start."""
        logger.info('Briefing officer for scenario: %s', self._scenario.name)
        handle = self.session.say(
            self._scenario.officer_briefing,
            allow_interruptions=False,
            add_to_chat_ctx=False,
        )
        await handle.wait_for_playout()
        logger.info('Briefing done — officer may start the conversation')


async def entrypoint(ctx: agents.JobContext) -> None:
    TTSProcessor()
    from pipeline.stt import STTProcessor

    STTProcessor()

    scenario = random_scenario()
    logger.info('Active scenario: %s', scenario.name)

    session = AgentSession(
        stt=LocalWhisperSTT(),
        llm=LocalOllamaLLM(),
        tts=LocalKokoroTTS(),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            endpointing={'min_delay': 0.6, 'max_delay': 2.5},
        ),
    )

    await session.start(room=ctx.room, agent=CustomerAgent(scenario))


if __name__ == '__main__':
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
