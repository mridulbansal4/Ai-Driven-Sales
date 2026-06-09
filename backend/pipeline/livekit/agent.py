"""CustomerAgent — LiveKit Agent that role-plays an HDFC banking customer."""

from __future__ import annotations

import logging

from livekit.agents import Agent

from pipeline.livekit.scenarios import random_scenario

logger = logging.getLogger(__name__)


class CustomerAgent(Agent):
    """
    AI   = HDFC banking customer (always).
    Human = loan sales officer trainee (always).

    Flow on session start
    ─────────────────────
    1. Briefing spoken via TTS (not through LLM) — tells the trainee
       who the customer is, what to sell, and to start when ready.
    2. AI generates the customer's opening line and waits for the trainee.
    """

    def __init__(self) -> None:
        scenario = random_scenario()
        self._scenario = scenario
        super().__init__(instructions=scenario["instructions"])

        banner = "=" * 55
        logger.info("\n%s\n  SCENARIO : %s\n%s", banner, scenario["name"], banner)

    async def on_enter(self) -> None:
        """Speak the briefing, then let the AI customer open the call."""

        # ── Step 1: Briefing (pure TTS — spoken as narrator, NOT added to
        #   the LLM chat context so it does not confuse the customer persona)
        logger.info("Speaking trainee briefing for scenario: %s", self._scenario["name"])
        briefing_handle = self.session.say(
            self._scenario["briefing"],
            allow_interruptions=False,
            add_to_chat_ctx=False,
        )
        await briefing_handle.wait_for_playout()

        # ── Step 2: AI customer picks up the phone and opens the conversation
        logger.info("Customer opening the call...")
        await self.session.generate_reply(
            instructions=self._scenario["greeting"]
        )
