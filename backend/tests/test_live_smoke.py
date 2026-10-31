"""Opt-in smoke test against the real Anthropic API.

Default runs never touch the network: the test requires both the `live` marker
selection and RUN_LIVE_TESTS=1, plus a configured API key.
"""

import os

import pytest

from app.agent.llm import AnthropicLLM
from app.core.config import Settings

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_anthropic_adapter_returns_text():
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 and select -m live to run.")
    settings = Settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not configured.")

    llm = AnthropicLLM(settings)
    turn = await llm.complete(
        system="You are a terse clinic assistant. Reply in one short sentence.",
        messages=[
            {
                "role": "user",
                "content": "Say that the clinic opens at 9 AM.",
            }
        ],
        tools=[],
    )

    assert turn.text
    assert "9" in turn.text
