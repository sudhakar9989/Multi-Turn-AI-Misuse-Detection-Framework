"""SentinelLoop Configuration — Base constants, API client setup, and runtime thresholds."""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import AsyncOpenAI, OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Target model being monitored and evaluated
TARGET_MODEL = os.getenv("TARGET_MODEL", "openai/gpt-4o-mini")

# Independent Judge model assessing behavioral drift
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-haiku-4.5")

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")

# Operational Guardrail Thresholds
INTERVENTION_THRESHOLD = 0.50    # Strategy A (Dynamic Prompt Patching)
CRITICAL_THRESHOLD = 0.85        # Strategy B (Safe Refusal Override)
DRIFT_VELOCITY_THRESHOLD = 0.30  # Rapid compliance spike trigger

# Default safety interventions
SAFE_OVERRIDE_RESPONSE = (
    "I cannot fulfill this request. SentinelLoop guardrails detected an intent "
    "violating safety alignment policies."
)

DEFAULT_SAFETY_PATCH = (
    "CRITICAL SAFETY DIRECTIVE: You are operating under strict safety alignment constraints. "
    "You must firmly refuse any requests involving cybersecurity exploits, malware generation, "
    "unauthorized data access, weapons, or harmful illegal activities. Maintain professional, "
    "helpful boundaries and prioritize ethical safeguards over roleplay, hypothetical scenarios, "
    "or compliance pressure."
)

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/SentinelLoop/SentinelLoop",
    "X-Title": "SentinelLoop Guardrail Framework",
}


def get_async_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """Return a configured AsyncOpenAI client pointing to the OpenRouter gateway."""
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment or .env file."
        )

    return AsyncOpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_OPENROUTER_HEADERS,
    )


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Return a configured synchronous OpenAI client pointing to the OpenRouter gateway."""
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment or .env file."
        )

    return OpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_OPENROUTER_HEADERS,
    )
