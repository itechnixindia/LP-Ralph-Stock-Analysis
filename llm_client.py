import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

_anthropic_client = None


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
        except Exception as e:
            logger.warning(f"Could not initialise Anthropic client: {e}")
            _anthropic_client = None
    return _anthropic_client


def call_intelligence_layer(
    agent_name: str,
    system_prompt: str,
    computed_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str = "",
    max_tokens: int = 1000,
    model: str = "claude-sonnet-4-20250514",
    retries: int = 3,
) -> dict:
    """
    Standard LLM call used by every agent's intelligence layer.
    Returns structured JSON verdict, or empty dict if unavailable.
    """
    client = _get_client()
    if client is None:
        logger.warning(f"{agent_name}: Anthropic client unavailable — skipping LLM layer.")
        return {}

    user_message = f"""COMPUTED RESULTS THIS ITERATION:
{json.dumps(computed_numbers, indent=2, default=str)}

CONTEXT FROM PRIOR AGENTS THIS ITERATION:
{json.dumps(accumulated_context, indent=2, default=str)}

MEMORY FROM PRIOR ITERATIONS:
{prior_iterations_summary}

Respond ONLY with a valid JSON object matching the schema in your system prompt.
Do not include markdown fences or any text outside the JSON."""

    for attempt in range(retries):
        try:
            import anthropic
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            return _parse_json_response(raw, agent_name)
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"{agent_name}: Rate limit — waiting {wait}s before retry.")
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.warning(f"{agent_name}: API error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"{agent_name}: Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return {}
            time.sleep(1)

    return {}


def _parse_json_response(raw: str, agent_name: str) -> dict:
    raw = raw.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    logger.warning(f"{agent_name}: Could not parse LLM JSON response.")
    return {}
