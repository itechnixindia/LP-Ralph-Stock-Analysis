"""
LLM Client — Shared intelligence layer for all RALPH agents.

Supports multiple LLM providers:
  - Google Gemini (default) — set GEMINI_API_KEY env var
  - Anthropic Claude — set ANTHROPIC_API_KEY env var

Uses the new google.genai SDK. Tracks token usage and cost per call.
"""

import json
import logging
import os
import re
import time
import threading

logger = logging.getLogger(__name__)

# ── Auto-load .env file ──────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

_gemini_client = None
_anthropic_client = None

# ── Gemini pricing (per 1M tokens) ───────────────────────────────────────────
GEMINI_INPUT_COST_PER_1M = 0.50
GEMINI_OUTPUT_COST_PER_1M = 3.00

# ── Cost tracker (thread-safe) ───────────────────────────────────────────────
_cost_lock = threading.Lock()
_iteration_cost = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost_usd": 0.0,
    "llm_calls": 0,
}


def reset_iteration_cost():
    """Reset cost counters at the start of each iteration."""
    with _cost_lock:
        _iteration_cost["input_tokens"] = 0
        _iteration_cost["output_tokens"] = 0
        _iteration_cost["total_cost_usd"] = 0.0
        _iteration_cost["llm_calls"] = 0


def get_iteration_cost() -> dict:
    """Return copy of current iteration cost data."""
    with _cost_lock:
        return dict(_iteration_cost)


def _track_cost(input_tokens: int, output_tokens: int):
    """Accumulate token usage and cost for this iteration."""
    cost = (
        (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M
        + (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M
    )
    with _cost_lock:
        _iteration_cost["input_tokens"] += input_tokens
        _iteration_cost["output_tokens"] += output_tokens
        _iteration_cost["total_cost_usd"] += cost
        _iteration_cost["llm_calls"] += 1


# ── Provider detection ────────────────────────────────────────────────────────

def _detect_provider() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            _gemini_client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"]
            )
            logger.info("LLM provider: Google Gemini (gemini-3-flash-preview)")
        except Exception as e:
            logger.warning(f"Could not initialise Gemini: {e}")
    return _gemini_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
            logger.info("LLM provider: Anthropic Claude")
        except Exception as e:
            logger.warning(f"Could not initialise Anthropic client: {e}")
    return _anthropic_client


# ── Main entry point ──────────────────────────────────────────────────────────

def call_intelligence_layer(
    agent_name: str,
    system_prompt: str,
    computed_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str = "",
    max_tokens: int = 4096,
    retries: int = 3,
) -> dict:
    """
    Standard LLM call used by every agent's intelligence layer.
    Returns structured JSON verdict, or empty dict if unavailable.
    """
    provider = _detect_provider()

    if provider == "gemini":
        return _call_gemini(
            agent_name, system_prompt, computed_numbers,
            accumulated_context, prior_iterations_summary,
            max_tokens, retries,
        )
    elif provider == "anthropic":
        return _call_anthropic(
            agent_name, system_prompt, computed_numbers,
            accumulated_context, prior_iterations_summary,
            max_tokens, retries,
        )

    logger.warning(f"{agent_name}: No LLM API key. Set GEMINI_API_KEY or ANTHROPIC_API_KEY.")
    return {}


# ── Gemini implementation ─────────────────────────────────────────────────────

def _call_gemini(
    agent_name, system_prompt, computed_numbers,
    accumulated_context, prior_iterations_summary,
    max_tokens, retries,
) -> dict:
    client = _get_gemini_client()
    if client is None:
        return {}

    from google.genai import types

    user_message = (
        f"COMPUTED RESULTS THIS ITERATION:\n"
        f"{json.dumps(computed_numbers, indent=2, default=str)}\n\n"
        f"CONTEXT FROM PRIOR AGENTS THIS ITERATION:\n"
        f"{json.dumps(accumulated_context, indent=2, default=str)}\n\n"
        f"MEMORY FROM PRIOR ITERATIONS:\n"
        f"{prior_iterations_summary}\n\n"
        f"IMPORTANT: Respond ONLY with a valid JSON object. "
        f"Keep responses concise. Do NOT include arrays of individual headline scores. "
        f"Your entire response must start with {{ and end with }}."
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            # Track cost
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                input_t = getattr(meta, "prompt_token_count", 0) or 0
                output_t = getattr(meta, "candidates_token_count", 0) or 0
                _track_cost(input_t, output_t)

            # Extract text
            raw = None
            try:
                raw = response.text
            except Exception:
                pass
            if not raw and hasattr(response, "candidates") and response.candidates:
                try:
                    raw = response.candidates[0].content.parts[0].text
                except Exception:
                    pass
            if not raw:
                logger.warning(f"{agent_name}: Empty Gemini response.")
                return {}

            raw = raw.strip()
            result = _parse_json_response(raw, agent_name)
            if not result:
                logger.warning(
                    f"{agent_name}: JSON parse failed. "
                    f"Response length: {len(raw)} chars. "
                    f"First 300: {raw[:300]}"
                )
            return result

        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = 2 ** attempt
                logger.warning(f"{agent_name}: Rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"{agent_name}: Gemini error attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    return {}
                time.sleep(2 ** attempt)

    return {}


# ── Anthropic implementation ──────────────────────────────────────────────────

def _call_anthropic(
    agent_name, system_prompt, computed_numbers,
    accumulated_context, prior_iterations_summary,
    max_tokens, retries,
) -> dict:
    client = _get_anthropic_client()
    if client is None:
        return {}

    user_message = (
        f"COMPUTED RESULTS THIS ITERATION:\n"
        f"{json.dumps(computed_numbers, indent=2, default=str)}\n\n"
        f"CONTEXT FROM PRIOR AGENTS THIS ITERATION:\n"
        f"{json.dumps(accumulated_context, indent=2, default=str)}\n\n"
        f"MEMORY FROM PRIOR ITERATIONS:\n"
        f"{prior_iterations_summary}\n\n"
        f"Respond ONLY with a valid JSON object."
    )

    for attempt in range(retries):
        try:
            import anthropic
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            if hasattr(response, "usage"):
                _track_cost(response.usage.input_tokens, response.usage.output_tokens)
            return _parse_json_response(raw, agent_name)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"{agent_name}: API error attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    return {}
                time.sleep(2 ** attempt)

    return {}


# ── JSON parser with truncation repair ────────────────────────────────────────

def _parse_json_response(raw: str, agent_name: str) -> dict:
    """Extract JSON from LLM response, with truncation repair."""
    raw = raw.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Brace-depth extraction
    depth = 0
    start_idx = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                try:
                    return json.loads(raw[start_idx:i + 1])
                except json.JSONDecodeError:
                    start_idx = None

    # 3. Truncation repair — close unclosed braces/brackets
    if raw.startswith("{"):
        repaired = _repair_truncated_json(raw)
        if repaired:
            try:
                result = json.loads(repaired)
                logger.info(f"{agent_name}: Repaired truncated JSON.")
                return result
            except json.JSONDecodeError:
                pass

    logger.warning(f"{agent_name}: Could not parse LLM JSON response.")
    return {}


def _repair_truncated_json(raw: str) -> str:
    """Fix JSON truncated by token limit by closing open structures."""
    if not raw.strip().startswith("{"):
        return ""

    raw = raw.strip()

    # Remove trailing incomplete string (cut mid-value)
    # Find last complete key-value separator
    last_comma = raw.rfind(",")
    last_close_brace = raw.rfind("}")
    last_close_bracket = raw.rfind("]")
    last_closer = max(last_close_brace, last_close_bracket)

    if last_comma > last_closer:
        # Truncated after a comma — remove the dangling part
        raw = raw[:last_comma]

    # Close any unclosed strings
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        raw += '"'

    # Close unclosed brackets and braces
    open_brackets = raw.count("[") - raw.count("]")
    open_braces = raw.count("{") - raw.count("}")

    raw += "]" * max(0, open_brackets)
    raw += "}" * max(0, open_braces)

    return raw
