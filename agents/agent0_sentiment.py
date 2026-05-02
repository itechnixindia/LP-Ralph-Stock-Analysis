"""
Agent 0 — Sentiment Signal

Computation layer: Fetches news headlines, computes basic text sentiment heuristics.
Intelligence layer: Claude LLM scores headlines and produces aggregate sentiment.

Runs BEFORE agent1 — provides a sentiment signal that enriches the entry condition.
"""

import logging
import re
from typing import Dict, List

from constants import EPSILON

logger = logging.getLogger(__name__)

# ── Basic keyword sentiment (fallback when LLM unavailable) ──────────────────

BULLISH_KEYWORDS = {
    "upgrade", "beat", "beats", "exceeds", "growth", "profit", "gains",
    "soars", "surges", "rallies", "rally", "bullish", "outperform",
    "record", "dividend", "buyback", "positive", "breakthrough",
    "expansion", "partnership", "acquisition", "revenue",
}

BEARISH_KEYWORDS = {
    "downgrade", "miss", "misses", "falls", "drops", "plunges", "crash",
    "bearish", "underperform", "loss", "losses", "decline", "cut",
    "layoffs", "fraud", "investigation", "lawsuit", "recall",
    "default", "bankruptcy", "warning", "concern", "risk",
}


def _keyword_score(headline: str) -> float:
    """Simple keyword-based sentiment score (-1 to +1)."""
    words = set(re.findall(r'\b\w+\b', headline.lower()))
    bull_count = len(words & BULLISH_KEYWORDS)
    bear_count = len(words & BEARISH_KEYWORDS)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    sentiment_history: list = None,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
) -> dict:
    """
    Stage 0: Fetch news and compute sentiment signal.
    Returns sentiment score, label, and insight payload.
    """
    sentiment_history = sentiment_history or []
    accumulated_context = accumulated_context or {}

    # ── Fetch headlines ───────────────────────────────────────────────────────
    try:
        from news_fetcher import fetch_headlines
        headlines = fetch_headlines(ticker)
    except Exception as e:
        logger.warning(f"News fetch failed: {e}. Using empty headlines.")
        headlines = []

    # ── Computation layer: keyword scoring ────────────────────────────────────
    keyword_scores = []
    for h in headlines:
        score = _keyword_score(h.get("title", ""))
        keyword_scores.append({
            "headline": h.get("title", "")[:80],
            "keyword_score": round(score, 3),
            "source": h.get("source", "unknown"),
        })

    # Aggregate keyword score
    if keyword_scores:
        total_score = sum(s["keyword_score"] for s in keyword_scores)
        keyword_aggregate = total_score / len(keyword_scores)
    else:
        keyword_aggregate = 0.0

    # ── Sentiment trend ───────────────────────────────────────────────────────
    sentiment_trend = "stable"
    if len(sentiment_history) >= 3:
        recent = sentiment_history[-3:]
        if recent[-1] > recent[0] + 0.1:
            sentiment_trend = "improving"
        elif recent[-1] < recent[0] - 0.1:
            sentiment_trend = "deteriorating"

    raw_numbers = {
        "n_headlines": len(headlines),
        "keyword_aggregate": round(keyword_aggregate, 4),
        "sentiment_trend": sentiment_trend,
        "top_headlines": [h.get("title", "")[:80] for h in headlines[:10]],
        "keyword_scores_summary": keyword_scores[:10],
    }

    # ── Intelligence layer: LLM sentiment scoring ─────────────────────────────
    insight = _run_intelligence_layer(
        headlines=headlines,
        raw_numbers=raw_numbers,
        ticker=ticker,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    # Use LLM aggregate if available, else fall back to keywords
    aggregate = float(insight.get("aggregate_sentiment", keyword_aggregate))
    label = insight.get("sentiment_label", _score_to_label(keyword_aggregate))

    return {
        "sentiment_score": round(aggregate, 4),
        "sentiment_label": label,
        "sentiment_trend": sentiment_trend,
        "n_headlines": len(headlines),
        "keyword_aggregate": round(keyword_aggregate, 4),
        "signal_strength": insight.get("signal_strength", _signal_strength(
            aggregate, len(headlines)
        )),
        "dominant_theme": insight.get("dominant_theme", "No clear theme"),
        "insight": insight,
    }


def _score_to_label(score: float) -> str:
    if score > 0.3:
        return "bullish"
    elif score > 0.1:
        return "mildly_bullish"
    elif score > -0.1:
        return "neutral"
    elif score > -0.3:
        return "mildly_bearish"
    return "bearish"


def _signal_strength(score: float, n_headlines: int) -> str:
    if n_headlines < 3:
        return "no_signal"
    abs_score = abs(score)
    if abs_score > 0.4:
        return "strong"
    elif abs_score > 0.2:
        return "moderate"
    elif abs_score > 0.05:
        return "weak"
    return "no_signal"


def _run_intelligence_layer(
    headlines: list,
    raw_numbers: dict,
    ticker: str,
    accumulated_context: dict,
    prior_iterations_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent0_prompt import AGENT0_SYSTEM_PROMPT
    except ImportError:
        return {}

    # Prepare headline text for LLM
    headline_text = "\n".join(
        f"- [{h.get('source', '?')}] {h.get('title', '')}"
        for h in headlines[:20]
    )

    computed_numbers = {
        "ticker": ticker,
        "n_headlines": len(headlines),
        "headlines": headline_text,
        "keyword_aggregate": raw_numbers.get("keyword_aggregate", 0.0),
    }

    result = call_intelligence_layer(
        agent_name="agent0_sentiment",
        system_prompt=AGENT0_SYSTEM_PROMPT,
        computed_numbers=computed_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    defaults = {
        "headline_scores": [],
        "aggregate_sentiment": raw_numbers.get("keyword_aggregate", 0.0),
        "sentiment_label": _score_to_label(
            raw_numbers.get("keyword_aggregate", 0.0)
        ),
        "dominant_theme": "Insufficient data for theme extraction",
        "signal_strength": "weak",
        "recommendation_for_next_agent": "No strong sentiment signal detected.",
        "confidence": 0.3,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = compute_sentiment(
        ticker="RELIANCE.NS",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(f"Sentiment score: {result['sentiment_score']:.3f}")
    print(f"Sentiment label: {result['sentiment_label']}")
    print(f"Signal strength: {result['signal_strength']}")
    print(f"Headlines found: {result['n_headlines']}")
    print(f"Dominant theme: {result['dominant_theme']}")
    print(f"Keyword aggregate: {result['keyword_aggregate']:.3f}")
    print("agent0_sentiment: OK")
