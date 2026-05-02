"""
Agent 1 — Signal Quality + Factor Alpha

Computation layer: IC, Fama-French regression, OU half-life.
Intelligence layer: LLM interprets signal quality and advises agent2.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from constants import (
    ANNUALIZATION_FACTOR,
    EPSILON,
    IC_EXCELLENT_THRESHOLD,
    IC_GOOD_THRESHOLD,
    IC_MIN_WINDOW_SIZE,
    IC_ROLLING_WINDOW,
    IC_TREND_DELTA,
    IC_TREND_LOOKBACK,
    OU_DEFAULT_HALF_LIFE,
    OU_MIN_OBSERVATIONS,
    OU_NON_MEAN_REVERTING_FALLBACK,
)
from data_loader import load_ff_factors, load_prices

logger = logging.getLogger(__name__)


# ── Computation layer ─────────────────────────────────────────────────────────

def compute_signal(
    ticker: str,
    start_date: str,
    end_date: str,
    params: dict,
    benchmark: str,
    ic_history: list,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
) -> dict:
    """
    Stage 1: Compute signal quality metrics and return insight payload.
    """
    accumulated_context = accumulated_context or {}

    sma_fast = int(params["sma_fast"])
    sma_slow = int(params["sma_slow"])
    rsi_period = int(params["rsi_period"])
    holding_days = int(params["holding_days"])

    df = load_prices(ticker, start_date, end_date)
    if df is None or len(df) < max(sma_slow, 100):
        raise ValueError(
            f"Insufficient data for {ticker}. Need at least {max(sma_slow, 100)} rows."
        )

    close = df["Close"].ffill()
    volume = df.get("Volume", pd.Series(dtype=float))

    # ── SMA and signal series ─────────────────────────────────────────────────
    sma_f = close.rolling(sma_fast).mean()
    sma_s = close.rolling(sma_slow).mean()
    raw_signal = (sma_f - sma_s) / close
    signal_zscore = (raw_signal - raw_signal.rolling(IC_ROLLING_WINDOW).mean()) / (
        raw_signal.rolling(IC_ROLLING_WINDOW).std() + EPSILON
    )

    # ── RSI ───────────────────────────────────────────────────────────────────
    try:
        import ta
        rsi = ta.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
    except Exception:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss + EPSILON)
        rsi = 100 - (100 / (1 + rs))

    # ── Daily returns ─────────────────────────────────────────────────────────
    daily_returns = close.pct_change().dropna()

    # ── Information Coefficient (IC) ──────────────────────────────────────────
    forward_returns = close.pct_change(holding_days).shift(-holding_days)
    aligned = pd.DataFrame(
        {"signal": signal_zscore, "fwd": forward_returns}
    ).dropna()

    ic_values = []
    for i in range(IC_ROLLING_WINDOW, len(aligned)):
        window = aligned.iloc[i - IC_ROLLING_WINDOW: i]
        if len(window) < IC_MIN_WINDOW_SIZE:
            continue
        corr, _ = spearmanr(window["signal"], window["fwd"])
        if not np.isnan(corr):
            ic_values.append(corr)

    ic_mean = float(np.mean(ic_values)) if ic_values else 0.0
    ic_std = float(np.std(ic_values)) if ic_values else 0.01
    ic_ir = ic_mean / (ic_std + EPSILON)

    # ── Fama-French factor regression ─────────────────────────────────────────
    ff = load_ff_factors(start_date, end_date)
    factor_alpha = 0.0
    factor_beta_mkt = 1.0
    factor_beta_smb = 0.0
    factor_beta_hml = 0.0
    factor_r2 = 0.0

    if not ff.empty:
        try:
            import statsmodels.api as sm
            ret = daily_returns.copy()
            ret.index = pd.to_datetime(ret.index)
            ff.index = pd.to_datetime(ff.index)

            merged = pd.DataFrame({"ret": ret}).join(ff, how="inner").dropna()
            if "RF" in merged.columns:
                merged["excess_ret"] = merged["ret"] - merged["RF"]
            else:
                merged["excess_ret"] = merged["ret"]

            available_factors = [c for c in ["Mkt-RF", "SMB", "HML"] if c in merged.columns]
            if available_factors:
                X = sm.add_constant(merged[available_factors])
                ols = sm.OLS(merged["excess_ret"], X).fit()
                factor_alpha = float(ols.params.get("const", 0.0))
                factor_beta_mkt = float(ols.params.get("Mkt-RF", 1.0))
                factor_beta_smb = float(ols.params.get("SMB", 0.0))
                factor_beta_hml = float(ols.params.get("HML", 0.0))
                factor_r2 = float(ols.rsquared)
        except Exception as e:
            logger.warning(f"Factor regression failed: {e}")
            factor_alpha = float(daily_returns.mean())
    else:
        factor_alpha = float(daily_returns.mean())
        factor_beta_mkt = 1.0

    # ── OU half-life ──────────────────────────────────────────────────────────
    half_life_days = _compute_ou_half_life(signal_zscore.dropna())

    # ── IC trend ─────────────────────────────────────────────────────────────
    ic_trend = "stable"
    if len(ic_history) >= IC_TREND_LOOKBACK:
        recent = ic_history[-IC_TREND_LOOKBACK:]
        if recent[-1] > recent[0] + IC_TREND_DELTA:
            ic_trend = "improving"
        elif recent[-1] < recent[0] - IC_TREND_DELTA:
            ic_trend = "declining"

    # ── Assemble raw output ───────────────────────────────────────────────────
    raw_numbers = {
        "ic_mean": round(ic_mean, 5),
        "ic_std": round(ic_std, 5),
        "ic_ir": round(ic_ir, 4),
        "factor_alpha": round(factor_alpha, 6),
        "factor_beta_mkt": round(factor_beta_mkt, 4),
        "factor_beta_smb": round(factor_beta_smb, 4),
        "factor_beta_hml": round(factor_beta_hml, 4),
        "factor_r2": round(factor_r2, 4),
        "half_life_days": round(half_life_days, 2),
        "holding_days": holding_days,
        "ic_trend": ic_trend,
        "n_ic_observations": len(ic_values),
        "recent_ic_history": ic_history[-10:] if ic_history else [],
    }

    # ── Intelligence layer ────────────────────────────────────────────────────
    insight = _run_intelligence_layer(
        raw_numbers, accumulated_context, prior_iterations_summary
    )

    return {
        **raw_numbers,
        "signal_series": signal_zscore.tolist(),
        "return_series": daily_returns.tolist(),
        "rsi_series": rsi.tolist(),
        "close_series": close.tolist(),
        "close_index": [str(d) for d in close.index],
        "insight": insight,
    }


def _compute_ou_half_life(series: pd.Series) -> float:
    try:
        import statsmodels.api as sm
        s = series.dropna()
        if len(s) < OU_MIN_OBSERVATIONS:
            return OU_DEFAULT_HALF_LIFE
        delta = s.diff().dropna()
        lagged = s.shift(1).dropna()
        aligned = pd.DataFrame({"delta": delta, "lag": lagged}).dropna()
        X = sm.add_constant(aligned["lag"])
        result = sm.OLS(aligned["delta"], X).fit()
        theta = result.params.get("lag", 0.0)
        if theta >= 0:
            return OU_NON_MEAN_REVERTING_FALLBACK
        return float(-np.log(2) / theta)
    except Exception:
        return OU_DEFAULT_HALF_LIFE


def _run_intelligence_layer(
    raw_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent1_prompt import AGENT1_SYSTEM_PROMPT
    except ImportError:
        return {}

    result = call_intelligence_layer(
        agent_name="agent1_signal",
        system_prompt=AGENT1_SYSTEM_PROMPT,
        computed_numbers=raw_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    ic = raw_numbers.get("ic_mean", 0)
    defaults = {
        "signal_quality": "strong" if ic >= IC_EXCELLENT_THRESHOLD else (
            "moderate" if ic >= IC_GOOD_THRESHOLD else "weak"
        ),
        "ic_interpretation": f"IC of {ic:.3f} computed.",
        "alpha_interpretation": "Factor alpha computed.",
        "half_life_verdict": "matched",
        "half_life_explanation": "Holding period alignment assessed.",
        "ic_trend": raw_numbers.get("ic_trend", "stable"),
        "warnings": [],
        "recommendation_for_next_agent": "Proceed with standard sizing.",
        "confidence": 0.5,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    params = {
        "sma_fast": 12,
        "sma_slow": 48,
        "rsi_period": 14,
        "holding_days": 9,
        "stop_loss_pct": 0.018,
        "take_profit_pct": 0.072,
        "vol_target_pct": 0.15,
        "kelly_fraction": 0.25,
    }
    result = compute_signal(
        ticker="RELIANCE.NS",
        start_date="2020-01-01",
        end_date="2024-12-31",
        params=params,
        benchmark="^NSEI",
        ic_history=[],
    )
    print(f"IC Mean: {result['ic_mean']:.4f}")
    print(f"IC IR: {result['ic_ir']:.3f}")
    print(f"Factor Alpha: {result['factor_alpha']:.6f}")
    print(f"Half-life: {result['half_life_days']:.1f} days")
    print(f"Signal quality: {result['insight'].get('signal_quality', 'N/A')}")
    print("agent1_signal: OK")
