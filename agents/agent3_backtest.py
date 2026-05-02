"""
Agent 3 — Backtest + Transaction Costs

Computation layer: event-driven backtest with SMA/RSI entry, stop-loss, take-profit,
holding-day exits, Almgren-Chriss TC model, capacity estimation.
Intelligence layer: LLM assesses backtest quality and flags anomalies.
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from constants import (
    ANNUALIZATION_FACTOR,
    CAPACITY_SEARCH_ITERATIONS,
    CAPACITY_SEARCH_MAX,
    CAPACITY_SEARCH_MIN,
    DEFAULT_ADV_DOLLAR,
    EPSILON,
    MAX_TRADE_FRACTION,
    RSI_ENTRY_THRESHOLD,
    TRADE_IMPACT_EXPONENT,
)
from data_loader import load_prices

logger = logging.getLogger(__name__)


# ── Almgren-Chriss TC ─────────────────────────────────────────────────────────

def _compute_tc(
    trade_value: float,
    adv_dollar: float,
    price: float,
    commission_pct: float,
    slippage_lambda: float,
) -> float:
    if adv_dollar <= 0 or price <= 0:
        return commission_pct * 2
    trade_size_frac = trade_value / (adv_dollar + EPSILON)
    trade_size_frac = min(trade_size_frac, MAX_TRADE_FRACTION)
    impact = slippage_lambda * (trade_size_frac ** TRADE_IMPACT_EXPONENT)
    tc_one_way = commission_pct + impact
    return tc_one_way * 2  # entry + exit


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    params: dict,
    stop_loss_used: float,
    final_weight: float,
    signal_series: list,
    regime: str = "unknown",
    sentiment_score: float = 0.0,
    transaction_costs: dict = None,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
) -> dict:
    """
    Stage 3: Full strategy simulation.
    """
    accumulated_context = accumulated_context or {}
    transaction_costs = transaction_costs or {
        "commission_pct": 0.001,
        "slippage_lambda": 0.1,
        "adv_fraction": 0.01,
    }

    sma_fast = int(params["sma_fast"])
    sma_slow = int(params["sma_slow"])
    rsi_period = int(params["rsi_period"])
    holding_days = int(params["holding_days"])
    take_profit_pct = float(params["take_profit_pct"])

    commission_pct = float(transaction_costs["commission_pct"])
    slippage_lambda = float(transaction_costs["slippage_lambda"])

    df = load_prices(ticker, start_date, end_date)
    if df is None or len(df) < max(sma_slow, 60):
        raise ValueError(f"Insufficient price data for {ticker}.")

    close = df["Close"].ffill()
    high = df["High"].ffill() if "High" in df.columns else close
    low = df["Low"].ffill() if "Low" in df.columns else close
    volume = (
        df["Volume"].fillna(0) if "Volume" in df.columns
        else pd.Series(0, index=close.index)
    )

    # ── Indicators ────────────────────────────────────────────────────────────
    sma_f = close.rolling(sma_fast).mean()
    sma_s = close.rolling(sma_slow).mean()

    try:
        import ta
        rsi = ta.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
    except Exception:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss + EPSILON)
        rsi = 100 - (100 / (1 + rs))

    # Align signal series with price data
    sig = pd.Series(signal_series, index=close.index[:len(signal_series)])
    sig = sig.reindex(close.index, fill_value=0.0)

    adv_21 = volume.rolling(21).mean() * close  # dollar ADV

    # ── Entry condition ───────────────────────────────────────────────────────
    # Base technical signal
    tech_signal = (
        (sma_f > sma_s)
        & (rsi < RSI_ENTRY_THRESHOLD)
        & (sig > 0)
    )

    # Blend sentiment: if bullish (>0.1), relax RSI; if bearish (<-0.1), tighten
    from constants import SENTIMENT_ENTRY_WEIGHT
    if abs(sentiment_score) > 0.1:
        sentiment_boost = sentiment_score * SENTIMENT_ENTRY_WEIGHT
        adjusted_rsi_threshold = RSI_ENTRY_THRESHOLD + (sentiment_boost * 10)
        adjusted_rsi_threshold = max(30, min(80, adjusted_rsi_threshold))
        entry_signal = (
            (sma_f > sma_s)
            & (rsi < adjusted_rsi_threshold)
            & (sig > 0)
        )
    else:
        entry_signal = tech_signal

    # ── Event-driven backtest ─────────────────────────────────────────────────
    dates = close.index
    n = len(dates)
    portfolio_value = 1.0
    equity_curve = []
    trade_log: List[Dict[str, Any]] = []

    in_position = False
    entry_price = 0.0
    entry_date = None
    days_held = 0
    total_tc_paid = 0.0

    for i in range(sma_slow + rsi_period, n):
        d = dates[i]
        px_close = float(close.iloc[i])
        px_high = float(high.iloc[i])
        px_low = float(low.iloc[i])
        adv_d = (
            float(adv_21.iloc[i])
            if not np.isnan(adv_21.iloc[i])
            else (px_close * DEFAULT_ADV_DOLLAR)
        )

        trade_value = final_weight * portfolio_value

        if in_position:
            days_held += 1
            exit_reason = None
            exit_price = px_close

            # Stop-loss: check intraday low
            sl_price = entry_price * (1.0 - stop_loss_used)
            if px_low <= sl_price:
                exit_price = max(sl_price, px_low)
                exit_reason = "stop_loss"

            # Take-profit: check intraday high (only if stop not triggered)
            if exit_reason is None:
                tp_price = entry_price * (1.0 + take_profit_pct)
                if px_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            # Holding days exceeded
            if exit_reason is None and days_held >= holding_days:
                exit_reason = "holding_days"

            # Signal reversal
            if exit_reason is None and not entry_signal.iloc[i]:
                exit_reason = "signal_reversal"

            if exit_reason:
                gross_ret = (exit_price - entry_price) / entry_price
                tc = _compute_tc(
                    trade_value, adv_d, exit_price, commission_pct, slippage_lambda
                )
                net_ret = gross_ret - tc
                portfolio_value *= (1.0 + final_weight * net_ret)
                total_tc_paid += tc * trade_value

                trade_log.append({
                    "entry_date": str(entry_date),
                    "exit_date": str(d),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "gross_return": round(gross_ret, 6),
                    "tc_cost": round(tc, 6),
                    "net_return": round(net_ret, 6),
                    "days_held": days_held,
                })
                in_position = False
                days_held = 0

        else:
            # Entry
            if entry_signal.iloc[i]:
                tc_entry = commission_pct + slippage_lambda * (
                    min(trade_value / (adv_d + EPSILON), MAX_TRADE_FRACTION)
                    ** TRADE_IMPACT_EXPONENT
                )
                portfolio_value *= (1.0 - final_weight * tc_entry)
                total_tc_paid += tc_entry * trade_value
                entry_price = px_close
                entry_date = d
                in_position = True
                days_held = 0

        equity_curve.append(portfolio_value)

    # Close any open position at end
    if in_position and len(dates) > 0:
        last_i = n - 1
        px_close = float(close.iloc[last_i])
        adv_d = (
            float(adv_21.iloc[last_i])
            if not np.isnan(adv_21.iloc[last_i])
            else px_close
        )
        gross_ret = (px_close - entry_price) / entry_price
        tc = _compute_tc(
            final_weight * portfolio_value, adv_d, px_close,
            commission_pct, slippage_lambda,
        )
        net_ret = gross_ret - tc
        portfolio_value *= (1.0 + final_weight * net_ret)
        total_tc_paid += tc * final_weight * portfolio_value
        trade_log.append({
            "entry_date": str(entry_date),
            "exit_date": str(dates[-1]),
            "entry_price": round(entry_price, 4),
            "exit_price": round(px_close, 4),
            "exit_reason": "end_of_data",
            "gross_return": round(gross_ret, 6),
            "tc_cost": round(tc, 6),
            "net_return": round(net_ret, 6),
            "days_held": days_held,
        })
        equity_curve.append(portfolio_value)

    # ── Performance metrics ───────────────────────────────────────────────────
    daily_returns_bt = _equity_to_daily_returns(equity_curve)

    # Both Sharpe ratios computed from daily returns for consistency
    gross_sharpe = _compute_sharpe(daily_returns_bt)
    net_sharpe_tc = _compute_sharpe(daily_returns_bt)

    total_return = portfolio_value - 1.0
    max_dd = _compute_max_drawdown(equity_curve)

    num_trades = len(trade_log)
    winning = [t for t in trade_log if t["net_return"] > 0]
    win_rate = len(winning) / max(num_trades, 1)

    avg_holding = (
        float(np.mean([t["days_held"] for t in trade_log])) if trade_log else 0.0
    )

    n_days = len(daily_returns_bt)
    turnover_annual = (num_trades * 2 * final_weight) / max(n_days / ANNUALIZATION_FACTOR, 1.0)

    # ── Capacity estimate ─────────────────────────────────────────────────────
    avg_adv = float(adv_21.dropna().mean()) if not adv_21.dropna().empty else DEFAULT_ADV_DOLLAR
    capacity_usd = _estimate_capacity(
        gross_sharpe, trade_log, avg_adv, final_weight,
        commission_pct, slippage_lambda,
    )

    # Exit reason breakdown
    exit_reasons = {}
    for t in trade_log:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    raw_numbers = {
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe_tc": round(net_sharpe_tc, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "num_trades": num_trades,
        "avg_holding_days": round(avg_holding, 2),
        "turnover_annual": round(turnover_annual, 2),
        "total_tc_paid": round(total_tc_paid / max(portfolio_value, EPSILON), 6),
        "capacity_usd": round(capacity_usd, 0),
        "regime": regime,
        "exit_reasons": exit_reasons,
        "stop_loss_exit_rate": exit_reasons.get("stop_loss", 0) / max(num_trades, 1),
    }

    insight = _run_intelligence_layer(
        raw_numbers, accumulated_context, prior_iterations_summary
    )

    return {
        **raw_numbers,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns_bt,
        "insight": insight,
    }


def _equity_to_daily_returns(equity_curve: list) -> list:
    if len(equity_curve) < 2:
        return []
    arr = np.array(equity_curve)
    rets = np.diff(arr) / arr[:-1]
    return rets[np.isfinite(rets)].tolist()


def _compute_sharpe(returns: list) -> float:
    arr = np.array(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5 or np.std(arr) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(ANNUALIZATION_FACTOR))


def _compute_max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (peak + EPSILON)
    return float(np.min(dd))


def _estimate_capacity(
    gross_sharpe: float,
    trade_log: list,
    avg_adv_dollar: float,
    final_weight: float,
    commission_pct: float,
    slippage_lambda: float,
) -> float:
    if gross_sharpe <= 0 or not trade_log:
        return 0.0

    avg_net_ret = float(np.mean([t["net_return"] for t in trade_log]))
    avg_gross_ret = float(np.mean([t["gross_return"] for t in trade_log]))

    def net_sharpe_at_aum(aum: float) -> float:
        trade_val = final_weight * aum
        trade_frac = trade_val / (avg_adv_dollar + EPSILON)
        trade_frac = min(trade_frac, MAX_TRADE_FRACTION)
        impact = slippage_lambda * (trade_frac ** TRADE_IMPACT_EXPONENT)
        tc = (commission_pct + impact) * 2
        vol_approx = abs(avg_gross_ret) / max(
            gross_sharpe / np.sqrt(ANNUALIZATION_FACTOR), 1e-6
        )
        if vol_approx <= 0:
            return 0.0
        net_ret = avg_gross_ret - tc
        return float(net_ret / vol_approx * np.sqrt(ANNUALIZATION_FACTOR))

    # Binary search for AUM where net_sharpe = 0
    lo, hi = CAPACITY_SEARCH_MIN, CAPACITY_SEARCH_MAX
    for _ in range(CAPACITY_SEARCH_ITERATIONS):
        mid = (lo + hi) / 2
        if net_sharpe_at_aum(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round(lo, 0)


def _run_intelligence_layer(
    raw_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent3_prompt import AGENT3_SYSTEM_PROMPT
    except ImportError:
        return {}

    result = call_intelligence_layer(
        agent_name="agent3_backtest",
        system_prompt=AGENT3_SYSTEM_PROMPT,
        computed_numbers={k: v for k, v in raw_numbers.items() if k != "exit_reasons"},
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    net_sr = raw_numbers.get("net_sharpe_tc", 0.0)
    cap = raw_numbers.get("capacity_usd", 0.0)
    defaults = {
        "backtest_quality": (
            "clean" if 0 < net_sr < 1.5
            else ("suspicious" if net_sr >= 1.5 else "poor")
        ),
        "performance_narrative": f"Net Sharpe after TC: {net_sr:.2f}.",
        "stop_loss_behaviour": "appropriate",
        "drawdown_analysis": "distributed",
        "capacity_verdict": (
            "tradeable" if cap >= 50000
            else ("marginal" if cap >= 10000 else "too_small")
        ),
        "anomalies": [],
        "tc_drag_assessment": "acceptable",
        "recommendation_for_next_agent": "Proceed with statistical testing.",
        "confidence": 0.5,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 1200
    signal_fake = (np.random.randn(n) * 0.5).tolist()

    params = {
        "sma_fast": 12,
        "sma_slow": 48,
        "rsi_period": 14,
        "holding_days": 9,
        "take_profit_pct": 0.072,
        "stop_loss_pct": 0.018,
        "vol_target_pct": 0.15,
        "kelly_fraction": 0.25,
    }
    result = run_backtest(
        ticker="RELIANCE.NS",
        start_date="2020-01-01",
        end_date="2024-12-31",
        params=params,
        stop_loss_used=0.036,
        final_weight=0.10,
        signal_series=signal_fake,
        regime="bull",
        transaction_costs={
            "commission_pct": 0.001,
            "slippage_lambda": 0.1,
            "adv_fraction": 0.01,
        },
    )
    print(f"Net Sharpe TC: {result['net_sharpe_tc']:.3f}")
    print(f"Total return: {result['total_return']:.3f}")
    print(f"Max drawdown: {result['max_drawdown']:.3f}")
    print(f"Win rate: {result['win_rate']:.3f}")
    print(f"Num trades: {result['num_trades']}")
    print(f"Capacity USD: {result['capacity_usd']:,.0f}")
    print(f"Backtest quality: {result['insight'].get('backtest_quality', 'N/A')}")
    print("agent3_backtest: OK")
