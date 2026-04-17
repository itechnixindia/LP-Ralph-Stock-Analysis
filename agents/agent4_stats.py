"""
Agent 4 — Statistical Tests

Computation layer: Jarque-Bera, Welch t-test, bootstrap Sharpe CI, CVaR/VaR,
skewness, kurtosis, Sortino, Calmar, Omega.
Intelligence layer: LLM evaluates statistical trustworthiness.
"""

import logging
from typing import List

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ── Computation layer ─────────────────────────────────────────────────────────

def run_stats(
    daily_returns: list,
    trade_log: list,
    gross_sharpe: float,
    net_sharpe_tc: float,
    regime: str = "unknown",
    alpha_level: float = 0.05,
    cvar_confidence: float = 0.95,
    n_bootstrap: int = 10000,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
) -> dict:
    """
    Stage 4: Run all hypothesis tests on daily returns.
    """
    accumulated_context = accumulated_context or {}

    arr = np.array(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 20:
        return _empty_stats(regime)

    # ── Normality test ────────────────────────────────────────────────────────
    jb_stat, jb_pval = stats.jarque_bera(arr)
    is_normal = bool(jb_pval >= alpha_level)

    # ── Welch t-test (H0: mean return <= 0) ───────────────────────────────────
    t_stat, t_pval_two = stats.ttest_1samp(arr, popmean=0)
    t_pval_one = t_pval_two / 2 if t_stat > 0 else 1.0
    ttest_reject_null = bool(t_pval_one < alpha_level and t_stat > 0)

    # ── Bootstrap Sharpe CI ───────────────────────────────────────────────────
    sharpe_ci_low, sharpe_ci_high = _bootstrap_sharpe_ci(arr, n_bootstrap)
    sharpe_ci_width = sharpe_ci_high - sharpe_ci_low

    # ── Tail risk ─────────────────────────────────────────────────────────────
    threshold = np.percentile(arr, (1 - cvar_confidence) * 100)
    var_95 = float(threshold)
    tail = arr[arr <= threshold]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    skewness = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis

    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    sortino = _compute_sortino(arr)
    total_ret = float(np.prod(1 + arr) - 1)
    max_dd = _compute_max_drawdown_from_returns(arr)
    calmar = (total_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0
    omega = _compute_omega(arr)

    # ── Regime-conditional stats ───────────────────────────────────────────────
    regime_sharpe = net_sharpe_tc
    regime_win_rate = 0.0
    if trade_log:
        regime_win_rate = sum(1 for t in trade_log if t.get("net_return", 0) > 0) / len(trade_log)

    raw_numbers = {
        "jarque_bera_stat": round(float(jb_stat), 4),
        "jarque_bera_pval": round(float(jb_pval), 6),
        "is_normal": is_normal,
        "ttest_stat": round(float(t_stat), 4),
        "ttest_pval": round(float(t_pval_one), 6),
        "ttest_reject_null": ttest_reject_null,
        "sharpe_ci_low": round(sharpe_ci_low, 4),
        "sharpe_ci_high": round(sharpe_ci_high, 4),
        "sharpe_ci_width": round(sharpe_ci_width, 4),
        "cvar_95": round(cvar_95, 6),
        "var_95": round(var_95, 6),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega": round(omega, 4),
        "regime": regime,
        "regime_sharpe": round(regime_sharpe, 4),
        "regime_win_rate": round(regime_win_rate, 4),
        "n_obs": len(arr),
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe_tc": round(net_sharpe_tc, 4),
    }

    insight = _run_intelligence_layer(
        raw_numbers, accumulated_context, prior_iterations_summary
    )

    return {**raw_numbers, "insight": insight}


def _bootstrap_sharpe_ci(
    arr: np.ndarray, n_bootstrap: int = 10000
) -> tuple:
    if len(arr) < 10:
        return 0.0, 0.0
    sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        std = np.std(sample)
        if std > 0:
            sharpes.append(float(np.mean(sample) / std * np.sqrt(252)))
    if not sharpes:
        return 0.0, 0.0
    return float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


def _compute_sortino(arr: np.ndarray) -> float:
    downside = arr[arr < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    downside_std = float(np.std(downside)) * np.sqrt(252)
    annual_ret = float(np.mean(arr)) * 252
    return annual_ret / downside_std


def _compute_max_drawdown_from_returns(arr: np.ndarray) -> float:
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(np.min(dd))


def _compute_omega(arr: np.ndarray, threshold: float = 0.0) -> float:
    gains = arr[arr > threshold] - threshold
    losses = threshold - arr[arr <= threshold]
    if losses.sum() == 0:
        return 10.0
    return float(gains.sum() / losses.sum())


def _empty_stats(regime: str) -> dict:
    return {
        "jarque_bera_stat": 0.0,
        "jarque_bera_pval": 1.0,
        "is_normal": True,
        "ttest_stat": 0.0,
        "ttest_pval": 1.0,
        "ttest_reject_null": False,
        "sharpe_ci_low": 0.0,
        "sharpe_ci_high": 0.0,
        "sharpe_ci_width": 0.0,
        "cvar_95": 0.0,
        "var_95": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "omega": 1.0,
        "regime": regime,
        "regime_sharpe": 0.0,
        "regime_win_rate": 0.0,
        "n_obs": 0,
        "gross_sharpe": 0.0,
        "net_sharpe_tc": 0.0,
        "insight": {},
    }


def _run_intelligence_layer(
    raw_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent4_prompt import AGENT4_SYSTEM_PROMPT
    except ImportError:
        return {}

    result = call_intelligence_layer(
        agent_name="agent4_stats",
        system_prompt=AGENT4_SYSTEM_PROMPT,
        computed_numbers=raw_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    jb_pval = raw_numbers.get("jarque_bera_pval", 1.0)
    kurt = raw_numbers.get("kurtosis", 3.0)
    use_bootstrap = bool(jb_pval < 0.05 and kurt > 5)

    defaults = {
        "stats_quality": "trustworthy" if raw_numbers.get("ttest_reject_null") else "uncertain",
        "primary_test_to_trust": "bootstrap" if use_bootstrap else "ttest",
        "cvar_interpretation": f"CVaR 95%: {raw_numbers.get('cvar_95', 0):.3f} per day.",
        "distribution_assessment": "normal_enough" if raw_numbers.get("is_normal") else "fat_tailed",
        "skew_risk": "high_crash_risk" if raw_numbers.get("skewness", 0) < -0.5 else "low",
        "sortino_insight": "Sortino computed.",
        "trustworthiness_flags": [],
        "overall_stats_verdict": "accept" if raw_numbers.get("ttest_reject_null") else "marginal",
        "recommendation_for_next_agent": "Proceed with DSR computation.",
        "confidence": 0.5,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = np.random.normal(0.0008, 0.015, 1000).tolist()
    fake_trades = [
        {"net_return": r, "exit_reason": "take_profit" if r > 0 else "stop_loss"}
        for r in np.random.normal(0.003, 0.02, 50)
    ]
    result = run_stats(
        daily_returns=fake_returns,
        trade_log=fake_trades,
        gross_sharpe=1.4,
        net_sharpe_tc=1.1,
        regime="bull",
        alpha_level=0.05,
        cvar_confidence=0.95,
        n_bootstrap=1000,
    )
    print(f"JB p-val: {result['jarque_bera_pval']:.4f}")
    print(f"T-test p-val: {result['ttest_pval']:.4f} (reject null: {result['ttest_reject_null']})")
    print(f"Bootstrap Sharpe CI: [{result['sharpe_ci_low']:.2f}, {result['sharpe_ci_high']:.2f}]")
    print(f"CVaR 95%: {result['cvar_95']:.4f}")
    print(f"Sortino: {result['sortino']:.3f}")
    print(f"Stats quality: {result['insight'].get('stats_quality', 'N/A')}")
    print("agent4_stats: OK")
