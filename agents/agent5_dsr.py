"""
Agent 5 — Deflated Sharpe Ratio (The Judge)

Computation layer: DSR, PBO, MinTRL (Bailey & López de Prado 2014).
Intelligence layer: LLM synthesises all prior agent insights into a final verdict.
"""

import logging

import numpy as np
from scipy.stats import norm

from constants import (
    ANNUALIZATION_FACTOR,
    DSR_ACCEPT_THRESHOLD,
    DSR_MARGINAL_THRESHOLD,
    EULER_MASCHERONI,
    MIN_TRL_MAX,
)

logger = logging.getLogger(__name__)


# ── DSR formula ───────────────────────────────────────────────────────────────

def compute_dsr(
    sr_hat: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sr_benchmark: float = 0.0,
) -> tuple:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns (dsr, expected_max_sr, trials_penalty).
    sr_hat is annualized. n_obs is number of daily observations.
    """
    n_trials = max(n_trials, 1)

    # Expected maximum Sharpe from N trials (Extreme Value Theory)
    if n_trials == 1:
        expected_max_sr = sr_benchmark
    else:
        try:
            term1 = (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
            term2 = EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
            expected_max_sr = term1 + term2
        except Exception:
            expected_max_sr = sr_benchmark

    # Convert annualized SR to per-period SR
    sr_period = sr_hat / np.sqrt(ANNUALIZATION_FACTOR)
    expected_period = expected_max_sr / np.sqrt(ANNUALIZATION_FACTOR)

    # Variance adjustment for non-normality
    variance_adj = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period ** 2
    variance_adj = max(variance_adj, 1e-6)

    numerator = (sr_period - expected_period) * np.sqrt(max(n_obs - 1, 1))
    denominator = np.sqrt(variance_adj)

    z = numerator / denominator
    dsr = float(norm.cdf(z))
    dsr = float(np.clip(dsr, 0.0, 1.0))

    trials_penalty = max(expected_max_sr - sr_benchmark, 0.0)
    return dsr, float(expected_max_sr), float(trials_penalty)


def compute_min_trl(
    sr_target: float,
    sr_benchmark: float,
    skew: float,
    kurt: float,
    alpha: float = 0.05,
) -> int:
    """Minimum track record length to confirm SR is real (in daily obs)."""
    if sr_target <= sr_benchmark:
        return MIN_TRL_MAX
    z = norm.ppf(1 - alpha)
    variance_adj = 1.0 - skew * sr_target + ((kurt - 1.0) / 4.0) * sr_target ** 2
    variance_adj = max(variance_adj, 1e-6)
    sr_period = (sr_target - sr_benchmark) / np.sqrt(ANNUALIZATION_FACTOR)
    if sr_period <= 0:
        return MIN_TRL_MAX
    min_trl = (1.0 + variance_adj) * (z / sr_period) ** 2
    return int(np.ceil(min_trl))


def compute_pbo(sr_hat: float, n_trials: int) -> float:
    """
    Probability of Backtest Overfitting — simplified combinatorial estimate.
    Uses the fraction of random trials that would achieve the observed SR.
    """
    if n_trials <= 1:
        return 0.5
    n_trials = max(n_trials, 1)
    _, expected_max, _ = compute_dsr(
        sr_hat, ANNUALIZATION_FACTOR, 0.0, 3.0, n_trials, sr_benchmark=0.0
    )
    if expected_max <= 0:
        return 0.1
    pbo = float(np.clip(expected_max / max(sr_hat, 0.01), 0.0, 1.0))
    return round(pbo, 4)


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_dsr_verdict(
    net_sharpe_tc: float,
    skewness: float,
    kurtosis: float,
    n_obs: int,
    n_trials: int,
    leaderboard: list,
    sharpe_ci_low: float,
    sharpe_ci_high: float,
    alpha_level: float = 0.05,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
    leaderboard_summary: str = "",
) -> dict:
    """
    Stage 5: Compute DSR verdict and LLM synthesis.
    """
    accumulated_context = accumulated_context or {}

    # Use excess kurtosis (scipy returns excess kurtosis, add 3 for Bailey formula)
    kurt_total = kurtosis + 3.0

    dsr, expected_max_sr, trials_penalty = compute_dsr(
        sr_hat=net_sharpe_tc,
        n_obs=n_obs,
        skew=skewness,
        kurt=kurt_total,
        n_trials=n_trials,
    )

    min_trl = compute_min_trl(
        sr_target=net_sharpe_tc / np.sqrt(ANNUALIZATION_FACTOR),
        sr_benchmark=0.0,
        skew=skewness,
        kurt=kurt_total,
        alpha=alpha_level,
    )
    min_trl_satisfied = bool(n_obs >= min_trl)

    pbo = compute_pbo(net_sharpe_tc, n_trials)

    # DSR verdict thresholds
    if dsr > DSR_ACCEPT_THRESHOLD:
        dsr_verdict = "accept"
    elif dsr > DSR_MARGINAL_THRESHOLD:
        dsr_verdict = "marginal"
    else:
        dsr_verdict = "reject"

    # Leaderboard comparison
    leader_dsr = 0.0
    is_new_leader = False
    if leaderboard:
        leader_dsr = float(leaderboard[0].get("dsr", 0.0))
        is_new_leader = dsr > leader_dsr
    else:
        is_new_leader = True

    raw_numbers = {
        "dsr": round(dsr, 4),
        "dsr_verdict": dsr_verdict,
        "min_trl_days": min_trl,
        "min_trl_satisfied": min_trl_satisfied,
        "pbo": round(pbo, 4),
        "is_new_leader": is_new_leader,
        "leader_dsr": round(leader_dsr, 4),
        "trials_penalty": round(trials_penalty, 4),
        "expected_max_sr": round(expected_max_sr, 4),
        "net_sharpe_tc": round(net_sharpe_tc, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "n_obs": n_obs,
        "n_trials": n_trials,
        "sharpe_ci_low": round(sharpe_ci_low, 4),
        "sharpe_ci_high": round(sharpe_ci_high, 4),
    }

    insight = _run_intelligence_layer(
        raw_numbers,
        accumulated_context,
        prior_iterations_summary,
        leaderboard_summary,
    )

    return {**raw_numbers, "insight": insight}


def _run_intelligence_layer(
    raw_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str,
    leaderboard_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent5_prompt import AGENT5_SYSTEM_PROMPT
    except ImportError:
        return {}

    enriched_summary = f"{prior_iterations_summary}\n\nLEADERBOARD:\n{leaderboard_summary}"

    result = call_intelligence_layer(
        agent_name="agent5_dsr",
        system_prompt=AGENT5_SYSTEM_PROMPT,
        computed_numbers=raw_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=enriched_summary,
        max_tokens=1200,
    )

    dsr = raw_numbers.get("dsr", 0.0)
    defaults = {
        "final_verdict": raw_numbers.get("dsr_verdict", "reject"),
        "dsr_interpretation": (
            f"DSR={dsr:.3f}. Verdict: {raw_numbers.get('dsr_verdict', 'reject')}."
        ),
        "pbo_interpretation": f"PBO={raw_numbers.get('pbo', 0.5):.2f}.",
        "quality_adjusted_verdict": raw_numbers.get("dsr_verdict", "reject"),
        "quality_adjustment_reason": "",
        "key_issues": [],
        "strengths": [],
        "leaderboard_position": (
            "new_leader" if raw_numbers.get("is_new_leader") else "middle"
        ),
        "verdict_narrative": f"Strategy scored DSR={dsr:.3f} on this iteration.",
        "mutator_instruction": "Continue Bayesian search with GP recommendations.",
        "confidence": 0.6,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    dsr, exp_sr, penalty = compute_dsr(
        sr_hat=1.34, n_obs=1008, skew=-0.31, kurt=4.1 + 3, n_trials=23
    )
    print(f"DSR: {dsr:.4f}")
    print(f"Expected max SR: {exp_sr:.4f}")
    print(f"Trials penalty: {penalty:.4f}")

    min_trl = compute_min_trl(
        sr_target=1.34 / np.sqrt(252), sr_benchmark=0.0,
        skew=-0.31, kurt=4.1 + 3,
    )
    print(f"Min TRL: {min_trl} days")

    pbo = compute_pbo(sr_hat=1.34, n_trials=23)
    print(f"PBO: {pbo:.4f}")

    result = compute_dsr_verdict(
        net_sharpe_tc=1.34,
        skewness=-0.31,
        kurtosis=4.1,
        n_obs=1008,
        n_trials=23,
        leaderboard=[{"dsr": 0.91, "rank": 1}],
        sharpe_ci_low=0.84,
        sharpe_ci_high=1.87,
    )
    print(f"Verdict: {result['dsr_verdict']}")
    print(f"Is new leader: {result['is_new_leader']}")
    print(f"Final verdict (LLM): {result['insight'].get('final_verdict', 'N/A')}")
    print("agent5_dsr: OK")
