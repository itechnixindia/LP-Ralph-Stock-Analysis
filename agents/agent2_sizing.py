"""
Agent 2 — Position Sizing + Dynamic Stop-Loss

Computation layer: Kelly fraction, GARCH(1,1) sigma, vol-targeting weight.
Intelligence layer: LLM assesses sizing appropriateness and volatility regime.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


# ── Computation layer ─────────────────────────────────────────────────────────

def compute_sizing(
    factor_alpha: float,
    return_series: list,
    garch_sigma: float,
    params: dict,
    dynamic_stop_multiplier: float = 2.0,
    accumulated_context: dict = None,
    prior_iterations_summary: str = "",
) -> dict:
    """
    Stage 2: Compute position size, dynamic stop-loss.
    """
    accumulated_context = accumulated_context or {}

    returns = np.array(return_series, dtype=float)
    returns = returns[np.isfinite(returns)]

    kelly_fraction_param = float(params.get("kelly_fraction", 0.25))
    vol_target_pct = float(params.get("vol_target_pct", 0.15))
    base_stop_pct = float(params.get("stop_loss_pct", 0.02))

    # ── Kelly fraction ────────────────────────────────────────────────────────
    annualized_alpha = factor_alpha * 252
    annualized_var = float(np.var(returns)) * 252 if len(returns) > 10 else 0.04

    if annualized_var > 0 and annualized_alpha > 0:
        kelly_full = annualized_alpha / annualized_var
    else:
        kelly_full = 0.0

    kelly_full = float(np.clip(kelly_full, 0.0, 5.0))
    kelly_applied = kelly_full * kelly_fraction_param
    kelly_applied = float(np.clip(kelly_applied, 0.0, 1.0))

    # ── GARCH(1,1) re-fit ─────────────────────────────────────────────────────
    new_garch_sigma = garch_sigma
    if len(returns) >= 100:
        new_garch_sigma = _fit_garch(returns, fallback_sigma=garch_sigma)
    else:
        new_garch_sigma = float(np.std(returns[-50:])) if len(returns) >= 10 else garch_sigma

    new_garch_sigma = max(new_garch_sigma, 1e-4)

    # ── Volatility-targeting weight ───────────────────────────────────────────
    realized_vol_ann = new_garch_sigma * np.sqrt(252)
    if realized_vol_ann > 0:
        vol_target_weight = vol_target_pct / realized_vol_ann
    else:
        vol_target_weight = 1.0
    vol_target_weight = float(np.clip(vol_target_weight, 0.0, 2.0))

    # ── Final weight ──────────────────────────────────────────────────────────
    final_weight = min(kelly_applied, vol_target_weight)
    final_weight = float(np.clip(final_weight, 0.01, 1.0))

    # ── Dynamic stop-loss ─────────────────────────────────────────────────────
    stop_loss_dynamic = dynamic_stop_multiplier * new_garch_sigma
    stop_loss_used = max(base_stop_pct, stop_loss_dynamic)

    raw_numbers = {
        "kelly_full": round(kelly_full, 4),
        "kelly_applied": round(kelly_applied, 4),
        "vol_target_weight": round(vol_target_weight, 4),
        "final_weight": round(final_weight, 4),
        "stop_loss_dynamic": round(stop_loss_dynamic, 4),
        "stop_loss_used": round(stop_loss_used, 4),
        "garch_sigma": round(new_garch_sigma, 6),
        "new_garch_sigma": round(new_garch_sigma, 6),
        "annualized_alpha": round(annualized_alpha, 4),
        "annualized_vol": round(realized_vol_ann, 4),
        "kelly_fraction_param": kelly_fraction_param,
        "vol_target_pct": vol_target_pct,
        "base_stop_pct": base_stop_pct,
    }

    insight = _run_intelligence_layer(
        raw_numbers, accumulated_context, prior_iterations_summary
    )

    return {**raw_numbers, "insight": insight}


def _fit_garch(returns: np.ndarray, fallback_sigma: float = 0.02) -> float:
    try:
        from arch import arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pct_returns = returns * 100
            model = arch_model(pct_returns, vol="Garch", p=1, q=1, dist="normal")
            res = model.fit(disp="off", show_warning=False)
            sigma_pct = float(res.conditional_volatility.iloc[-1])
            return sigma_pct / 100.0
    except Exception as e:
        logger.warning(f"GARCH fit failed: {e}. Using rolling std fallback.")
        return float(np.std(returns[-60:])) if len(returns) >= 60 else fallback_sigma


def _run_intelligence_layer(
    raw_numbers: dict,
    accumulated_context: dict,
    prior_iterations_summary: str,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent2_prompt import AGENT2_SYSTEM_PROMPT
    except ImportError:
        return {}

    result = call_intelligence_layer(
        agent_name="agent2_sizing",
        system_prompt=AGENT2_SYSTEM_PROMPT,
        computed_numbers=raw_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    sigma = raw_numbers.get("garch_sigma", 0.02)
    if sigma < 0.01:
        vol_regime = "low"
    elif sigma < 0.02:
        vol_regime = "normal"
    elif sigma < 0.03:
        vol_regime = "elevated"
    else:
        vol_regime = "extreme"

    defaults = {
        "sizing_verdict": "appropriate",
        "kelly_interpretation": "Kelly fraction within acceptable range.",
        "volatility_regime": vol_regime,
        "stop_loss_assessment": "appropriate",
        "adjustment_applied": False,
        "adjustment_reason": "",
        "adjusted_weight_recommendation": raw_numbers.get("final_weight", 0.1),
        "warnings": [],
        "recommendation_for_next_agent": "Proceed with backtest.",
        "confidence": 0.5,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = np.random.normal(0.0005, 0.015, 500).tolist()
    params = {
        "kelly_fraction": 0.25,
        "vol_target_pct": 0.15,
        "stop_loss_pct": 0.018,
    }
    result = compute_sizing(
        factor_alpha=0.00031,
        return_series=fake_returns,
        garch_sigma=0.018,
        params=params,
        dynamic_stop_multiplier=2.0,
    )
    print(f"Kelly full: {result['kelly_full']:.3f}")
    print(f"Kelly applied: {result['kelly_applied']:.3f}")
    print(f"Final weight: {result['final_weight']:.3f}")
    print(f"Stop loss used: {result['stop_loss_used']:.4f}")
    print(f"GARCH sigma: {result['garch_sigma']:.5f}")
    print(f"Volatility regime: {result['insight'].get('volatility_regime', 'N/A')}")
    print("agent2_sizing: OK")
