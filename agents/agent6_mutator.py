"""
Agent 6 — Bayesian Parameter Mutator

Computation layer: Gaussian Process via scikit-optimize.
On iteration 1: random sample.
On iteration N>1: GP Expected Improvement acquisition → top-3 candidates.
Intelligence layer: LLM selects one candidate and provides rationale.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

from constants import (
    GP_MAX_CANDIDATE_ATTEMPTS,
    GP_MIN_OBSERVATIONS_FOR_TRUST,
    GP_N_CANDIDATES,
    GP_N_INITIAL_POINTS,
    PRUNE_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

PARAM_KEYS = [
    "stop_loss_pct",
    "take_profit_pct",
    "holding_days",
    "sma_fast",
    "sma_slow",
    "rsi_period",
    "vol_target_pct",
    "kelly_fraction",
]

INTEGER_PARAMS = {"holding_days", "sma_fast", "sma_slow", "rsi_period"}


# ── Search space builder ──────────────────────────────────────────────────────

def _build_skopt_space(space_config: dict):
    from skopt.space import Integer, Real

    dims = []
    for key in PARAM_KEYS:
        bounds = space_config.get(key, [0.01, 0.1])
        lo, hi = float(bounds[0]), float(bounds[1])
        if key in INTEGER_PARAMS:
            dims.append(Integer(int(lo), int(hi), name=key))
        else:
            dims.append(Real(lo, hi, name=key))
    return dims


def _x_to_params(x: list, space_config: dict) -> dict:
    params = {}
    for i, key in enumerate(PARAM_KEYS):
        if key in INTEGER_PARAMS:
            params[key] = int(round(x[i]))
        else:
            params[key] = float(x[i])

    # Enforce sma_fast < sma_slow
    if params.get("sma_fast", 5) >= params.get("sma_slow", 20):
        params["sma_fast"] = max(5, params["sma_slow"] - 5)

    return params


def _params_to_x(params: dict) -> list:
    return [params.get(k, 0) for k in PARAM_KEYS]


def _random_params(space_config: dict, rng) -> dict:
    params = {}
    for key in PARAM_KEYS:
        bounds = space_config.get(key, [0.01, 0.1])
        lo, hi = float(bounds[0]), float(bounds[1])
        if key in INTEGER_PARAMS:
            params[key] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            params[key] = float(rng.uniform(lo, hi))

    if params.get("sma_fast", 5) >= params.get("sma_slow", 20):
        params["sma_fast"] = max(5, params["sma_slow"] - 5)
    return params


# ── GP proposal ───────────────────────────────────────────────────────────────

def _gp_propose_candidates(
    gp_observations: list,
    space_config: dict,
    n_candidates: int = GP_N_CANDIDATES,
    pruned_params: list = None,
    memory=None,
) -> tuple:
    """
    Fit GP on existing observations and return top-N candidates by EI.
    Returns (candidates_list, acquisition_values_list).
    """
    pruned_params = pruned_params or []

    try:
        from skopt import Optimizer
    except ImportError:
        logger.warning("scikit-optimize not installed — falling back to random.")
        return None, None

    dims = _build_skopt_space(space_config)

    optimizer = Optimizer(
        dimensions=dims,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=GP_N_INITIAL_POINTS,
        random_state=42,
    )

    # Feed existing observations to the GP
    for obs in gp_observations:
        p = obs["params"]
        x = _params_to_x(p)
        y = -float(obs.get("dsr", 0.0))  # minimize negative DSR
        try:
            optimizer.tell(x, y)
        except Exception as e:
            logger.debug(f"GP tell failed for obs: {e}")

    # Ask for multiple candidates
    candidates = []
    ei_scores = []
    seen = set()

    for _ in range(GP_MAX_CANDIDATE_ATTEMPTS):
        if len(candidates) >= n_candidates:
            break
        try:
            xs = optimizer.ask(n_points=n_candidates * 2)
        except Exception:
            xs = [optimizer.ask() for _ in range(n_candidates * 2)]

        for x in xs:
            if len(candidates) >= n_candidates:
                break
            params = _x_to_params(x, space_config)
            key = tuple(round(v, 3) for v in x)
            if key in seen:
                continue
            if memory and memory.is_pruned(params):
                continue
            # Check against already-pruned list directly
            is_pruned = _is_near_pruned(params, pruned_params)
            if is_pruned:
                continue
            seen.add(key)
            candidates.append(params)

            # Approximate EI by predicting on this point
            try:
                pred = optimizer.models[-1].predict([x]) if optimizer.models else [0]
                ei_scores.append(float(-pred[0]))
            except Exception:
                ei_scores.append(0.0)

    return candidates, ei_scores


def _is_near_pruned(params: dict, pruned_params: list) -> bool:
    """Check if params are too similar to any pruned parameter set."""
    for pp in pruned_params:
        diffs = []
        for k in PARAM_KEYS:
            v1, v2 = float(params.get(k, 0)), float(pp.get(k, 0))
            mx = max(abs(v1), abs(v2))
            if mx > 0:
                diffs.append(abs(v1 - v2) / mx)
        if diffs and max(diffs) < PRUNE_SIMILARITY_THRESHOLD:
            return True
    return False


# ── Main entry point ──────────────────────────────────────────────────────────

def propose_params(
    gp_observations: list,
    pruned_params: list,
    space: dict,
    iteration: int,
    memory=None,
    prior_iterations_summary: str = "",
    leaderboard_summary: str = "",
    regime_history: list = None,
    ic_history: list = None,
    last_agent5_insight: dict = None,
) -> dict:
    """
    R stage: propose next parameter set to test.
    """
    regime_history = regime_history or []
    ic_history = ic_history or []
    last_agent5_insight = last_agent5_insight or {}

    rng = np.random.default_rng(iteration)

    # ── Iteration 1 or too few observations: random ───────────────────────────
    if len(gp_observations) < GP_MIN_OBSERVATIONS_FOR_TRUST:
        params = _random_params(space, rng)
        # Ensure not pruned
        for _ in range(GP_MAX_CANDIDATE_ATTEMPTS):
            if not (memory and memory.is_pruned(params)):
                break
            params = _random_params(space, rng)

        return {
            "proposed_params": params,
            "acquisition_value": 0.0,
            "is_random": True,
            "selection_rationale": "Random exploration — too few GP observations.",
            "insight": {},
        }

    # ── GP proposal ───────────────────────────────────────────────────────────
    candidates, ei_scores = _gp_propose_candidates(
        gp_observations=gp_observations,
        space_config=space,
        n_candidates=GP_N_CANDIDATES,
        pruned_params=pruned_params,
        memory=memory,
    )

    if not candidates:
        # GP failed — fall back to random
        params = _random_params(space, rng)
        return {
            "proposed_params": params,
            "acquisition_value": 0.0,
            "is_random": True,
            "selection_rationale": "GP failed — random fallback.",
            "insight": {},
        }

    # Pad to exactly N if fewer returned
    while len(candidates) < GP_N_CANDIDATES:
        candidates.append(_random_params(space, rng))
        ei_scores.append(0.0)
    candidates = candidates[:GP_N_CANDIDATES]
    ei_scores = ei_scores[:GP_N_CANDIDATES]

    # ── Intelligence layer: LLM picks one candidate ───────────────────────────
    insight = _run_intelligence_layer(
        candidates=candidates,
        ei_scores=ei_scores,
        last_agent5_insight=last_agent5_insight,
        prior_iterations_summary=prior_iterations_summary,
        leaderboard_summary=leaderboard_summary,
        regime_history=regime_history[-20:],
        ic_history=ic_history[-10:],
        pruned_count=len(pruned_params),
        n_observations=len(gp_observations),
    )

    selected_idx = int(insight.get("selected_candidate_index", 0))
    selected_idx = max(0, min(selected_idx, len(candidates) - 1))

    # Handle GP override
    if insight.get("gp_override") and insight.get("override_params"):
        override = insight["override_params"]
        if isinstance(override, dict):
            valid = all(k in override for k in PARAM_KEYS)
            if valid:
                return {
                    "proposed_params": _validate_params(override, space),
                    "acquisition_value": 0.0,
                    "is_random": False,
                    "selection_rationale": insight.get("override_reason", "LLM override."),
                    "insight": insight,
                }

    selected_params = candidates[selected_idx]
    selected_ei = ei_scores[selected_idx] if ei_scores else 0.0

    return {
        "proposed_params": selected_params,
        "acquisition_value": round(float(selected_ei), 4),
        "is_random": False,
        "selection_rationale": insight.get(
            "selection_rationale", f"GP candidate {selected_idx}."
        ),
        "insight": insight,
    }


def _validate_params(params: dict, space: dict) -> dict:
    valid = {}
    for key in PARAM_KEYS:
        bounds = space.get(key, [0.01, 0.1])
        lo, hi = float(bounds[0]), float(bounds[1])
        v = params.get(key, (lo + hi) / 2)
        if key in INTEGER_PARAMS:
            valid[key] = int(np.clip(round(float(v)), int(lo), int(hi)))
        else:
            valid[key] = float(np.clip(float(v), lo, hi))
    if valid.get("sma_fast", 5) >= valid.get("sma_slow", 20):
        valid["sma_fast"] = max(5, valid["sma_slow"] - 5)
    return valid


def _run_intelligence_layer(
    candidates: list,
    ei_scores: list,
    last_agent5_insight: dict,
    prior_iterations_summary: str,
    leaderboard_summary: str,
    regime_history: list,
    ic_history: list,
    pruned_count: int,
    n_observations: int,
) -> dict:
    try:
        from llm_client import call_intelligence_layer
        from agents.prompts.agent6_prompt import AGENT6_SYSTEM_PROMPT
    except ImportError:
        return {"selected_candidate_index": 0}

    computed_numbers = {
        "gp_candidates": [
            {
                "index": i,
                "params": c,
                "ei_score": round(
                    ei_scores[i] if i < len(ei_scores) else 0.0, 4
                ),
            }
            for i, c in enumerate(candidates)
        ],
        "n_gp_observations": n_observations,
        "pruned_regions_count": pruned_count,
        "ic_history_recent": ic_history,
        "regime_history_recent": regime_history,
    }

    accumulated_context = {
        "last_agent5_verdict": last_agent5_insight,
        "leaderboard": leaderboard_summary,
    }

    result = call_intelligence_layer(
        agent_name="agent6_mutator",
        system_prompt=AGENT6_SYSTEM_PROMPT,
        computed_numbers=computed_numbers,
        accumulated_context=accumulated_context,
        prior_iterations_summary=prior_iterations_summary,
    )

    defaults = {
        "selected_candidate_index": 0,
        "selection_rationale": "Selected highest EI GP candidate.",
        "gp_override": False,
        "override_params": None,
        "override_reason": None,
        "regime_consideration": "No special regime consideration.",
        "expected_improvement_direction": "Improving DSR via GP guidance.",
        "confidence": 0.5,
    }
    return {**defaults, **result}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    space = {
        "stop_loss_pct":   [0.005, 0.10],
        "take_profit_pct": [0.01,  0.30],
        "holding_days":    [1, 60],
        "sma_fast":        [5, 50],
        "sma_slow":        [20, 200],
        "rsi_period":      [7, 21],
        "vol_target_pct":  [0.10, 0.25],
        "kelly_fraction":  [0.10, 0.50],
    }

    # Test iteration 1 — random
    result1 = propose_params(
        gp_observations=[], pruned_params=[], space=space, iteration=1
    )
    print(f"Iter 1 (random): {result1['proposed_params']}")
    print(f"Is random: {result1['is_random']}")

    # Test iteration 5 — GP with observations
    obs = [
        {"params": result1["proposed_params"], "dsr": 0.45, "iter": 1},
        {"params": _random_params(space, np.random.default_rng(2)), "dsr": 0.62, "iter": 2},
        {"params": _random_params(space, np.random.default_rng(3)), "dsr": 0.71, "iter": 3},
        {"params": _random_params(space, np.random.default_rng(4)), "dsr": 0.58, "iter": 4},
    ]
    result5 = propose_params(
        gp_observations=obs, pruned_params=[], space=space, iteration=5
    )
    print(f"Iter 5 (GP): {result5['proposed_params']}")
    print(f"EI score: {result5['acquisition_value']}")
    print(f"Rationale: {result5['selection_rationale']}")
    print("agent6_mutator: OK")
