"""
Orchestrator — RALPH Loop Controller

R → A → L → P → H → (repeat or halt)

Run: python orchestrator.py
     python orchestrator.py --config path/to/config.yaml
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # Group A — metadata
    "iter", "timestamp", "iter_duration_s", "ticker", "n_trials_so_far",
    "mutator_ei", "is_random",
    # Group B — params
    "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast", "sma_slow",
    "rsi_period", "vol_target_pct", "kelly_fraction",
    # Group C — signal
    "ic_mean", "ic_std", "ic_ir", "factor_alpha", "factor_beta_mkt",
    "factor_beta_smb", "factor_beta_hml", "factor_r2", "half_life_days",
    # Group D — sizing
    "kelly_full", "kelly_applied", "vol_target_weight", "final_weight",
    "stop_loss_dynamic", "stop_loss_used", "garch_sigma",
    # Group E — backtest
    "gross_sharpe", "net_sharpe_tc", "total_return", "max_drawdown", "win_rate",
    "num_trades", "avg_holding_days", "turnover_annual", "total_tc_paid",
    "capacity_usd", "regime",
    # Group F — stats
    "jarque_bera_pval", "is_normal", "ttest_stat", "ttest_pval", "ttest_reject_null",
    "sharpe_ci_low", "sharpe_ci_high", "sharpe_ci_width", "cvar_95", "var_95",
    "skewness", "kurtosis", "sortino", "calmar", "omega",
    "regime_sharpe", "regime_win_rate",
    # Group G — DSR verdict
    "dsr", "dsr_verdict", "min_trl_days", "min_trl_satisfied", "pbo",
    "is_new_leader", "leader_dsr", "trials_penalty", "expected_max_sr",
    # Group H — RALPH metadata
    "pruned", "prune_reasons", "cumul_best_dsr", "dsr_delta",
    "halt_triggered", "halt_reason",
    # Group I — LLM narratives
    "agent1_signal_quality", "agent1_narrative",
    "agent2_sizing_verdict",
    "agent3_backtest_quality", "agent3_anomalies",
    "agent4_stats_quality",
    "agent5_final_verdict", "agent5_verdict_narrative", "agent5_mutator_instruction",
    "agent6_selection_rationale",
]


def init_csv(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_csv_row(csv_path: str, row: dict):
    full_row = {col: row.get(col, "") for col in CSV_COLUMNS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(full_row)


def update_last_csv_row(csv_path: str, **updates):
    """Re-write the last row with updated halt fields."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    rows[-1].update(updates)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ── Regime detection ──────────────────────────────────────────────────────────

def compute_regime(daily_returns: list) -> str:
    import numpy as np
    if not daily_returns or len(daily_returns) < 20:
        return "unknown"
    arr = np.array(daily_returns[-60:])
    total_ret = float(np.prod(1 + arr) - 1) * (252 / len(arr))
    if total_ret > 0.10:
        return "bull"
    elif total_ret < -0.10:
        return "bear"
    return "sideways"


# ── Final report ──────────────────────────────────────────────────────────────

def write_final_report(memory, config: dict, halt_reason: str, total_runtime_s: float):
    report_path = config["output"].get("report_path", "final_report.txt")
    ticker = config["stock"]["ticker"]
    start_d = config["stock"]["start_date"]
    end_d = config["stock"]["end_date"]

    lines = [
        "=" * 60,
        "QUANT RALPH LOOP — FINAL REPORT",
        "=" * 60,
        f"Stock:          {ticker}",
        f"Period:         {start_d} to {end_d}",
        f"Iterations run: {memory.n_trials}",
        f"Halt reason:    {halt_reason}",
        f"Total runtime:  {int(total_runtime_s // 60)}m {int(total_runtime_s % 60)}s",
        "=" * 60,
        "",
    ]

    if memory.leaderboard:
        best = memory.leaderboard[0]
        dsr_val = best.get("dsr", 0.0)
        verdict = "ACCEPTED" if dsr_val > 0.95 else ("MARGINAL" if dsr_val > 0.5 else "REJECTED")
        lines += [
            "BEST STRATEGY FOUND",
            "-" * 50,
            f"DSR verdict:      {verdict} (DSR = {dsr_val:.3f})",
            f"Iteration:        {best.get('iter', '?')}",
            f"Net Sharpe TC:    {best.get('net_sharpe_tc', 0):.3f}",
            f"PBO:              {best.get('pbo', 0):.2f}",
            "",
            "Parameters:",
        ]
        for k, v in best.get("params", {}).items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines += [
        "-" * 50,
        "HYPOTHESIS SUMMARY",
        "-" * 50,
    ]
    n_accept = sum(1 for e in memory.leaderboard if e.get("dsr", 0) > 0.95)
    n_marginal = sum(1 for e in memory.leaderboard if 0.5 < e.get("dsr", 0) <= 0.95)
    lines += [
        f"Accepted (DSR>0.95): {n_accept}",
        f"Marginal (DSR 0.5-0.95): {n_marginal}",
        f"Total iterations: {memory.n_trials}",
        f"Total pruned: {len(memory.pruned_params)}",
        "",
        "=" * 60,
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Final report written to {report_path}")


# ── Halt check ────────────────────────────────────────────────────────────────

def check_halt(memory, config: dict, iteration: int) -> tuple:
    r = config["ralph"]
    if iteration >= r["max_iterations"]:
        return True, "max_iterations_reached"
    if memory.dsr_plateau_detected(
        window=r["dsr_plateau_window"],
        threshold=r["dsr_plateau_threshold"],
    ):
        return True, "dsr_plateau"
    if memory.leaderboard and memory.leaderboard[0].get("pbo", 1.0) < r["halt_pbo_threshold"]:
        return True, "pbo_threshold_satisfied"
    return False, None


# ── Main RALPH loop ───────────────────────────────────────────────────────────

def run_ralph_loop(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ticker = config["stock"]["ticker"]
    start_date = config["stock"]["start_date"]
    end_date = config["stock"]["end_date"]
    benchmark = config["stock"]["benchmark"]
    memory_dir = config["output"]["memory_dir"]
    csv_path = config["output"]["csv_path"]

    from memory_store import MemoryStore
    memory = MemoryStore(memory_dir=memory_dir)
    memory.load()

    iteration = memory.n_trials + 1
    init_csv(csv_path)

    loop_start = time.time()
    halt_reason_final = "unknown"

    logger.info(f"Starting RALPH loop from iteration {iteration}")
    logger.info(f"Ticker: {ticker} | Period: {start_date} to {end_date}")

    while True:
        iter_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"  ITERATION {iteration}")
        logger.info(f"{'='*50}")

        try:
            # ── R: REPEAT — propose params ──────────────────────────────────
            from agents.agent6_mutator import propose_params

            last_a5_insight = {}
            if memory.narrative_history:
                last_a5_insight = memory.narrative_history[-1].get("agent5", {})

            mutator_out = propose_params(
                gp_observations=memory.gp_observations,
                pruned_params=memory.pruned_params,
                space=config["hyperparameter_space"],
                iteration=iteration,
                memory=memory,
                prior_iterations_summary=memory.get_prior_iterations_summary(last_n=5),
                leaderboard_summary=memory.get_leaderboard_summary(),
                regime_history=memory.regime_history,
                ic_history=memory.ic_history,
                last_agent5_insight=last_a5_insight,
            )
            params = mutator_out["proposed_params"]
            logger.info(f"  R: params={params}")

            # Accumulated context — grows as agents run
            accumulated_context = {
                "agent6": mutator_out.get("insight", {}),
            }
            prior_summary = memory.get_prior_iterations_summary(last_n=5)

            # ── A: ASSESS — pipeline stages 1→4→5 ──────────────────────────

            # Stage 1: Signal
            logger.info("  A: Stage 1 — signal")
            from agents.agent1_signal import compute_signal
            sig_out = compute_signal(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                params=params,
                benchmark=benchmark,
                ic_history=memory.ic_history,
                accumulated_context=accumulated_context,
                prior_iterations_summary=prior_summary,
            )
            accumulated_context["agent1"] = sig_out.get("insight", {})
            logger.info(f"     IC={sig_out['ic_mean']:.3f}  alpha={sig_out['factor_alpha']:.5f}  hl={sig_out['half_life_days']:.1f}d")

            # Stage 2: Sizing
            logger.info("  A: Stage 2 — sizing")
            from agents.agent2_sizing import compute_sizing
            siz_out = compute_sizing(
                factor_alpha=sig_out["factor_alpha"],
                return_series=sig_out["return_series"],
                garch_sigma=memory.garch_sigma.get(ticker, 0.02),
                params=params,
                dynamic_stop_multiplier=config["risk"]["dynamic_stop_multiplier"],
                accumulated_context=accumulated_context,
                prior_iterations_summary=prior_summary,
            )
            accumulated_context["agent2"] = siz_out.get("insight", {})
            logger.info(f"     kelly={siz_out['kelly_applied']:.3f}  weight={siz_out['final_weight']:.3f}  stop={siz_out['stop_loss_used']:.4f}")

            # Stage 3: Backtest
            logger.info("  A: Stage 3 — backtest")
            regime = memory.regime_history[-1] if memory.regime_history else "unknown"
            from agents.agent3_backtest import run_backtest
            bt_out = run_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                params=params,
                stop_loss_used=siz_out["stop_loss_used"],
                final_weight=siz_out["final_weight"],
                signal_series=sig_out["signal_series"],
                regime=regime,
                transaction_costs=config["transaction_costs"],
                accumulated_context=accumulated_context,
                prior_iterations_summary=prior_summary,
            )
            accumulated_context["agent3"] = bt_out.get("insight", {})
            logger.info(f"     net_sharpe={bt_out['net_sharpe_tc']:.3f}  trades={bt_out['num_trades']}  cap=${bt_out['capacity_usd']:,.0f}")

            # Stage 4: Stats
            logger.info("  A: Stage 4 — statistics")
            from agents.agent4_stats import run_stats
            stats_out = run_stats(
                daily_returns=bt_out["daily_returns"],
                trade_log=bt_out["trade_log"],
                gross_sharpe=bt_out["gross_sharpe"],
                net_sharpe_tc=bt_out["net_sharpe_tc"],
                regime=regime,
                alpha_level=config["risk"]["alpha_level"],
                cvar_confidence=config["risk"]["cvar_confidence"],
                accumulated_context=accumulated_context,
                prior_iterations_summary=prior_summary,
            )
            accumulated_context["agent4"] = stats_out.get("insight", {})
            logger.info(f"     pval={stats_out['ttest_pval']:.4f}  cvar={stats_out['cvar_95']:.4f}  sortino={stats_out['sortino']:.3f}")

            # Stage 5: DSR verdict
            logger.info("  A: Stage 5 — DSR verdict")
            from agents.agent5_dsr import compute_dsr_verdict
            dsr_out = compute_dsr_verdict(
                net_sharpe_tc=bt_out["net_sharpe_tc"],
                skewness=stats_out["skewness"],
                kurtosis=stats_out["kurtosis"],
                n_obs=len(bt_out["daily_returns"]),
                n_trials=memory.n_trials + 1,
                leaderboard=memory.leaderboard,
                sharpe_ci_low=stats_out["sharpe_ci_low"],
                sharpe_ci_high=stats_out["sharpe_ci_high"],
                alpha_level=config["risk"]["alpha_level"],
                accumulated_context=accumulated_context,
                prior_iterations_summary=prior_summary,
                leaderboard_summary=memory.get_leaderboard_summary(),
            )
            logger.info(f"     DSR={dsr_out['dsr']:.3f}  verdict={dsr_out['dsr_verdict']}  PBO={dsr_out['pbo']:.2f}")

            iter_duration = time.time() - iter_start

            # ── L: LEARN — update memory ────────────────────────────────────
            logger.info("  L: Updating memory")
            memory.gp_observations.append({
                "params": params,
                "dsr": dsr_out["dsr"],
                "iter": iteration,
            })
            memory.ic_history.append(sig_out["ic_mean"])
            memory.dsr_history.append(dsr_out["dsr"])
            memory.n_trials = iteration
            memory.garch_sigma[ticker] = siz_out["new_garch_sigma"]
            memory.update_leaderboard({
                "params": params,
                "dsr": dsr_out["dsr"],
                "iter": iteration,
                "pbo": dsr_out["pbo"],
                "net_sharpe_tc": bt_out["net_sharpe_tc"],
                "capacity_usd": bt_out["capacity_usd"],
                "cvar_95": stats_out["cvar_95"],
                "sortino": stats_out["sortino"],
                "max_drawdown": bt_out["max_drawdown"],
                "win_rate": bt_out["win_rate"],
            })

            # Update regime every 10 iterations
            if iteration % 10 == 0:
                regime_label = compute_regime(bt_out["daily_returns"])
                memory.regime_history.append(regime_label)
                logger.info(f"     Regime update: {regime_label}")

            # Save LLM narratives
            narrative_entry = {
                "iter": iteration,
                "agent1": sig_out.get("insight", {}),
                "agent2": siz_out.get("insight", {}),
                "agent3": bt_out.get("insight", {}),
                "agent4": stats_out.get("insight", {}),
                "agent5": dsr_out.get("insight", {}),
                "agent6": mutator_out.get("insight", {}),
            }
            memory.narrative_history.append(narrative_entry)

            memory.save()

            # ── P: PRUNE ────────────────────────────────────────────────────
            prune_reasons = []
            if dsr_out["dsr"] < 0.0:
                prune_reasons.append("dsr_negative")
            if dsr_out["pbo"] > 0.5:
                prune_reasons.append("pbo_high")
            if bt_out["net_sharpe_tc"] < 0:
                prune_reasons.append("negative_net_sharpe")
            if bt_out["capacity_usd"] < config["risk"]["min_capacity_usd"]:
                prune_reasons.append("insufficient_capacity")

            pruned = len(prune_reasons) > 0
            if pruned:
                memory.add_pruned(params)
                memory.save()
                logger.info(f"  P: PRUNED — {prune_reasons}")
            else:
                logger.info(f"  P: Not pruned")

            # ── Write CSV row ────────────────────────────────────────────────
            cumul_best_dsr = memory.leaderboard[0]["dsr"] if memory.leaderboard else 0.0
            dsr_delta = dsr_out["dsr"] - (
                memory.dsr_history[-2] if len(memory.dsr_history) > 1 else 0.0
            )

            a1_insight = sig_out.get("insight", {})
            a2_insight = siz_out.get("insight", {})
            a3_insight = bt_out.get("insight", {})
            a4_insight = stats_out.get("insight", {})
            a5_insight = dsr_out.get("insight", {})
            a6_insight = mutator_out.get("insight", {})

            row = {
                # Group A
                "iter": iteration,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "iter_duration_s": round(iter_duration, 2),
                "ticker": ticker,
                "n_trials_so_far": memory.n_trials,
                "mutator_ei": mutator_out.get("acquisition_value", 0.0),
                "is_random": mutator_out.get("is_random", False),
                # Group B — params
                **{k: params.get(k, "") for k in [
                    "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
                    "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction"
                ]},
                # Group C — signal
                **{k: sig_out.get(k, "") for k in [
                    "ic_mean", "ic_std", "ic_ir", "factor_alpha", "factor_beta_mkt",
                    "factor_beta_smb", "factor_beta_hml", "factor_r2", "half_life_days"
                ]},
                # Group D — sizing
                **{k: siz_out.get(k, "") for k in [
                    "kelly_full", "kelly_applied", "vol_target_weight", "final_weight",
                    "stop_loss_dynamic", "stop_loss_used", "garch_sigma"
                ]},
                # Group E — backtest
                **{k: bt_out.get(k, "") for k in [
                    "gross_sharpe", "net_sharpe_tc", "total_return", "max_drawdown",
                    "win_rate", "num_trades", "avg_holding_days", "turnover_annual",
                    "total_tc_paid", "capacity_usd", "regime"
                ]},
                # Group F — stats
                **{k: stats_out.get(k, "") for k in [
                    "jarque_bera_pval", "is_normal", "ttest_stat", "ttest_pval",
                    "ttest_reject_null", "sharpe_ci_low", "sharpe_ci_high", "sharpe_ci_width",
                    "cvar_95", "var_95", "skewness", "kurtosis", "sortino", "calmar", "omega",
                    "regime_sharpe", "regime_win_rate"
                ]},
                # Group G — DSR
                **{k: dsr_out.get(k, "") for k in [
                    "dsr", "dsr_verdict", "min_trl_days", "min_trl_satisfied",
                    "pbo", "is_new_leader", "leader_dsr", "trials_penalty", "expected_max_sr"
                ]},
                # Group H — RALPH metadata
                "pruned": pruned,
                "prune_reasons": "|".join(prune_reasons),
                "cumul_best_dsr": round(cumul_best_dsr, 4),
                "dsr_delta": round(dsr_delta, 4),
                "halt_triggered": False,
                "halt_reason": "",
                # Group I — LLM narratives
                "agent1_signal_quality": a1_insight.get("signal_quality", ""),
                "agent1_narrative": a1_insight.get("signal_narrative",
                                    a1_insight.get("ic_interpretation", "")),
                "agent2_sizing_verdict": a2_insight.get("sizing_verdict", ""),
                "agent3_backtest_quality": a3_insight.get("backtest_quality", ""),
                "agent3_anomalies": "|".join(a3_insight.get("anomalies", [])),
                "agent4_stats_quality": a4_insight.get("stats_quality", ""),
                "agent5_final_verdict": a5_insight.get("final_verdict",
                                        dsr_out.get("dsr_verdict", "")),
                "agent5_verdict_narrative": a5_insight.get("verdict_narrative", ""),
                "agent5_mutator_instruction": a5_insight.get("mutator_instruction", ""),
                "agent6_selection_rationale": a6_insight.get("selection_rationale",
                                              mutator_out.get("selection_rationale", "")),
            }
            append_csv_row(csv_path, row)
            logger.info(f"  CSV row written (iter {iteration})")

            # ── H: HALT ─────────────────────────────────────────────────────
            halt, halt_reason = check_halt(memory, config, iteration)
            if halt:
                halt_reason_final = halt_reason
                update_last_csv_row(csv_path, halt_triggered=True, halt_reason=halt_reason)
                logger.info(f"\n{'='*50}")
                logger.info(f"  HALT: {halt_reason}")
                logger.info(f"{'='*50}")
                break

            iteration += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            halt_reason_final = "user_interrupt"
            update_last_csv_row(csv_path, halt_triggered=True, halt_reason=halt_reason_final)
            break
        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
            logger.info("Skipping to next iteration.")
            iteration += 1
            if iteration > config["ralph"]["max_iterations"]:
                break

    total_runtime = time.time() - loop_start
    write_final_report(memory, config, halt_reason_final, total_runtime)

    if memory.leaderboard:
        best = memory.leaderboard[0]
        logger.info(f"\nBest strategy: DSR={best.get('dsr', 0):.3f}  params={best.get('params', {})}")
    logger.info(f"Results written to: {csv_path}")
    logger.info(f"Total runtime: {int(total_runtime // 60)}m {int(total_runtime % 60)}s")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant RALPH Multi-Agent Loop")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    run_ralph_loop(config_path=args.config)
