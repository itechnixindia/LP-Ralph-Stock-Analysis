"""
Walk-Forward Validation Engine

Implements rolling and expanding walk-forward splits for out-of-sample testing.
Ensures the RALPH loop trains on historical data and validates on unseen data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple

logger = logging.getLogger(__name__)


class WalkForwardSplit(NamedTuple):
    """A single train/test split."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def compute_splits(
    start_date: str,
    end_date: str,
    n_folds: int = 3,
    mode: str = "expanding",
    test_ratio: float = 0.20,
) -> List[WalkForwardSplit]:
    """
    Generate walk-forward splits.

    Modes:
      - "expanding": training window grows each fold (anchored start)
      - "rolling": training window slides (fixed width)
      - "single": one split — train on (1-test_ratio), test on test_ratio

    Returns list of WalkForwardSplit tuples.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    if mode == "single":
        return _single_split(start, end, test_ratio)

    if n_folds < 2:
        return _single_split(start, end, test_ratio)

    return _multi_fold_split(start, end, n_folds, mode, test_ratio)


def _single_split(
    start: datetime, end: datetime, test_ratio: float
) -> List[WalkForwardSplit]:
    total_days = (end - start).days
    train_days = int(total_days * (1 - test_ratio))
    train_end = start + timedelta(days=train_days)
    test_start = train_end + timedelta(days=1)

    return [WalkForwardSplit(
        fold=1,
        train_start=start.strftime("%Y-%m-%d"),
        train_end=train_end.strftime("%Y-%m-%d"),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=end.strftime("%Y-%m-%d"),
    )]


def _multi_fold_split(
    start: datetime,
    end: datetime,
    n_folds: int,
    mode: str,
    test_ratio: float,
) -> List[WalkForwardSplit]:
    total_days = (end - start).days
    test_window = int(total_days * test_ratio / n_folds)
    test_window = max(test_window, 60)  # minimum 60 days per test window

    splits = []

    for fold in range(n_folds):
        test_end_dt = end - timedelta(days=(n_folds - fold - 1) * test_window)
        test_start_dt = test_end_dt - timedelta(days=test_window)

        if mode == "expanding":
            train_start_dt = start
        else:
            train_days = int(test_window / test_ratio * (1 - test_ratio))
            train_start_dt = test_start_dt - timedelta(days=train_days)
            train_start_dt = max(train_start_dt, start)

        train_end_dt = test_start_dt - timedelta(days=1)

        if train_end_dt <= train_start_dt:
            continue
        if test_end_dt <= test_start_dt:
            continue

        splits.append(WalkForwardSplit(
            fold=fold + 1,
            train_start=train_start_dt.strftime("%Y-%m-%d"),
            train_end=train_end_dt.strftime("%Y-%m-%d"),
            test_start=test_start_dt.strftime("%Y-%m-%d"),
            test_end=test_end_dt.strftime("%Y-%m-%d"),
        ))

    return splits


def run_oos_validation(
    strategy_params: dict,
    split: WalkForwardSplit,
    ticker: str,
    benchmark: str,
    config: dict,
) -> Dict:
    """
    Run a single out-of-sample validation for given params on the test period.
    Returns OOS metrics dict.
    """
    from agents.agent1_signal import compute_signal
    from agents.agent2_sizing import compute_sizing
    from agents.agent3_backtest import run_backtest
    from agents.agent4_stats import run_stats
    from agents.agent5_dsr import compute_dsr_verdict

    # Stage 1: Signal (computed on full period for indicator warm-up,
    # but measured on test period)
    sig_out = compute_signal(
        ticker=ticker,
        start_date=split.train_start,  # need train data for SMA warm-up
        end_date=split.test_end,
        params=strategy_params,
        benchmark=benchmark,
        ic_history=[],
    )

    # Stage 2: Sizing (use train-period returns for GARCH fit)
    siz_out = compute_sizing(
        factor_alpha=sig_out["factor_alpha"],
        return_series=sig_out["return_series"],
        garch_sigma=0.02,
        params=strategy_params,
        dynamic_stop_multiplier=config["risk"]["dynamic_stop_multiplier"],
    )

    # Stage 3: Backtest on TEST period only
    bt_out = run_backtest(
        ticker=ticker,
        start_date=split.test_start,
        end_date=split.test_end,
        params=strategy_params,
        stop_loss_used=siz_out["stop_loss_used"],
        final_weight=siz_out["final_weight"],
        signal_series=sig_out["signal_series"],
        regime="unknown",
        transaction_costs=config.get("transaction_costs", {}),
    )

    # Stage 4: Stats on OOS returns
    stats_out = run_stats(
        daily_returns=bt_out["daily_returns"],
        trade_log=bt_out["trade_log"],
        gross_sharpe=bt_out["gross_sharpe"],
        net_sharpe_tc=bt_out["net_sharpe_tc"],
    )

    # Stage 5: OOS DSR (n_trials=1 since this is validation, not search)
    dsr_out = compute_dsr_verdict(
        net_sharpe_tc=bt_out["net_sharpe_tc"],
        skewness=stats_out["skewness"],
        kurtosis=stats_out["kurtosis"],
        n_obs=len(bt_out["daily_returns"]),
        n_trials=1,
        leaderboard=[],
        sharpe_ci_low=stats_out["sharpe_ci_low"],
        sharpe_ci_high=stats_out["sharpe_ci_high"],
    )

    return {
        "fold": split.fold,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "oos_sharpe": bt_out["net_sharpe_tc"],
        "oos_return": bt_out["total_return"],
        "oos_max_drawdown": bt_out["max_drawdown"],
        "oos_win_rate": bt_out["win_rate"],
        "oos_num_trades": bt_out["num_trades"],
        "oos_dsr": dsr_out["dsr"],
        "oos_cvar": stats_out["cvar_95"],
        "oos_sortino": stats_out["sortino"],
        "oos_ttest_pval": stats_out["ttest_pval"],
    }


def validate_leaderboard(
    leaderboard: list,
    splits: List[WalkForwardSplit],
    ticker: str,
    benchmark: str,
    config: dict,
    top_n: int = 5,
) -> List[Dict]:
    """
    Run OOS validation on the top-N leaderboard strategies.
    Returns list of dicts with IS (in-sample) + OOS (out-of-sample) metrics.
    """
    results = []

    for rank, entry in enumerate(leaderboard[:top_n], 1):
        params = entry.get("params", {})
        is_dsr = entry.get("dsr", 0.0)
        is_sharpe = entry.get("net_sharpe_tc", 0.0)

        logger.info(f"  OOS validation: Rank {rank} (IS DSR={is_dsr:.3f})")

        fold_results = []
        for split in splits:
            try:
                oos = run_oos_validation(
                    strategy_params=params,
                    split=split,
                    ticker=ticker,
                    benchmark=benchmark,
                    config=config,
                )
                fold_results.append(oos)
                logger.info(
                    f"    Fold {split.fold}: OOS Sharpe={oos['oos_sharpe']:.3f}, "
                    f"OOS DSR={oos['oos_dsr']:.3f}"
                )
            except Exception as e:
                logger.warning(f"    Fold {split.fold} failed: {e}")

        if fold_results:
            import numpy as np
            avg_oos_sharpe = float(np.mean([f["oos_sharpe"] for f in fold_results]))
            avg_oos_dsr = float(np.mean([f["oos_dsr"] for f in fold_results]))
            avg_oos_return = float(np.mean([f["oos_return"] for f in fold_results]))

            sharpe_degradation = (
                (is_sharpe - avg_oos_sharpe) / abs(is_sharpe)
                if abs(is_sharpe) > 0.01 else 0.0
            )

            results.append({
                "rank": rank,
                "params": params,
                "is_dsr": round(is_dsr, 4),
                "is_sharpe": round(is_sharpe, 4),
                "oos_avg_sharpe": round(avg_oos_sharpe, 4),
                "oos_avg_dsr": round(avg_oos_dsr, 4),
                "oos_avg_return": round(avg_oos_return, 4),
                "sharpe_degradation_pct": round(sharpe_degradation * 100, 1),
                "n_folds_tested": len(fold_results),
                "fold_details": fold_results,
                "oos_verdict": _oos_verdict(
                    is_dsr, avg_oos_dsr, sharpe_degradation
                ),
            })

    return results


def _oos_verdict(
    is_dsr: float, oos_dsr: float, sharpe_degradation: float
) -> str:
    """
    Classify OOS performance.
    - "confirmed": OOS performance validates IS findings
    - "degraded": OOS is weaker but still positive
    - "failed": OOS shows the strategy doesn't hold
    """
    if oos_dsr > 0.50 and sharpe_degradation < 0.50:
        return "confirmed"
    elif oos_dsr > 0.20 and sharpe_degradation < 0.75:
        return "degraded"
    return "failed"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test split generation
    splits = compute_splits(
        start_date="2020-01-01",
        end_date="2024-12-31",
        n_folds=3,
        mode="expanding",
        test_ratio=0.20,
    )
    for s in splits:
        print(f"Fold {s.fold}: train=[{s.train_start} → {s.train_end}] "
              f"test=[{s.test_start} → {s.test_end}]")

    # Test single split
    single = compute_splits(
        start_date="2020-01-01",
        end_date="2024-12-31",
        mode="single",
        test_ratio=0.25,
    )
    for s in single:
        print(f"Single: train=[{s.train_start} → {s.train_end}] "
              f"test=[{s.test_start} → {s.test_end}]")

    print("walk_forward.py: OK")
