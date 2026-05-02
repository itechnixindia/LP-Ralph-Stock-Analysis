"""
Setup script: adds headers + formatting to an existing Google Sheet.

Usage:
  python3 setup_sheet.py SPREADSHEET_ID

The spreadsheet must be shared (Editor) with the service account email found
in credentials/service_account.json.
"""
import sys
import gspread

if len(sys.argv) < 2:
    print("Usage: python3 setup_sheet.py SPREADSHEET_ID_OR_URL")
    sys.exit(1)

sheet_id = sys.argv[1]
gc = gspread.service_account(filename="credentials/service_account.json")

if sheet_id.startswith("http"):
    sh = gc.open_by_url(sheet_id)
else:
    sh = gc.open_by_key(sheet_id)

print(f"Opened: {sh.title}")

# Tab 1: RALPH Results
ws = sh.sheet1
ws.update_title("RALPH Results")
headers = [
    # Group A — metadata
    "iter", "timestamp", "iter_duration_s", "ticker", "n_trials_so_far",
    "mutator_ei", "is_random",
    # Group B — params
    "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
    "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction",
    # Group B2 — sentiment
    "sentiment_score", "sentiment_label", "n_headlines",
    # Group C — signal
    "ic_mean", "ic_std", "ic_ir", "factor_alpha", "factor_beta_mkt",
    "factor_beta_smb", "factor_beta_hml", "factor_r2", "half_life_days",
    # Group D — sizing
    "kelly_full", "kelly_applied", "vol_target_weight", "final_weight",
    "stop_loss_dynamic", "stop_loss_used", "garch_sigma",
    # Group E — backtest
    "gross_sharpe", "net_sharpe_tc", "total_return", "max_drawdown",
    "win_rate", "num_trades", "avg_holding_days", "turnover_annual",
    "total_tc_paid", "capacity_usd", "regime",
    # Group F — stats
    "jarque_bera_pval", "is_normal", "ttest_stat", "ttest_pval",
    "ttest_reject_null", "sharpe_ci_low", "sharpe_ci_high",
    "sharpe_ci_width", "cvar_95", "var_95", "skewness", "kurtosis",
    "sortino", "calmar", "omega", "regime_sharpe", "regime_win_rate",
    # Group G — DSR
    "dsr", "dsr_verdict", "min_trl_days", "min_trl_satisfied",
    "pbo", "is_new_leader", "leader_dsr", "trials_penalty", "expected_max_sr",
    # Group H — RALPH metadata
    "pruned", "prune_reasons", "cumul_best_dsr", "dsr_delta",
    "halt_triggered", "halt_reason",
    # Group I — LLM narratives
    "agent0_sentiment_label",
    "agent1_signal_quality", "agent1_narrative",
    "agent2_sizing_verdict",
    "agent3_backtest_quality", "agent3_anomalies",
    "agent4_stats_quality",
    "agent5_final_verdict", "agent5_verdict_narrative",
    "agent5_mutator_instruction", "agent6_selection_rationale",
    # Group J — LLM cost tracking
    "llm_input_tokens", "llm_output_tokens", "llm_calls", "llm_cost_usd",
]
ws.update(values=[headers], range_name="A1")
ws.format("A1:CG1", {
    "textFormat": {"bold": True},
    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.95},
})
ws.freeze(rows=1)
print(f"Tab 1 (RALPH Results): {len(headers)} headers written")

# Tab 2: OOS Validation
try:
    oos = sh.worksheet("OOS Validation")
except gspread.exceptions.WorksheetNotFound:
    oos = sh.add_worksheet("OOS Validation", rows=100, cols=10)

oos_headers = [
    "rank", "is_dsr", "is_sharpe", "oos_avg_dsr", "oos_avg_sharpe",
    "oos_avg_return", "sharpe_degradation_pct", "n_folds_tested",
    "oos_verdict",
]
oos.update(values=[oos_headers], range_name="A1")
oos.format("A1:I1", {
    "textFormat": {"bold": True},
    "backgroundColor": {"red": 0.9, "green": 0.95, "blue": 0.9},
})
oos.freeze(rows=1)
print(f"Tab 2 (OOS Validation): {len(oos_headers)} headers written")

print(f"\nSPREADSHEET_ID={sh.id}")
print("DONE!")
