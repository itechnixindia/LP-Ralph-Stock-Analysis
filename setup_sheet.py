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
    print("  1. Create a blank Google Sheet")
    print("  2. Share it with: ralph-stock-anlaysis@tatvic-langchain-dev.iam.gserviceaccount.com")
    print("  3. Copy the spreadsheet ID from the URL and pass it here")
    sys.exit(1)

sheet_id = sys.argv[1]
gc = gspread.service_account(filename="credentials/service_account.json")

# Open existing sheet
if sheet_id.startswith("http"):
    sh = gc.open_by_url(sheet_id)
else:
    sh = gc.open_by_key(sheet_id)

print(f"Opened: {sh.title}")

# Tab 1: RALPH Results
ws = sh.sheet1
ws.update_title("RALPH Results")
headers = [
    "iter", "timestamp", "iter_duration_s", "ticker", "n_trials_so_far",
    "mutator_ei", "is_random",
    "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
    "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction",
    "sentiment_score", "sentiment_label", "n_headlines",
    "ic_mean", "ic_std", "ic_ir", "factor_alpha", "factor_beta_mkt",
    "factor_beta_smb", "factor_beta_hml", "factor_r2", "half_life_days",
    "kelly_full", "kelly_applied", "vol_target_weight", "final_weight",
    "stop_loss_dynamic", "stop_loss_used", "garch_sigma",
    "gross_sharpe", "net_sharpe_tc", "total_return", "max_drawdown",
    "win_rate", "num_trades", "avg_holding_days", "turnover_annual",
    "total_tc_paid", "capacity_usd", "regime",
    "jarque_bera_pval", "is_normal", "ttest_stat", "ttest_pval",
    "ttest_reject_null", "sharpe_ci_low", "sharpe_ci_high",
    "sharpe_ci_width", "cvar_95", "var_95", "skewness", "kurtosis",
    "sortino", "calmar", "omega",
    "dsr", "dsr_verdict", "min_trl_days", "min_trl_satisfied",
    "pbo", "is_new_leader", "leader_dsr",
    "pruned", "prune_reasons", "cumul_best_dsr", "dsr_delta",
    "halt_triggered", "halt_reason",
    "agent0_sentiment_label", "agent1_signal_quality",
    "agent2_sizing_verdict", "agent3_backtest_quality",
    "agent4_stats_quality", "agent5_final_verdict",
    "agent5_mutator_instruction",
]
ws.update("A1", [headers])
ws.format("A1:CC1", {
    "textFormat": {"bold": True},
    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.95},
})
ws.freeze(rows=1)
print("Tab 1 (RALPH Results): headers written + formatted")

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
oos.update("A1", [oos_headers])
oos.format("A1:I1", {
    "textFormat": {"bold": True},
    "backgroundColor": {"red": 0.9, "green": 0.95, "blue": 0.9},
})
oos.freeze(rows=1)
print("Tab 2 (OOS Validation): headers written + formatted")

print(f"\nSPREADSHEET_ID={sh.id}")
print(f"\nNow update config.yaml:")
print(f"  sheets_logger:")
print(f"    enabled: true")
print(f'    spreadsheet_id: "{sh.id}"')
print(f'    sheet_name: "RALPH Results"')
print("\nDONE!")
