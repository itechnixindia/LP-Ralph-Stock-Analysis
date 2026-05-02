"""
Google Sheets Logger — Appends each RALPH iteration's results to a Google Sheet.

Setup:
  1. Enable Google Sheets API in your GCP project
  2. Create a service account and download JSON key
  3. Save key to credentials/service_account.json (or set RALPH_SHEETS_CREDS env var)
  4. Create a Google Sheet and share it (Editor) with the service account email
  5. Set the sheet URL/ID in config.yaml under sheets_logger.spreadsheet_id

Usage:
  Automatically called by orchestrator.py after each iteration if enabled.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CREDS_DEFAULT_PATH = "credentials/service_account.json"
HEADER_ROW_MARKER = "iter"  # Used to detect if header exists


def _get_client():
    """Authenticate and return a gspread client."""
    import gspread

    creds_path = os.environ.get("RALPH_SHEETS_CREDS", CREDS_DEFAULT_PATH)

    if not os.path.exists(creds_path):
        raise FileNotFoundError(
            f"Service account key not found at '{creds_path}'. "
            f"Set RALPH_SHEETS_CREDS env var or place key at {CREDS_DEFAULT_PATH}"
        )

    return gspread.service_account(filename=creds_path)


def _get_or_create_worksheet(spreadsheet, sheet_name: str):
    """Get existing worksheet or create a new one."""
    try:
        return spreadsheet.worksheet(sheet_name)
    except Exception:
        return spreadsheet.add_worksheet(
            title=sheet_name, rows=500, cols=80
        )


def _build_row(iteration_data: dict) -> List:
    """Convert iteration dict to ordered list of values for the sheet."""
    columns = get_column_order()
    row = []
    for col in columns:
        val = iteration_data.get(col, "")
        # Convert non-serializable types
        if isinstance(val, bool):
            val = str(val)
        elif isinstance(val, (list, dict)):
            val = str(val)[:500]  # Truncate long values
        elif val is None:
            val = ""
        row.append(val)
    return row


def get_column_order() -> List[str]:
    """Define the column order for the Google Sheet."""
    return [
        # Group A — Metadata
        "iter", "timestamp", "iter_duration_s", "ticker", "n_trials_so_far",
        "mutator_ei", "is_random",
        # Group B — Parameters
        "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
        "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction",
        # Group C — Sentiment (NEW)
        "sentiment_score", "sentiment_label", "n_headlines",
        # Group D — Signal
        "ic_mean", "ic_std", "ic_ir", "factor_alpha", "factor_beta_mkt",
        "factor_beta_smb", "factor_beta_hml", "factor_r2", "half_life_days",
        # Group E — Sizing
        "kelly_full", "kelly_applied", "vol_target_weight", "final_weight",
        "stop_loss_dynamic", "stop_loss_used", "garch_sigma",
        # Group F — Backtest
        "gross_sharpe", "net_sharpe_tc", "total_return", "max_drawdown",
        "win_rate", "num_trades", "avg_holding_days", "turnover_annual",
        "total_tc_paid", "capacity_usd", "regime",
        # Group G — Stats
        "jarque_bera_pval", "is_normal", "ttest_stat", "ttest_pval",
        "ttest_reject_null", "sharpe_ci_low", "sharpe_ci_high", "sharpe_ci_width",
        "cvar_95", "var_95", "skewness", "kurtosis", "sortino", "calmar", "omega",
        # Group H — DSR
        "dsr", "dsr_verdict", "min_trl_days", "min_trl_satisfied",
        "pbo", "is_new_leader", "leader_dsr",
        # Group I — RALPH metadata
        "pruned", "prune_reasons", "cumul_best_dsr", "dsr_delta",
        "halt_triggered", "halt_reason",
        # Group J — LLM verdicts
        "agent0_sentiment_label", "agent1_signal_quality",
        "agent2_sizing_verdict", "agent3_backtest_quality",
        "agent4_stats_quality", "agent5_final_verdict",
        "agent5_mutator_instruction",
    ]


class SheetsLogger:
    """Manages Google Sheets logging for RALPH iterations."""

    def __init__(
        self,
        spreadsheet_id: str,
        sheet_name: str = "RALPH Results",
        enabled: bool = True,
    ):
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name
        self.enabled = enabled
        self._client = None
        self._spreadsheet = None
        self._worksheet = None
        self._header_written = False

    def _connect(self):
        """Lazy connection — only connects when first row is written."""
        if self._worksheet is not None:
            return

        self._client = _get_client()

        # Open by ID or URL
        if self.spreadsheet_id.startswith("http"):
            self._spreadsheet = self._client.open_by_url(self.spreadsheet_id)
        else:
            self._spreadsheet = self._client.open_by_key(self.spreadsheet_id)

        self._worksheet = _get_or_create_worksheet(
            self._spreadsheet, self.sheet_name
        )

        # Check if header row exists
        try:
            first_cell = self._worksheet.acell("A1").value
            if first_cell == HEADER_ROW_MARKER:
                self._header_written = True
        except Exception:
            pass

    def _ensure_header(self):
        """Write header row if it doesn't exist yet."""
        if self._header_written:
            return

        columns = get_column_order()
        self._worksheet.update("A1", [columns])

        # Bold + freeze header row
        try:
            self._worksheet.format("A1:BZ1", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.95},
            })
            self._worksheet.freeze(rows=1)
        except Exception:
            pass  # Formatting is optional

        self._header_written = True

    def log_iteration(self, row_data: dict):
        """Append one iteration's results to the Google Sheet."""
        if not self.enabled:
            return

        try:
            self._connect()
            self._ensure_header()

            row = _build_row(row_data)
            self._worksheet.append_row(row, value_input_option="USER_ENTERED")

            logger.info(
                f"  Sheets: logged iter {row_data.get('iter', '?')} "
                f"to '{self.sheet_name}'"
            )

        except ImportError:
            logger.warning(
                "  Sheets: gspread not installed. "
                "Run: pip install gspread"
            )
            self.enabled = False

        except FileNotFoundError as e:
            logger.warning(f"  Sheets: {e}")
            self.enabled = False

        except Exception as e:
            logger.warning(f"  Sheets: failed to log — {e}")

    def log_oos_results(self, oos_results: list):
        """Write OOS validation results to a separate worksheet."""
        if not self.enabled or not oos_results:
            return

        try:
            self._connect()
            oos_sheet = _get_or_create_worksheet(
                self._spreadsheet, "OOS Validation"
            )

            # Header
            oos_columns = [
                "rank", "is_dsr", "is_sharpe",
                "oos_avg_dsr", "oos_avg_sharpe", "oos_avg_return",
                "sharpe_degradation_pct", "n_folds_tested", "oos_verdict",
            ]
            oos_sheet.update("A1", [oos_columns])

            # Data rows
            for r in oos_results:
                row = [r.get(col, "") for col in oos_columns]
                oos_sheet.append_row(row, value_input_option="USER_ENTERED")

            logger.info(f"  Sheets: OOS results logged ({len(oos_results)} strategies)")

        except Exception as e:
            logger.warning(f"  Sheets: OOS logging failed — {e}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SheetsLogger module loaded successfully.")
    print(f"Column count: {len(get_column_order())}")
    print(f"Columns: {get_column_order()[:10]}...")

    # Test row building
    test_data = {"iter": 1, "ticker": "RELIANCE.NS", "dsr": 0.847}
    row = _build_row(test_data)
    print(f"Test row (first 5): {row[:5]}")
    print("sheets_logger.py: OK")
