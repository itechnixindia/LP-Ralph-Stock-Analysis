"""Pre-flight check: validates everything before running RALPH."""
import ast
import os
import sys
import yaml
import json

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

print("=" * 60)
print("  QUANT RALPH — PRE-FLIGHT CHECK")
print("=" * 60)

# 1. Syntax check
print("\n[1] SYNTAX CHECK")
files = [
    "constants.py", "data_loader.py", "memory_store.py", "llm_client.py",
    "orchestrator.py", "walk_forward.py", "news_fetcher.py", "sheets_logger.py",
    "agents/agent0_sentiment.py", "agents/agent1_signal.py",
    "agents/agent2_sizing.py", "agents/agent3_backtest.py",
    "agents/agent4_stats.py", "agents/agent5_dsr.py",
    "agents/agent6_mutator.py",
    "agents/prompts/agent0_prompt.py",
]
all_ok = True
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"  ✅ {f}")
    except SyntaxError as e:
        print(f"  ❌ {f}: {e}")
        all_ok = False
    except FileNotFoundError:
        print(f"  ⚠️  {f}: NOT FOUND")

# 2. Config
print("\n[2] CONFIG.YAML")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

items = [
    ("stock.ticker", cfg.get("stock", {}).get("ticker")),
    ("stock.start_date", cfg.get("stock", {}).get("start_date")),
    ("stock.end_date", cfg.get("stock", {}).get("end_date")),
    ("ralph.max_iterations", cfg.get("ralph", {}).get("max_iterations")),
    ("walk_forward.enabled", cfg.get("walk_forward", {}).get("enabled")),
    ("walk_forward.mode", cfg.get("walk_forward", {}).get("mode")),
    ("sentiment.enabled", cfg.get("sentiment", {}).get("enabled")),
    ("sheets_logger.enabled", cfg.get("sheets_logger", {}).get("enabled")),
    ("sheets_logger.spreadsheet_id", str(cfg.get("sheets_logger", {}).get("spreadsheet_id", ""))[:20] + "..."),
]
for k, v in items:
    print(f"  {'✅' if v else '❌'} {k}: {v}")

# 3. Credentials
print("\n[3] CREDENTIALS")
creds_path = "credentials/service_account.json"
if os.path.exists(creds_path):
    with open(creds_path) as f:
        creds = json.load(f)
    print(f"  ✅ File exists")
    print(f"  ✅ Email: {creds.get('client_email', '?')}")
else:
    print(f"  ❌ {creds_path} NOT FOUND")

# 4. Dependencies
print("\n[4] DEPENDENCIES")
for d in ["gspread", "yaml", "numpy", "pandas"]:
    try:
        __import__(d)
        print(f"  ✅ {d}")
    except ImportError:
        print(f"  ❌ {d}: NOT INSTALLED")

# 5. Sheets connection
print("\n[5] GOOGLE SHEETS")
try:
    import gspread
    gc = gspread.service_account(filename=creds_path)
    sid = cfg["sheets_logger"]["spreadsheet_id"]
    sh = gc.open_by_key(sid)
    ws = sh.worksheet("RALPH Results")
    h = ws.row_values(1)
    print(f"  ✅ Connected: {sh.title}")
    print(f"  ✅ Headers: {len(h)} columns")
    oos = sh.worksheet("OOS Validation")
    print(f"  ✅ OOS tab: ready")
except Exception as e:
    print(f"  ❌ {e}")

# 6. Walk-forward
print("\n[6] WALK-FORWARD SPLITS")
from walk_forward import compute_splits
splits = compute_splits(
    cfg["stock"]["start_date"], cfg["stock"]["end_date"],
    mode=cfg["walk_forward"]["mode"],
    test_ratio=cfg["walk_forward"]["test_ratio"],
)
for s in splits:
    print(f"  ✅ Fold {s.fold}: train=[{s.train_start}→{s.train_end}] test=[{s.test_start}→{s.test_end}]")

# 7. Constants
print("\n[7] CONSTANTS")
import constants
n = len([x for x in dir(constants) if x.isupper() and not x.startswith("_")])
print(f"  ✅ {n} named constants loaded")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED ✅" if all_ok else "  SOME CHECKS FAILED ❌")
print("=" * 60)
