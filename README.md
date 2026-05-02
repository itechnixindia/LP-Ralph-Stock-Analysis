# Quant RALPH — Multi-Agent Stock Strategy Optimiser

**RALPH** = **R**epeat → **A**ssess → **L**earn → **P**rune → **H**alt

A 7-agent + orchestrator system that uses Bayesian optimisation (Gaussian Process), LLM-extracted sentiment signals, and Claude LLM intelligence to find statistically-validated equity momentum strategies. Each iteration tests a new parameter set, evaluates it through a rigorous quant pipeline, and learns from the result to propose better parameters next time. Walk-forward validation ensures results hold on unseen data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key (for LLM intelligence layers)
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Edit config.yaml — set your ticker and date range
# 4. Run
python orchestrator.py
```

Results appear in `results.csv` after every iteration. The loop is crash-safe — restart anytime and it resumes from where it stopped.

> **Note:** The system works without an Anthropic API key — agents fall back to heuristic defaults. You just lose the LLM intelligence layer.

---

## What It Does

Given only a stock ticker and date range, the system:

1. **Searches** 8-dimensional hyperparameter space (SMA periods, RSI, stop-loss, take-profit, holding period, Kelly fraction, vol target)
2. **Evaluates** each candidate through a 5-stage quant pipeline
3. **Learns** which parameter regions are good vs bad via Gaussian Process posterior
4. **Prunes** dead regions permanently
5. **Halts** when it finds a statistically-validated strategy (DSR > 0.95) or plateaus

After 50–100 iterations you get exact, deployable trading rules with statistical proof they're not luck.

---

## Architecture

```
config.yaml
    │
    ▼
orchestrator.py  ←──────────────────────────────────────────────┐
    │                                                            │
    │  R: agent6_mutator  ── Gaussian Process → propose params  │
    │                                                            │
    │  A: Pipeline                                               │
    │     agent1_signal   ── IC, Factor Alpha, OU Half-life      │
    │     agent2_sizing   ── Kelly, GARCH σ, Dynamic Stop        │
    │     agent3_backtest ── Event-driven backtest, TC model     │
    │     agent4_stats    ── t-test, Bootstrap CI, CVaR, Sortino │
    │     agent5_dsr      ── Deflated Sharpe Ratio (THE JUDGE)   │
    │                                                            │
    │  L: memory_store.py ── persist GP obs, GARCH, leaderboard  │
    │  P: Prune dead regions                                     │
    │  H: Check halt conditions                                  │
    │                                                            │
    └──── results.csv (one row per iteration) ───────────────────┘
```

### Two-Layer Agent Design

Every agent runs two layers in sequence:

```
┌─────────────────────────────────────────┐
│  COMPUTATION LAYER (Python/scipy/arch)  │
│  Fast, deterministic maths.             │
│  Produces raw numbers.                  │
├─────────────────────────────────────────┤
│  INTELLIGENCE LAYER (Claude LLM)        │
│  Reads numbers + all prior insights.    │
│  Interprets what they mean.             │
│  Produces JSON verdict + narrative.     │
│  Narrative flows to the next agent.     │
└─────────────────────────────────────────┘
```

By iteration 20, each agent receives 4–5 rich contextual narratives from prior agents and the LLM reasons across all of them simultaneously.

---

## File Structure

```
quant_ralph/
├── orchestrator.py          # RALPH loop controller + config validation
├── memory_store.py          # Persistent memory (JSON-based, crash-safe)
├── llm_client.py            # Shared Anthropic API wrapper
├── constants.py             # All named constants (no magic numbers)
├── data_loader.py           # Shared price + factor data loader (cached)
├── walk_forward.py          # Walk-forward OOS validation engine
├── news_fetcher.py          # News headline fetcher (Yahoo/Google RSS)
├── config.yaml              # All user settings
├── requirements.txt
├── agents/
│   ├── agent0_sentiment.py  # LLM-as-Alpha: news sentiment signal
│   ├── agent1_signal.py     # IC + Fama-French regression + OU half-life
│   ├── agent2_sizing.py     # Kelly fraction + GARCH(1,1) + vol targeting
│   ├── agent3_backtest.py   # Event-driven backtest + Almgren-Chriss TC
│   ├── agent4_stats.py      # Hypothesis tests + CVaR + Sortino/Calmar/Omega
│   ├── agent5_dsr.py        # Deflated Sharpe Ratio + PBO + MinTRL
│   ├── agent6_mutator.py    # Gaussian Process EI + LLM candidate selection
│   └── prompts/
│       ├── agent0_prompt.py  # Sentiment scoring prompt
│       ├── agent1_prompt.py
│       ├── agent2_prompt.py
│       ├── agent3_prompt.py
│       ├── agent4_prompt.py
│       ├── agent5_prompt.py
│       └── agent6_prompt.py
├── results.csv              # Auto-created, one row per iteration
├── memory/                  # Auto-created, crash-safe persistence (all JSON)
│   ├── gp_state.json
│   ├── garch_state.json
│   ├── pruned.json
│   ├── leaderboard.json
│   ├── ic_history.json
│   ├── narrative_history.json
│   └── oos_validation.json  # Walk-forward OOS results
└── cache/                   # Auto-created, price + news data cached here
    ├── prices_RELIANCE.NS_*.csv
    ├── ff_factors_*.csv
    └── news_*.json           # Cached news headlines (6-hour TTL)
```

### Key Shared Modules

| Module | Purpose |
|--------|---------|
| `constants.py` | ~85 named constants replacing magic numbers — signal thresholds, GARCH parameters, capacity limits, DSR thresholds, walk-forward settings, sentiment weights. |
| `data_loader.py` | `load_prices()` and `load_ff_factors()` — yfinance + Fama-French data, cached to disk. Single source of truth used by agent1 and agent3. |
| `memory_store.py` | JSON-based persistence (no pickle). Auto-migrates legacy `.pkl` files to `.json` on first load. Crash-safe — saved after every iteration. |
| `walk_forward.py` | Walk-forward split generator + OOS validation engine. Supports single, expanding, and rolling modes. |
| `news_fetcher.py` | News headline fetcher using Yahoo Finance RSS + Google News RSS. Cached with 6-hour TTL. |

---

## Configuration

Edit `config.yaml` before running:

```yaml
stock:
  ticker: "RELIANCE.NS"      # Any yfinance ticker
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  benchmark: "^NSEI"         # ^NSEI for NSE, ^GSPC for S&P 500

ralph:
  max_iterations: 100
  dsr_plateau_window: 10     # Halt if no improvement for N iters
  halt_pbo_threshold: 0.20   # Halt early if leader PBO < this

hyperparameter_space:
  stop_loss_pct:   [0.005, 0.10]
  take_profit_pct: [0.01,  0.30]
  holding_days:    [1, 60]
  sma_fast:        [5, 50]
  sma_slow:        [20, 200]
  rsi_period:      [7, 21]
  vol_target_pct:  [0.10, 0.25]
  kelly_fraction:  [0.10, 0.50]

risk:
  dynamic_stop_multiplier: 2.0   # stop = 2 × GARCH σ_t
  min_capacity_usd: 10000

transaction_costs:
  commission_pct: 0.001          # 0.1% per trade
  slippage_lambda: 0.1           # Almgren-Chriss λ
```

### Config Validation

The orchestrator validates your config at startup:

- Required sections: `stock`, `ralph`, `hyperparameter_space`, `risk`, `output`
- Date format: `YYYY-MM-DD`, `start_date` must be before `end_date`
- Hyperparameter bounds: all `[min, max]` pairs validated (`min < max`)
- SMA overlap warning: alerts if `sma_fast` upper bound ≥ `sma_slow` lower bound

### Supported Tickers

| Exchange | Format | Examples |
|----------|--------|---------|
| NSE India | `TICKER.NS` | `RELIANCE.NS`, `TCS.NS`, `INFY.NS` |
| BSE India | `TICKER.BO` | `RELIANCE.BO` |
| US stocks | `TICKER` | `AAPL`, `NVDA`, `MSFT` |
| NSE index | `^NSEI` | Nifty 50 (benchmark) |
| S&P 500 | `^GSPC` | US benchmark |

---

## Agent Reference

### Agent 0 — Sentiment Signal (`agent0_sentiment.py`) — NEW

**What it does:** Fetches live news headlines and scores them for market sentiment.

**Data sources:**
- Yahoo Finance RSS (free, no API key)
- Google News RSS (company name search)

**Two-layer scoring:**
- **Computation layer**: Keyword-based heuristic scoring (bullish/bearish word lists)
- **Intelligence layer**: Claude LLM reads up to 20 headlines, scores each -1.0 to +1.0, and produces an aggregate sentiment signal

**Output:** Aggregate sentiment score (-1 to +1), label (bullish/bearish/neutral), dominant theme

**Key:** If LLM is unavailable, the keyword fallback still produces a signal. Sentiment flows through accumulated context to all downstream agents.

### Agent 1 — Signal Quality (`agent1_signal.py`)

**What it computes:**
- **Information Coefficient (IC)**: Spearman correlation between signal and forward returns over 60-day rolling windows
- **Fama-French regression**: OLS of excess returns on Mkt-RF, SMB, HML → extracts genuine alpha
- **OU Half-life**: Ornstein-Uhlenbeck fit to signal series → natural holding period

**Key thresholds** (defined in `constants.py`):
- IC > 0.10 → strong | IC 0.05–0.10 → moderate | IC < 0.05 → weak
- IC_IR > 0.5 required for reliable signal
- Factor α < 0.02%/day is marginal

### Agent 2 — Position Sizing (`agent2_sizing.py`)

**What it computes:**
- **Kelly fraction**: f* = α/σ² × kelly_fraction_param
- **GARCH(1,1)**: Fits arch model on daily returns → σ_t (conditional volatility)
- **Dynamic stop**: stop = max(base_stop, 2 × σ_t)
- **Final weight**: min(kelly_applied, vol_target_weight)

### Agent 3 — Backtest (`agent3_backtest.py`)

**Entry rule:** SMA_fast > SMA_slow AND RSI < 60 AND signal > 0  
**Exit rules:** Stop-loss | Take-profit | Holding days exceeded | Signal reversal  
**TC model:** Almgren-Chriss — impact = λ × (trade_size/ADV)^0.6  
**Capacity:** Binary search for AUM where net Sharpe → 0

### Agent 4 — Statistics (`agent4_stats.py`)

| Test | What it checks |
|------|----------------|
| Jarque-Bera | Normality of returns |
| Welch t-test | H₀: mean return ≤ 0 |
| Bootstrap Sharpe CI | 10,000 resamples, 95% CI |
| CVaR 95% | Expected loss on worst 5% of days |
| Sortino | Sharpe using only downside deviation |
| Calmar | Annual return / max drawdown |
| Omega | Probability-weighted gains/losses ratio |

### Agent 5 — DSR Judge (`agent5_dsr.py`)

Implements **Bailey & López de Prado (2014) Deflated Sharpe Ratio**:

```
DSR = Φ( (SR̂ - E[max SR]) × √(N-1) / √Var_adj )
```

Where E[max SR] is the expected maximum Sharpe from N i.i.d. trials (penalises data mining).

| DSR | Verdict |
|-----|---------|
| > 0.95 | **Accept** — genuine edge |
| 0.50–0.95 | **Marginal** — real signal, insufficient confidence |
| < 0.50 | **Reject** — likely noise or overfit |

Also computes:
- **PBO** (Probability of Backtest Overfitting) — want < 0.20
- **MinTRL** (Minimum Track Record Length) — must be satisfied

### Agent 6 — Mutator (`agent6_mutator.py`)

- **Iterations 1–2**: Random sampling
- **Iteration 3+**: Gaussian Process (scikit-optimize) with Expected Improvement acquisition
- Generates **top-3 GP candidates** → Claude LLM picks one based on prior agent context
- Respects pruned regions — never proposes a dead parameter combination again

---

## Walk-Forward Validation (Out-of-Sample Testing)

The system supports **walk-forward validation** — the gold standard for proving a trading strategy works on unseen data.

### How It Works

```
Full data range: [2020-01-01 ————————————————————— 2024-12-31]

With walk-forward enabled:
  RALPH trains on:  [2020-01-01 ———————————— 2023-10-17]  (80%)
  OOS validates on: [2023-10-18 ——— 2024-12-31]           (20%)
```

1. The RALPH loop runs all 100 iterations on the **training period only**
2. After halt, the top-5 leaderboard strategies are automatically tested on the **out-of-sample period**
3. Results are classified as:

| OOS Verdict | Meaning |
|-------------|---------|
| **CONFIRMED** | OOS Sharpe ≥ 50% of IS Sharpe, OOS DSR > 0.50 |
| **DEGRADED** | OOS is weaker but still positive |
| **FAILED** | Strategy doesn't hold on unseen data |

### Configuration

```yaml
walk_forward:
  enabled: true
  mode: "single"        # "single", "expanding", or "rolling"
  test_ratio: 0.20      # reserve 20% of data for OOS
  n_folds: 3            # folds for expanding/rolling mode
  top_n_validate: 5     # validate top-N leaderboard strategies
```

### Modes

| Mode | Description |
|------|-------------|
| `single` | One split: train on 80%, test on 20% |
| `expanding` | N folds with growing training window (anchored start) |
| `rolling` | N folds with fixed-width sliding training window |

### Output

- `memory/oos_validation.json` — Full OOS results for each validated strategy
- `final_report.txt` — Appended with OOS validation summary

---

## Sentiment Signal (LLM-as-Alpha)

The system can use **news sentiment** as an additional alpha source alongside SMA/RSI technical signals.

### How It Works

1. `news_fetcher.py` pulls headlines from Yahoo Finance RSS + Google News RSS (free, no API key)
2. `agent0_sentiment.py` scores each headline:
   - **Computation layer**: Keyword-based scoring (30+ bullish/bearish word lists)
   - **Intelligence layer**: Claude LLM reads 20 headlines and produces a -1.0 to +1.0 aggregate score
3. Sentiment flows through `accumulated_context` to all downstream agents
4. All agents can factor sentiment into their decisions

### Configuration

```yaml
sentiment:
  enabled: true
  entry_weight: 0.15     # how much sentiment affects entry decision
  min_headlines: 3        # minimum headlines for a valid signal
```

### Example Output

```
  A: Stage 0 — sentiment
     sentiment=0.342 (mildly_bullish) headlines=12
```

### Graceful Degradation

- No internet? → Empty headlines → keyword_aggregate = 0.0 (neutral)
- No Anthropic API key? → Keyword scoring still works
- News fetch fails? → Pipeline continues without sentiment (logged as warning)

---

## Output

### `results.csv` — 66 columns, one row per iteration

Groups:

| Group | Columns | Source |
|-------|---------|--------|
| A — Metadata | iter, timestamp, duration | orchestrator |
| B — Parameters | stop_loss_pct, sma_fast/slow, rsi_period, ... | agent6 |
| C — Signal | ic_mean, ic_ir, factor_alpha, half_life_days | agent1 |
| D — Sizing | kelly_applied, final_weight, stop_loss_used, garch_sigma | agent2 |
| E — Backtest | net_sharpe_tc, max_drawdown, win_rate, capacity_usd | agent3 |
| F — Stats | ttest_pval, cvar_95, sortino, calmar, omega | agent4 |
| G — DSR | **dsr**, dsr_verdict, pbo, min_trl_satisfied | agent5 |
| H — RALPH | pruned, cumul_best_dsr, halt_triggered | orchestrator |
| I — Narratives | agent1–6 LLM interpretations | LLM layers |

### `final_report.txt` — Written on halt

Human-readable summary with best strategy parameters, statistical proof, capacity estimate, and recommended next steps.

### `memory/leaderboard.json` — Top-10 strategies by DSR

Updated after every iteration. Includes full params, DSR, PBO, Sharpe, capacity.

---

## The Four Questions It Answers

**1. What are the exact trading rules?**
> SMA(12) > SMA(48) AND RSI(14) < 58. Stop: max(1.8%, 2×GARCH-σ). Target: 7.2%. Size: 11% of capital. Hold max 9 days.

**2. Is this result real or data-mined luck?**
> DSR = 0.963. After testing 67 combinations, the DSR formula penalises for all 67 trials. 96.3% probability of genuine alpha. PBO = 0.17 — only 17% chance of overfit.

**3. Can I trade this at my capital size?**
> Capacity ceiling: ₹1.6 Cr. Above that, transaction costs kill the edge.

**4. What is the worst-case daily loss?**
> CVaR 95% = -1.8%/day. On your worst 1 day/month, expect ~1.8% of capital at risk.

---

## LLM Cost Estimate

| Agent | ~Input tokens | ~Output tokens | Cost/call |
|-------|--------------|----------------|-----------|
| agent1 | 800 | 300 | ~$0.003 |
| agent2 | 1,000 | 300 | ~$0.004 |
| agent3 | 1,200 | 400 | ~$0.005 |
| agent4 | 1,400 | 400 | ~$0.006 |
| agent5 | 2,000 | 600 | ~$0.009 |
| agent6 | 1,500 | 300 | ~$0.006 |
| **Total/iter** | | | **~$0.033** |
| **100 iterations** | | | **~$3.30** |

---

## Testing Individual Agents

Every agent has a `__main__` block for standalone testing:

```bash
python agents/agent1_signal.py     # Tests IC, alpha, half-life on RELIANCE.NS
python agents/agent2_sizing.py     # Tests Kelly + GARCH with fake returns
python agents/agent3_backtest.py   # Tests full backtest simulation
python agents/agent4_stats.py      # Tests all hypothesis tests
python agents/agent5_dsr.py        # Verifies DSR formula against paper values
python agents/agent6_mutator.py    # Tests random (iter 1) + GP (iter 5)
python memory_store.py             # Tests JSON save/load cycle
```

---

## Halt Conditions

The loop stops when any of these are true:

| Condition | Setting in config.yaml |
|-----------|----------------------|
| Max iterations reached | `max_iterations: 100` |
| DSR plateau — no improvement for N iters | `dsr_plateau_window: 10` |
| Leader PBO drops below threshold | `halt_pbo_threshold: 0.20` |
| Keyboard interrupt (Ctrl+C) | — |

On halt: `results.csv` final row gets `halt_triggered=True`, `final_report.txt` is written.

---

## Data Sources (All Automatic)

| Data | Source | Cached |
|------|--------|--------|
| OHLCV price data | `yfinance` | `cache/prices_*.csv` |
| Benchmark prices | `yfinance` | `cache/prices_*.csv` |
| Fama-French 3 factors | `pandas_datareader` (Kenneth French website) | `cache/ff_factors_*.csv` |

First run downloads and caches everything (~5 seconds). All 100 iterations read from cache (~0.1s each).

---

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No magic numbers** | All thresholds and constants defined in `constants.py` |
| **DRY** | Shared `data_loader.py` for price/factor data (used by agent1 + agent3) |
| **Secure persistence** | JSON-only storage — no pickle (eliminates arbitrary code execution risk) |
| **Validated config** | `validate_config()` runs at startup — catches bad dates, bounds, missing keys |
| **Graceful LLM degradation** | System works without Anthropic API key — agents use heuristic defaults |
| **Crash-safe** | Memory saved after every iteration; restart resumes from `n_trials + 1` |

---

## Migrating from Legacy Versions

If you have existing `.pkl` files in `memory/`:

- **Auto-migration**: On first load, `memory_store.py` converts `gp_state.pkl` and `garch_state.pkl` to `.json` equivalents
- **Backup**: Original `.pkl` files are renamed to `.pkl.bak`
- **No action needed**: Just run the system and it migrates automatically

---

## Pre-Deployment Checklist

Before trading real capital:

- [ ] DSR > 0.95 (accept verdict)
- [ ] PBO < 0.30
- [ ] MinTRL satisfied (enough history)
- [ ] `ttest_pval` < 0.05
- [ ] `net_sharpe_tc` > 1.0 (after costs)
- [ ] Your AUM < `capacity_usd`
- [ ] CVaR acceptable for your risk tolerance
- [ ] Max drawdown acceptable psychologically
- [ ] Top-3 leaderboard entries have similar params (GP converged)
- [ ] **Paper trade for 90 days before live capital**

---

## References

- Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions.* Journal of Risk.
- Ornstein, L. S., & Uhlenbeck, G. E. (1930). *On the theory of Brownian motion.* Physical Review.
