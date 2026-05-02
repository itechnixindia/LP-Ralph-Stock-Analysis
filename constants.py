"""
Named constants — eliminates magic numbers across the codebase.
"""

# ── Signal Analysis ───────────────────────────────────────────────────────────
IC_ROLLING_WINDOW = 60
IC_MIN_WINDOW_SIZE = 20
IC_EXCELLENT_THRESHOLD = 0.10
IC_GOOD_THRESHOLD = 0.05
IC_IR_RELIABLE_THRESHOLD = 0.5
IC_TREND_DELTA = 0.01
IC_TREND_LOOKBACK = 5

# ── OU Half-life ──────────────────────────────────────────────────────────────
OU_MIN_OBSERVATIONS = 30
OU_DEFAULT_HALF_LIFE = 10.0
OU_NON_MEAN_REVERTING_FALLBACK = 30.0

# ── GARCH ─────────────────────────────────────────────────────────────────────
GARCH_MIN_OBSERVATIONS = 100
GARCH_ROLLING_FALLBACK_WINDOW = 60
GARCH_MIN_SIGMA = 1e-4

# ── Position Sizing ───────────────────────────────────────────────────────────
KELLY_MAX_FULL = 5.0
KELLY_MAX_APPLIED = 1.0
VOL_TARGET_MAX_WEIGHT = 2.0
FINAL_WEIGHT_MIN = 0.01
FINAL_WEIGHT_MAX = 1.0
ANNUALIZATION_FACTOR = 252

# ── Backtest ──────────────────────────────────────────────────────────────────
RSI_ENTRY_THRESHOLD = 60
TRADE_IMPACT_EXPONENT = 0.6
MAX_TRADE_FRACTION = 1.0
DEFAULT_ADV_DOLLAR = 1e6

# ── Capacity ──────────────────────────────────────────────────────────────────
CAPACITY_SEARCH_MIN = 1000.0
CAPACITY_SEARCH_MAX = 10_000_000.0
CAPACITY_SEARCH_ITERATIONS = 30

# ── Statistics ────────────────────────────────────────────────────────────────
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42
MIN_RETURNS_FOR_STATS = 20
MIN_RETURNS_FOR_BOOTSTRAP = 10
OMEGA_CAPPED_VALUE = 10.0

# ── DSR ───────────────────────────────────────────────────────────────────────
EULER_MASCHERONI = 0.5772156649
DSR_ACCEPT_THRESHOLD = 0.95
DSR_MARGINAL_THRESHOLD = 0.50
PBO_HIGH_THRESHOLD = 0.50
MIN_TRL_MAX = 999999

# ── Pruning ───────────────────────────────────────────────────────────────────
PRUNE_SIMILARITY_THRESHOLD = 0.08

# ── GP / Mutator ──────────────────────────────────────────────────────────────
GP_MIN_OBSERVATIONS_FOR_TRUST = 3
GP_N_INITIAL_POINTS = 5
GP_MAX_CANDIDATE_ATTEMPTS = 50
GP_N_CANDIDATES = 3

# ── Regime ────────────────────────────────────────────────────────────────────
REGIME_LOOKBACK_DAYS = 60
REGIME_MIN_DAYS = 20
REGIME_BULL_THRESHOLD = 0.10
REGIME_BEAR_THRESHOLD = -0.10
REGIME_UPDATE_FREQUENCY = 10

# ── Walk-Forward Validation ───────────────────────────────────────────────────
WF_DEFAULT_TEST_RATIO = 0.20
WF_DEFAULT_N_FOLDS = 3
WF_DEFAULT_MODE = "expanding"
WF_MIN_TEST_DAYS = 60
WF_SHARPE_DEGRADATION_WARN = 0.50  # warn if OOS Sharpe drops >50% from IS
WF_TOP_N_VALIDATE = 5              # validate top-N leaderboard strategies

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_BULLISH_THRESHOLD = 0.3
SENTIMENT_BEARISH_THRESHOLD = -0.3
SENTIMENT_SIGNAL_MIN_HEADLINES = 3
SENTIMENT_ENTRY_WEIGHT = 0.15      # how much sentiment affects entry decision
SENTIMENT_CACHE_HOURS = 6

# ── Epsilon for division safety ───────────────────────────────────────────────
EPSILON = 1e-9
