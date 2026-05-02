"""
Persistent memory shared across all agents and RALPH iterations.
Loaded at loop start, saved after every L stage.

All persistence uses JSON — no pickle, eliminating arbitrary code execution risk.
"""

import json
import logging
import os
from typing import Optional

from constants import PRUNE_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

PARAM_KEYS = [
    "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
    "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction",
]


class MemoryStore:
    """
    Persistent memory shared across all agents and RALPH iterations.
    Loaded at loop start, saved after every L stage.
    """

    def __init__(self, memory_dir: str = "memory/"):
        self.memory_dir = memory_dir
        self.gp_observations: list = []
        self.garch_sigma: dict = {}
        self.pruned_params: list = []
        self.leaderboard: list = []
        self.ic_history: list = []
        self.n_trials: int = 0
        self.dsr_history: list = []
        self.regime_history: list = []
        self.narrative_history: list = []

    # ── Persistence (JSON-only — no pickle) ──────────────────────────────────

    def save(self):
        os.makedirs(self.memory_dir, exist_ok=True)

        self._write_json("gp_state.json", {
            "gp_observations": self.gp_observations,
            "n_trials": self.n_trials,
            "dsr_history": self.dsr_history,
        })

        self._write_json("garch_state.json", {
            "garch_sigma": self.garch_sigma,
        })

        self._write_json("pruned.json", {
            "pruned_params": self.pruned_params,
        })

        self._write_json("leaderboard.json", self.leaderboard)

        self._write_json("ic_history.json", {
            "ic_history": self.ic_history,
            "regime_history": self.regime_history,
        })

        self._write_json("narrative_history.json", {
            "narratives": self.narrative_history,
        })

    def load(self):
        if not os.path.exists(self.memory_dir):
            return

        # ── Migrate from legacy pickle files ─────────────────────────────────
        self._migrate_pickle_to_json()

        # ── Load JSON files ──────────────────────────────────────────────────
        gp_data = self._read_json("gp_state.json")
        if gp_data:
            self.gp_observations = gp_data.get("gp_observations", [])
            self.n_trials = gp_data.get("n_trials", 0)
            self.dsr_history = gp_data.get("dsr_history", [])

        garch_data = self._read_json("garch_state.json")
        if garch_data:
            self.garch_sigma = garch_data.get("garch_sigma", {})

        pruned_data = self._read_json("pruned.json")
        if pruned_data:
            self.pruned_params = pruned_data.get("pruned_params", [])

        leaderboard_data = self._read_json("leaderboard.json")
        if leaderboard_data is not None:
            self.leaderboard = leaderboard_data if isinstance(leaderboard_data, list) else []

        ic_data = self._read_json("ic_history.json")
        if ic_data:
            self.ic_history = ic_data.get("ic_history", [])
            self.regime_history = ic_data.get("regime_history", [])

        narrative_data = self._read_json("narrative_history.json")
        if narrative_data:
            self.narrative_history = narrative_data.get("narratives", [])

    def _write_json(self, filename: str, data):
        path = os.path.join(self.memory_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _read_json(self, filename: str):
        path = os.path.join(self.memory_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def _migrate_pickle_to_json(self):
        """One-time migration from legacy .pkl files to .json equivalents."""
        import pickle

        migrated = False

        gp_pkl = os.path.join(self.memory_dir, "gp_state.pkl")
        gp_json = os.path.join(self.memory_dir, "gp_state.json")
        if os.path.exists(gp_pkl) and not os.path.exists(gp_json):
            try:
                with open(gp_pkl, "rb") as f:
                    data = pickle.load(f)
                self._write_json("gp_state.json", data)
                os.rename(gp_pkl, gp_pkl + ".bak")
                migrated = True
            except Exception as e:
                logger.warning(f"Could not migrate {gp_pkl}: {e}")

        garch_pkl = os.path.join(self.memory_dir, "garch_state.pkl")
        garch_json = os.path.join(self.memory_dir, "garch_state.json")
        if os.path.exists(garch_pkl) and not os.path.exists(garch_json):
            try:
                with open(garch_pkl, "rb") as f:
                    data = pickle.load(f)
                self._write_json("garch_state.json", data)
                os.rename(garch_pkl, garch_pkl + ".bak")
                migrated = True
            except Exception as e:
                logger.warning(f"Could not migrate {garch_pkl}: {e}")

        if migrated:
            logger.info("Migrated legacy pickle files to JSON. Old files renamed to .bak")

    # ── Pruning ───────────────────────────────────────────────────────────────

    def is_pruned(self, params: dict) -> bool:
        return any(
            self._params_similar(params, pruned)
            for pruned in self.pruned_params
        )

    def _params_similar(
        self, p1: dict, p2: dict, threshold: float = PRUNE_SIMILARITY_THRESHOLD
    ) -> bool:
        for k in PARAM_KEYS:
            v1 = float(p1.get(k, 0))
            v2 = float(p2.get(k, 0))
            max_v = max(abs(v1), abs(v2))
            if max_v > 0 and abs(v1 - v2) / max_v > threshold:
                return False
        return True

    def add_pruned(self, params: dict):
        if not self.is_pruned(params):
            self.pruned_params.append(dict(params))

    # ── Leaderboard ───────────────────────────────────────────────────────────

    def update_leaderboard(self, entry: dict):
        self.leaderboard.append(dict(entry))
        self.leaderboard.sort(key=lambda x: x.get("dsr", 0), reverse=True)
        self.leaderboard = self.leaderboard[:10]
        for i, item in enumerate(self.leaderboard):
            item["rank"] = i + 1

    # ── Halt detection ────────────────────────────────────────────────────────

    def dsr_plateau_detected(self, window: int, threshold: float) -> bool:
        if len(self.dsr_history) < window:
            return False
        recent = self.dsr_history[-window:]
        best_recent = max(recent)
        best_overall = max(self.dsr_history)
        return (best_overall - best_recent) < threshold

    # ── Summaries for LLM context ─────────────────────────────────────────────

    def get_prior_iterations_summary(self, last_n: int = 5) -> str:
        if not self.narrative_history:
            return "No prior iterations."
        recent = self.narrative_history[-last_n:]
        lines = []
        for entry in recent:
            iter_num = entry.get("iter", "?")
            a5 = entry.get("agent5", {})
            verdict = a5.get("final_verdict", "unknown")
            instruction = a5.get("mutator_instruction", "")
            a1 = entry.get("agent1", {})
            sig_q = a1.get("signal_quality", "unknown")
            lines.append(
                f"Iter {iter_num}: verdict={verdict}, signal={sig_q}. "
                f"Mutator note: {instruction}"
            )
        return "\n".join(lines)

    def get_leaderboard_summary(self) -> str:
        if not self.leaderboard:
            return "Leaderboard is empty (no iterations completed yet)."
        lines = ["Top strategies by DSR:"]
        for entry in self.leaderboard[:5]:
            lines.append(
                f"  Rank {entry.get('rank', '?')}: DSR={entry.get('dsr', 0):.3f}, "
                f"iter={entry.get('iter', '?')}, pbo={entry.get('pbo', 0):.2f}, "
                f"sharpe={entry.get('net_sharpe_tc', 0):.2f}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    mem = MemoryStore(memory_dir="memory_test/")
    mem.gp_observations.append({"params": {"sma_fast": 12}, "dsr": 0.75, "iter": 1})
    mem.ic_history.append(0.065)
    mem.dsr_history.append(0.75)
    mem.n_trials = 1
    mem.save()

    mem2 = MemoryStore(memory_dir="memory_test/")
    mem2.load()
    assert mem2.n_trials == 1
    assert len(mem2.gp_observations) == 1
    print("MemoryStore save/load: OK")

    import shutil
    shutil.rmtree("memory_test/", ignore_errors=True)
