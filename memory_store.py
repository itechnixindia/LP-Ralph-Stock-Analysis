import os
import json
import pickle
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(self.memory_dir, exist_ok=True)

        with open(os.path.join(self.memory_dir, "gp_state.pkl"), "wb") as f:
            pickle.dump(
                {
                    "gp_observations": self.gp_observations,
                    "n_trials": self.n_trials,
                    "dsr_history": self.dsr_history,
                },
                f,
            )

        with open(os.path.join(self.memory_dir, "garch_state.pkl"), "wb") as f:
            pickle.dump({"garch_sigma": self.garch_sigma}, f)

        with open(os.path.join(self.memory_dir, "pruned.json"), "w") as f:
            json.dump({"pruned_params": self.pruned_params}, f)

        with open(os.path.join(self.memory_dir, "leaderboard.json"), "w") as f:
            json.dump(self.leaderboard, f, indent=2)

        with open(os.path.join(self.memory_dir, "ic_history.json"), "w") as f:
            json.dump(
                {
                    "ic_history": self.ic_history,
                    "regime_history": self.regime_history,
                },
                f,
            )

        with open(os.path.join(self.memory_dir, "narrative_history.json"), "w") as f:
            json.dump({"narratives": self.narrative_history}, f, indent=2)

    def load(self):
        if not os.path.exists(self.memory_dir):
            return

        gp_path = os.path.join(self.memory_dir, "gp_state.pkl")
        if os.path.exists(gp_path):
            with open(gp_path, "rb") as f:
                data = pickle.load(f)
            self.gp_observations = data.get("gp_observations", [])
            self.n_trials = data.get("n_trials", 0)
            self.dsr_history = data.get("dsr_history", [])

        garch_path = os.path.join(self.memory_dir, "garch_state.pkl")
        if os.path.exists(garch_path):
            with open(garch_path, "rb") as f:
                data = pickle.load(f)
            self.garch_sigma = data.get("garch_sigma", {})

        pruned_path = os.path.join(self.memory_dir, "pruned.json")
        if os.path.exists(pruned_path):
            with open(pruned_path, "r") as f:
                data = json.load(f)
            self.pruned_params = data.get("pruned_params", [])

        leaderboard_path = os.path.join(self.memory_dir, "leaderboard.json")
        if os.path.exists(leaderboard_path):
            with open(leaderboard_path, "r") as f:
                self.leaderboard = json.load(f)

        ic_path = os.path.join(self.memory_dir, "ic_history.json")
        if os.path.exists(ic_path):
            with open(ic_path, "r") as f:
                data = json.load(f)
            self.ic_history = data.get("ic_history", [])
            self.regime_history = data.get("regime_history", [])

        narrative_path = os.path.join(self.memory_dir, "narrative_history.json")
        if os.path.exists(narrative_path):
            with open(narrative_path, "r") as f:
                data = json.load(f)
            self.narrative_history = data.get("narratives", [])

    # ── Pruning ───────────────────────────────────────────────────────────────

    def is_pruned(self, params: dict) -> bool:
        for pruned in self.pruned_params:
            if self._params_similar(params, pruned):
                return True
        return False

    def _params_similar(self, p1: dict, p2: dict, threshold: float = 0.08) -> bool:
        keys = [
            "stop_loss_pct", "take_profit_pct", "holding_days", "sma_fast",
            "sma_slow", "rsi_period", "vol_target_pct", "kelly_fraction",
        ]
        for k in keys:
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
