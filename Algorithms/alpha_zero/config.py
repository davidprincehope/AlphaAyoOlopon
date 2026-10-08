"""Validated experiment settings, independent of the optional training packages."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    max_moves: int = 1000
    cutoff: str = "collect"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    decouple_weight_decay: bool = False
    train_batch_size: int = 128
    replay_buffer_size: int = 16384
    replay_buffer_reuse: int = 4
    max_steps: int = 100
    checkpoint_freq: int = 10
    resume_checkpoint_freq: int = 10
    actors: int = 2
    evaluators: int = 0
    evaluation_games: int = 100
    evaluation_opponent: dict = field(default_factory=lambda: {"name": "RAND", "params": {}})
    evaluation_seed: int = 0
    evaluation_window: int = 50
    eval_levels: int = 3
    uct_c: float = 1.5
    max_simulations: int = 64
    policy_alpha: float = 1.0
    policy_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop: int = 20
    nn_model: str = "ayo_mlp"
    nn_width: int = 256
    nn_depth: int = 3
    nn_api_version: str = "linen"
    verbose: bool = False
    quiet: bool = True

    def __post_init__(self):
        self._integer("evaluation_games", 0)
        self._integer("evaluation_seed", 0)
        if self.evaluation_games % 2:
            raise ValueError("evaluation_games must be zero or even for balanced seats")
        if (not isinstance(self.evaluation_opponent, dict)
                or not isinstance(self.evaluation_opponent.get("name"), str)
                or not isinstance(self.evaluation_opponent.get("params", {}), dict)):
            raise ValueError("evaluation_opponent requires a name and a params object")
        for name in (
            "max_moves", "train_batch_size", "replay_buffer_size",
            "replay_buffer_reuse", "checkpoint_freq", "resume_checkpoint_freq", "actors",
            "evaluation_window", "nn_width", "nn_depth",
        ):
            self._integer(name, 1)
        for name in ("max_steps", "evaluators", "temperature_drop"):
            self._integer(name, 0)
        # Upstream divides by eval_levels - 1 even with no evaluator workers.
        self._integer("eval_levels", 2)
        # One simulation only visits the root, leaving zero child visit counts.
        self._integer("max_simulations", 2)
        for name in ("learning_rate", "uct_c", "policy_alpha", "temperature"):
            self._number(name, strictly_positive=True)
        for name in ("weight_decay", "policy_epsilon"):
            self._number(name)
        if self.policy_epsilon > 1:
            raise ValueError("policy_epsilon must be in [0, 1]")
        if self.replay_buffer_size // self.replay_buffer_reuse < self.train_batch_size:
            raise ValueError("replay_buffer_size // replay_buffer_reuse must cover a batch")
        if self.cutoff != "collect":
            raise ValueError("cutoff must be collect")
        if self.nn_model != "ayo_mlp" or self.nn_width != 256 or self.nn_depth != 3:
            raise ValueError("The canonical model requires ayo_mlp, width=256, depth=3")
        if self.nn_api_version != "linen":
            raise ValueError("The canonical model currently requires nn_api_version=linen")
        if self.decouple_weight_decay:
            raise ValueError("The canonical model uses Adam with explicit L2, not AdamW")
        for name in ("decouple_weight_decay", "verbose", "quiet"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def _integer(self, name, minimum):
        value = getattr(self, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def _number(self, name, strictly_positive=False):
        value = getattr(self, name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (strictly_positive and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")

    @property
    def game_string(self):
        return f"ayo_olopon_alpha_zero(max_moves={self.max_moves},cutoff={self.cutoff})"

    def upstream_kwargs(self, path):
        values = asdict(self)
        values.pop("max_moves")
        values.pop("cutoff")
        values.update(game=self.game_string, path=str(Path(path).resolve()),
                      observation_shape=None, output_size=None)
        return values


def load_settings(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = data.keys() - Settings.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return Settings(**data)
