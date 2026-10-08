"""Minimal Stage 8 JAX compute benchmark; does not tune training settings.

Run: python -m experiments.alpha_zero.benchmark --output runs/alpha_zero/benchmark.json
"""

import argparse
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
import pyspiel  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402,F401
from open_spiel.python.algorithms.alpha_zero import alpha_zero, evaluator, utils  # noqa: E402


class TimedEvaluator(evaluator.AlphaZeroEvaluator):
    def __init__(self, game, model):
        super().__init__(game, model)
        self.nn_seconds = 0.0
        self.nn_calls = 0

    def _inference(self, state):
        started = time.perf_counter()
        result = super()._inference(state)
        jax.block_until_ready(result)
        self.nn_seconds += time.perf_counter() - started
        self.nn_calls += 1
        return result


def hardware():
    cpu = platform.processor() or platform.uname().processor or "unknown"
    ram_gib = None
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong),
                        ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram_gib = status.total_physical / 2**30
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram_gib = int(line.split()[1]) * 1024 / 2**30
                    break
        except OSError:
            pass
    devices = jax.devices()
    accelerators = [str(device) for device in devices if device.platform != "cpu"]
    return {"cpu": cpu, "gpu_or_accelerator": accelerators or "none used by JAX",
            "ram_gib": ram_gib, "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in devices],
            "jax_default_device": str(devices[0])}


def benchmark_self_play(settings, game, model, games, budgets):
    results = []
    for budget in budgets:
        config = alpha_zero.Config(**replace(settings, max_simulations=budget)
                                   .upstream_kwargs("."))
        network = TimedEvaluator(game, model)
        bots = tuple(alpha_zero._init_bot(config, game, network, False)
                     for _ in range(2))
        logger = type("QuietLogger", (), {"print": lambda *a: None,
                                          "opt_print": lambda *a: None})()
        lengths = []
        started = time.perf_counter()
        for game_number in range(games):
            trajectory = alpha_zero._play_game(
                logger, game_number, game, bots, config.temperature,
                config.temperature_drop)
            lengths.append(len(trajectory.states))
            print(f"self-play {budget}: game {game_number + 1}/{games}, "
                  f"{lengths[-1]} moves", flush=True)
        jax.block_until_ready(model._state.params)
        wall = time.perf_counter() - started
        moves = sum(lengths)
        simulations = budget * moves
        results.append({
            "simulations_per_move": budget, "games": games,
            "total_moves": moves, "mean_game_length": moves / games,
            "game_lengths": lengths, "wall_seconds": wall,
            "seconds_per_game": wall / games, "games_per_hour": 3600 * games / wall,
            "simulations_per_second": simulations / wall,
            "replay_positions_per_hour": 3600 * moves / wall,
            "nn_evaluator_seconds": network.nn_seconds,
            "nn_evaluator_calls": network.nn_calls,
            "nn_fraction": network.nn_seconds / wall,
            "search_environment_fraction": (wall - network.nn_seconds) / wall,
        })
    return results


def benchmark_training(model, game, sizes, warmups, repeats):
    state = game.new_initial_state()
    observation = jnp.asarray(state.observation_tensor(), dtype=jnp.float32)
    mask = jnp.asarray(state.legal_actions_mask(), dtype=jnp.bool)
    policy = mask.astype(jnp.float32) / mask.sum()
    results = []
    for size in sizes:
        batch = utils.TrainInput(
            observation=jnp.broadcast_to(observation, (size, 15)),
            legals_mask=jnp.broadcast_to(mask, (size, 6)),
            policy=jnp.broadcast_to(policy, (size, 6)),
            value=jnp.zeros((size,), dtype=jnp.float32),
        )
        for _ in range(warmups):
            model.update(batch)
        jax.block_until_ready(model._state)
        started = time.perf_counter()
        for _ in range(repeats):
            model.update(batch)
        jax.block_until_ready(model._state)
        wall = time.perf_counter() - started
        results.append({"batch_size": size, "warmup_updates": warmups,
                        "timed_updates": repeats, "wall_seconds": wall,
                        "milliseconds_per_update": 1000 * wall / repeats,
                        "examples_per_second": size * repeats / wall})
        print(f"training B={size}: {1000 * wall / repeats:.2f} ms/update",
              flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--updates", type=int, default=20)
    args = parser.parse_args()
    if min(args.games, args.max_moves, args.warmups, args.updates) < 1:
        parser.error("all counts must be positive")
    settings = Settings(max_moves=args.max_moves)
    game = pyspiel.load_game(settings.game_string)
    config = alpha_zero.Config(**settings.upstream_kwargs(args.output.parent)).replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())
    model = alpha_zero._init_model_from_config(config)
    if int(model.num_trainable_variables) != 185863:
        raise AssertionError("unexpected model architecture")
    state = game.new_initial_state()
    jax.block_until_ready(model.inference(state.observation_tensor(),
                                          state.legal_actions_mask()))
    environment = hardware()
    print(json.dumps(environment), flush=True)
    self_play = benchmark_self_play(settings, game, model, args.games,
                                    (32, 64, 128, 256))
    training = benchmark_training(model, game, (128, 256, 512),
                                  args.warmups, args.updates)
    report = {
        "label": "MINIMAL COMPUTE BENCHMARK - NOT HYPERPARAMETER TUNING",
        "hardware": environment,
        "configuration": {"games_per_budget": args.games,
                          "max_moves": args.max_moves,
                          "same_initial_network_for_all_search_budgets": True,
                          "mcts_budgets": [32, 64, 128, 256],
                          "training_batches": [128, 256, 512],
                          "warmup_updates_per_batch": args.warmups,
                          "timed_updates_per_batch": args.updates,
                          "self_play_actors": 1,
                          "search_has_dirichlet_noise": True},
        "self_play": self_play, "training": training,
        "coarse_time_breakdown": {
            "budget": 128,
            "nn_fraction": self_play[2]["nn_fraction"],
            "search_environment_fraction": self_play[2]["search_environment_fraction"],
            "method": "wall time in AlphaZeroEvaluator._inference, including cache lookup and JAX synchronization; remainder is search, environment, and Python overhead",
        },
        "runtime_projections": {
            "self_play": [
                {"simulations_per_move": row["simulations_per_move"],
                 "games": {str(n): row["seconds_per_game"] * n
                           for n in (100, 1000, 10000)}}
                for row in self_play],
            "training": [
                {"batch_size": row["batch_size"],
                 "updates": {str(n): row["wall_seconds"] / args.updates * n
                             for n in (1000, 10000)}}
                for row in training],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
