"""One-process Stage 7 integration probe (SANITY TEST, not a strength run).

Run after a full sanity training run, or standalone:
  python -m experiments.alpha_zero.sanity_check --output runs/alpha_zero/diagnostic
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from Algorithms.alpha_zero.config import load_settings
from Algorithms.alpha_zero.runtime import ROOT, use_repository_open_spiel

use_repository_open_spiel()
import pyspiel  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402
from open_spiel.python.algorithms.alpha_zero import alpha_zero, evaluator, replay_buffer  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def tree_equal(left, right):
    return all(np.allclose(a, b, rtol=1e-6, atol=1e-7)
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def loss_and_grads(model, data, coefficient):
    def objective(params):
        logits, values = jax.vmap(
            lambda obs: model._state.apply_fn({"params": params}, obs, training=False)
        )(data.observation)
        logits = jnp.where(data.legals_mask, logits, jnp.finfo(jnp.float32).min)
        policy_loss = optax.softmax_cross_entropy(logits, data.policy).mean()
        value_loss = jnp.square(values - data.value).mean()
        l2_loss = coefficient * sum(
            jnp.square(layer["kernel"]).sum() for layer in params.values()
        )
        return policy_loss + value_loss + l2_loss, (policy_loss, value_loss, l2_loss)

    (total, parts), grads = jax.value_and_grad(objective, has_aux=True)(
        model._state.params
    )
    return [float(x) for x in (*parts, total)], grads


def search_details(config, game, neural, state):
    raw_value, nn_policy = neural._inference(state)
    legal = state.legal_actions()
    root = alpha_zero._init_bot(config, game, neural, True).mcts_search(state)
    priors = {int(c.action): float(c.prior) for c in root.children}
    counts = {int(c.action): int(c.explore_count) for c in root.children}
    nn_policy = np.asarray(nn_policy)
    for action in legal:
        np.testing.assert_allclose(priors[action], nn_policy[action], atol=1e-6)
    powered = np.array([counts.get(a, 0) ** (1 / config.temperature)
                        for a in range(6)], dtype=float)
    search_policy = powered / powered.sum()
    return {
        "legal": legal,
        "nn_policy": nn_policy.tolist(),
        "raw_value": float(raw_value),
        "root_priors": [priors.get(a, 0.0) for a in range(6)],
        "visit_counts": [counts.get(a, 0) for a in range(6)],
        "search_policy": search_policy.tolist(),
    }


class RecordingBot:
    def __init__(self, bot, records):
        self.bot = bot
        self.records = records

    def mcts_search(self, state):
        root = self.bot.mcts_search(state)
        self.records.append({
            "player": state.current_player(),
            "legal": state.legal_actions(),
            "priors": {c.action: float(c.prior) for c in root.children},
            "visits": {c.action: int(c.explore_count) for c in root.children},
        })
        return root


def play_recorded(config, game, neural, game_number):
    roots = []
    bots = tuple(RecordingBot(alpha_zero._init_bot(config, game, neural, False),
                              roots) for _ in range(2))
    logger = SimpleNamespace(print=lambda *args: None,
                             opt_print=lambda *args: None)
    trajectory = alpha_zero._play_game(
        logger, game_number, game, bots, config.temperature,
        config.temperature_drop,
    )
    check(len(roots) == len(trajectory.states), "MCTS roots and trajectory differ")
    replay_state = game.new_initial_state()
    moves = []
    for index, (position, root) in enumerate(zip(trajectory.states, roots)):
        check(replay_state.current_player() == position.current_player,
              "stored player differs from game state")
        np.testing.assert_allclose(position.observation,
                                   replay_state.observation_tensor(), atol=1e-6)
        np.testing.assert_allclose(position.observation[14],
                                   (config.game_max_moves - index) / config.game_max_moves)
        powered = np.array([root["visits"].get(a, 0) ** (1 / config.temperature)
                            for a in range(6)], dtype=float)
        np.testing.assert_allclose(position.policy, powered / powered.sum(), atol=1e-6)
        check(position.action in root["legal"], "played illegal action")
        moves.append({"move": index, "player": int(position.current_player),
                      "legal": root["legal"], "action": int(position.action)})
        replay_state.apply_action(position.action)
    check(replay_state.is_terminal(), "self-play did not terminate")
    np.testing.assert_allclose(trajectory.returns, replay_state.returns())
    return trajectory, moves, ("horizon" if replay_state.truncated else "natural")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "experiments/alpha_zero/configs/sanity.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    game = pyspiel.load_game(settings.game_string)
    config = alpha_zero.Config(**settings.upstream_kwargs(output)).replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions(),
    )
    # Diagnostic-only move counter; not an upstream training configuration key.
    config = SimpleNamespace(**config.__dict__, game_max_moves=settings.max_moves)
    model = alpha_zero._init_model_from_config(config)
    check(model.num_trainable_variables == 185863, "parameter count")
    check(game.observation_tensor_shape() == [15], "observation size")
    check(game.num_distinct_actions() == 6, "policy size")
    neural = evaluator.AlphaZeroEvaluator(game, model)
    initial_state = game.new_initial_state()
    initial = search_details(config, game, neural, initial_state)
    check(np.isfinite(initial["raw_value"]) and abs(initial["raw_value"]) <= 1,
          "initial value range")
    check(np.isclose(sum(initial["nn_policy"]), 1), "initial policy normalization")
    p0_vector = neural.evaluate(initial_state)
    np.testing.assert_allclose(p0_vector,
                               [initial["raw_value"], -initial["raw_value"]])
    player_one_state = game.new_initial_state()
    player_one_state.apply_action(0)
    p1_raw = float(neural._inference(player_one_state)[0])
    np.testing.assert_allclose(neural.evaluate(player_one_state), [-p1_raw, p1_raw])

    np.random.seed(17)
    trajectories = []
    move_logs = []
    terminations = []
    example_records = []
    buffer = replay_buffer.Buffer(settings.replay_buffer_size, seed=17)
    for game_number in range(2):
        trajectory, moves, reason = play_recorded(config, game, neural, game_number)
        trajectories.append(trajectory)
        move_logs.append(moves)
        terminations.append(reason)
        trace_state = game.new_initial_state()
        for position, example in zip(trajectory.states,
                                     alpha_zero.replay_examples(trajectory)):
            check(example.observation.shape == (15,) and example.policy.shape == (6,),
                  "replay shape")
            check(np.isfinite(example.observation).all() and
                  np.isfinite(example.policy).all() and np.isfinite(example.value),
                  "non-finite replay record")
            check(np.all(np.asarray(example.policy) >= 0) and
                  np.isclose(float(example.policy.sum()), 1), "invalid replay policy")
            check(np.all(np.asarray(example.policy)[~np.asarray(example.legals_mask)] == 0),
                  "illegal action has target mass")
            np.testing.assert_allclose(example.value,
                                       trajectory.returns[position.current_player])
            check(-1 <= example.value <= 1, "replay value out of range")
            check(0 <= example.observation[14] <= 1, "horizon out of range")
            raw_value, nn_policy = neural._inference(trace_state)
            check(np.isfinite(raw_value) and -1 <= raw_value <= 1,
                  "non-finite or out-of-range self-play NN value")
            check(np.isfinite(nn_policy).all() and np.all(np.asarray(nn_policy) >= 0)
                  and np.isclose(np.sum(nn_policy), 1), "invalid self-play NN policy")
            check(np.all(np.asarray(nn_policy)[~np.asarray(example.legals_mask)] == 0),
                  "NN policy has illegal action mass")
            if len(example_records) < 4:
                example_records.append({
                    "game": game_number,
                    "player": int(position.current_player),
                    "x": np.asarray(example.observation).tolist(),
                    "legal": np.flatnonzero(example.legals_mask).tolist(),
                    "nn_policy": np.asarray(nn_policy).tolist(),
                    "nn_value": float(raw_value),
                    "pi": np.asarray(example.policy).tolist(),
                    "z": float(example.value),
                })
            buffer.append(example)
            trace_state.apply_action(position.action)
    check(len(buffer) == 2 * settings.max_moves, "replay position count")
    data = buffer.sample(settings.train_batch_size)
    check(data.observation.shape == (settings.train_batch_size, 15), "batch X shape")
    check(data.policy.shape == (settings.train_batch_size, 6), "batch policy shape")
    check(data.value.shape == (settings.train_batch_size,), "batch value shape")
    check(data.observation.dtype == data.policy.dtype == data.value.dtype == jnp.float32,
          "batch dtype")
    check(data.legals_mask.dtype == jnp.bool, "mask dtype")

    before_loss, grads = loss_and_grads(model, data, settings.weight_decay)
    check(all(np.isfinite(before_loss)), "non-finite initial loss")
    np.testing.assert_allclose(before_loss[3], sum(before_loss[:3]), atol=1e-6)
    grad_norm = float(np.sqrt(sum(float(jnp.square(g).sum())
                                  for g in jax.tree.leaves(grads))))
    check(np.isfinite(grad_norm), "non-finite gradient norm")
    layer_names = ("trunk_1", "trunk_2", "trunk_3", "policy_hidden",
                   "policy_logits", "value_hidden", "value_output")
    for name in layer_names:
        leaves = jax.tree.leaves(grads[name])
        check(all(np.isfinite(x).all() for x in leaves), f"non-finite {name} gradient")
        check(any(np.any(np.asarray(x) != 0) for x in leaves),
              f"zero {name} gradient")
    original_params = jax.tree.map(lambda x: np.array(x, copy=True), model._state.params)
    history = []
    first = model.update(data)
    np.testing.assert_allclose([first.policy, first.value, first.l2, first.total],
                               before_loss, rtol=1e-5, atol=1e-5)
    history.append([float(first.policy), float(first.value),
                    float(first.l2), float(first.total)])
    for name in layer_names:
        check(np.any(original_params[name]["kernel"] !=
                     np.asarray(model._state.params[name]["kernel"])),
              f"{name} did not update")
    check(all(np.isfinite(np.asarray(x)).all() for x in
              jax.tree.leaves(model._state.params)), "non-finite parameters")
    for _ in range(3):
        loss = model.update(buffer.sample(settings.train_batch_size))
        row = [float(loss.policy), float(loss.value), float(loss.l2),
               float(loss.total)]
        check(all(np.isfinite(row)), "non-finite training loss")
        check(all(np.isfinite(np.asarray(x)).all() for x in
                  jax.tree.leaves(model._state.params)), "non-finite parameters")
        history.append(row)

    fixed_observation = initial_state.observation_tensor()
    fixed_mask = initial_state.legal_actions_mask()
    before_checkpoint = model.inference(fixed_observation, fixed_mask)
    model.save_checkpoint(7)
    restored = alpha_zero._init_model_from_config(config)
    restored.load_checkpoint(7)
    after_checkpoint = restored.inference(fixed_observation, fixed_mask)
    for a, b in zip(before_checkpoint, after_checkpoint):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    check(restored._state.step == model._state.step == 4, "checkpoint training step")
    check(tree_equal(restored._state.opt_state, model._state.opt_state),
          "checkpoint optimizer state")
    resume_batch = buffer.sample(settings.train_batch_size)
    model.update(resume_batch)
    restored.update(resume_batch)
    check(restored._state.step == model._state.step == 5, "resume step")
    check(tree_equal(restored._state.params, model._state.params),
          "resume parameters")
    updated_neural = evaluator.AlphaZeroEvaluator(game, restored)
    updated = search_details(config, game, updated_neural, game.new_initial_state())
    check(not np.allclose(updated["nn_policy"], initial["nn_policy"]),
          "updated evaluator still uses old policy")
    post_trajectory, post_moves, post_reason = play_recorded(
        config, game, updated_neural, 2
    )
    check(len(post_trajectory.states) > 0, "no updated self-play positions")

    report = {
        "label": "SANITY TEST VALUES - NOT EXPERIMENTAL HYPERPARAMETERS",
        "config": str(args.config.resolve()),
        "parameters": int(model.num_trainable_variables),
        "observation_shape": game.observation_tensor_shape(),
        "initial_search": initial,
        "p0_evaluator": np.asarray(p0_vector).tolist(),
        "p1_raw_value": p1_raw,
        "p1_evaluator": np.asarray(neural.evaluate(player_one_state)).tolist(),
        "self_play_games": len(trajectories),
        "generated_positions": sum(len(t.states) for t in trajectories),
        "game_moves": move_logs,
        "final_returns": [list(map(float, t.returns)) for t in trajectories],
        "example_records": example_records,
        "termination_reasons": terminations,
        "replay_size": len(buffer),
        "batch_shapes": [list(data.observation.shape), list(data.policy.shape),
                         list(data.value.shape)],
        "batch_dtypes": [str(data.observation.dtype), str(data.policy.dtype),
                         str(data.value.dtype)],
        "batch_device": str(data.observation.device),
        "pre_update_losses": before_loss,
        "global_gradient_norm": grad_norm,
        "training_losses": history,
        "optimizer_steps_after_resume": int(restored._state.step),
        "updated_search": updated,
        "updated_self_play_moves": post_moves,
        "updated_termination_reason": post_reason,
    }
    (output / "sanity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
