"""Stage 4 checks for the canonical policy-value training objective."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from types import SimpleNamespace
from flax.core import unfreeze

from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
from open_spiel.python.algorithms.alpha_zero import model_linen, utils  # noqa: E402
from open_spiel.python.algorithms.alpha_zero import alpha_zero  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402,F401
import pyspiel  # noqa: E402


def make_model(tmp_path, weight_decay=0.0):
    return model_linen.Model.build_model(
        "ayo_mlp", (15,), 6, 256, 3, weight_decay, 0.001, str(tmp_path)
    )


def batch(observations, masks, policies, values):
    return utils.TrainInput(
        observation=jnp.asarray(observations, dtype=jnp.float32),
        legals_mask=jnp.asarray(masks, dtype=bool),
        policy=jnp.asarray(policies, dtype=jnp.float32),
        value=jnp.asarray(values, dtype=jnp.float32),
    )


def test_self_play_policy_is_normalized_visit_counts_in_local_action_order():
    game = pyspiel.load_game(ayo_game.GAME_NAME, {"max_moves": 1})

    class Bot:
        def mcts_search(self, state):
            children = [SimpleNamespace(action=i, explore_count=count)
                        for i, count in enumerate([7, 2, 1, 0, 0, 0])]
            return SimpleNamespace(
                children=children, total_reward=0, explore_count=10,
                best_child=lambda: children[0],
            )

    logger = SimpleNamespace(opt_print=lambda *args: None, print=lambda *args: None)
    trajectory = alpha_zero._play_game(logger, 0, game, (Bot(), Bot()), 1.0, 0)
    assert len(trajectory.states) == 1
    stored = trajectory.states[0]
    np.testing.assert_allclose(stored.policy, [0.7, 0.2, 0.1, 0, 0, 0])
    assert stored.action == 0
    assert all(stored.policy[i] == 0 for i in range(6) if not stored.legals_mask[i])


def test_exact_cross_entropy_squared_value_and_illegal_mask(tmp_path):
    model = make_model(tmp_path)
    params = unfreeze(model._state.params)
    for layer in params.values():
        for name in layer:
            layer[name] = jnp.zeros_like(layer[name])
    params["policy_logits"]["bias"] = jnp.array(
        [np.log(0.6), np.log(0.3), np.log(0.1), 10.0, 10.0, 10.0]
    )
    params["value_output"]["bias"] = jnp.array([np.arctanh(0.5)])
    model._state = model._state.replace(params=params)
    data = batch(
        np.zeros((1, 15)), [[1, 1, 1, 0, 0, 0]],
        [[0.7, 0.2, 0.1, 0, 0, 0]], [1.0],
    )
    losses = model.update(data)
    expected = -(0.7 * np.log(0.6) + 0.2 * np.log(0.3) + 0.1 * np.log(0.1))
    np.testing.assert_allclose(losses.policy, expected, rtol=1e-6)
    np.testing.assert_allclose(losses.value, 0.25, rtol=1e-6)
    np.testing.assert_allclose(losses.l2, 0.0, atol=1e-7)
    assert np.isfinite(losses.total)


@pytest.mark.parametrize("target,prediction,expected", [
    (1.0, 0.5, 0.25), (-1.0, -0.5, 0.25), (0.0, 0.2, 0.04),
])
def test_value_loss_is_full_square_for_either_player(tmp_path, target, prediction, expected):
    model = make_model(tmp_path)
    params = unfreeze(model._state.params)
    for layer in params.values():
        for name in layer:
            layer[name] = jnp.zeros_like(layer[name])
    params["value_output"]["bias"] = jnp.array([np.arctanh(prediction)])
    model._state = model._state.replace(params=params)
    data = batch(np.zeros((2, 15)), np.ones((2, 6)),
                 np.full((2, 6), 1 / 6), [target, target])
    losses = model.update(data)
    np.testing.assert_allclose(losses.value, expected, rtol=1e-6, atol=1e-7)


def test_weight_only_l2_and_mean_batch_reduction(tmp_path):
    model = make_model(tmp_path, weight_decay=0.0001)
    params = model._state.params
    expected_l2 = 0.0001 * sum(
        float(jnp.sum(layer["kernel"] ** 2)) for layer in params.values()
    )
    obs = np.full((1, 15), 4 / 48, dtype=np.float32)
    mask = np.array([[1, 1, 1, 0, 0, 0]])
    policy = np.array([[0.7, 0.2, 0.1, 0, 0, 0]])
    single = batch(obs, mask, policy, [1.0])
    repeated = batch(np.repeat(obs, 3, axis=0), np.repeat(mask, 3, axis=0),
                     np.repeat(policy, 3, axis=0), [1.0] * 3)
    one = model.update(single)
    model_again = make_model(tmp_path, weight_decay=0.0001)
    three = model_again.update(repeated)
    for key in ("policy", "value", "l2"):
        np.testing.assert_allclose(getattr(one, key), getattr(three, key), rtol=1e-5)
    np.testing.assert_allclose(one.l2, expected_l2, rtol=1e-5)
    np.testing.assert_allclose(one.total, one.policy + one.value + one.l2)


def test_data_gradients_reach_both_heads_and_shared_trunk(tmp_path):
    model = make_model(tmp_path)
    data = batch(
        [[4 / 48] * 12 + [0, 0, 1],
         [3 / 48] * 6 + [5 / 48] * 6 + [0, 0, 0.999]],
        [[1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]],
        [[0.7, 0.2, 0.1, 0, 0, 0], [0, 0.2, 0, 0.3, 0, 0.5]],
        [1.0, -1.0],
    )

    def loss_fn(params):
        logits, values = jax.vmap(
            lambda obs: model._state.apply_fn({"params": params}, obs, training=False)
        )(data.observation)
        masked = jnp.where(data.legals_mask, logits, jnp.finfo(jnp.float32).min)
        policy_loss = optax.softmax_cross_entropy(masked, data.policy).mean()
        value_loss = jnp.square(values - data.value).mean()
        return policy_loss + value_loss

    total, grads = jax.value_and_grad(loss_fn)(model._state.params)
    assert np.isfinite(total)
    for name in ("trunk_1", "trunk_2", "trunk_3", "policy_hidden",
                 "policy_logits", "value_hidden", "value_output"):
        assert np.all(np.isfinite(grads[name]["kernel"]))
        assert float(jnp.linalg.norm(grads[name]["kernel"])) > 0
    losses = model.update(data)
    assert all(np.isfinite(getattr(losses, name)) for name in ("policy", "value", "l2", "total"))


def test_tiny_fixed_batch_overfits(tmp_path):
    model = make_model(tmp_path)
    observations = np.array([
        [4 / 48] * 12 + [0, 0, 1],
        [3 / 48] * 6 + [5 / 48] * 6 + [0, 0, 0.999],
    ], dtype=np.float32)
    masks = [[1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]]
    targets = [[0.7, 0.2, 0.1, 0, 0, 0], [0, 0.2, 0, 0.3, 0, 0.5]]
    values = [1.0, -1.0]
    data = batch(observations, masks, targets, values)

    def errors():
        predictions = [model.inference(obs, mask) for obs, mask in zip(observations, masks)]
        policy_ce = np.mean([
            -sum(t * np.log(float(p[a])) for a, t in enumerate(target) if t)
            for (_, p), target in zip(predictions, targets)
        ])
        value_mse = np.mean([(float(v) - z) ** 2 for (v, _), z in zip(predictions, values)])
        return policy_ce, value_mse

    before = errors()
    for _ in range(80):
        model.update(data)
    after = errors()
    assert after[0] < before[0], (before, after)
    assert after[1] < before[1], (before, after)
