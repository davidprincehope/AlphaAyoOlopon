"""Stage 5 Adam update, shared gradients, and state persistence checks."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
from open_spiel.python.algorithms.alpha_zero import model_linen, utils  # noqa: E402
from open_spiel.python.algorithms.alpha_zero import evaluator  # noqa: E402
from open_spiel.python.algorithms import mcts  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402,F401
import pyspiel  # noqa: E402


def make_model(path):
    return model_linen.Model.build_model(
        "ayo_mlp", (15,), 6, 256, 3, 0.0001, 0.001, str(path), seed=17
    )


def fixed_batch():
    return utils.TrainInput(
        observation=jnp.asarray([
            [4 / 48] * 12 + [0, 0, 1],
            [3 / 48] * 6 + [5 / 48] * 6 + [0, 0, 0.999],
        ], dtype=jnp.float32),
        legals_mask=jnp.asarray([
            [1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1],
        ], dtype=bool),
        policy=jnp.asarray([
            [0.7, 0.2, 0.1, 0, 0, 0], [0, 0.2, 0, 0.3, 0, 0.5],
        ], dtype=jnp.float32),
        value=jnp.asarray([1, -1], dtype=jnp.float32),
    )


def assert_trees_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_ayo_rejects_adamw_and_uses_adam_state(tmp_path):
    with pytest.raises(ValueError, match="AdamW"):
        Settings(decouple_weight_decay=True)
    with pytest.raises(ValueError, match="Adam"):
        model_linen.Model.build_model(
            "ayo_mlp", (15,), 6, 256, 3, 0.0001, 0.001,
            str(tmp_path), decouple_weight_decay=True,
        )
    model = make_model(tmp_path)
    adam = model._state.opt_state[0]
    assert adam.count == 0 and model._state.step == 0
    assert set(adam.mu) == set(model._state.params)
    assert set(adam.nu) == set(model._state.params)
    assert sum(p.size for p in jax.tree.leaves(adam.mu)) == 185863
    assert sum(p.size for p in jax.tree.leaves(adam.nu)) == 185863
    assert all(np.all(np.asarray(p) == 0) for p in jax.tree.leaves(adam.mu))
    assert all(np.all(np.asarray(p) == 0) for p in jax.tree.leaves(adam.nu))


def test_optimizer_has_no_weight_decay_and_default_adam_moments(tmp_path):
    model = make_model(tmp_path)
    params = model._state.params
    zero_grads = jax.tree.map(jnp.zeros_like, params)
    zero_updates, _ = model._state.tx.update(
        zero_grads, model._state.opt_state, params
    )
    assert all(np.all(np.asarray(update) == 0)
               for update in jax.tree.leaves(zero_updates))

    ones = jax.tree.map(jnp.ones_like, params)
    updates, next_state = model._state.tx.update(ones, model._state.opt_state, params)
    first_mu = next_state[0].mu["trunk_1"]["kernel"][0, 0]
    first_nu = next_state[0].nu["trunk_1"]["kernel"][0, 0]
    np.testing.assert_allclose(first_mu, 0.1, rtol=1e-6)
    np.testing.assert_allclose(first_nu, 0.001, rtol=1e-6)
    np.testing.assert_allclose(updates["trunk_1"]["kernel"][0, 0], -0.001,
                               rtol=1e-5)
    _, third_state = model._state.tx.update(ones, next_state, params)
    np.testing.assert_allclose(third_state[0].mu["trunk_1"]["kernel"][0, 0],
                               0.19, rtol=1e-6)
    np.testing.assert_allclose(third_state[0].nu["trunk_1"]["kernel"][0, 0],
                               0.001999, rtol=1e-6)


def test_policy_and_value_separately_reach_shared_trunk(tmp_path):
    model = make_model(tmp_path)
    data = fixed_batch()

    def component_loss(params, which):
        logits, values = jax.vmap(
            lambda obs: model._state.apply_fn({"params": params}, obs, training=False)
        )(data.observation)
        if which == "policy":
            masked = jnp.where(data.legals_mask, logits, jnp.finfo(jnp.float32).min)
            return optax.softmax_cross_entropy(masked, data.policy).mean()
        return jnp.square(values - data.value).mean()

    for which in ("policy", "value"):
        grads = jax.grad(component_loss)(model._state.params, which)
        for name in ("trunk_1", "trunk_2", "trunk_3"):
            kernel = np.asarray(grads[name]["kernel"])
            assert np.all(np.isfinite(kernel))
            assert np.linalg.norm(kernel) > 0, (which, name)


def test_adam_updates_every_layer_and_inference_is_read_only(tmp_path):
    model = make_model(tmp_path)
    data = fixed_batch()
    initial_params = model._state.params
    initial_state = model._state.opt_state
    model.inference(data.observation[0], data.legals_mask[0])
    assert model._state.step == 0
    assert_trees_equal(model._state.params, initial_params)
    assert_trees_equal(model._state.opt_state, initial_state)

    losses = model.update(data)
    assert all(np.isfinite(getattr(losses, name))
               for name in ("policy", "value", "l2", "total"))
    assert model._state.step == 1
    assert model._state.opt_state[0].count == 1
    for name in ("trunk_1", "trunk_2", "trunk_3", "policy_hidden",
                 "policy_logits", "value_hidden", "value_output"):
        old = np.asarray(initial_params[name]["kernel"])
        new = np.asarray(model._state.params[name]["kernel"])
        assert np.all(np.isfinite(new))
        assert np.any(old != new), name
        for moment in (model._state.opt_state[0].mu, model._state.opt_state[0].nu):
            assert moment[name]["kernel"].shape == old.shape
            assert np.all(np.isfinite(moment[name]["kernel"]))
            assert np.any(np.asarray(moment[name]["kernel"]) != 0)

    updated_params = model._state.params
    updated_state = model._state.opt_state
    model.inference(data.observation[1], data.legals_mask[1])
    game = pyspiel.load_game(ayo_game.GAME_NAME)
    bot = mcts.MCTSBot(
        game, 1.5, 3, evaluator.AlphaZeroEvaluator(game, model), solve=False,
        random_state=np.random.RandomState(4),
        child_selection_fn=mcts.SearchNode.puct_value,
    )
    bot.mcts_search(game.new_initial_state())
    assert model._state.step == 1
    assert_trees_equal(model._state.params, updated_params)
    assert_trees_equal(model._state.opt_state, updated_state)


def test_deterministic_update_and_optimizer_checkpoint(tmp_path):
    first = make_model(tmp_path)
    second = make_model(tmp_path)
    data = fixed_batch()
    assert_trees_equal(first._state.params, second._state.params)
    assert_trees_equal(first._state.opt_state, second._state.opt_state)
    first.update(data)
    second.update(data)
    assert first._state.step == second._state.step == 1
    assert_trees_equal(first._state.params, second._state.params)
    assert_trees_equal(first._state.opt_state, second._state.opt_state)

    first.save_checkpoint(3)
    restored = make_model(tmp_path)
    restored.load_checkpoint(3)
    assert restored._state.step == 1
    assert restored._state.opt_state[0].count == 1
    assert_trees_equal(restored._state.params, first._state.params)
    assert_trees_equal(restored._state.opt_state, first._state.opt_state)
    first.update(data)
    restored.update(data)
    assert first._state.step == restored._state.step == 2
    assert_trees_equal(restored._state.params, first._state.params)
    assert_trees_equal(restored._state.opt_state, first._state.opt_state)
