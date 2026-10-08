"""Stage 2 architecture and inference contract for the canonical Ayo model."""

import numpy as np
import pytest

from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
from open_spiel.python.algorithms.alpha_zero import model_linen  # noqa: E402


def make_model(path):
    return model_linen.Model.build_model(
        "ayo_mlp", (15,), 6, 256, 3, 0.0001, 0.001, str(path)
    )


def test_architecture_shapes_and_parameter_count(tmp_path):
    model = make_model(tmp_path)
    assert model.num_trainable_variables == 185863
    params = model._state.params
    expected = {
        "trunk_1": ((15, 256), (256,)),
        "trunk_2": ((256, 256), (256,)),
        "trunk_3": ((256, 256), (256,)),
        "policy_hidden": ((256, 128), (128,)),
        "policy_logits": ((128, 6), (6,)),
        "value_hidden": ((256, 64), (64,)),
        "value_output": ((64, 1), (1,)),
    }
    assert set(params) == set(expected)
    for name, (kernel, bias) in expected.items():
        assert params[name]["kernel"].shape == kernel
        assert params[name]["bias"].shape == bias
    logits, values = model._state.apply_fn(
        {"params": params}, np.ones((15, 1, 1), dtype=np.float32), training=False
    )
    assert logits.shape == (6,)
    assert values.shape == ()  # Existing JAX API squeezes the final dimension.
    batch_logits, batch_values = model._state.apply_fn(
        {"params": params}, np.ones((4, 15, 1, 1), dtype=np.float32), training=False
    )
    assert batch_logits.shape == (4, 6)
    assert batch_values.shape == (4,)


@pytest.mark.parametrize("legal", [
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
])
def test_masked_policy_and_bounded_value(tmp_path, legal):
    model = make_model(tmp_path)
    observation = np.linspace(0, 1, 15, dtype=np.float32)
    value, policy = model.inference(observation, legal)
    policy = np.asarray(policy)
    assert policy.shape == (6,)
    assert np.all(np.isfinite(policy))
    assert np.all(policy >= 0)
    assert np.all(policy[np.logical_not(legal)] == 0)
    np.testing.assert_allclose(policy.sum(), 1, atol=1e-6)
    assert np.isfinite(value) and -1 <= value <= 1


def test_checkpoint_round_trip(tmp_path):
    model = make_model(tmp_path)
    observation = np.arange(15, dtype=np.float32) / 48
    legal = [1, 0, 1, 0, 1, 1]
    before = model.inference(observation, legal)
    model.save_checkpoint(7)
    restored = make_model(tmp_path)
    restored.load_checkpoint(7)
    after = restored.inference(observation, legal)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
