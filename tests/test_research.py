import numpy as np
import pytest
import torch

from maplearn.data import Grid, generate
from maplearn.model import make_model
from maplearn.research import (
    baselines, encoding_basis, graph_embedding, predictions,
    ridge_fit, predict_probe, shortest_paths, wilson,
)


def test_portal_inverse_and_shortcut():
    ordinary, portal = Grid(), Grid(topology="portal")
    assert ordinary.step(0, 1) == 1
    assert portal.step(0, 1) == 37
    for cell in range(64):
        for direction in range(4):
            assert portal.step(portal.step(cell, direction), (direction + 2) % 4) == cell
    d = shortest_paths(portal)
    assert d[0, 37] == 1
    assert shortest_paths(ordinary)[0, 37] == 7
    np.testing.assert_allclose(d, d.T)
    assert np.isfinite(graph_embedding(portal)).all()


def test_portal_route_targets_and_disjointness():
    grid = Grid(topology="portal")
    train = generate(grid, 256, "train", 42)
    held = generate(grid, 256, "test", 42)
    assert not {r["key"] for r in train} & {r["key"] for r in held}
    for row in held:
        cell = row["key"][0]
        for direction in row["key"][1:]:
            cell = grid.step(cell, direction)
        assert row["target"] == grid.labels[cell]


def test_probe_held_out_and_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 8))
    w = rng.normal(size=(8, 2))
    y = x @ w + 3
    fit = ridge_fit(x[:75], y[:75], 1e-9)
    np.testing.assert_allclose(predict_probe(fit, x[75:]), y[75:], atol=1e-8)
    _, _, basis = encoding_basis(y, x)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-8)


def test_intervention_identity_and_hook_cleanup():
    torch.set_num_threads(2)
    grid = Grid()
    model = make_model(grid, d_model=32)
    model.eval()
    rows = generate(grid, 8, "test", 17)
    original = predictions(model, rows, grid, "cpu")
    edited = predictions(model, rows, grid, "cpu", "blocks.0.hook_resid_post", np.zeros((8, 32)))
    np.testing.assert_array_equal(original, edited)
    delta = np.random.default_rng(1).normal(size=(8, 32))
    predictions(model, rows, grid, "cpu", "blocks.0.hook_resid_post", delta)
    np.testing.assert_array_equal(original, predictions(model, rows, grid, "cpu"))


def test_intervals_and_baseline_leak_detection():
    assert wilson(0, 0)["accuracy"] is None
    assert wilson(0, 10)["ci95"][1] > 0
    assert wilson(10, 10)["ci95"][0] < 1
    grid = Grid()
    train = generate(grid, 10, "train", 42)
    with pytest.raises(AssertionError, match="leakage"):
        baselines(train, train, grid)
