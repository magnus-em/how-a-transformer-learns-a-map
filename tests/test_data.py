import pytest

from maplearn.data import Grid, batch, generate


def test_topologies():
    assert Grid().step(7, 1) == 0
    assert Grid().step(0, 0) == 56
    with pytest.raises(ValueError):
        Grid(topology="bounded").step(7, 1)


@pytest.mark.parametrize("topology", ["wrap", "bounded"])
def test_routes_and_splits(topology):
    grid = Grid(topology=topology)
    splits = [generate(grid, 120, s, 7) for s in ("train", "val", "test")]
    keys = [{r["key"] for r in rows} for rows in splits]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert splits[0] == generate(grid, 120, "train", 7)
    for rows in splits:
        for row in rows:
            cell = row["key"][0]
            for move in row["key"][1:]:
                cell = grid.step(cell, move)
            assert grid.labels[cell] == row["target"]
            assert row["tokens"][0] == grid.labels[row["key"][0]]
            assert all(t >= 64 for t in row["tokens"][1:])
    _tokens, positions, targets = batch(splits[0], grid)
    assert positions.tolist() == [len(r["tokens"]) - 1 for r in splits[0]]
    assert targets.tolist() == [r["target"] for r in splits[0]]
