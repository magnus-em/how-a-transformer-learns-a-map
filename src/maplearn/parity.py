"""Exploratory follow-up: does parity constrain coordinate-based steering?

Designed after observing weak one-cell steering in wrap seed 42. Report the
second seed separately as a replication; do not treat this as preregistered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analyze import collect, coordinates
from .data import generate
from .research import (DIRECTION_NAMES, encoding_basis, load_checkpoint,
                       predictions, wilson, ridge_fit, predict_probe, r2)


def score(mask, eligible):
    return wilson(int((mask & eligible).sum()), int(eligible.sum()))


def run(args):
    torch.set_num_threads(args.threads)
    saved, grid, model = load_checkpoint(args.checkpoint, "cpu")
    if grid.topology != "wrap":
        raise ValueError("This diagnostic is defined for ordinary wrap grids")
    config = saved["config"]
    fit = generate(grid, 2048, "val", grid.seed + 201, max_moves=config["max_moves"])
    held = generate(grid, 2048, "test", grid.seed + 202, max_moves=config["max_moves"])
    fit_cells = np.asarray([r["cell"] for r in fit])
    cells = np.asarray([r["cell"] for r in held])
    labels = np.asarray(grid.labels)
    inverse = np.argsort(labels)
    hook = "blocks.0.hook_resid_post"
    a = collect(model, fit, grid, hook, "cpu", 256)
    b = collect(model, held, grid, hook, "cpu", 256)
    features = coordinates(range(grid.size**2), grid)
    mean, encoder, _ = encoding_basis(features[fit_cells], a)
    baseline = predictions(model, held, grid, "cpu")
    eligible = baseline == labels[cells]
    parity = lambda c: (c % grid.size + c // grid.size) % 2
    result = {"seed": grid.seed, "hook": hook, "scope": "Exploratory follow-up to observed one-cell steering failure",
              "fit_seed": grid.seed + 201, "held_seed": grid.seed + 202,
              "baseline_correct": int(eligible.sum()), "routes": len(held), "axis_ablation": {}, "steering": []}
    for axis, indices in (("x", slice(0, 2)), ("y", slice(2, 4))):
        center, _, span = encoding_basis(features[fit_cells, indices], a)
        delta = -(b - center) @ span @ span.T
        predicted_cells = inverse[predictions(model, held, grid, "cpu", hook, delta)]
        result["axis_ablation"][axis] = {
            "x_correct_on_baseline_correct": score(predicted_cells % grid.size == cells % grid.size, eligible),
            "y_correct_on_baseline_correct": score(predicted_cells // grid.size == cells // grid.size, eligible),
            "parity_preserved_on_baseline_correct": score(parity(predicted_cells) == parity(cells), eligible)}
    for direction, name in enumerate(DIRECTION_NAMES):
        for distance in (1, 2):
            shifted = cells.copy()
            for _ in range(distance):
                shifted = np.asarray([grid.step(int(c), direction) for c in shifted])
            delta = (features[shifted] - features[cells]) @ encoder
            edited = predictions(model, held, grid, "cpu", hook, delta)
            predicted_cells = inverse[edited]
            entry = {"direction": name, "distance": distance, "strength": 1,
                     "target_on_baseline_correct": score(edited == labels[shifted], eligible),
                     "original_parity_preserved": score(parity(predicted_cells) == parity(cells), eligible),
                     "random_controls": []}
            for control in range(5):
                rng = np.random.default_rng(grid.seed + 700 + control)
                random_delta = rng.normal(size=delta.shape)
                random_delta *= np.linalg.norm(delta, axis=1, keepdims=True) / np.linalg.norm(random_delta, axis=1, keepdims=True)
                control_pred = predictions(model, held, grid, "cpu", hook, random_delta)
                entry["random_controls"].append(score(control_pred == labels[shifted], eligible))
            result["steering"].append(entry)
    # A second exploratory test discriminates missing checkerboard information
    # from separate x/y parity: diagonal shifts change both axis parities while
    # preserving checkerboard parity. Also add that explicit parity feature.
    result["diagonal_steering"] = []
    for horizontal, vertical, name in ((1, 0, "northeast"), (1, 2, "southeast"),
                                       (3, 2, "southwest"), (3, 0, "northwest")):
        shifted = np.asarray([grid.step(grid.step(int(c), horizontal), vertical) for c in cells])
        delta = (features[shifted] - features[cells]) @ encoder
        pred = predictions(model, held, grid, "cpu", hook, delta)
        result["diagonal_steering"].append({"direction": name,
            "target_on_baseline_correct": score(pred == labels[shifted], eligible)})
    all_cells = np.arange(grid.size**2)
    parity_features = (1 - 2 * parity(all_cells)).reshape(-1, 1)
    augmented = np.column_stack((features, parity_features))
    _, augmented_encoder, _ = encoding_basis(augmented[fit_cells], a)
    parity_probe = ridge_fit(a, parity_features[fit_cells])
    result["parity_probe_r2"] = r2(parity_features[cells], predict_probe(parity_probe, b))
    result["parity_augmented_steering"] = []
    for direction, name in enumerate(DIRECTION_NAMES):
        shifted = np.asarray([grid.step(int(c), direction) for c in cells])
        delta = (augmented[shifted] - augmented[cells]) @ augmented_encoder
        pred = predictions(model, held, grid, "cpu", hook, delta)
        entry = {"direction": name, "distance": 1, "strength": 1,
                 "target_on_baseline_correct": score(pred == labels[shifted], eligible),
                 "target_all_routes": score(pred == labels[shifted], np.ones(len(cells), dtype=bool)),
                 "random_controls": []}
        for control in range(5):
            rng = np.random.default_rng(grid.seed + 700 + control)
            random_delta = rng.normal(size=delta.shape)
            random_delta *= np.linalg.norm(delta, axis=1, keepdims=True) / np.linalg.norm(random_delta, axis=1, keepdims=True)
            control_pred = predictions(model, held, grid, "cpu", hook, random_delta)
            entry["random_controls"].append(score(control_pred == labels[shifted], eligible))
        result["parity_augmented_steering"].append(entry)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
