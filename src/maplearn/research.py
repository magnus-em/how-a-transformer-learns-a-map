"""Reproducible held-out tests of spatial decoding, interventions and emergence.

A good coordinate probe alone does not identify the model's algorithm. In
particular, the shuffled-cell control can be decoded from destination identity.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from .analyze import collect, coordinates
from .data import DIRECTIONS, Grid, batch, generate
from .model import make_model

HOOKS = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
DIRECTION_NAMES = ["north", "east", "south", "west"]


def wilson(successes, total):
    if not total:
        return {"n": 0, "successes": 0, "accuracy": None, "ci95": [None, None]}
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return {"n": total, "successes": int(successes), "accuracy": p,
            "ci95": [max(0, center - half), min(1, center + half)]}


def ridge_fit(x, y, ridge=1.0):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    weights = np.linalg.solve(xc.T @ xc + ridge * np.eye(x.shape[1]), xc.T @ yc)
    return mx, my, weights


def predict_probe(fit, x):
    mx, my, weights = fit
    return (x - mx) @ weights + my


def r2(y, predicted):
    denominator = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - predicted) ** 2).sum() / denominator) if denominator else None


def encoding_basis(features, activations, ridge=1.0):
    _, center, encoder = ridge_fit(features, activations, ridge)
    u, singular, _ = np.linalg.svd(encoder.T, full_matrices=False)
    return center, encoder, u[:, singular > 1e-7]


def shortest_paths(grid):
    distances = np.full((grid.size**2, grid.size**2), np.inf)
    for source in range(grid.size**2):
        distances[source, source] = 0
        queue = deque([source])
        while queue:
            cell = queue.popleft()
            for direction in range(4):
                try:
                    neighbor = grid.step(cell, direction)
                except ValueError:
                    continue
                if np.isinf(distances[source, neighbor]):
                    distances[source, neighbor] = distances[source, cell] + 1
                    queue.append(neighbor)
    if not np.isfinite(distances).all():
        raise ValueError("World must be connected")
    return distances


def graph_embedding(grid, dimensions=4):
    """Classical MDS of shortest-path distances; a target hypothesis, not a finding."""
    distances = shortest_paths(grid)
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(-0.5 * centering @ distances**2 @ centering)
    order = np.argsort(values)[::-1][:dimensions]
    embedding = vectors[:, order] * np.sqrt(np.maximum(values[order], 0))
    embedding /= np.maximum(embedding.std(0), 1e-12)
    return embedding


def load_checkpoint(path, device):
    # TorchVersion is a string subclass used by the first run's environment log.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    grid = Grid(**config["grid"])
    model = make_model(grid, config["seed"], config["d_model"], config["max_moves"], device)
    model.load_state_dict(saved["model"])
    model.eval()
    return saved, grid, model


@torch.no_grad()
def predictions(model, rows, grid, device, hook=None, edits=None, batch_size=256):
    result = []
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        tokens, positions, _ = batch(group, grid, device)
        indices = torch.arange(len(group), device=device)
        if edits is None:
            logits = model(tokens)
        else:
            delta = torch.as_tensor(edits[start:start + len(group)], dtype=torch.float32, device=device)
            def intervention(value, hook):
                value = value.clone()
                value[indices, positions] += delta
                return value
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook, intervention)])
        result.extend(logits[indices, positions].argmax(-1).cpu().tolist())
    return np.asarray(result)


def metrics_for_predictions(predicted, rows, grid):
    target = np.asarray([r["target"] for r in rows])
    correct = predicted == target
    by_length = {}
    for length in sorted({len(r["key"]) - 1 for r in rows}):
        mask = np.asarray([len(r["key"]) - 1 == length for r in rows])
        by_length[str(length)] = wilson(int(correct[mask].sum()), int(mask.sum()))
    return {"overall": wilson(int(correct.sum()), len(rows)), "by_length": by_length}


def probe_metrics(a, b, fit_cells, held_cells, grid, seed, controls=5):
    if set(fit_cells) != set(range(grid.size**2)):
        raise ValueError("Increase fit routes to cover every destination cell")
    targets = coordinates(range(grid.size**2), grid)
    result = {"coordinate_r2": r2(targets[held_cells], predict_probe(
        ridge_fit(a, targets[fit_cells]), b))}
    shuffled = []
    for index in range(controls):
        permutation = np.random.default_rng(seed + 900 + index).permutation(len(targets))
        labels = targets[permutation]
        shuffled.append(r2(labels[held_cells], predict_probe(ridge_fit(a, labels[fit_cells]), b)))
    result["shuffled_cell_r2"] = shuffled
    result["coordinate_selectivity"] = result["coordinate_r2"] - float(np.mean(shuffled))
    graph = graph_embedding(grid)
    result["graph_distance_mds_r2"] = r2(graph[held_cells], predict_probe(
        ridge_fit(a, graph[fit_cells]), b))
    # Destination centroids are averaged on fit routes only. Correlation is
    # descriptive: cell pairs are not independent samples for a significance test.
    means = np.asarray([a[fit_cells == c].mean(0) for c in range(grid.size**2)])
    activation_distances = np.linalg.norm(means[:, None] - means[None], axis=-1)
    triangle = np.triu_indices(grid.size**2, 1)
    for name, distances in (("actual_graph", shortest_paths(grid)),
                            ("ordinary_wrap", shortest_paths(Grid(grid.size, "wrap", grid.seed)))):
        xx, yy = activation_distances[triangle], distances[triangle]
        result[name + "_distance_correlation"] = float(np.corrcoef(xx, yy)[0, 1])
    return result


def emergence(args):
    torch.set_num_threads(args.threads)
    paths = sorted(Path(args.run).glob("step-*.pt"))
    if not paths:
        raise ValueError("No checkpoints found")
    if not (Path(args.run) / "test.json").exists():
        raise ValueError("Wait for training to finish before analyzing checkpoint history")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    saved, grid, _ = load_checkpoint(paths[0], args.device)
    config = saved["config"]
    fit = generate(grid, args.routes, "val", config["seed"] + 101, max_moves=config["max_moves"])
    held = generate(grid, args.routes, "test", config["seed"] + 102, max_moves=config["max_moves"])
    fit_cells = np.asarray([r["cell"] for r in fit])
    held_cells = np.asarray([r["cell"] for r in held])
    if set(fit_cells) != set(range(grid.size**2)):
        raise ValueError("Increase routes to cover every fit destination")
    history = []
    for path in paths:
        saved, grid, model = load_checkpoint(path, args.device)
        result = {"step": saved["step"], "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "navigation": metrics_for_predictions(predictions(model, held, grid, args.device), held, grid),
                  "layers": {}}
        for hook in HOOKS:
            a = collect(model, fit, grid, hook, args.device, 256)
            b = collect(model, held, grid, hook, args.device, 256)
            result["layers"][hook] = probe_metrics(a, b, fit_cells, held_cells, grid, config["seed"])
        history.append(result)
        print(json.dumps({"step": saved["step"], "accuracy": result["navigation"]["overall"]["accuracy"],
                          "probes": {k: v["coordinate_r2"] for k, v in result["layers"].items()}}), flush=True)
        (out / "emergence.json").write_text(json.dumps({"config": config, "probe_routes_per_split": args.routes,
                                                    "fit_seed": config["seed"] + 101,
                                                    "held_seed": config["seed"] + 102, "history": history}, indent=2, allow_nan=False))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Held-out navigation", "Linear decoding: geometry and shuffled-cell control"))
    steps = [r["step"] for r in history]
    fig.add_trace(go.Scatter(x=steps, y=[r["navigation"]["overall"]["accuracy"] for r in history], name="Navigation"), row=1, col=1)
    for hook in HOOKS:
        for field, label in (("coordinate_r2", "coordinates"), ("shuffled_cell_r2", "shuffled cells")):
            values = [r["layers"][hook][field] for r in history]
            if field == "shuffled_cell_r2":
                values = [float(np.mean(v)) for v in values]
            fig.add_trace(go.Scatter(x=steps, y=values, name=f"{hook}: {label}"), row=2, col=1)
    fig.update_layout(title=f"{grid.topology}, seed {grid.seed}: when does spatial information become decodable?", height=850, template="plotly_white")
    fig.update_xaxes(title_text="Optimizer step", row=2, col=1)
    fig.write_html(out / "emergence.html", include_plotlyjs="cdn")


def intervention_metrics(model, rows, grid, hook, activations, fit_activations,
                         fit_cells, device, seed, controls=5):
    cells = np.asarray([r["cell"] for r in rows])
    labels = np.asarray(grid.labels)
    targets = labels[cells]
    baseline = predictions(model, rows, grid, device)
    baseline_correct = baseline == targets
    coordinates_all = coordinates(range(grid.size**2), grid)
    center, encoder, basis = encoding_basis(coordinates_all[fit_cells], fit_activations)
    result = {"baseline": wilson(int(baseline_correct.sum()), len(rows)), "ablation": {}, "steering": []}
    feature_sets = {"all_coordinates": coordinates_all,
                    "x_only": coordinates_all[:, :2] if grid.topology != "bounded" else coordinates_all[:, :1],
                    "y_only": coordinates_all[:, 2:] if grid.topology != "bounded" else coordinates_all[:, 1:]}
    for feature_name, features in feature_sets.items():
        mean, _, span = encoding_basis(features[fit_cells], fit_activations)
        removed = (activations - mean) @ span @ span.T
        edited = predictions(model, rows, grid, device, hook, -removed)
        condition = {"rank": span.shape[1], "spatial": wilson(int((edited == targets).sum()), len(rows)), "random_controls": []}
        for index in range(controls):
            rng = np.random.default_rng(seed + 500 + index)
            random_span, _ = np.linalg.qr(rng.normal(size=span.shape))
            random_removed = (activations - mean) @ random_span @ random_span.T
            # Equal rank AND per-example perturbation norm. Match the actual
            # intervention size, not merely the number of removed dimensions.
            random_removed *= (np.linalg.norm(removed, axis=1, keepdims=True) /
                               np.maximum(np.linalg.norm(random_removed, axis=1, keepdims=True), 1e-12))
            control_pred = predictions(model, rows, grid, device, hook, -random_removed)
            condition["random_controls"].append(wilson(int((control_pred == targets).sum()), len(rows)))
        result["ablation"][feature_name] = condition
    # Oracle destination coordinates define each requested shift. These are
    # diagnostic interventions, not a navigation method or input-only steering.
    for direction, name in enumerate(DIRECTION_NAMES):
        shifted, eligible = [], []
        for cell in cells:
            try:
                shifted.append(grid.step(int(cell), direction)); eligible.append(True)
            except ValueError:
                shifted.append(int(cell)); eligible.append(False)
        shifted, eligible = np.asarray(shifted), np.asarray(eligible)
        shifted_labels = labels[shifted]
        delta = (coordinates_all[shifted] - coordinates_all[cells]) @ encoder
        for strength in (0.5, 1.0, 2.0):
            edit = strength * delta
            pred = predictions(model, rows, grid, device, hook, edit)
            inverse = np.argsort(labels)
            predicted_cells = inverse[pred]
            # For wrap/bounded, the orthogonal coordinate should stay fixed.
            # Portal directions can change both coordinates, so score relative
            # to the specified graph neighbor for all worlds.
            orthogonal = ((predicted_cells // grid.size == shifted // grid.size) if direction % 2
                          else (predicted_cells % grid.size == shifted % grid.size))
            correct_subset = eligible & baseline_correct
            entry = {"direction": name, "strength": strength,
                     "target": wilson(int(((pred == shifted_labels) & eligible).sum()), int(eligible.sum())),
                     "baseline_target": wilson(int(((baseline == shifted_labels) & eligible).sum()), int(eligible.sum())),
                     "target_on_baseline_correct": wilson(int(((pred == shifted_labels) & correct_subset).sum()), int(correct_subset.sum())),
                     "orthogonal_coordinate_preserved": wilson(int((orthogonal & correct_subset).sum()), int(correct_subset.sum())),
                     "original_target": wilson(int(((pred == targets) & eligible).sum()), int(eligible.sum())),
                     "random_controls": []}
            for index in range(controls):
                rng = np.random.default_rng(seed + 700 + index)
                random_delta = rng.normal(size=edit.shape)
                random_delta *= np.linalg.norm(edit, axis=1, keepdims=True) / np.linalg.norm(random_delta, axis=1, keepdims=True)
                control_pred = predictions(model, rows, grid, device, hook, random_delta)
                entry["random_controls"].append(wilson(int(((control_pred == shifted_labels) & eligible).sum()), int(eligible.sum())))
            result["steering"].append(entry)
    return result


def displacement_key(row):
    dx = sum(DIRECTIONS[d][0] for d in row["key"][1:])
    dy = sum(DIRECTIONS[d][1] for d in row["key"][1:])
    return row["key"][0], dx, dy


def baselines(train_rows, held, grid):
    majority = Counter(r["target"] for r in train_rows).most_common(1)[0][0]
    exact = {r["key"]: r["target"] for r in train_rows}
    pooled = {}
    for row in train_rows:
        pooled.setdefault(displacement_key(row), Counter())[row["target"]] += 1
    pooled = {k: v.most_common(1)[0][0] for k, v in pooled.items()}
    collisions = sum(r["key"] in exact for r in held)
    if collisions:
        raise AssertionError("Train/test complete-route leakage")
    return {"exact_route_overlap": collisions,
            "majority": metrics_for_predictions(np.full(len(held), majority), held, grid),
            "return_start": metrics_for_predictions(np.asarray([grid.labels[r["key"][0]] for r in held]), held, grid),
            "exact_route_lookup_majority_fallback": metrics_for_predictions(np.asarray([exact.get(r["key"], majority) for r in held]), held, grid),
            "displacement_lookup": metrics_for_predictions(np.asarray([pooled.get(displacement_key(r), majority) for r in held]), held, grid),
            "displacement_lookup_coverage": sum(displacement_key(r) in pooled for r in held) / len(held)}


def final_analysis(args):
    torch.set_num_threads(args.threads)
    saved, grid, model = load_checkpoint(args.checkpoint, args.device)
    config = saved["config"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    fit = generate(grid, args.fit_routes, "val", config["seed"] + 201, max_moves=config["max_moves"])
    held = generate(grid, args.routes, "test", config["seed"] + 202, max_moves=config["max_moves"])
    train_rows = generate(grid, config["routes"], "train", config["seed"], max_moves=config["max_moves"])
    fit_cells = np.asarray([r["cell"] for r in fit])
    held_cells = np.asarray([r["cell"] for r in held])
    result = {"config": config, "checkpoint_step": saved["step"],
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "fit_routes": len(fit), "held_routes": len(held), "fit_seed": config["seed"] + 201,
              "held_seed": config["seed"] + 202, "random_controls": args.controls,
              "navigation": metrics_for_predictions(predictions(model, held, grid, args.device), held, grid),
              "baselines": baselines(train_rows, held, grid), "layers": {}}
    for hook in HOOKS:
        a = collect(model, fit, grid, hook, args.device, 256)
        b = collect(model, held, grid, hook, args.device, 256)
        result["layers"][hook] = {
            "probes": probe_metrics(a, b, fit_cells, held_cells, grid, config["seed"], args.controls),
            "interventions": intervention_metrics(model, held, grid, hook, b, a, fit_cells, args.device, config["seed"], args.controls)}
        (out / "final.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({"hook": hook, "probes": result["layers"][hook]["probes"]}), flush=True)
    # Common legal direction sequences with changed destinations distinguish
    # shortcut-aware navigation from treating all moves as ordinary grid steps.
    if grid.topology == "portal":
        ordinary = Grid(grid.size, "wrap", grid.seed)
        changed = []
        for row in held:
            cell = row["key"][0]
            for direction in row["key"][1:]:
                cell = ordinary.step(cell, direction)
            if cell != row["cell"]:
                changed.append(row)
        result["portal_changed_destination"] = metrics_for_predictions(predictions(model, changed, grid, args.device), changed, grid) if changed else None
    (out / "final.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"navigation": result["navigation"]["overall"], "output": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    history = commands.add_parser("emergence")
    history.add_argument("--run", required=True)
    history.add_argument("--routes", type=int, default=1024)
    final = commands.add_parser("final")
    final.add_argument("--checkpoint", required=True)
    final.add_argument("--routes", type=int, default=2048)
    final.add_argument("--fit-routes", type=int, default=2048)
    final.add_argument("--controls", type=int, default=5)
    for command in (history, final):
        command.add_argument("--out", required=True)
        command.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.routes < 256 or args.threads < 1:
        parser.error("Use at least 256 routes and positive threads")
    if args.command == "final" and (args.fit_routes < 256 or args.controls < 1):
        parser.error("Use at least 256 fit routes and at least one control")
    (emergence if args.command == "emergence" else final_analysis)(args)


if __name__ == "__main__":
    main()
