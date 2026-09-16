"""Held-out geometry probes and interventions, without assuming a torus emerges."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from .data import Grid, batch, generate
from .model import make_model


def coordinates(cells, grid):
    cells = np.asarray(cells)
    x, y = cells % grid.size, cells // grid.size
    if grid.topology in ("wrap", "portal"):
        return np.column_stack(
            (
                np.cos(2 * np.pi * x / grid.size),
                np.sin(2 * np.pi * x / grid.size),
                np.cos(2 * np.pi * y / grid.size),
                np.sin(2 * np.pi * y / grid.size),
            )
        )
    return np.column_stack((x, y)) / (grid.size - 1) * 2 - 1


@torch.no_grad()
def collect(model, rows, grid, hook_name, device, batch_size=128):
    values = []
    for start in range(0, len(rows), batch_size):
        tokens, positions, _ = batch(rows[start : start + batch_size], grid, device)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        values.append(
            cache[hook_name][torch.arange(len(tokens), device=device), positions]
            .cpu()
            .numpy()
        )
    return np.concatenate(values)


def analyze(args):
    if args.routes < 3 or args.ridge <= 0:
        raise ValueError(
            "Analysis needs at least three routes and positive ridge regularization"
        )
    torch.set_num_threads(args.threads)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    grid = Grid(**config["grid"])
    model = make_model(
        grid, config["seed"], config["d_model"], config["max_moves"], args.device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    fit = generate(
        grid, args.routes, "val", config["seed"] + 101, max_moves=config["max_moves"]
    )
    held = generate(
        grid, args.routes, "test", config["seed"] + 102, max_moves=config["max_moves"]
    )
    hook_name = f"blocks.{args.layer}.hook_resid_pre"
    a = collect(model, fit, grid, hook_name, args.device)
    b = collect(model, held, grid, hook_name, args.device)
    y = coordinates([r["cell"] for r in fit], grid)
    target = coordinates([r["cell"] for r in held], grid)
    mean_a, mean_y = a.mean(0), y.mean(0)
    ac, yc = a - mean_a, y - mean_y
    ridge = args.ridge
    probe = np.linalg.solve(ac.T @ ac + ridge * np.eye(ac.shape[1]), ac.T @ yc)
    predicted = (b - mean_a) @ probe + mean_y
    r2 = 1 - ((target - predicted) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()
    # Fit an encoding map on validation routes; orthonormalize its spatial span.
    encoder = np.linalg.solve(yc.T @ yc + ridge * np.eye(yc.shape[1]), yc.T @ ac)
    basis, singular, _ = np.linalg.svd(encoder.T, full_matrices=False)
    basis = basis[:, singular > 1e-7]
    projection = torch.tensor(basis @ basis.T, dtype=torch.float32, device=args.device)
    center = torch.tensor(mean_a, dtype=torch.float32, device=args.device)
    # Matched-rank random ablation gives a basic control for generic damage.
    rng = np.random.default_rng(config["seed"])
    random_basis, _ = np.linalg.qr(rng.normal(size=basis.shape))
    control = torch.tensor(
        random_basis @ random_basis.T, dtype=torch.float32, device=args.device
    )
    counts = {
        "baseline": 0,
        "spatial_ablation": 0,
        "random_ablation": 0,
        "steered_east": 0,
        "baseline_east": 0,
    }
    east_valid = 0
    with torch.no_grad():
        for start in range(0, len(held), 128):
            rows = held[start : start + 128]
            tokens, positions, labels = batch(rows, grid, args.device)
            indices = torch.arange(len(rows), device=args.device)
            logits = model(tokens)[indices, positions]
            counts["baseline"] += (logits.argmax(-1) == labels).sum().item()
            for name, matrix in (
                ("spatial_ablation", projection),
                ("random_ablation", control),
            ):

                def ablate(
                    value, hook, indices=indices, positions=positions, matrix=matrix
                ):
                    value = value.clone()
                    value[indices, positions] -= (
                        value[indices, positions] - center
                    ) @ matrix
                    return value

                edited = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate)])[
                    indices, positions
                ]
                counts[name] += (edited.argmax(-1) == labels).sum().item()
            shifted, valid = [], []
            for row in rows:
                try:
                    shifted.append(grid.step(row["cell"], 1))
                    valid.append(True)
                except ValueError:
                    shifted.append(row["cell"])
                    valid.append(False)
            delta = (
                coordinates(shifted, grid)
                - coordinates([r["cell"] for r in rows], grid)
            ) @ encoder
            delta = torch.tensor(delta, dtype=torch.float32, device=args.device)

            def steer(value, hook, indices=indices, positions=positions, delta=delta):
                value = value.clone()
                value[indices, positions] += args.strength * delta
                return value

            edited = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steer)])[
                indices, positions
            ]
            shifted_labels = torch.tensor(
                [grid.labels[c] for c in shifted], device=args.device
            )
            mask = torch.tensor(valid, device=args.device)
            east_valid += mask.sum().item()
            counts["steered_east"] += (
                ((edited.argmax(-1) == shifted_labels) & mask).sum().item()
            )
            counts["baseline_east"] += (
                ((logits.argmax(-1) == shifted_labels) & mask).sum().item()
            )
    metrics = {
        "checkpoint_step": checkpoint["step"],
        "hook": hook_name,
        "probe_r2": float(r2),
        "routes": len(held),
        "spatial_rank": basis.shape[1],
        "east_eligible_routes": east_valid,
        "accuracy": {
            k: v / (east_valid if k.endswith("east") else len(held))
            if (east_valid or not k.endswith("east"))
            else None
            for k, v in counts.items()
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(json.dumps(metrics, indent=2))
    # PCA is unsupervised and fitted on validation activations only.
    _, _, vt = np.linalg.svd(ac, full_matrices=False)
    pcs = (b - mean_a) @ vt[:3].T
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        subplot_titles=(
            "Held-out residual states (PCA)",
            "Coordinate probe: dimensions 1 and 2",
        ),
    )
    cells = np.asarray([r["cell"] for r in held])
    marker = {"size": 3, "color": cells, "colorscale": "Viridis", "opacity": 0.65}
    fig.add_trace(
        go.Scatter3d(
            x=pcs[:, 0],
            y=pcs[:, 1],
            z=pcs[:, 2],
            mode="markers",
            marker=marker,
            text=[f"cell=({c % grid.size}, {c // grid.size})" for c in cells],
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=predicted[:, 0],
            y=predicted[:, 1],
            mode="markers",
            marker=marker,
            name="Probe",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"{grid.topology} grid · step {checkpoint['step']} · held-out probe R² {r2:.3f}",
        height=1000,
        margin={"l": 35, "r": 25, "t": 100, "b": 40},
        title_font_size=14,
        showlegend=False,
    )
    fig.write_html(out / "map.html", include_plotlyjs=True)
    print(json.dumps(metrics, indent=2))
