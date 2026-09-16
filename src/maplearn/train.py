import json
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Grid, batch, generate
from .model import final_logits, make_model


@torch.no_grad()
def evaluate(model, rows, grid, batch_size, device):
    model.eval()
    loss, correct = 0.0, 0
    for start in range(0, len(rows), batch_size):
        tokens, pos, target = batch(rows[start : start + batch_size], grid, device)
        logits = final_logits(model, tokens, pos)
        loss += F.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(-1) == target).sum().item()
    return {"loss": loss / len(rows), "accuracy": correct / len(rows)}


def train(args):
    if (
        min(
            args.routes,
            args.eval_routes,
            args.epochs,
            args.batch_size,
            args.checkpoints,
        )
        < 1
    ):
        raise ValueError("Counts must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    grid = Grid(args.size, args.topology, args.seed)
    rows = generate(grid, args.routes, "train", args.seed, max_moves=args.max_moves)
    val = generate(
        grid, args.eval_routes, "val", args.seed + 1, max_moves=args.max_moves
    )
    test = generate(
        grid, args.eval_routes, "test", args.seed + 2, max_moves=args.max_moves
    )
    model = make_model(grid, args.seed, args.d_model, args.max_moves, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(rows) / args.batch_size) * args.epochs
    schedule = {
        math.ceil(i * steps / min(steps, args.checkpoints))
        for i in range(1, min(steps, args.checkpoints) + 1)
    }
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config["grid"] = asdict(grid)
    config["labels"] = grid.labels
    source_root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    config["environment"] = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "platform": platform.platform(),
        "git_revision": revision, "git_dirty": dirty,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source_root.glob("*.py"))},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    torch.save({"model": model.state_dict(), "config": config, "step": 0},
               out / "step-000000.pt")
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    started = time.monotonic()
    with (out / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            order = torch.randperm(len(rows), generator=generator).tolist()
            for start in range(0, len(rows), args.batch_size):
                model.train()
                tokens, pos, target = batch(
                    [rows[i] for i in order[start : start + args.batch_size]],
                    grid,
                    args.device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(final_logits(model, tokens, pos), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step in schedule:
                    metrics = {
                        "step": step,
                        "elapsed_seconds": time.monotonic() - started,
                        "epoch": epoch + 1,
                        "train_batch_loss": loss.item(),
                        "val": evaluate(model, val, grid, args.batch_size, args.device),
                    }
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    print(json.dumps(metrics), flush=True)
                    torch.save(
                        {"model": model.state_dict(), "config": config, "step": step},
                        out / f"step-{step:06d}.pt",
                    )
    result = evaluate(model, test, grid, args.batch_size, args.device)
    (out / "test.json").write_text(json.dumps(result, indent=2))
    print("Held-out test:", result)
