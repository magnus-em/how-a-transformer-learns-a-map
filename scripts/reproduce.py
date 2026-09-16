"""Run the fixed six-experiment study; preserve completed runs on restart.

Usage: python scripts/reproduce.py [--train-only] [--workers 2]
Run from the repository root after installing maplearn in a virtual environment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

RUNS = [(topology, seed) for seed in (42, 43) for topology in ("wrap", "bounded", "portal")]


def run(command, log_path):
    print(" ".join(command), flush=True)
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def train_one(spec):
    topology, seed = spec
    target = Path(f"runs/{topology}-{seed}")
    if (target / "test.json").exists():
        config = json.loads((target / "config.json").read_text())
        expected = {"routes": 250000, "eval_routes": 2000, "epochs": 8,
                    "d_model": 128, "checkpoints": 40, "seed": seed, "topology": topology}
        if any(config[k] != v for k, v in expected.items()) or len(list(target.glob("step-*.pt"))) != 41:
            raise ValueError(f"Completed run has unexpected configuration: {target}")
        print(f"Keeping completed {target}", flush=True)
        return
    if target.exists():
        raise ValueError(f"Incomplete run at {target}; move it aside before restarting")
    run([sys.executable, "-m", "maplearn.cli", "train", "--topology", topology,
         "--seed", str(seed), "--out", str(target), "--routes", "250000", "--eval-routes", "2000",
         "--epochs", "8", "--d-model", "128", "--checkpoints", "40", "--threads", "4"],
        Path(f"runs/{topology}-{seed}-train.log"))


def analyze_one(spec):
    topology, seed = spec
    name = f"{topology}-{seed}"
    run_dir = Path("runs") / name
    for command in ("emergence", "final"):
        target = Path("runs") / f"{name}-{command}"
        expected = target / ("emergence.html" if command == "emergence" else "final.json")
        if expected.exists():
            saved = json.loads((target / f"{command}.json").read_text())
            if command == "emergence" and len(saved["history"]) == 41:
                continue
            if command == "final" and len(saved["layers"]) == 2 and (topology != "portal" or "portal_changed_destination" in saved):
                continue
        if target.exists():
            raise ValueError(f"Incomplete analysis at {target}; move it aside before restarting")
        source_args = ["--run", str(run_dir)] if command == "emergence" else ["--checkpoint", str(run_dir / "step-007816.pt")]
        run([sys.executable, "-m", "maplearn.research", command, *source_args, "--out", str(target)],
            Path(f"runs/{name}-{command}.log"))
    if topology == "wrap":
        target = Path("runs") / f"{name}-parity-augmented"
        if (target / "parity.json").exists():
            saved = json.loads((target / "parity.json").read_text())
            if "parity_augmented_steering" in saved:
                return
        if target.exists():
            raise ValueError(f"Incomplete parity analysis at {target}; move it aside before restarting")
        run([sys.executable, "-m", "maplearn.parity", "--checkpoint", str(run_dir / "step-007816.pt"),
             "--out", str(target)], Path(f"runs/{name}-parity-augmented.log"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    if args.train_only and args.analysis_only:
        p.error("Choose only one of --train-only and --analysis-only")
    Path("runs").mkdir(exist_ok=True)
    if not args.analysis_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(train_one, RUNS))
    if not args.train_only:
        # Analysis can be memory intensive; run sequentially by default.
        for spec in RUNS:
            analyze_one(spec)


if __name__ == "__main__":
    main()
