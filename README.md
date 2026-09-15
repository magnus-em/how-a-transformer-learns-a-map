# How a Transformer Learns a Map

Reconstruction of a lost research project: train a small transformer to follow routes on randomly labeled 8×8 grids, then inspect whether it develops reusable spatial representations.

**Status:** working reconstruction, not recovered original code. Original weights, exact hyperparameters, and historical findings are unavailable. The torus, boundary warping, and causal direction-feature findings are hypotheses to reproduce. A short smoke run verifies execution only.

## What is implemented

- Bounded and periodic (wraparound) grids with a fixed, seeded random permutation of cell labels.
- Unique routes with 2–12 legal moves; deterministic hash partitions prevent identical complete routes from crossing train/validation/test splits.
- A two-layer, four-head PyTorch transformer built with TransformerLens, trained to predict the final cell from the initial cell and direction sequence.
- Defaults of 250,000 **training** routes and up to 40 evenly spaced checkpoints per topology, plus 5,000 validation and 5,000 test routes. Counts are reconstruction choices; the résumé did not specify allocation.
- Validation-fitted coordinate probes, held-out probe R², residual PCA, spatial-subspace ablation with a matched-rank random control, and eastward activation steering.
- Standalone interactive Plotly HTML generated from actual checkpoint activations.

## Setup

Use Python 3.11 or 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

`requirements-tested.txt` records the exact local test environment. For a closer recreation, install it before `pip install -e '.[dev]'`; some wheels may differ by operating system.

## Quick end-to-end check

```sh
maplearn train --topology wrap --out runs/smoke \
  --routes 256 --eval-routes 64 --epochs 2 --batch-size 32 \
  --d-model 32 --checkpoints 2
maplearn analyze --checkpoint runs/smoke/step-000016.pt \
  --out runs/smoke-analysis --routes 128
```

Open `runs/smoke-analysis/map.html`. This tiny run is intentionally too short to establish any research result. Existing output directories are rejected to protect previous experiments.

## Full experiments

```sh
maplearn train --topology wrap --out runs/wrap --device cuda
maplearn train --topology bounded --out runs/bounded --device cuda
maplearn analyze --checkpoint runs/wrap/step-003908.pt \
  --out runs/wrap-analysis --device cuda
maplearn analyze --checkpoint runs/bounded/step-003908.pt \
  --out runs/bounded-analysis --device cuda
```

Use `--device cpu` for CPU or `--device mps` on compatible Apple Silicon machines. CUDA is suggested for full experiments, but no full-run runtime or convergence guarantee is established. The final step shown above follows the default counts and batch size; use the actual filename if changing them.

Each training directory contains the grid labeling and configuration, validation metrics at checkpoints, saved model states, and one final held-out test result. Checkpoints are inference/analysis snapshots, not exact optimizer-resume snapshots. Run files and weights are ignored by Git. Publish validated weights separately as release assets when ready.

## Interpretation and experimental limits

The task uses one fixed map per run. Randomly changing labels per route would make navigation ambiguous without supplying a map. Bounded routes sample only valid moves, with no wall-clamping convention. Directions are N, E, S, W with north decreasing y.

Only the final destination is supervised. The input contains no intermediate destination labels. Padding follows the last move; causal attention ensures that it cannot influence the queried position. This is a reconstruction decision, not a claim about the original tokenizer or objective.

Complete-route holdouts can share prefixes and subpaths with training. They test novel route sequences on a familiar labeling, not unseen-map transfer or length extrapolation. Route-memorization baselines, longer-path testing, and multiple seeds remain follow-up work.

Wraparound probes target `(cos x, sin x, cos y, sin y)`; bounded probes target scaled `(x, y)`. Probe targets encode a geometric hypothesis; a probe plot is **not** evidence that the network spontaneously forms a torus. The PCA panel is unsupervised. Compare layers, seeds, shuffled-label controls, generalization, and intervention outcomes before drawing conclusions.

Analysis fits probes and spatial encoding directions on validation routes, then scores disjoint test routes. Ablation removes the centered activation component in the fitted spatial span. Steering adds an encoded delta from the true destination to its eastern neighbor (excluding illegal bounded moves). This uses known coordinates as an experimental intervention, not as a deployable navigation algorithm. Compare shifted-target accuracy with `baseline_east`; compare ablation damage with the random control. A single intervention is not enough to establish causality.

## Next research milestones

1. Train both topologies to meaningful held-out navigation accuracy and repeat across seeds.
2. Add path-length and route-memorization baselines, shuffled-coordinate probes, and uncertainty estimates.
3. Track geometry across all 40 checkpoints; test all four directions and intervention strengths.
4. Publish reproducible result tables and trained weights only after evaluating these checks.

## Reference

Uses the documented [TransformerLens hook system](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.HookedRootModule.html) for caching residual states and applying temporary interventions. TransformerLens is held below version 3 to keep the reconstruction on the tested API family.
