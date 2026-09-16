# Study validation

Executed locally on macOS Apple Silicon, Python 3.12, PyTorch 2.14.0, CPU. Exact packages are recorded in `requirements-tested.txt`.

## Code checks

`PYTHONPATH=src python -m pytest -q`: **9 passed**.

Coverage includes bounded/wrap movement, deterministic disjoint route generation, correct destinations, input-answer separation, causal padding invariance, a small overfit learning check, TransformerLens hooks, portal reversibility and shortcut distances, portal targets/splits, held-out ridge-probe recovery, orthonormal encoding spans, intervention identity and hook cleanup, confidence-interval edge cases, and explicit train/test leakage rejection.

`python -m compileall -q src scripts` and `git diff --check` passed.

## Executed study

- All six configurations completed eight epochs on 250,000 training routes each.
- All **246 snapshots** (initialization plus 40 trained snapshots for each model) were loaded and analyzed.
- Final analysis used **2,048 held-out test routes** and 2,048 validation-fit routes per model; exact complete-route overlap with training was zero.
- Both residual layers were probed and intervened on. Ablation uses five matched random controls; primary steering covers four directions at three strengths with five controls each.
- The parity follow-up ran on both wraparound seeds, including two-cell shifts, diagonal shifts, per-axis ablation checks, and the parity-augmented encoder.
- Portal evaluation includes the changed-destination subset and a training-fitted displacement baseline. That subset has 157 routes for seed 42 and 130 for seed 43.

`python scripts/validate_results.py` verified all six runs, all 246 checkpoint hashes, all published result-file hashes, sample counts, and confidence intervals. `scripts/package_checkpoints.py` independently checked each checkpoint against its analyzed hash before creating the six release archives.

## Presentation checks

The offline dashboard was rendered in a browser; chart rendering and changing the experiment selector were verified.

## Interpretation

[Findings](reports/FINDINGS.md) reports measured values, negative results, controls, and limitations. The parity follow-up was motivated by observed seed-42 behavior and then applied to seed 43; it is exploratory, not preregistered. Passing tests or high navigation accuracy does not establish a literal torus or a complete mechanistic circuit. Historical smoke validation is retained separately in [INITIAL_RECONSTRUCTION.md](reports/INITIAL_RECONSTRUCTION.md).
