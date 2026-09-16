# How a Transformer Learns a Map

**Can a transformer learn a reusable map—and can we make it get lost in a predictable way?**

This project trains small transformers to navigate randomly labeled grids, follows their internal representations across training, and tests those representations with controlled interventions. Six trained models cover bounded, wraparound, and shortcut-connected worlds across two seeds.

The most useful result came from an experiment that failed. A linear coordinate intervention barely moved predictions by one cell, although two-cell and diagonal moves worked. Adding a checkerboard-parity feature repaired one-cell steering, reaching **97.5–99.3% target success** across directions and both seeds on initially correct routes. These interventions use known destination coordinates: they are diagnostic tests, not a deployable navigation method.

[**Read the findings**](reports/FINDINGS.md) · [**Methods and limitations**](reports/PROTOCOL.md) · [**Raw metrics**](reports/results/) · [**Interactive dashboard file**](reports/dashboard.html)

Download `reports/dashboard.html` with GitHub's download button and open it in a browser; the interactive charts work offline. GitHub's source view does not run the dashboard.

## Three questions

1. **When does the map become decodable?** Track navigation, coordinate probes, and five shuffled-cell controls at 40 trained checkpoints plus initialization, in both transformer layers.
2. **Does the model use those features?** Remove x/y features and compare against rank- and norm-matched random controls. Test all four steering directions, strengths, two-cell shifts, diagonal shifts, and a parity-augmented encoder.
3. **What if the world has a shortcut?** Rewire two edges so route order matters, then compare a separately trained model against a displacement-lookup baseline, including the subset of routes with changed destinations.

## Study at a glance

- 8×8 cells with one fixed random label permutation per seed; inputs contain only the start cell and moves.
- Two layers, four attention heads, width 128, MLP width 512; final destination supervision.
- 250,000 unique training routes per model, 2–12 legal moves, eight epochs, AdamW at 0.001.
- Six models, 246 saved snapshots, 2,048 final held-out analysis routes per model.
- Deterministic complete-route partitions, validation-fitted probes, random controls, path-length breakdowns, and route-sampling confidence intervals.

The evidence supports behaviorally relevant coordinate and parity features. It does **not** establish a complete circuit, a literal internal torus, an edge-warping result, unseen-map transfer, or a general mechanism across architectures. The parity follow-up is exploratory and replicated once. See the protocol for all qualifications.

## Install and test

Use Python 3.11 or 3.12. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt
pip install -e '.[dev]'
pytest -q
```

The tested study ran on macOS Apple Silicon using CPU. Some package wheels differ across operating systems. `requirements-tested.txt` records the actual environment; `pyproject.toml` provides broader dependency ranges.

## Reproduce the full study

```sh
python scripts/reproduce.py --workers 2
python scripts/publish_results.py
python scripts/write_readme.py
```

The runner trains all six fixed configurations, evaluates all checkpoints, runs final probes/interventions/baselines, and performs the wraparound parity follow-up. `--workers 2` runs two training processes at a time; analysis runs sequentially. Use `--train-only` or `--analysis-only` to run one phase. Complete matching outputs are reused. Incomplete output directories raise an error; move them aside to rerun. Training snapshots do not contain optimizer states.

The full study uses local computation and takes substantially longer than the smoke check; runtime depends on hardware. The committed metrics are the actual measured run outputs, not example values.

## Analyze saved weights

All 246 checkpoints were generated and packaged into six hash-verified archives. Public archive distribution is pending; the source code and all measured results are available here. Train locally with the full-study command above, or, if you have a checkpoint archive, extract and analyze it as follows:

```sh
mkdir -p runs
tar -xzf wrap-42-checkpoints.tar.gz -C runs
python -m maplearn.research emergence --run runs/wrap-42 --out runs/wrap-42-emergence
python -m maplearn.research final --checkpoint runs/wrap-42/step-007816.pt --out runs/wrap-42-final
python -m maplearn.parity --checkpoint runs/wrap-42/step-007816.pt --out runs/wrap-42-parity-augmented
```

Each archive has all 41 snapshots, training configuration, validation history, and final training-script test result. Check archive hashes against `reports/checkpoint-manifest.json`; individual checkpoint hashes are in the committed emergence histories. Result-file hashes are in `reports/results-sha256.json`.

## Small smoke check

```sh
maplearn train --topology wrap --out runs/smoke-new --routes 256 --eval-routes 64 --epochs 2 --batch-size 32 --d-model 32 --checkpoints 2
maplearn analyze --checkpoint runs/smoke-new/step-000016.pt --out runs/smoke-new-analysis --routes 128
```

This short run checks execution only, not research conclusions. `maplearn analyze` is the original compact single-checkpoint visualization; `maplearn.research` contains the controlled study.

## Files

- `src/maplearn/data.py`: deterministic route generation and all three worlds.
- `src/maplearn/train.py`: training, checkpointing, and provenance.
- `src/maplearn/research.py`: checkpoint probes, interventions, baselines, and graph comparisons.
- `src/maplearn/parity.py`: the exploratory parity follow-up.
- `scripts/reproduce.py`: fixed full-study runner.
- `scripts/publish_results.py`: findings and offline dashboard, generated from raw results.
- `scripts/package_checkpoints.py`: hash-verified release archives.

## Project history

The repository began as a reconstruction of an earlier project; its initial smoke tests did not verify historical geometry or causality claims. This study produces new, narrower evidence. [Initial reconstruction validation](reports/INITIAL_RECONSTRUCTION.md) is retained for provenance, and [current validation](VALIDATION.md) records the expanded checks. No claim of novelty over prior literature is made.

Built with PyTorch, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), NumPy, and Plotly.
