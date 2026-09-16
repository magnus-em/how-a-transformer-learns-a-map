# Initial reconstruction validation

Local environment: macOS Apple Silicon, Python 3.12, CPU. Exact installed versions are recorded in `requirements-tested.txt`.

## Checks completed

- Four automated tests cover both grid topologies, deterministic disjoint route generation, correct targets, absence of intermediate answers in inputs, causal padding invariance, an overfit learning check, and TransformerLens activation access.
- Wrap smoke experiment: 256 training routes, 64 validation and 64 test routes, two epochs, width 32, batch size 32. Both requested checkpoints were saved; the final checkpoint was loaded for analysis.
- Bounded smoke experiment: 64 training routes, 32 validation and 32 test routes, one epoch, width 32. The checkpoint was loaded for bounded-coordinate probes and legal eastward steering.
- Both analysis pipelines produced JSON metrics and standalone Plotly HTML.

## Smoke results (not research evidence)

| Measurement | Wrap | Bounded |
| --- | ---: | ---: |
| Final held-out navigation accuracy | 1.5625% | 6.25% |
| Analysis route count | 128 | 64 |
| Coordinate probe R² | -0.3173 | -0.5601 |

Uniform chance accuracy for 64 cells is 1.5625%. Small test sets have substantial sampling variation. Negative probe R² and these navigation accuracies do not support learned spatial structure. These runs establish that the reconstruction executes, saves checkpoints, reloads them, and produces analysis artifacts. Full-scale training and replication remain outstanding.
