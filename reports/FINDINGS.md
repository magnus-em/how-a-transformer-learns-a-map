# What does a navigation transformer actually learn?

A small transformer can solve unfamiliar routes while representing destination coordinates, but coordinates alone do not explain all of its behavior. In two wraparound runs, a missing checkerboard-parity feature explains a striking steering failure: coordinate-only one-cell shifts work 0.8–2.9%, while adding parity raises success to 97.5–99.3%. This is a new controlled experiment in this repository, not a recovered historical result or a claim of first discovery in the literature.

## Setup and navigation

Six two-layer, four-head transformers (width 128), each trained on 250,000 unique routes for eight epochs. Worlds are bounded, wraparound, and wraparound with two rewired edges; each has two seeds. Forty trained checkpoints plus initialization per model give 246 snapshots. Reported final analysis uses 2,048 held-out routes per model. Seeds jointly change map labels and model initialization.

| World / seed | Navigation (95% CI) | Displacement lookup | Coordinate R² | Shuffled R², mean |
| --- | ---: | ---: | ---: | ---: |
| wrap-42 | 99.51% (99.10%–99.73%) | 99.46% | 0.978 | 0.161 |
| wrap-43 | 99.32% (98.86%–99.59%) | 99.76% | 0.974 | 0.184 |
| bounded-42 | 100.00% (99.81%–100.00%) | 99.95% | 0.996 | 0.408 |
| bounded-43 | 100.00% (99.81%–100.00%) | 100.00% | 0.994 | 0.368 |
| portal-42 | 98.93% (98.38%–99.29%) | 93.12% | 0.709 | 0.111 |
| portal-43 | 99.12% (98.61%–99.44%) | 93.65% | 0.748 | 0.111 |

Coordinate probes use two dimensions for bounded worlds and four sine/cosine dimensions for wrap/portal worlds. Probe R² is not directly comparable across different target definitions. The shuffled control permutes cell-to-coordinate labels, not train/test route assignments. A displacement lookup is an intentionally strong algebra-aware baseline; ordinary-grid success does not by itself prove a learned general graph algorithm. Uniform guessing is 1.5625%, but the learned baselines are more informative. Complete-route lookup has zero train/test overlap and falls back to the training majority.

## When does spatial information appear?

- **wrap-42, step 196:** navigation 21.19%, coordinate R² 0.371, shuffled-cell R² -0.085.
- **wrap-43, step 196:** navigation 61.91%, coordinate R² 0.631, shuffled-cell R² -0.051.

Both runs show decodable spatial information before high navigation accuracy. This observation is bounded by the checkpoint spacing; it does not establish an exact onset, a discrete phase transition, or that representation formation causes later learning. All 41 checkpoints and both layers are included in the raw results; the dashboard plots the first layer. Shuffled-cell decoding grows substantially in the final layer, illustrating why probing alone is insufficient.

## Does the model use those features?

In the first residual block, removing the four-dimensional fitted coordinate span reduces wraparound accuracy from 99.3–99.5% to 6.5–13.0%. Five random controls per model, matched for rank and each example's perturbation norm, retain 95.3–98.8%. Removing just x or just y features preserves the other coordinate in 97.3–98.6% of baseline-correct routes. This is evidence of behaviorally relevant, partly separable spatial features; it is not a complete circuit identification or proof of a literal internal torus.

## The failed steering experiment led to a better explanation

The primary one-cell coordinate intervention mostly failed, despite strong probes and ablation effects. Rather than select a favorable strength, a follow-up tested parity-preserving shifts and then a parity-augmented encoder:

| Intervention | Target success across directions and two seeds |
| --- | ---: |
| One cell, coordinates only | 0.8–2.9% |
| Two cells, coordinates only | 92.5–98.8% |
| Diagonal, coordinates only | 96.1–98.9% |
| One cell, coordinates + checkerboard parity | 97.5–99.3% |

These numbers condition on the original navigation being correct; denominators and Wilson intervals are stored per condition. The encoder is fitted on validation routes. Diagonal moves flip both individual axis parities while retaining checkerboard parity. Augmented steering includes five random-direction controls matched for perturbation norm. The pattern supports a missing checkerboard-parity component; it does not localize a unique parity circuit or show that no other features matter. Interventions use the true destination to calculate the desired shift, so they are oracle-assisted diagnostics, not an input-only navigation or control algorithm.

This follow-up was developed after inspecting wrap seed 42, then applied unchanged to seed 43. It is exploratory and replicated once, not preregistered. Both positive and negative outcomes are retained.

## What changes when routes pass through a shortcut?

- **portal-42:** 92.99% on 157 test routes whose destination differs from the ordinary wrap world, versus 24.84% for the training-fitted displacement lookup.
- **portal-43:** 97.69% on 130 test routes whose destination differs from the ordinary wrap world, versus 16.15% for the training-fitted displacement lookup.

The portal world swaps two east/west edges while retaining four neighbors and reversible moves. Unlike ordinary grids, paths with the same net displacement can lead to different destinations. Its displacement baseline and path-length results help distinguish this issue from complete-route memorization. Each world is trained separately; these experiments do not test instant adaptation after editing an already learned map.

Graph-distance MDS probes and centroid-distance correlations are included as exploratory measurements. They do not establish that the portal model organizes its states by graph distance, and should not be presented as a topological discovery. The old claim that a bounded map is warped near edges has not been established by this study.

## Reproduction, provenance, and limits

- [Protocol and caveats](PROTOCOL.md), [raw metrics](results/), and [offline interactive dashboard](dashboard.html).
- `python scripts/reproduce.py --workers 2` reruns all training and analysis, including the parity follow-up. `python scripts/publish_results.py` rebuilds this report and the dashboard.
- `results-sha256.json` records raw-result hashes. Every analyzed checkpoint has a SHA-256 in its history; checkpoint archives are published with the GitHub release. Published snapshots are for inference/analysis, not exact optimizer-state resumption.
- CPU runs use the repository's pinned tested environment. Floating-point details can vary across platforms. Some initial runs record a dirty reconstruction revision; later configs also record source hashes. See the protocol for the development chronology.
- No identical complete train/test routes; prefixes and subpaths can overlap. No unseen-labeling generalization, longer-route extrapolation, language-model transfer, or multiple architectural replications is claimed. Two joint seeds are a limited replication. Wilson intervals concern route sampling, not training-seed uncertainty.
