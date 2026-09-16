# Experiment protocol

## Questions

1. When do destination coordinates become linearly decodable relative to navigation accuracy?
2. Does removing or shifting those features change predictions more than a matched random perturbation?
3. Does the same architecture navigate a world where ordinary displacement is insufficient?

## Fixed study

Train six models: wraparound, bounded, and portal worlds, each at seeds 42 and 43. Each seed controls both the fixed random cell labeling and model initialization; this is replication across joint seeds, not an isolated initialization experiment. Each model has 2 layers, 4 heads, width 128, and an MLP width of 512. Train on 250,000 unique routes of 2–12 moves for 8 epochs with AdamW (learning rate 0.001, weight decay 0.01), batch size 256, gradient clipping at norm 1. Save 40 evenly spaced training checkpoints plus the untrained initialization. Use the final checkpoint, without selecting by test performance.

A deterministic hash of (start cell, complete move sequence) assigns routes to train/validation/test partitions. The generator checks route uniqueness. The 2,000 validation routes monitored during training and the final 2,000-route training-script test set are separate from the 250,000 training routes. Analysis generates additional samples from the same validation/test partitions using documented seeds. They may overlap the training script's validation/test samples, but cannot overlap training routes. Complete routes can share prefixes and subpaths. This is neither unseen-map transfer nor length extrapolation.

## Emergence

Use the same 1,024 validation and 1,024 test routes at all 41 checkpoints. At the final input position, extract residual states after each transformer block. Fit centered ridge probes (lambda = 1) on validation activations and score test R². Wrap and portal coordinate targets are cos/sin pairs for each axis; bounded targets are scaled x/y. Five fixed random permutations of the mapping from destination cells to coordinate targets control for arbitrary destination-label decoding. A probe's high R² does not establish an unsupervised torus. Sampled checkpoints bound the temporal resolution of any emergence statement.

## Final evaluation and causal interventions

Use 2,048 validation routes to fit probes/encoders and 2,048 held-out test routes for all final metrics, with fresh analysis seeds. Report navigation accuracy by path length and Wilson 95% intervals. Intervals describe sampled routes conditional on a trained model; two seeds do not establish population-level certainty.

Fit an encoder from the coordinate features to residual states. Remove the centered component in its full, x-only, or y-only span. Compare with five random subspaces of the same rank, scaling each route's perturbation to match the spatial perturbation norm. Thus controls match rank and per-route magnitude. Scaled random projections are perturbation controls, not literal orthogonal ablations after scaling.

For each of N/E/S/W, add the encoded difference between the true destination and its legal neighbor at strengths 0.5, 1 and 2. Compare with five random directions of the same per-route norm. Report target-neighbor accuracy, original-destination accuracy, success conditional on a correct baseline, and preservation of the orthogonal coordinate. Strength 1 is the primary comparison; other strengths are sensitivity checks, not test-set tuning. These interventions use true destination coordinates, so they are diagnostic, oracle-assisted experiments. Success establishes sensitivity to the fitted features, not a complete circuit or a deployable steering method.

## Changed world

The portal world starts from the same wraparound grid, swaps the east-going edges from cells 0 and 36 (and their reverse west edges), and therefore connects 0 to 37 and 36 to 1. All nodes retain four directions and reverse moves work. Labels stay fixed for a given seed. Models are trained separately on each world; this does not test immediate adaptation to a changed map.

Evaluate portal routes whose destinations differ from those in the ordinary wraparound world. Compare against a lookup baseline grouped by (start cell, net dx, net dy), fitted only on training routes. Net displacement is sufficient in ordinary grids but may be ambiguous with portals. Also report majority, return-to-start, and exact-route lookup baselines. Exact-route lookup falls back to the training majority because complete test sequences never occur in training.

Probe a four-dimensional classical-MDS embedding of graph shortest-path distances and report descriptive correlations between validation destination-centroid distances and actual-graph/ordinary-wrap distances. These are exploratory: graph-MDS and coordinate targets differ, pair distances are dependent, and dimensionality can affect probe comparisons. Do not infer a topological manifold or graph representation from them alone.

## Provenance

The repository began as a reconstruction with only smoke tests. Historical torus/causality claims were not verified. This study generates new evidence. The first wrap/bounded runs were launched while instrumentation was being developed; their configs record a dirty reconstruction revision. Subsequent runs record the committed source revision and source hashes. The training objective, model, split scheme, and fixed hyperparameters are shared. The initial checkpoint is additional to the 40 trained snapshots.

The first emergence inspection of wrap seed 42 was exploratory and covered an incomplete live checkpoint list; it is excluded from published study artifacts. The final analysis repeats all checkpoints. This protocol was written after inspecting that exploratory history, so it is an analysis plan, not a preregistration.

## Exploratory parity follow-up

After seeing weak one-cell coordinate steering in wrap seed 42, test two-cell shifts at strength 1, keeping the original baseline-correct denominator. Measure whether predictions retain checkerboard parity `(x + y) mod 2`, and whether x-only/y-only ablations preserve the other coordinate. Next, test diagonal shifts (which flip individual axis parities but preserve checkerboard parity), fit a fifth encoder feature `(-1)^(x+y)`, and repeat one-cell steering with that augmented encoder. Fit on the same 2,048 validation routes and evaluate on the same 2,048 held-out routes as the primary analysis. Compare augmented steering against five per-route norm-matched random perturbations and record a held-out parity-probe R². Apply the resulting fixed procedure to seed 43 as one replication.

This sequence of tests was motivated by observed test-set behavior and reuses that evaluation set, so its numerical results are exploratory rather than an unbiased estimate after independent hypothesis selection. Replication on the second independently trained joint seed adds evidence, but two seeds and a single architecture do not establish a universal mechanism. All interventions use known true coordinates/parity. A high linear parity-probe score does not by itself locate a unique parity circuit.
