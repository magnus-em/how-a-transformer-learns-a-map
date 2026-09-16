"""Build committed tables, provenance, and an offline Plotly dashboard from raw runs."""
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

RUNS = [f"{world}-{seed}" for world in ("wrap", "bounded", "portal") for seed in (42, 43)]
HOOK = "blocks.0.hook_resid_post"


def load(path):
    return json.loads(Path(path).read_text())


def percent(x):
    return f"{100 * x:.2f}%"


def span(values):
    return f"{100 * min(values):.1f}–{100 * max(values):.1f}%"


def chart(fig, name):
    fig.update_layout(template="plotly_white", font={"family": "Arial, sans-serif", "size": 13},
                      margin={"l": 65, "r": 35, "b": 65, "t": 70}, height=450,
                      legend={"orientation": "h", "y": -0.23})
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=name,
                       config={"responsive": True, "displaylogo": False})


def main():
    report = Path("reports")
    results = report / "results"
    results.mkdir(exist_ok=True)
    data, parity, histories = {}, {}, {}
    manifest = {}
    for name in RUNS:
        target = results / name
        target.mkdir(exist_ok=True)
        for filename in ("config.json", "metrics.jsonl", "test.json"):
            shutil.copyfile(Path("runs") / name / filename, target / filename)
        shutil.copyfile(f"runs/{name}-final/final.json", target / "final.json")
        shutil.copyfile(f"runs/{name}-emergence/emergence.json", target / "emergence.json")
        data[name] = load(target / "final.json")
        histories[name] = load(target / "emergence.json")["history"]
        assert len(histories[name]) == 41 and histories[name][-1]["step"] == 7816
        assert data[name]["held_routes"] == 2048
        if name.startswith("wrap"):
            shutil.copyfile(f"runs/{name}-parity-augmented/parity.json", target / "parity.json")
            parity[name] = load(target / "parity.json")
        for path in sorted(target.iterdir()):
            manifest[str(path.relative_to(report))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (report / "results-sha256.json").write_text(json.dumps(manifest, indent=2))
    table = ["| World / seed | Navigation (95% CI) | Displacement lookup | Coordinate R² | Shuffled R², mean |",
             "| --- | ---: | ---: | ---: | ---: |"]
    ablations = []
    random = []
    nav = []
    one, two, corrected, orthogonal, diagonal = [], [], [], [], []
    for name in RUNS:
        r = data[name]
        m = r["navigation"]["overall"]
        p = r["layers"][HOOK]["probes"]
        table.append(f"| {name} | {percent(m['accuracy'])} ({percent(m['ci95'][0])}–{percent(m['ci95'][1])}) | "
                     f"{percent(r['baselines']['displacement_lookup']['overall']['accuracy'])} | "
                     f"{p['coordinate_r2']:.3f} | {np.mean(p['shuffled_cell_r2']):.3f} |")
        if name.startswith("wrap"):
            nav.append(m["accuracy"])
            a = r["layers"][HOOK]["interventions"]["ablation"]["all_coordinates"]
            ablations.append(a["spatial"]["accuracy"])
            random.extend(c["accuracy"] for c in a["random_controls"])
            for s in parity[name]["steering"]:
                (one if s["distance"] == 1 else two).append(s["target_on_baseline_correct"]["accuracy"])
            corrected.extend(s["target_on_baseline_correct"]["accuracy"] for s in parity[name]["parity_augmented_steering"])
            diagonal.extend(s["target_on_baseline_correct"]["accuracy"] for s in parity[name]["diagonal_steering"])
            orthogonal.append(parity[name]["axis_ablation"]["x"]["y_correct_on_baseline_correct"]["accuracy"])
            orthogonal.append(parity[name]["axis_ablation"]["y"]["x_correct_on_baseline_correct"]["accuracy"])
    emergence_rows = []
    for name in ("wrap-42", "wrap-43"):
        r = histories[name][1]
        p = r["layers"][HOOK]
        emergence_rows.append(f"- **{name}, step {r['step']}:** navigation {percent(r['navigation']['overall']['accuracy'])}, "
                              f"coordinate R² {p['coordinate_r2']:.3f}, shuffled-cell R² {np.mean(p['shuffled_cell_r2']):.3f}.")
    portal_rows = []
    for name in ("portal-42", "portal-43"):
        m = data[name]["portal_changed_destination"]["overall"]
        baseline = data[name]["portal_changed_baselines"]["displacement_lookup"]["overall"]["accuracy"]
        portal_rows.append(f"- **{name}:** {percent(m['accuracy'])} on {m['n']} test routes whose destination differs from the ordinary wrap world, versus {percent(baseline)} for the training-fitted displacement lookup.")
    document = f'''# What does a navigation transformer actually learn?

A small transformer can solve unfamiliar routes while representing destination coordinates, but coordinates alone do not explain all of its behavior. In two wraparound runs, a missing checkerboard-parity feature explains a striking steering failure: coordinate-only one-cell shifts work {span(one)}, while adding parity raises success to {span(corrected)}. This is a new controlled experiment in this repository, not a recovered historical result or a claim of first discovery in the literature.

## Setup and navigation

Six two-layer, four-head transformers (width 128), each trained on 250,000 unique routes for eight epochs. Worlds are bounded, wraparound, and wraparound with two rewired edges; each has two seeds. Forty trained checkpoints plus initialization per model give 246 snapshots. Reported final analysis uses 2,048 held-out routes per model. Seeds jointly change map labels and model initialization.

{chr(10).join(table)}

Coordinate probes use two dimensions for bounded worlds and four sine/cosine dimensions for wrap/portal worlds. Probe R² is not directly comparable across different target definitions. The shuffled control permutes cell-to-coordinate labels, not train/test route assignments. A displacement lookup is an intentionally strong algebra-aware baseline; ordinary-grid success does not by itself prove a learned general graph algorithm. Uniform guessing is 1.5625%, but the learned baselines are more informative. Complete-route lookup has zero train/test overlap and falls back to the training majority.

## When does spatial information appear?

{chr(10).join(emergence_rows)}

Both runs show decodable spatial information before high navigation accuracy. This observation is bounded by the checkpoint spacing; it does not establish an exact onset, a discrete phase transition, or that representation formation causes later learning. All 41 checkpoints and both layers are included in the raw results; the dashboard plots the first layer. Shuffled-cell decoding grows substantially in the final layer, illustrating why probing alone is insufficient.

## Does the model use those features?

In the first residual block, removing the four-dimensional fitted coordinate span reduces wraparound accuracy from {span(nav)} to {span(ablations)}. Five random controls per model, matched for rank and each example's perturbation norm, retain {span(random)}. Removing just x or just y features preserves the other coordinate in {span(orthogonal)} of baseline-correct routes. This is evidence of behaviorally relevant, partly separable spatial features; it is not a complete circuit identification or proof of a literal internal torus.

## The failed steering experiment led to a better explanation

The primary one-cell coordinate intervention mostly failed, despite strong probes and ablation effects. Rather than select a favorable strength, a follow-up tested parity-preserving shifts and then a parity-augmented encoder:

| Intervention | Target success across directions and two seeds |
| --- | ---: |
| One cell, coordinates only | {span(one)} |
| Two cells, coordinates only | {span(two)} |
| Diagonal, coordinates only | {span(diagonal)} |
| One cell, coordinates + checkerboard parity | {span(corrected)} |

These numbers condition on the original navigation being correct; denominators and Wilson intervals are stored per condition. The encoder is fitted on validation routes. Diagonal moves flip both individual axis parities while retaining checkerboard parity. Augmented steering includes five random-direction controls matched for perturbation norm. The pattern supports a missing checkerboard-parity component; it does not localize a unique parity circuit or show that no other features matter. Interventions use the true destination to calculate the desired shift, so they are oracle-assisted diagnostics, not an input-only navigation or control algorithm.

This follow-up was developed after inspecting wrap seed 42, then applied unchanged to seed 43. It is exploratory and replicated once, not preregistered. Both positive and negative outcomes are retained.

## What changes when routes pass through a shortcut?

{chr(10).join(portal_rows)}

The portal world swaps two east/west edges while retaining four neighbors and reversible moves. Unlike ordinary grids, paths with the same net displacement can lead to different destinations. Its displacement baseline and path-length results help distinguish this issue from complete-route memorization. Each world is trained separately; these experiments do not test instant adaptation after editing an already learned map.

Graph-distance MDS probes and centroid-distance correlations are included as exploratory measurements. They do not establish that the portal model organizes its states by graph distance, and should not be presented as a topological discovery. The old claim that a bounded map is warped near edges has not been established by this study.

## Reproduction, provenance, and limits

- [Protocol and caveats](PROTOCOL.md), [raw metrics](results/), and [offline interactive dashboard](dashboard.html).
- `python scripts/reproduce.py --workers 2` reruns all training and analysis, including the parity follow-up. `python scripts/publish_results.py` rebuilds this report and the dashboard.
- `results-sha256.json` records raw-result hashes. Every analyzed checkpoint has a SHA-256 in its history; checkpoint archives have been created and hash-verified, with public distribution pending. Saved snapshots are for inference/analysis, not exact optimizer-state resumption.
- CPU runs use the repository's pinned tested environment. Floating-point details can vary across platforms. Some initial runs record a dirty reconstruction revision; later configs also record source hashes. See the protocol for the development chronology.
- No identical complete train/test routes; prefixes and subpaths can overlap. No unseen-labeling generalization, longer-route extrapolation, language-model transfer, or multiple architectural replications is claimed. Two joint seeds are a limited replication. Wilson intervals concern route sampling, not training-seed uncertainty.
'''
    (report / "FINDINGS.md").write_text(document)

    # Checkpoint comparison with a run selector, preserving all observations.
    emergence = go.Figure()
    for i, name in enumerate(RUNS):
        h = histories[name]
        for label, color, values in (
            ("Navigation accuracy", "#205bce", [r["navigation"]["overall"]["accuracy"] for r in h]),
            ("Coordinate probe R²", "#d46720", [r["layers"][HOOK]["coordinate_r2"] for r in h]),
            ("Shuffled-cell probe R²", "#718096", [np.mean(r["layers"][HOOK]["shuffled_cell_r2"]) for r in h])):
            emergence.add_trace(go.Scatter(x=[r["step"] for r in h], y=values, name=label,
                                          line={"color": color}, mode="lines+markers", visible=i == 0))
    emergence.update_layout(title="Spatial information across all training checkpoints",
        xaxis_title="Optimizer step", yaxis_title="Accuracy / held-out R²",
        updatemenus=[{"buttons": [{"label": name, "method": "update", "args": [
            {"visible": [j // 3 == i for j in range(18)]}]} for i, name in enumerate(RUNS)],
            "x": 1, "xanchor": "right", "y": 1.22}])
    ablation = go.Figure()
    for name in ("wrap-42", "wrap-43"):
        r = data[name]
        a = r["layers"][HOOK]["interventions"]["ablation"]
        ablation.add_trace(go.Bar(name=name, x=["Unmodified", "All spatial features removed", "Matched random perturbation", "x features removed", "y features removed"],
            y=[r["navigation"]["overall"]["accuracy"], a["all_coordinates"]["spatial"]["accuracy"],
               np.mean([c["accuracy"] for c in a["all_coordinates"]["random_controls"]]),
               a["x_only"]["spatial"]["accuracy"], a["y_only"]["spatial"]["accuracy"]]))
    ablation.update_layout(title="Spatial ablation damages navigation much more than random controls", barmode="group", yaxis={"tickformat": ".0%", "range": [0, 1.05]})
    steering = go.Figure()
    labels = [f"{name} · {d}" for name in ("wrap-42", "wrap-43") for d in ("north", "east", "south", "west")]
    for distance, title, color in ((1, "Coordinates: one cell", "#b4bbc7"), (2, "Coordinates: two cells", "#205bce")):
        values = [r["target_on_baseline_correct"]["accuracy"] for name in ("wrap-42", "wrap-43") for r in parity[name]["steering"] if r["distance"] == distance]
        steering.add_trace(go.Bar(x=labels, y=values, name=title, marker_color=color))
    values = [r["target_on_baseline_correct"]["accuracy"] for name in ("wrap-42", "wrap-43") for r in parity[name]["parity_augmented_steering"]]
    steering.add_trace(go.Bar(x=labels, y=values, name="Coordinates + parity: one cell", marker_color="#d46720"))
    steering.update_layout(title="Adding parity repairs the one-cell steering failure", barmode="group", yaxis={"tickformat": ".0%", "range": [0, 1.05]})
    lengths = go.Figure()
    colors = {"wrap": "#205bce", "bounded": "#238c68", "portal": "#d46720"}
    for name in RUNS:
        values = data[name]["navigation"]["by_length"]
        lengths.add_trace(go.Scatter(x=[int(k) for k in values], y=[v["accuracy"] for v in values.values()], name=name,
            mode="lines+markers", line={"color": colors[name.split('-')[0]], "dash": "dash" if name.endswith('43') else "solid"}))
    lengths.update_layout(title="Navigation by path length", xaxis_title="Number of moves", yaxis={"title": "Held-out accuracy", "tickformat": ".0%"})
    cards = f'<div class="cards"><div><b>6</b><span>trained models</span></div><div><b>1.5M</b><span>training routes across runs</span></div><div><b>246</b><span>saved snapshots</span></div><div><b>{html.escape(span(corrected))}</b><span>parity-augmented steering</span></div></div>'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>How a Transformer Learns a Map — Findings</title>
<style>body{{margin:0;background:#f3f5f8;color:#18253d;font:16px/1.6 Arial,sans-serif}}main{{max-width:1150px;margin:45px auto;padding:0 22px}}h1{{font-size:38px;line-height:1.2;letter-spacing:-1px}}h2{{font-size:24px;margin-bottom:8px}}.eyebrow{{text-transform:uppercase;letter-spacing:2px;color:#205bce;font-size:12px;font-weight:bold}}.lead{{font-size:19px;max-width:900px}}section{{background:white;border:1px solid #dde3ec;border-radius:12px;padding:22px;margin:25px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}.cards>div{{background:#18253d;color:white;padding:20px;border-radius:10px}}.cards b{{display:block;font-size:28px}}.cards span{{font-size:13px;color:#d2daeb}}.note{{color:#516079;font-size:14px}}a{{color:#205bce}}@media(max-width:750px){{.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:30px}}section{{padding:10px}}}}</style>
<script>{get_plotlyjs()}</script></head><body><main><div class="eyebrow">An experiment in mechanistic interpretability</div><h1>What does a navigation transformer actually learn?</h1><p class="lead">It learns useful spatial features. But making it move one cell exposes something the coordinates leave out: checkerboard parity.</p>{cards}
<section><h2>1. Watch spatial information emerge</h2><p>The first block's coordinate probe separates from shuffled labels before navigation reaches high accuracy. Select a world and seed; drag to zoom into early checkpoints.</p>{chart(emergence,'emergence')}<p class="note">Probes are fitted on validation routes and evaluated on held-out routes. Coordinate targets impose a geometric hypothesis; this is not an unsupervised torus visualization.</p></section>
<section><h2>2. Remove features and test their effect</h2><p>Ablating the fitted coordinate span reduces accuracy to {html.escape(span(ablations))}, versus {html.escape(span(random))} under matched random perturbations.</p>{chart(ablation,'ablation')}<p class="note">Random bars are means of five controls per model. Controls match subspace rank and each route's perturbation norm. Full counts and intervals are in the raw results.</p></section>
<section><h2>3. Follow the failed experiment</h2><p>One-cell coordinate steering mostly fails. Two-cell and diagonal shifts preserve checkerboard parity and work much better. Adding that parity feature repairs one-cell steering.</p>{chart(steering,'steering')}<p class="note">Success is conditional on initially correct navigation. These diagnostic interventions use the true destination. This follow-up was developed after observing seed 42 and replicated on seed 43; it is exploratory.</p></section>
<section><h2>4. Change the world</h2><p>Two rewired edges make route order matter: net displacement alone no longer determines the destination. Separately trained portal models test whether the architecture can handle this change.</p>{chart(lengths,'lengths')}<p class="note">Paths have 2–12 moves. This is a comparison of separately trained worlds, not immediate adaptation to an edited map. Two joint map/model seeds are a limited replication.</p></section>
<section><h2>Reproduce and inspect</h2><p><a href="https://github.com/magnus-em/how-a-transformer-learns-a-map/blob/main/reports/FINDINGS.md">Full findings</a> · <a href="https://github.com/magnus-em/how-a-transformer-learns-a-map/blob/main/reports/PROTOCOL.md">Methods and limitations</a> · <a href="https://github.com/magnus-em/how-a-transformer-learns-a-map/tree/main/reports/results">Raw metrics</a></p><p class="note">All charts are generated from saved measurements. This dashboard works offline. It establishes neither a complete circuit nor a literal internal torus, and makes no claim of priority over prior research.</p></section></main></body></html>'''
    (report / "dashboard.html").write_text(page)
    print(report / "FINDINGS.md")
    print(report / "dashboard.html")


if __name__ == "__main__":
    main()
