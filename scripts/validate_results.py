"""Audit the published result bundle against its source checkpoints and counts."""
import hashlib
import json
import math
from pathlib import Path

RUNS = [f'{world}-{seed}' for world in ('wrap','bounded','portal') for seed in (42,43)]


def check_intervals(value):
    if isinstance(value, dict):
        if {'n','successes','accuracy','ci95'} <= value.keys():
            n, k = value['n'], value['successes']
            assert 0 <= k <= n
            if n:
                assert abs(value['accuracy'] - k/n) < 1e-12
                lo, hi = value['ci95']
                assert 0 <= lo <= k/n + 1e-12 <= hi + 1e-12 <= 1 + 1e-12
            else:
                assert value['accuracy'] is None
        for v in value.values():
            check_intervals(v)
    elif isinstance(value, list):
        for v in value:
            check_intervals(v)
    elif isinstance(value, float):
        assert math.isfinite(value)


def main():
    root = Path('reports/results')
    manifest = json.loads(Path('reports/results-sha256.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((Path('reports') / name).read_bytes()).hexdigest() == expected
    snapshots = 0
    for name in RUNS:
        folder = root / name
        config = json.loads((folder/'config.json').read_text())
        assert config['routes'] == 250000 and config['epochs'] == 8
        assert config['d_model'] == 128 and config['batch_size'] == 256
        assert config['max_moves'] == 12 and config['checkpoints'] == 40
        final = json.loads((folder/'final.json').read_text())
        history = json.loads((folder/'emergence.json').read_text())['history']
        assert len(history) == 41 and history[0]['step'] == 0 and history[-1]['step'] == 7816
        assert final['checkpoint_sha256'] == history[-1]['checkpoint_sha256']
        assert final['held_routes'] == final['fit_routes'] == 2048
        assert final['baselines']['exact_route_overlap'] == 0
        assert len(final['layers']) == 2
        for checkpoint in history:
            path = Path('runs')/name/f"step-{checkpoint['step']:06d}.pt"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['checkpoint_sha256']
            snapshots += 1
        for layer in final['layers'].values():
            interventions = layer['interventions']
            assert len(interventions['steering']) == 12
            assert len(layer['probes']['shuffled_cell_r2']) == 5
            for condition in interventions['ablation'].values():
                assert len(condition['random_controls']) == 5
        if name.startswith('portal'):
            assert final['portal_changed_destination']['overall']['n'] > 0
            assert final['portal_changed_baselines']['exact_route_overlap'] == 0
        if name.startswith('wrap'):
            parity = json.loads((folder/'parity.json').read_text())
            assert len(parity['steering']) == 8
            assert len(parity['parity_augmented_steering']) == 4
            assert parity['baseline_correct'] == final['navigation']['overall']['successes']
        for path in folder.glob('*.json'):
            check_intervals(json.loads(path.read_text()))
        print(f'{name}: verified', flush=True)
    assert snapshots == 246
    assert Path('reports/dashboard.html').stat().st_size > 1_000_000
    print(f'Validated {len(RUNS)} runs, {snapshots} checkpoint hashes, all result-file hashes, counts and confidence intervals.')


if __name__ == '__main__':
    main()
