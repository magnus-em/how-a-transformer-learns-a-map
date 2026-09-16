"""Create release archives without adding model binaries to Git history."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('runs')
OUT = Path('release-assets')
RUNS = [f'{world}-{seed}' for world in ('wrap','bounded','portal') for seed in (42,43)]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for name in RUNS:
        run = ROOT / name
        checkpoints = sorted(run.glob('step-*.pt'))
        if not (run / 'test.json').exists() or len(checkpoints) != 41:
            raise ValueError(f'{name}: incomplete training')
        history = json.loads((Path('reports/results') / name / 'emergence.json').read_text())['history']
        for item in history:
            path = run / f"step-{item['step']:06d}.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['checkpoint_sha256']:
                raise ValueError(f'Checkpoint hash differs from analyzed file: {path}')
        target = OUT / f'{name}-checkpoints.tar.gz'
        with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
            for path in [run / 'config.json', run / 'metrics.jsonl', run / 'test.json', *checkpoints]:
                archive.add(path, arcname=str(Path(name) / path.name))
        manifest[target.name] = {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                 'bytes': target.stat().st_size, 'checkpoints': len(checkpoints)}
        print(target, flush=True)
    (OUT / 'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
