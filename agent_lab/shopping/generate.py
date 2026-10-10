"""Generate isolated task worlds and SFT trajectories without an external LLM."""
import argparse
import hashlib
import json
from pathlib import Path

from .tasks import CASES, make_task
from .trajectory import dumps, sft_row


def generate_dataset(output, train=1000, val=200, test=200, seed=42, overwrite=False):
    output = Path(output)
    sizes = {'train': train, 'val': val, 'test': test}
    if any(type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError('Each split must have at least one task')
    names = [f'{split}.{kind}.jsonl' for split in sizes for kind in ('tasks', 'sft')]
    if not overwrite and any((output / name).exists() for name in names + ['manifest.json']):
        raise FileExistsError('Dataset exists; choose another output directory or --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'seed': seed, 'counts': sizes, 'files': {}, 'case_counts': {}}
    all_ids = set()
    for split, count in sizes.items():
        tasks = [make_task(seed, split, i) for i in range(count)]
        for task in tasks:
            ids = {p['product_id'] for p in task['catalog']}
            if all_ids.intersection(ids):
                raise AssertionError('World overlap detected')
            all_ids.update(ids)
        manifest['case_counts'][split] = {case: sum(t['case'] == case for t in tasks) for case in CASES}
        for kind, rows in (('tasks', tasks), ('sft', [sft_row(t) for t in tasks])):
            name = f'{split}.{kind}.jsonl'
            data = ('\n'.join(dumps(row) for row in rows) + '\n').encode()
            (output / name).write_bytes(data)
            manifest['files'][name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('workspace/shopping/data'))
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--val', type=int, default=200)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    manifest = generate_dataset(**vars(args))
    print(json.dumps({'output': str(args.output), 'counts': manifest['counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
