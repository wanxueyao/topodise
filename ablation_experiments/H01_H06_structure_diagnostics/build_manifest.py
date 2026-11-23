"""Build the grouped STR-A probe split from the user's local CORN-1 images."""
import argparse
from collections import Counter
from pathlib import Path
import re

from topodise.training.runtime.protocol import file_hash, write


def build_manifest(root):
    samples = []
    seen = set()
    owners = {}
    counts = Counter()
    for split, folder in (('train', 'train'), ('validation', 'val'), ('test', 'test')):
        images = sorted((root / 'images' / f'{folder}_orig').glob('*.tif'))
        if not images:
            raise FileNotFoundError(f'No CORN-1 images for {split} in {root}')
        retained = []
        for image in images:
            mask = root / 'labels' / f'{folder}_label' / image.name
            if not mask.is_file():
                raise FileNotFoundError(mask)
            match = re.match(r'(batch_\d+)_', image.name)
            if not match:
                raise ValueError(f'Unrecognized acquisition group: {image.name}')
            digest = file_hash(image)
            if digest in seen:
                continue
            seen.add(digest)
            group = match.group(1)
            if owners.setdefault(group, split) != split:
                raise ValueError(f'Acquisition group spans multiple probe splits: {group}')
            retained.append(dict(id=f'corn1:{split}:{image.stem}', image=str(image.resolve()),
                                 mask=str(mask.resolve()), split=split, group_id=group))
        if not retained:
            raise ValueError(f'No unique images left in {split}')
        counts[split] = len(retained)
        samples.extend(sorted(retained, key=lambda row: (row['group_id'], row['id'])))
    return dict(schema_version=1, protocol_id='STR-A', samples=samples, retained_by_split=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Local CORN-1 directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, build_manifest(args.root))


if __name__ == '__main__':
    main()
