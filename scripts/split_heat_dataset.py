#!/usr/bin/env python3
"""Split temperature6000.h5 into train / val / test for repository hosting."""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# Match benchmark/config.py
TRAIN = list(range(4000))
VAL = list(range(4000, 5000))
TEST = list(range(5000, 6000))

SPLITS = {
    'train': TRAIN,
    'val': VAL,
    'test': TEST,
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _field_key(h5_path: str) -> str:
    with h5py.File(h5_path, 'r') as f:
        if 'u' in f:
            return 'u'
        if 'sol' in f:
            return 'sol'
    raise KeyError(f"no 'u' or 'sol' in {h5_path}")


def split_dataset(src: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    key = _field_key(src)

    with h5py.File(src, 'r') as f_in:
        full = f_in[key]
        n, c, h, w = full.shape
        if n < 6000:
            raise ValueError(f'expected >=6000 samples, got {n}')

        manifest = {
            'source': os.path.basename(src),
            'field_key': key,
            'shape_per_sample': [c, h, w],
            'splits': {},
        }

        for name, indices in SPLITS.items():
            out_path = os.path.join(out_dir, f'{name}.h5')
            data = full[indices]
            with h5py.File(out_path, 'w') as f_out:
                dset = f_out.create_dataset(
                    key, data=data, dtype='float32',
                    compression='gzip', compression_opts=4,
                )
                dset.attrs['global_indices'] = np.array(indices, dtype=np.int32)
            manifest['splits'][name] = {
                'file': f'{name}.h5',
                'count': len(indices),
                'global_index_range': [indices[0], indices[-1]],
            }
            mb = os.path.getsize(out_path) / (1024 ** 2)
            print(f'Wrote {out_path}  ({len(indices)} samples, {mb:.1f} MB)', flush=True)

    manifest_path = os.path.join(out_dir, 'splits.json')
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)
    print(f'Wrote {manifest_path}', flush=True)


def main() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = argparse.ArgumentParser(description='Split heat H5 into train/val/test')
    parser.add_argument('--src', default=None, help='source temperature6000.h5')
    parser.add_argument('--out-dir', default=os.path.join(root, 'data', 'heat'))
    args = parser.parse_args()

    if args.src is None:
        from data.paths import HEAT_H5
        args.src = HEAT_H5

    if not os.path.isfile(args.src):
        print(f'Source not found: {args.src}', file=sys.stderr)
        print('Set --src or RECFNO_DATA_ROOT to your temperature6000.h5', file=sys.stderr)
        sys.exit(1)

    split_dataset(args.src, args.out_dir)


if __name__ == '__main__':
    main()
