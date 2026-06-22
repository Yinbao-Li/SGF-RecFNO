# Heat conduction dataset (train / val / test)

Steady 2D heat conduction: **6000** samples, **200×200** grid, stored as HDF5.

| File | Global indices | Samples | Split |
|------|----------------|---------|-------|
| `train.h5` | 0 – 3999 | 4000 | Training |
| `val.h5` | 4000 – 4999 | 1000 | Validation |
| `test.h5` | 5000 – 5999 | 1000 | Test |

Field key: `sol` or `u` (shape `(N, 1, 200, 200)` per file).

Split metadata: `splits.json`

## Download (Git LFS)

```bash
git lfs install
git clone https://github.com/Yinbao-Li/SGF-RecFNO.git
cd SGF-RecFNO
git lfs pull
```

Verify (each file should be **hundreds of MB**, not ~130 bytes):

```bash
ls -lh data/heat/*.h5
```

## Maintainer: create and upload splits

```bash
# from full temperature6000.h5
bash scripts/push_data_lfs.sh /path/to/temperature6000.h5

git commit -m "Add train/val/test heat dataset (Git LFS)"
git lfs push origin main --all
git push origin main
```

Or generate from scratch:

```bash
python scripts/generate_temperature6000.py
bash scripts/push_data_lfs.sh
```

## Alternative: monolithic file

If `temperature6000.h5` exists under `RECFNO_DATA_ROOT`, loaders still work without split files.
