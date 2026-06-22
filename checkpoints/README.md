# Pre-trained checkpoints (300 epochs, heat conduction)

Best validation checkpoints for the unified benchmark.

| Directory | Model | Test MAE (K) |
|-----------|-------|--------------|
| `benchmark_sgf-recfno_300/` | **SGF-RecFNO** (primary) | 0.00346 |
| `benchmark_isorecfno_300/` | IsoRecFNO | 0.00512 |
| `benchmark_recfno_300/` | RecFNO (Zhao et al.) | 0.00727 |
| `benchmark_pino_300/` | PINO | 0.00351 |
| `benchmark_geofno_300/` | Geo-FNO | 0.0373 |
| `benchmark_gino_300/` | GINO | 0.3658 |

## Download after clone

```bash
git lfs install
git clone https://github.com/Yinbao-Li/SGF-RecFNO.git
cd SGF-RecFNO
git lfs pull
```

## Verify checkpoints are real (not broken pointers)

Each `.pth` should be **megabytes**, not ~130 bytes:

```bash
ls -lh checkpoints/*/*.pth
file checkpoints/benchmark_sgf-recfno_300/*.pth
# Good: 175M, "Zip archive data" or PyTorch pickle
# Bad:  130 bytes, ASCII text "version https://git-lfs.github.com/spec/v1"
```

If files are ~130 bytes, LFS objects were not uploaded. Open an issue or pull latest after maintainer runs `git lfs push`.

## Maintainer: upload LFS binaries

```bash
sudo apt install git-lfs
bash scripts/push_checkpoints_lfs.sh
git lfs push origin main --all
git push origin main
```

New training runs save to `heat2D/logs/ckpt/` (gitignored).
