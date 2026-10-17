"""Create a noisy variant of a lerobot v3 dataset by injecting Gaussian noise into action labels.

Simulates imperfect teleoperation demonstrations: the recorded action at each frame is
perturbed by N(0, (level * std_dim)^2), where std_dim is the per-dimension std of the
original actions. Preprocessing stats (meta/stats.json) are kept identical on purpose:
the normalizer stays the same across variants so the only changed variable is the labels.

Usage:
    python scripts/make_noise_dataset.py datasets/lerobot/pusht datasets/local/pusht_noise005 0.05
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source dataset dir (contains data/ meta/ videos/)")
    parser.add_argument("dst", help="destination dataset dir")
    parser.add_argument("level", type=float, help="noise level as a fraction of per-dim action std, e.g. 0.05")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    data_file = src / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_file)
    action_type = table.schema.field("action").type

    actions = np.stack(table.column("action").to_pylist()).astype(np.float64)
    std = actions.std(axis=0)
    rng = np.random.default_rng(args.seed)
    noisy = actions + rng.normal(0.0, 1.0, actions.shape) * std * args.level
    noisy = np.clip(noisy, actions.min(axis=0), actions.max(axis=0))

    new_col = pa.array(noisy.astype(np.float32).tolist(), type=action_type)
    idx = table.column_names.index("action")
    new_table = table.set_column(idx, "action", new_col)
    pq.write_table(new_table, dst / "data/chunk-000/file-000.parquet")

    print(f"dataset written to {dst}")
    print(f"noise level {args.level:.2f} * std {std.round(2)} -> sigma {(std * args.level).round(2)} per dim")
    print(f"action std before: {actions.std(axis=0).round(2)}, after: {noisy.std(axis=0).round(2)}")


if __name__ == "__main__":
    main()
