"""Parse lerobot training console log and plot the loss curve.

Usage: .venv\\Scripts\\python.exe scripts\\plot_loss.py outputs/train/act_pusht/train.log outputs/train/act_pusht/loss_curve.png
"""
import re
import sys

import matplotlib.pyplot as plt


def main(log_path: str, out_path: str, log_freq: int = 100) -> None:
    losses = []
    success = []
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        m = re.search(r"loss:([\d.]+)", line)
        if m:
            losses.append(float(m.group(1)))
        if "overall aggregated" in line:
            m = re.search(r"'pc_success': ([\d.]+)", line)
            if m:
                success.append(float(m.group(1)) / 100.0)  # percentage -> rate
    if not losses:
        raise SystemExit(f"no loss lines found in {log_path}")
    steps = [(i + 1) * log_freq for i in range(len(losses))]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, losses, color="tab:blue", label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")
    if success:
        ax2 = ax1.twinx()
        ax2.plot(
            [(i + 1) * 3000 for i in range(len(success))],
            success,
            "o-",
            color="tab:green",
            label="eval success",
        )
        ax2.set_ylabel("success rate", color="tab:green")
        ax2.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path} ({len(losses)} loss points, {len(success)} eval points)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
