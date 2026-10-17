"""Aggregate the data-study runs and plot success rate vs data quantity / noise level.

Reads the final in-training eval (step 6000, 10 episodes) from each run's log:
  - scaling: logs/scale25.log, logs/scale50.log, train_act.log (baseline 100%)
  - noise:   train_act.log (0%), logs/noise005/010/020.log

Usage: .venv\\Scripts\\python.exe scripts\\plot_data_study.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

SCALING = [  # (label, log path, episode count)
    ("25%", BASE / "logs/scale25.log", 51),
    ("50%", BASE / "logs/scale50.log", 103),
    ("100%", BASE / "train_act.log", 206),
]
NOISE = [  # (label, log path, noise level)
    ("0%", BASE / "train_act.log", 0.0),
    ("5%", BASE / "logs/noise005.log", 0.05),
    ("10%", BASE / "logs/noise010.log", 0.10),
    ("20%", BASE / "logs/noise020.log", 0.20),
]


def final_metrics(log: Path) -> tuple[float, float]:
    txt = log.read_text(encoding="utf-8", errors="ignore")
    succ = re.findall(r"'pc_success': ([\d.]+)", txt)
    rew = re.findall(r"'avg_max_reward': ([\d.]+)", txt)
    if not succ:
        raise SystemExit(f"no eval metrics found in {log}")
    return float(succ[-1]) / 100.0, float(rew[-1])  # last = step-6000 overall


def main() -> None:
    out_dir = BASE / "outputs/data_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    s_succ, s_rew = zip(*(final_metrics(p) for _, p, _ in SCALING))
    n_succ, n_rew = zip(*(final_metrics(p) for _, p, _ in NOISE))

    print(f"{'run':<12}{'success':>10}{'avg_max_reward':>18}")
    for (label, _, _), s, r in zip(SCALING, s_succ, s_rew):
        print(f"scale {label:<6}{s:>9.0%}{r:>18.3f}")
    for (label, _, _), s, r in zip(NOISE, n_succ, n_rew):
        print(f"noise {label:<6}{s:>9.0%}{r:>18.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = [n for _, _, n in SCALING]
    ax1.plot(x, s_succ, "o-", color="tab:blue", label="success rate")
    ax1.plot(x, s_rew, "s--", color="tab:gray", label="avg max reward (coverage)")
    ax1.set_xlabel("training episodes (of 206)")
    ax1.set_ylabel("rate")
    ax1.set_title("Effect of demonstration quantity")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend()

    lv = [l for _, _, l in NOISE]
    ax2.plot(lv, n_succ, "o-", color="tab:red", label="success rate")
    ax2.plot(lv, n_rew, "s--", color="tab:gray", label="avg max reward (coverage)")
    ax2.set_xlabel("action noise level (fraction of per-dim action std)")
    ax2.set_title("Effect of demonstration quality (noise injection)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = out_dir / "data_study.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
