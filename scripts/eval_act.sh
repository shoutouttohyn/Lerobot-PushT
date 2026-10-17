#!/usr/bin/env bash
# Evaluate a trained checkpoint with 20 rollouts + auto-recorded videos.
# Usage: scripts/eval_act.sh <path-to-checkpoint-pretrained_model> [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN="$HOME/miniconda3/envs/lerobot/bin"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 SDL_VIDEODRIVER=dummy
"$PYBIN/lerobot-eval" \
  --policy.path="$1" \
  --env.type=pusht \
  --env.observation_height=96 \
  --env.observation_width=96 \
  --eval.n_episodes=20 \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --output_dir="${2:-outputs/eval/act_pusht_final}"
