#!/usr/bin/env bash
# Baseline ACT training on pusht (6000 steps, ~10-15 min on RTX 3070)
set -euo pipefail
cd "$(dirname "$0")/.."
PYBIN="$HOME/miniconda3/envs/lerobot/bin"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 SDL_VIDEODRIVER=dummy
"$PYBIN/lerobot-train" \
  --policy.type=act \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --policy.push_to_hub=false \
  --dataset.repo_id=pusht_clean \
  --dataset.root=datasets/pusht_clean \
  --env.type=pusht \
  --env.observation_height=96 \
  --env.observation_width=96 \
  --output_dir=outputs/train/act_pusht \
  --job_name=act_pusht_smoke \
  --steps=6000 \
  --batch_size=32 \
  --num_workers=4 \
  --save_freq=3000 \
  --env_eval_freq=3000 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --log_freq=100
