#!/usr/bin/env bash
# Data study: 5 sequential ACT training runs (6000 steps each, ~10-15 min each)
#   scale25 / scale50 : episode-count scaling on clean data
#   noise005/010/020  : action-label noise injection (5/10/20% of per-dim action std)
# Baseline (100% clean) = outputs/train/act_pusht (train_act.sh).
set -u
cd "$(dirname "$0")/.."
PYBIN="$HOME/miniconda3/envs/lerobot/bin"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 SDL_VIDEODRIVER=dummy
mkdir -p logs

EP25=$("$PYBIN/python" -c "print(','.join(map(str, range(51))))")
EP50=$("$PYBIN/python" -c "print(','.join(map(str, range(103))))")

COMMON=(--policy.type=act --policy.chunk_size=8 --policy.n_action_steps=8 --policy.push_to_hub=false
  --env.type=pusht --env.observation_height=96 --env.observation_width=96
  --steps=6000 --batch_size=32 --num_workers=4 --save_freq=3000
  --env_eval_freq=3000 --eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false --log_freq=100)

run() {
  name=$1; shift
  echo "=== RUN $name  $(date +%H:%M:%S) ==="
  if "$PYBIN/lerobot-train" "$@" > "logs/${name}.log" 2>&1; then echo "RUN $name OK"; else echo "RUN $name FAILED (see logs/${name}.log)"; fi
}

run scale25 "${COMMON[@]}" --dataset.repo_id=pusht_clean --dataset.root=datasets/pusht_clean \
  "--dataset.episodes=[${EP25}]" --output_dir=outputs/train/scale25 --job_name=scale25
run scale50 "${COMMON[@]}" --dataset.repo_id=pusht_clean --dataset.root=datasets/pusht_clean \
  "--dataset.episodes=[${EP50}]" --output_dir=outputs/train/scale50 --job_name=scale50
run noise005 "${COMMON[@]}" --dataset.repo_id=pusht_noise005 --dataset.root=datasets/pusht_noise005 \
  --output_dir=outputs/train/noise005 --job_name=noise005
run noise010 "${COMMON[@]}" --dataset.repo_id=pusht_noise010 --dataset.root=datasets/pusht_noise010 \
  --output_dir=outputs/train/noise010 --job_name=noise010
run noise020 "${COMMON[@]}" --dataset.repo_id=pusht_noise020 --dataset.root=datasets/pusht_noise020 \
  --output_dir=outputs/train/noise020 --job_name=noise020

echo ALL_RUNS_FINISHED
