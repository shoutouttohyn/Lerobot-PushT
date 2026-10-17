#!/usr/bin/env bash
# WSL env installer v2: TUNA-only index (pypi.org hangs from this network),
# lerobot/gym-pusht wheels provided locally via --find-links. Idempotent.
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL
PIP="$HOME/miniconda3/envs/lerobot/bin/pip"
PY="$HOME/miniconda3/envs/lerobot/bin/python"
WHEELS="/mnt/d/Kimi code work/lerobot-pusht-wsl/_wheels"
LOG=/tmp/wsl_pip_install.log
: > "$LOG"
echo "=== install $(date) ===" >> "$LOG"
if $PIP install --index-url "$PIP_INDEX_URL" --find-links "$WHEELS" \
    "lerobot[pusht]==0.6.1" accelerate matplotlib >> "$LOG" 2>&1; then
  $PY -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" >> "$LOG" 2>&1
  echo DONE_OK > /tmp/wsl_install_status
else
  echo DONE_FAIL > /tmp/wsl_install_status
fi
tail -5 "$LOG"
