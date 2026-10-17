# LeRobot PushT 模仿学习最小闭环

本仓库是基于Hugging Face Lerobot的复现 + 对照试验 + 推理基准增强版**：
在 WSL2 (Ubuntu 26.04) + Miniconda 环境下从零搭建，单机 RTX 3070 完成。

## 环境搭建（已在 WSL2 + RTX 3070 验证）

```bash
# 1. Miniconda（免 sudo，用户目录安装）
curl -sL -o ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh -b -p ~/miniconda3

# 2. 建环境（lerobot 0.6.1 要求 Python >= 3.12）
~/miniconda3/bin/conda create -y -n lerobot python=3.12

# 3. 安装（Linux 下 PyPI 的 torch 默认即 CUDA 版，无需官方源）
~/miniconda3/envs/lerobot/bin/pip install -U pip
~/miniconda3/envs/lerobot/bin/pip install "lerobot[pusht]==0.6.1" accelerate matplotlib
```

## 快速复现

```bash
cd /mnt/d/Kimi\ code\ work/lerobot-pusht-wsl   # 或你 clone 的位置

# 训练（6000 步，RTX 3070 约 10-15 分钟）
bash scripts/train_act.sh > train_act.log 2>&1

# 评测（20 条 rollout + 自动录屏）
bash scripts/eval_act.sh outputs/train/act_pusht/checkpoints/last/pretrained_model

# 数据研究（5 组对照，约 80 分钟）
bash scripts/run_data_study.sh

# 画曲线
~/miniconda3/envs/lerobot/bin/python scripts/plot_data_study.py

# 推理延迟基准（见下节）
~/miniconda3/envs/lerobot/bin/python scripts/benchmark_inference.py
```

## 推理延迟基准

`scripts/benchmark_inference.py` 测量训练 checkpoint 的部署侧延迟，模拟真实推理链路：
`观测预处理 → 归一化 → select_action → 反归一化`。

- **full-forward**：每次调用前 `policy.reset()`，每次都走完整 ResNet18+Transformer 前向（冷启动/最坏情况）
- **steady-state**：连续调用，ACT 从 8 步动作队列中取（真实均摊成本，7/8 次是缓存命中）
- 变体：CUDA FP32（基线）/ CUDA FP16（autocast）/ torch.compile / CPU FP32（无端侧 GPU 场景）
- 输出：控制台表格 + `outputs/benchmark/inference_benchmark.{json,md}`

结果（RTX 3070 Laptop + WSL2， checkpoint：ACT 6000 步）：

<!-- BENCHMARK_RESULTS_START -->
| variant | mean ms | p50 ms | p99 ms |
|---|---|---|---|
| CUDA FP32 full-forward（基线） | 8.94 | 8.70 | 20.53 |
| CUDA FP32 steady-state（队列均摊） | 1.55 | 0.69 | 9.73 |
| CUDA FP16 autocast full-forward | 8.75 | 8.46 | 16.94 |
| torch.compile full-forward | 7.20 | 6.90 | 14.67 |
| CPU FP32 full-forward | 13.83 | 13.91 | 15.45 |

**结论**：所有配置都远低于 10 fps 控制环预算（full-forward 均摊 1.12 ms/env-step vs 100 ms 预算，余量约 90 倍）。
有意思的发现：**这个小模型上 FP16 相对 FP32 几乎无收益**（8.75 vs 8.94 ms——模型太小，autocast 的转换/调度开销抵消了半精度计算收益）；
torch.compile 有约 20% 的稳定收益；纯 CPU 推理（13.8 ms）也能满足 10 fps 实时机型，无端侧 GPU 可行。
数字为单卡 RTX 3070 Laptop (WSL2) 实测，run-to-run 波动约 ±15%。
<!-- BENCHMARK_RESULTS_END -->

控制环换算：checkpoint 的 `n_action_steps=8`，即每 8 个环境步调用一次策略；
10 fps 控制环的单步预算是 100 ms/env-step。

## 项目结构

```
lerobot-pusht-wsl/
├── scripts/
│   ├── train_act.sh            # 基线训练（ACT / 6000 步）
│   ├── eval_act.sh             # 评测 + 录屏
│   ├── run_data_study.sh       # 5 组数据对照实验一键跑批
│   ├── make_noise_dataset.py   # 动作噪声注入（数据质量变体）
│   ├── plot_loss.py            # 训练 loss/成功率曲线
│   ├── plot_data_study.py      # 数据研究汇总曲线
│   └── benchmark_inference.py  # 推理延迟基准（本版新增）
├── datasets/                   # pusht_clean + 3 个噪声变体（7.4MB × 4）
└── outputs/                    # checkpoints / 评测视频 / 曲线（gitignored）
```

数据集说明（v3.0 格式、`root` 语义、噪声注入方法）与实验结论详见 Windows 版 README 的
「数据研究」章节，两版数据与结论完全一致。
