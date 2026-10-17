"""Benchmark end-to-end inference latency of a trained LeRobot policy.

What is measured, mirroring the real deployment path of lerobot_eval:
    prepare_observation_for_inference -> preprocessor (normalization)
    -> policy.select_action -> postprocessor (denormalization)

Two call patterns:
    full-forward : policy.reset() before every call -> every call runs ResNet18+Transformer
                   (worst-case latency, what a cold step costs)
    steady-state : consecutive calls without reset -> ACT pops from its 8-step action
                   queue, 7 of 8 calls are cache hits (real amortized cost)

Variants: fp32 CUDA (baseline), fp16 autocast CUDA, torch.compile, CPU fp32.

Usage:
    python scripts/benchmark_inference.py \
        --ckpt outputs/train/act_pusht/checkpoints/last/pretrained_model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

N_ACTION_STEPS = 8  # of the trained checkpoint (chunk 8, exec 8)


def build_pipeline(ckpt: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)  # PreTrainedPolicy is abstract; resolve e.g. ACTPolicy
    policy = policy_cls.from_pretrained(ckpt, config=cfg)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt))
    # Processor pipelines are saved with the training device baked in (DeviceProcessorStep /
    # normalizer buffers); retarget them when benchmarking on a different device.
    import torch.nn as nn

    for pipe in (pre, post):
        for step in getattr(pipe, "steps", []):
            if hasattr(step, "device"):
                step.device = torch.device(device)
            if isinstance(step, nn.Module):
                step.to(device)
    return policy, pre, post, preprocess_observation


def make_obs() -> dict:
    """Raw vector-env observation (numpy, batched, channel-last uint8 image)."""
    rng = np.random.default_rng(0)
    return {
        "pixels": rng.integers(0, 256, (1, 96, 96, 3), dtype=np.uint8),
        "agent_pos": np.array([[256.0, 256.0]], dtype=np.float32),
    }


def bench(fn, iters: int, warmup: int = 30) -> dict:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"mean_ms": round(float(ts.mean()), 2), "p50_ms": round(float(np.percentile(ts, 50)), 2),
            "p99_ms": round(float(np.percentile(ts, 99)), 2)}


def run_variant(name: str, ckpt: str, device: str, pattern: str, mode: str, iters: int, results: dict) -> None:
    try:
        policy, pre, post, prep = build_pipeline(ckpt, device)
        dev = next(policy.parameters()).device
        obs_np = make_obs()

        def call():
            with torch.inference_mode():
                obs = prep(obs_np)  # numpy env obs -> lerobot key tensors
                obs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in obs.items()}
                batch = pre(obs)
                # the saved pipeline hardcodes its training device; honor the requested one
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                action = policy.select_action(batch)
                return post(action)

        if mode == "fp16":
            def call_fp16():
                with torch.autocast("cuda", dtype=torch.float16):
                    return call()
            fn = call_fp16
        elif mode == "compile":
            policy_compiled = torch.compile(policy, mode="reduce-overhead")

            def call_compiled():
                with torch.inference_mode():
                    obs = prep(obs_np)
                    obs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in obs.items()}
                    batch = pre(obs)
                    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    return post(policy_compiled.select_action(batch))
            fn = call_compiled
        else:
            fn = call

        if pattern == "full":
            def timed():
                policy.reset()
                return fn()
        else:
            timed = fn

        results[name] = bench(timed, iters)
        results[name]["ok"] = True
    except Exception as e:  # noqa: BLE001 - report N/A instead of crashing the suite
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/train/act_pusht/checkpoints/last/pretrained_model")
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    results = {}
    run_variant("cuda_fp32_full_forward", args.ckpt, "cuda", "full", "fp32", args.iters, results)
    run_variant("cuda_fp32_steady_state", args.ckpt, "cuda", "steady", "fp32", args.iters, results)
    run_variant("cuda_fp16_full_forward", args.ckpt, "cuda", "full", "fp16", args.iters, results)
    run_variant("cuda_compile_full_forward", args.ckpt, "cuda", "full", "compile", args.iters, results)
    run_variant("cpu_fp32_full_forward", args.ckpt, "cpu", "full", "fp32", max(args.iters // 4, 50), results)

    out_dir = Path("outputs/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference_benchmark.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'variant':<28}{'mean ms':>10}{'p50 ms':>10}{'p99 ms':>10}")
    for k, v in results.items():
        if v.get("ok"):
            print(f"{k:<28}{v['mean_ms']:>10}{v['p50_ms']:>10}{v['p99_ms']:>10}")
        else:
            print(f"{k:<28}  N/A ({v['error'][:60]})")
    base = results.get("cuda_fp32_full_forward", {}).get("mean_ms")
    if base:
        eff = base / N_ACTION_STEPS
        print(f"\ncontrol-loop context: {N_ACTION_STEPS} env steps per call -> "
              f"amortized full-forward latency {eff:.2f} ms/env-step "
              f"(10 fps control loop budget is 100 ms/env-step)")
    (out_dir / "inference_benchmark.md").write_text(
        "| variant | mean ms | p50 ms | p99 ms |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {k} | {v.get('mean_ms', 'N/A')} | {v.get('p50_ms', 'N/A')} | {v.get('p99_ms', 'N/A')} |"
            for k, v in results.items()
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
