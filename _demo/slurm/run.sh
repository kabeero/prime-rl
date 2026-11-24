#!/usr/bin/env bash
# Run from the prime-rl checkout on the SLURM login node (not locally).
# Extra args are forwarded, e.g. `./_demo/slurm/run.sh --dry-run`.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run rl @ examples/advanced/qwen3-30b-a3b/math.toml \
    --deployment.num-train-nodes 1 \
    --deployment.num-infer-nodes 1 \
    --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --seq-len 8192 \
    --orchestrator.train.sampling.max-completion-tokens 8192 \
    --orchestrator.batch-size 128 \
    --orchestrator.concurrency.initial-inflight 256 \
    --max-steps 30 \
    --no-monitors.wandb \
    --output-dir "$HOME/outputs" \
    --run.name qwen30b-math-mini \
    --inference.vllm.tensor-parallel-size 2 \
    --inference.vllm.max-model-len 16384 \
    "$@"
