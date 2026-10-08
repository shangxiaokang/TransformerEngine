# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare QSA indexed-SDPA and the single-launch Triton prototype."""

import argparse
import gc
import importlib.util
import math
import statistics
from pathlib import Path

import torch

_QSA_PATH = (
    Path(__file__).resolve().parents[2] / "transformer_engine" / "pytorch" / "attention" / "qsa.py"
)
_SPEC = importlib.util.spec_from_file_location("qsa_benchmark_under_test", _QSA_PATH)
_QSA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_QSA)


def _selection(sequence_length, block_budget, device):
    """Build deterministic unique causal selections without a sequence-square tensor."""
    blocks = torch.arange(block_budget, dtype=torch.int32, device=device)[None, None, :]
    blocks = blocks.expand(1, sequence_length, -1)
    queries = torch.arange(sequence_length, device=device)[None, :, None]
    return torch.where(blocks * 4 + 3 <= queries, blocks, -1)


def _inputs(args):
    """Allocate the requested target-like GQA geometry."""
    shapes = (
        (1, args.sequence_length, args.query_heads, args.head_dim),
        (1, args.sequence_length, args.kv_heads, args.head_dim),
        (1, args.sequence_length, args.kv_heads, args.value_dim),
    )
    return [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_() for shape in shapes
    ]


def _step(name, inputs, selected, query_chunk_size):
    """Run one differentiable forward/backward pair."""
    if name == "indexed_sdpa":
        output = _QSA.qsa_indexed_sdpa_attention(
            *inputs,
            selected,
            query_chunk_size=query_chunk_size,
            checkpoint_chunks=True,
            validate_indices=False,
        )
    else:
        output = _QSA.qsa_triton_attention(
            *inputs,
            selected,
            validate_indices=False,
        )
    output.float().square().mean().backward()


def _measure(name, args, selected):
    """Return median iteration milliseconds and peak bytes above the input baseline."""
    inputs = _inputs(args)
    for _ in range(args.warmup):
        _step(name, inputs, selected, args.query_chunk_size)
        for tensor in inputs:
            tensor.grad = None
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _step(name, inputs, selected, args.query_chunk_size)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        for tensor in inputs:
            tensor.grad = None
    peak_bytes = torch.cuda.max_memory_allocated() - baseline
    return statistics.median(times), peak_bytes


def main():
    """Run both trainable paths with the same tensors and trusted index contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--block-budget", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_qsa.py requires CUDA")

    selected = _selection(args.sequence_length, args.block_budget, torch.device("cuda"))
    print("backend,median_train_ms,peak_bytes,python_attention_dispatches")
    for name in ("indexed_sdpa", "triton"):
        median_ms, peak_bytes = _measure(name, args, selected)
        dispatches = (
            math.ceil(args.sequence_length / args.query_chunk_size) if name == "indexed_sdpa" else 1
        )
        print(f"{name},{median_ms:.3f},{peak_bytes},{dispatches}")


if __name__ == "__main__":
    main()
