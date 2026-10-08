# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank DDP correctness smoke for the experimental QSA Triton path.

Run with::

    python -m torch.distributed.run --nproc_per_node=2 \
        tests/pytorch/attention/run_qsa_triton_distributed.py
"""

import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

_QSA_PATH = (
    Path(__file__).resolve().parents[3] / "transformer_engine" / "pytorch" / "attention" / "qsa.py"
)
_SPEC = importlib.util.spec_from_file_location("qsa_distributed_under_test", _QSA_PATH)
_QSA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_QSA)


class _QSAParameters(torch.nn.Module):
    """Keep Q/K/V as replicated parameters so DDP exercises all gradients."""

    def __init__(self, query, key, value):
        super().__init__()
        self.query = torch.nn.Parameter(query)
        self.key = torch.nn.Parameter(key)
        self.value = torch.nn.Parameter(value)

    def forward(self, selected):
        """Apply QSA to the replicated parameter tensors."""
        return _QSA.qsa_triton_attention(self.query, self.key, self.value, selected)


def _selected_blocks(rank, sequence_length, device):
    """Use a different valid selection on each data-parallel rank."""
    selected = torch.full((1, sequence_length, 2), -1, dtype=torch.int32, device=device)
    for query_position in range(sequence_length):
        complete_blocks = (query_position + 1) // 4
        if complete_blocks:
            block = (complete_blocks - 1 - rank) % complete_blocks
            selected[0, query_position, 0] = block
    return selected


def main():
    """Compare DDP-reduced Triton gradients with reduced reference gradients."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1931)
    query = torch.randn(1, 11, 4, 16, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 11, 2, 16, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 11, 2, 12, device=device, dtype=torch.bfloat16)
    selected = _selected_blocks(rank, query.shape[1], device)

    model = _QSAParameters(query.clone(), key.clone(), value.clone())
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    output = ddp_model(selected)
    torch.manual_seed(7000 + rank)
    output_weight = torch.randn_like(output)
    (output.float() * output_weight.float()).sum().backward()

    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
    reference_output = _QSA.qsa_block_sparse_attention(
        *reference_inputs, selected, query_chunk_size=3, checkpoint_chunks=False
    )
    (reference_output.float() * output_weight.float()).sum().backward()
    for tensor in reference_inputs:
        dist.all_reduce(tensor.grad)
        tensor.grad /= dist.get_world_size()

    for name, parameter, reference in zip(
        ("query", "key", "value"), model.parameters(), reference_inputs
    ):
        difference = parameter.grad.float() - reference.grad.float()
        reference_rms = reference.grad.float().square().mean().sqrt().clamp_min(1e-12)
        if difference.float().square().mean().sqrt() / reference_rms >= 2e-2:
            raise AssertionError(f"{name} DDP gradient RMS mismatch")
        if difference.abs().max() / reference_rms >= 2e-1:
            raise AssertionError(f"{name} DDP gradient maximum mismatch")

    checksum = output.float().square().sum()
    dist.all_reduce(checksum)
    if rank == 0:
        print(
            "QSA Triton DDP PASS: "
            f"world_size={dist.get_world_size()} checksum={checksum.item():.6f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
