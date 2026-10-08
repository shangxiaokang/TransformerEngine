# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare the differentiable QSA reference with independent dense attention."""

import importlib.util
from pathlib import Path

import pytest
import torch


# This pure PyTorch reference can be tested without building TE's native extension.
_QSA_PATH = (
    Path(__file__).resolve().parents[3]
    / "transformer_engine"
    / "pytorch"
    / "attention"
    / "qsa.py"
)
_SPEC = importlib.util.spec_from_file_location("qsa_reference_under_test", _QSA_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
qsa_block_sparse_attention = _MODULE.qsa_block_sparse_attention


def _selected_blocks(batch: int, seq_len: int, budget: int, device: torch.device) -> torch.Tensor:
    """Make causal per-token selections that change within each query block."""
    selected = torch.full((batch, seq_len, budget), -1, dtype=torch.int32, device=device)
    for batch_index in range(batch):
        for query_index in range(seq_len):
            full_blocks = (query_index + 1) // 4
            candidates = [
                block
                for block in range(full_blocks)
                if (block + batch_index + query_index) % 2 == 0
            ]
            if not candidates and full_blocks:
                candidates = [full_blocks - 1]
            candidates = candidates[-budget:]
            if candidates:
                selected[batch_index, query_index, : len(candidates)] = torch.tensor(
                    candidates, dtype=torch.int32, device=device
                )
    return selected


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Build a full token mask independently from the sparse gather implementation."""
    batch, seq_len, query_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    allowed = torch.zeros((batch, seq_len, seq_len), dtype=torch.bool, device=query.device)
    for batch_index in range(batch):
        for query_index in range(seq_len):
            chosen = set(selected[batch_index, query_index].tolist())
            for key_index in range(seq_len):
                key_block = key_index // 4
                complete = (key_block + 1) * 4 <= query_index + 1
                tail = (query_index + 1) % 4 != 0 and (
                    key_block == query_index // 4 and key_index <= query_index
                )
                allowed[batch_index, query_index, key_index] = (
                    key_block in chosen and complete
                ) or tail
    grouped_query = query.reshape(batch, seq_len, kv_heads, query_heads // kv_heads, head_dim)
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.einsum("bqhgd,bthd->bqhgt", grouped_query.float(), key.float())
        logits = logits * head_dim**-0.5
        visible = allowed[:, :, None, None, :]
        logits = logits.masked_fill(~visible, torch.finfo(torch.float32).min)
        probabilities = torch.softmax(logits, dim=-1).masked_fill(~visible, 0.0)
        output = torch.einsum("bqhgt,bthv->bqhgv", probabilities, value.float())
    return output.reshape(batch, seq_len, query_heads, value.shape[-1]).to(query.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [3, 4, 7, 11])
@pytest.mark.parametrize("head_dim", [8, 256])
@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("autocast_enabled", [False, True])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_qsa_matches_dense_forward_backward(
    dtype, seq_len, head_dim, checkpoint_chunks, autocast_enabled, device_name
):
    """Exercise causal boundaries, independent token indices, GQA, and gradients."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(1729)
    device = torch.device(device_name)
    batch, query_heads, kv_heads = 2, 24, 2
    value_dim = head_dim if head_dim == 256 else 12
    shapes = (
        (batch, seq_len, query_heads, head_dim),
        (batch, seq_len, kv_heads, head_dim),
        (batch, seq_len, kv_heads, value_dim),
    )
    sparse_inputs = [
        (0.3 * torch.randn(shape, device=device)).to(dtype).requires_grad_() for shape in shapes
    ]
    dense_inputs = [tensor.detach().clone().requires_grad_() for tensor in sparse_inputs]
    selected = _selected_blocks(batch, seq_len, budget=3, device=device)
    with torch.autocast(device_type=device_name, dtype=torch.bfloat16, enabled=autocast_enabled):
        sparse = qsa_block_sparse_attention(
            *sparse_inputs, selected, query_chunk_size=3, checkpoint_chunks=checkpoint_chunks
        )
        dense = _dense_attention(*dense_inputs, selected)
    forward_tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        sparse, dense, atol=forward_tolerance, rtol=forward_tolerance
    )

    output_weight = torch.randn_like(sparse, dtype=torch.float32)
    (sparse.float() * output_weight).sum().backward()
    (dense.float() * output_weight).sum().backward()
    # BF16 leaf gradients have coarser rounding than the FP32 attention output.
    gradient_atol = 1e-5 if dtype == torch.float32 else 4e-2
    gradient_rtol = 1e-5 if dtype == torch.float32 else 3e-2
    for sparse_input, dense_input in zip(sparse_inputs, dense_inputs):
        torch.testing.assert_close(
            sparse_input.grad, dense_input.grad, atol=gradient_atol, rtol=gradient_rtol
        )


def test_qsa_empty_selection_has_finite_output_and_gradient():
    """An all-masked complete-block row returns zero instead of NaN."""
    query = torch.randn(1, 4, 2, 8, requires_grad=True)
    key = torch.randn(1, 4, 1, 8, requires_grad=True)
    value = torch.randn(1, 4, 1, 8, requires_grad=True)
    selected = torch.full((1, 4, 0), -1, dtype=torch.int64)
    output = qsa_block_sparse_attention(query, key, value, selected, query_chunk_size=2)
    assert output.isfinite().all()
    torch.testing.assert_close(output[:, 3], torch.zeros_like(output[:, 3]))
    output.sum().backward()
    assert all(
        tensor.grad is not None and tensor.grad.isfinite().all()
        for tensor in (query, key, value)
    )
    torch.testing.assert_close(query.grad[:, 3], torch.zeros_like(query.grad[:, 3]))


@pytest.mark.parametrize(
    "seq_len, query_index, bad_selection",
    [(4, 3, [-2]), (4, 3, [1]), (8, 3, [1]), (7, 6, [1]), (4, 3, [0, 0])],
)
def test_qsa_rejects_invalid_block_indices(seq_len, query_index, bad_selection):
    """Invalid, future/incomplete, and duplicate blocks must not skew softmax."""
    query = torch.randn(1, seq_len, 2, 8)
    key = torch.randn(1, seq_len, 1, 8)
    value = torch.randn(1, seq_len, 1, 8)
    selected = torch.full((1, seq_len, len(bad_selection)), -1, dtype=torch.int32)
    selected[0, query_index] = torch.tensor(bad_selection, dtype=torch.int32)
    with pytest.raises(ValueError):
        qsa_block_sparse_attention(query, key, value, selected)
