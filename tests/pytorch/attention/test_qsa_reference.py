# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare the differentiable QSA reference with independent dense attention."""

import contextlib
import gc
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
qsa_indexed_sdpa_attention = _MODULE.qsa_indexed_sdpa_attention


@contextlib.contextmanager
def _cuda_matmul_precision(precision):
    """Set CUDA matmul precision and restore the caller's setting."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        original = matmul.fp32_precision
        try:
            matmul.fp32_precision = precision
            yield
        finally:
            matmul.fp32_precision = original
        return
    original = matmul.allow_tf32
    try:
        matmul.allow_tf32 = precision == "tf32"
        yield
    finally:
        matmul.allow_tf32 = original


def _ieee_cuda_matmul():
    """Keep forward and backward in IEEE mode."""
    return _cuda_matmul_precision("ieee")


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
    with _ieee_cuda_matmul():
        _compare_qsa_forward_backward(
            qsa_block_sparse_attention,
            dtype,
            seq_len,
            head_dim,
            checkpoint_chunks,
            autocast_enabled,
            device_name,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [3, 7, 11])
@pytest.mark.parametrize("head_dim", [8, 256])
@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_qsa_indexed_sdpa_matches_dense_forward_backward(
    dtype, seq_len, head_dim, checkpoint_chunks, device_name
):
    """Indexed SDPA preserves per-token selection and all Q/K/V gradients."""
    with _ieee_cuda_matmul():
        _compare_qsa_forward_backward(
            qsa_indexed_sdpa_attention,
            dtype,
            seq_len,
            head_dim,
            checkpoint_chunks,
            True,
            device_name,
        )


def _compare_qsa_forward_backward(
    attention_function, dtype, seq_len, head_dim, checkpoint_chunks, autocast_enabled, device_name
):
    """Compare one shape and precision configuration."""
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
        sparse = attention_function(
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
        if attention_function is qsa_indexed_sdpa_attention and dtype == torch.bfloat16:
            # SDPA's native BF16 backward reduces over 12 query heads per KV
            # head. Small cancellation residuals need an aggregate comparison.
            difference = sparse_input.grad.float() - dense_input.grad.float()
            reference = dense_input.grad.float()
            reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
            difference_rms = difference.square().mean().sqrt()
            assert difference_rms / reference_rms < 1e-2
            assert difference.abs().max() / reference_rms < 1e-1
        else:
            torch.testing.assert_close(
                sparse_input.grad, dense_input.grad, atol=gradient_atol, rtol=gradient_rtol
            )


def test_qsa_chunk_selection_size_is_independent_of_sequence_length():
    """A query chunk selects a bounded K/V list, even at target context length."""
    selected = torch.arange(512, dtype=torch.int32)[None, None, :].expand(1, 8, -1)
    for seq_len in (4096, 131072):
        query_positions = torch.arange(seq_len - 8, seq_len)
        positions, visible = _MODULE._qsa_chunk_positions(
            selected, query_positions, seq_len
        )
        assert positions.shape == visible.shape == (1, 8, 4 * (512 + 1))
        assert positions.numel() == 8 * 2052
        assert positions.max() < seq_len


def test_qsa_indexed_sdpa_peak_memory_scales_with_query_count():
    """With fixed selection budget and chunk size, memory grows below S squared."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def peak_for_length(seq_len):
        gc.collect()
        torch.cuda.empty_cache()
        query = torch.randn(1, seq_len, 24, 64, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, seq_len, 2, 64, device="cuda", dtype=torch.bfloat16)
        query.requires_grad_()
        key.requires_grad_()
        value.requires_grad_()
        blocks = torch.arange(4, device="cuda", dtype=torch.int32)[None, None, :]
        blocks = blocks.expand(1, seq_len, -1)
        query_positions = torch.arange(seq_len, device="cuda")[None, :, None]
        selected = torch.where(blocks * 4 + 3 <= query_positions, blocks, -1)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = qsa_indexed_sdpa_attention(
            query, key, value, selected, query_chunk_size=8, checkpoint_chunks=True
        )
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - baseline

    peak_for_length(16)  # initialize the SDPA backend before measuring
    short_peak = peak_for_length(128)
    long_peak = peak_for_length(512)
    assert short_peak > 0
    assert long_peak < 6 * short_peak


def test_qsa_rejects_tf32_cuda():
    """Fail closed because forward-only precision changes cannot cover backward."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    query = torch.randn(1, 1, 2, 8, device="cuda")
    key = torch.randn(1, 1, 1, 8, device="cuda")
    value = torch.randn(1, 1, 1, 8, device="cuda")
    selected = torch.empty((1, 1, 0), dtype=torch.int32, device="cuda")
    with _cuda_matmul_precision("tf32"):
        with pytest.raises(RuntimeError, match="allow_tf32=False"):
            qsa_block_sparse_attention(query, key, value, selected)


@pytest.mark.parametrize(
    "attention_function", [qsa_block_sparse_attention, qsa_indexed_sdpa_attention]
)
def test_qsa_bf16_runs_with_tf32_cuda(attention_function):
    """BF16 training smoke remains available under a TF32-enabled process."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    query = torch.randn(1, 5, 2, 8, device="cuda", dtype=torch.bfloat16).requires_grad_()
    key = torch.randn(1, 5, 1, 8, device="cuda", dtype=torch.bfloat16).requires_grad_()
    value = torch.randn(1, 5, 1, 8, device="cuda", dtype=torch.bfloat16).requires_grad_()
    selected = torch.tensor([[[-1], [-1], [-1], [0], [0]]], device="cuda", dtype=torch.int32)
    with _cuda_matmul_precision("tf32"):
        output = attention_function(query, key, value, selected)
        output.float().sum().backward()
    assert output.isfinite().all()
    assert all(tensor.grad is not None for tensor in (query, key, value))


@pytest.mark.parametrize(
    "attention_function", [qsa_block_sparse_attention, qsa_indexed_sdpa_attention]
)
def test_qsa_empty_selection_has_finite_output_and_gradient(attention_function):
    """An all-masked complete-block row returns zero instead of NaN."""
    query = torch.randn(1, 4, 2, 8, requires_grad=True)
    key = torch.randn(1, 4, 1, 8, requires_grad=True)
    value = torch.randn(1, 4, 1, 8, requires_grad=True)
    selected = torch.full((1, 4, 0), -1, dtype=torch.int64)
    output = attention_function(query, key, value, selected, query_chunk_size=2)
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
