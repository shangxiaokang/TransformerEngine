# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Differentiable PyTorch reference for QwenAir block-sparse attention.

This implementation gathers selected keys and values with PyTorch operations.
It is intended for numerical validation and small-scale training bring-up, not
as a production sparse-attention kernel.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_QSA_BLOCK_SIZE = 4


def _cuda_tf32_matmul_enabled() -> bool:
    """Query CUDA matmul precision across the old and new PyTorch settings."""
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision == "tf32":
        return True
    if precision == "ieee":
        return False
    if hasattr(torch, "get_float32_matmul_precision"):
        return torch.get_float32_matmul_precision() != "highest"
    return torch.backends.cuda.matmul.allow_tf32


def _validate_qsa_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    query_chunk_size: int,
    *,
    validate_indices: bool = True,
) -> None:
    """Check the self-attention layout and the indexer contract."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have [batch, sequence, heads, dim] layout")
    batch, seq_len, num_query_heads, query_dim = query.shape
    if seq_len == 0 or query_dim == 0:
        raise ValueError("sequence length and query/key head dimension must be positive")
    if key.shape[:2] != (batch, seq_len) or value.shape[:2] != (batch, seq_len):
        raise ValueError("QSA reference requires equal, nonempty query/key/value sequence lengths")
    if key.shape[-1] != query_dim or key.shape[2] != value.shape[2]:
        raise ValueError("query/key dimensions and key/value head counts must match")
    if (
        value.shape[-1] == 0
        or num_query_heads == 0
        or key.shape[2] == 0
        or num_query_heads % key.shape[2]
    ):
        raise ValueError("value dimension must be positive and query heads divisible by KV heads")
    if query.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("QSA reference supports BF16 and FP32 query/key/value tensors")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must reside on the same device")
    if selected_key_blocks.ndim != 3 or selected_key_blocks.shape[:2] != (batch, seq_len):
        raise ValueError("selected_key_blocks must have [batch, query_sequence, budget] layout")
    if selected_key_blocks.dtype not in (torch.int32, torch.int64):
        raise TypeError("selected_key_blocks must have int32 or int64 dtype")
    if selected_key_blocks.device != query.device:
        raise ValueError("selected_key_blocks and query must reside on the same device")
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    if query.device.type == "cuda" and query.dtype == torch.float32 and _cuda_tf32_matmul_enabled():
        raise RuntimeError(
            "QSA reference requires torch.backends.cuda.matmul.allow_tf32=False "
            "throughout forward and backward for IEEE FP32 score accumulation"
        )

    if not validate_indices:
        return

    # A nonnegative index identifies a *complete* four-token key block visible
    # to this query token. The current incomplete block is handled separately.
    valid = selected_key_blocks >= 0
    if torch.any(selected_key_blocks < -1) or torch.any(
        selected_key_blocks >= math.ceil(seq_len / _QSA_BLOCK_SIZE)
    ):
        raise ValueError("selected key block indices are outside the sequence")
    query_positions = torch.arange(seq_len, device=query.device)[None, :, None]
    if torch.any(
        valid & (selected_key_blocks * _QSA_BLOCK_SIZE + _QSA_BLOCK_SIZE - 1 > query_positions)
    ):
        raise ValueError("selected key blocks must be complete, causal blocks or -1")
    if selected_key_blocks.shape[-1] > 1:
        sorted_blocks = selected_key_blocks.sort(dim=-1).values
        duplicate_blocks = (sorted_blocks[..., 1:] == sorted_blocks[..., :-1]) & (
            sorted_blocks[..., 1:] >= 0
        )
        if torch.any(duplicate_blocks):
            raise ValueError("selected key blocks must be unique for each query token")


if triton is not None:

    @triton.jit
    def _qsa_triton_forward_kernel(
        query,
        key,
        value,
        selected_blocks,
        output,
        logsumexp,
        sequence_length,
        query_heads,
        kv_heads,
        selected_block_count,
        scale,
        stride_q_batch,
        stride_q_sequence,
        stride_q_head,
        stride_q_dim,
        stride_k_batch,
        stride_k_sequence,
        stride_k_head,
        stride_k_dim,
        stride_v_batch,
        stride_v_sequence,
        stride_v_head,
        stride_v_dim,
        stride_b_batch,
        stride_b_sequence,
        stride_b_block,
        HEAD_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
        BLOCK_VALUE_DIM: tl.constexpr,
        QSA_BLOCK_SIZE: tl.constexpr,
    ):
        """One-program-per-query-head indexed attention with online softmax."""
        row = tl.program_id(0)
        query_head = row % query_heads
        query_position = (row // query_heads) % sequence_length
        batch_index = row // (query_heads * sequence_length)
        kv_head = query_head // (query_heads // kv_heads)

        head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
        value_offsets = tl.arange(0, BLOCK_VALUE_DIM)
        head_mask = head_offsets < HEAD_DIM
        value_mask = value_offsets < VALUE_DIM
        query_ptrs = (
            query
            + batch_index * stride_q_batch
            + query_position * stride_q_sequence
            + query_head * stride_q_head
            + head_offsets * stride_q_dim
        )
        query_row = tl.load(query_ptrs, mask=head_mask, other=0.0).to(tl.float32)

        row_max = -float("inf")
        row_sum = 0.0
        accumulator = tl.zeros((BLOCK_VALUE_DIM,), dtype=tl.float32)
        block_row = (
            selected_blocks + batch_index * stride_b_batch + query_position * stride_b_sequence
        )

        # selected_block_count is a runtime value so this remains a device loop
        # instead of generating one host launch (or one unrolled graph) per block.
        for block_slot in tl.range(0, selected_block_count):
            block_index = tl.load(block_row + block_slot * stride_b_block)
            block_valid = block_index >= 0
            safe_block = tl.maximum(block_index, 0)
            for token_offset in range(QSA_BLOCK_SIZE):
                key_position = safe_block * QSA_BLOCK_SIZE + token_offset
                key_ptrs = (
                    key
                    + batch_index * stride_k_batch
                    + key_position * stride_k_sequence
                    + kv_head * stride_k_head
                    + head_offsets * stride_k_dim
                )
                key_row = tl.load(key_ptrs, mask=block_valid & head_mask, other=0.0).to(tl.float32)
                score = tl.sum(query_row * key_row, axis=0) * scale
                next_max = tl.maximum(row_max, score)
                old_weight = tl.where(block_valid, tl.exp(row_max - next_max), 1.0)
                new_weight = tl.where(block_valid, tl.exp(score - next_max), 0.0)
                value_ptrs = (
                    value
                    + batch_index * stride_v_batch
                    + key_position * stride_v_sequence
                    + kv_head * stride_v_head
                    + value_offsets * stride_v_dim
                )
                value_row = tl.load(value_ptrs, mask=block_valid & value_mask, other=0.0).to(
                    tl.float32
                )
                accumulator = accumulator * old_weight + value_row * new_weight
                row_sum = row_sum * old_weight + new_weight
                row_max = tl.where(block_valid, next_max, row_max)

        # The current block is appended only while it is incomplete. At the
        # four-token boundary it must be selected through the normal index list.
        tail_size = (query_position % QSA_BLOCK_SIZE) + 1
        tail_present = tail_size != QSA_BLOCK_SIZE
        tail_start = (query_position // QSA_BLOCK_SIZE) * QSA_BLOCK_SIZE
        for token_offset in range(QSA_BLOCK_SIZE):
            token_valid = tail_present & (token_offset < tail_size)
            key_position = tail_start + token_offset
            key_ptrs = (
                key
                + batch_index * stride_k_batch
                + key_position * stride_k_sequence
                + kv_head * stride_k_head
                + head_offsets * stride_k_dim
            )
            key_row = tl.load(key_ptrs, mask=token_valid & head_mask, other=0.0).to(tl.float32)
            score = tl.sum(query_row * key_row, axis=0) * scale
            next_max = tl.maximum(row_max, score)
            old_weight = tl.where(token_valid, tl.exp(row_max - next_max), 1.0)
            new_weight = tl.where(token_valid, tl.exp(score - next_max), 0.0)
            value_ptrs = (
                value
                + batch_index * stride_v_batch
                + key_position * stride_v_sequence
                + kv_head * stride_v_head
                + value_offsets * stride_v_dim
            )
            value_row = tl.load(value_ptrs, mask=token_valid & value_mask, other=0.0).to(tl.float32)
            accumulator = accumulator * old_weight + value_row * new_weight
            row_sum = row_sum * old_weight + new_weight
            row_max = tl.where(token_valid, next_max, row_max)

        output_row = accumulator / tl.maximum(row_sum, 1.0)
        output_ptrs = output + row * VALUE_DIM + value_offsets
        tl.store(output_ptrs, output_row, mask=value_mask)
        row_logsumexp = tl.where(row_sum > 0.0, row_max + tl.log(row_sum), -float("inf"))
        tl.store(logsumexp + row, row_logsumexp)

    @triton.jit
    def _qsa_triton_backward_kernel(
        query,
        key,
        value,
        selected_blocks,
        output,
        output_gradient,
        logsumexp,
        query_gradient,
        key_gradient,
        value_gradient,
        sequence_length,
        query_heads,
        kv_heads,
        selected_block_count,
        scale,
        stride_q_batch,
        stride_q_sequence,
        stride_q_head,
        stride_q_dim,
        stride_k_batch,
        stride_k_sequence,
        stride_k_head,
        stride_k_dim,
        stride_v_batch,
        stride_v_sequence,
        stride_v_head,
        stride_v_dim,
        stride_b_batch,
        stride_b_sequence,
        stride_b_block,
        HEAD_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_HEAD_DIM: tl.constexpr,
        BLOCK_VALUE_DIM: tl.constexpr,
        QSA_BLOCK_SIZE: tl.constexpr,
    ):
        """Recompute probabilities and accumulate exact Q/K/V gradient formulas."""
        row = tl.program_id(0)
        query_head = row % query_heads
        query_position = (row // query_heads) % sequence_length
        batch_index = row // (query_heads * sequence_length)
        kv_head = query_head // (query_heads // kv_heads)

        head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
        value_offsets = tl.arange(0, BLOCK_VALUE_DIM)
        head_mask = head_offsets < HEAD_DIM
        value_mask = value_offsets < VALUE_DIM
        query_ptrs = (
            query
            + batch_index * stride_q_batch
            + query_position * stride_q_sequence
            + query_head * stride_q_head
            + head_offsets * stride_q_dim
        )
        query_row = tl.load(query_ptrs, mask=head_mask, other=0.0).to(tl.float32)
        output_row = tl.load(output + row * VALUE_DIM + value_offsets, mask=value_mask).to(
            tl.float32
        )
        output_gradient_row = tl.load(
            output_gradient + row * VALUE_DIM + value_offsets, mask=value_mask
        ).to(tl.float32)
        delta = tl.sum(output_row * output_gradient_row, axis=0)
        row_logsumexp = tl.load(logsumexp + row)
        query_gradient_row = tl.zeros((BLOCK_HEAD_DIM,), dtype=tl.float32)
        block_row = (
            selected_blocks + batch_index * stride_b_batch + query_position * stride_b_sequence
        )

        for block_slot in tl.range(0, selected_block_count):
            block_index = tl.load(block_row + block_slot * stride_b_block)
            block_valid = block_index >= 0
            safe_block = tl.maximum(block_index, 0)
            for token_offset in range(QSA_BLOCK_SIZE):
                key_position = safe_block * QSA_BLOCK_SIZE + token_offset
                key_ptrs = (
                    key
                    + batch_index * stride_k_batch
                    + key_position * stride_k_sequence
                    + kv_head * stride_k_head
                    + head_offsets * stride_k_dim
                )
                key_row = tl.load(key_ptrs, mask=block_valid & head_mask, other=0.0).to(tl.float32)
                value_ptrs = (
                    value
                    + batch_index * stride_v_batch
                    + key_position * stride_v_sequence
                    + kv_head * stride_v_head
                    + value_offsets * stride_v_dim
                )
                value_row = tl.load(value_ptrs, mask=block_valid & value_mask, other=0.0).to(
                    tl.float32
                )
                score = tl.sum(query_row * key_row, axis=0) * scale
                probability = tl.where(block_valid, tl.exp(score - row_logsumexp), 0.0)
                value_dot = tl.sum(output_gradient_row * value_row, axis=0)
                score_gradient = probability * (value_dot - delta)
                query_gradient_row += score_gradient * scale * key_row
                key_gradient_ptrs = (
                    key_gradient
                    + ((batch_index * sequence_length + key_position) * kv_heads + kv_head)
                    * HEAD_DIM
                    + head_offsets
                )
                value_gradient_ptrs = (
                    value_gradient
                    + ((batch_index * sequence_length + key_position) * kv_heads + kv_head)
                    * VALUE_DIM
                    + value_offsets
                )
                tl.atomic_add(
                    key_gradient_ptrs,
                    score_gradient * scale * query_row,
                    mask=block_valid & head_mask,
                )
                tl.atomic_add(
                    value_gradient_ptrs,
                    probability * output_gradient_row,
                    mask=block_valid & value_mask,
                )

        tail_size = (query_position % QSA_BLOCK_SIZE) + 1
        tail_present = tail_size != QSA_BLOCK_SIZE
        tail_start = (query_position // QSA_BLOCK_SIZE) * QSA_BLOCK_SIZE
        for token_offset in range(QSA_BLOCK_SIZE):
            token_valid = tail_present & (token_offset < tail_size)
            key_position = tail_start + token_offset
            key_ptrs = (
                key
                + batch_index * stride_k_batch
                + key_position * stride_k_sequence
                + kv_head * stride_k_head
                + head_offsets * stride_k_dim
            )
            key_row = tl.load(key_ptrs, mask=token_valid & head_mask, other=0.0).to(tl.float32)
            value_ptrs = (
                value
                + batch_index * stride_v_batch
                + key_position * stride_v_sequence
                + kv_head * stride_v_head
                + value_offsets * stride_v_dim
            )
            value_row = tl.load(value_ptrs, mask=token_valid & value_mask, other=0.0).to(tl.float32)
            score = tl.sum(query_row * key_row, axis=0) * scale
            probability = tl.where(token_valid, tl.exp(score - row_logsumexp), 0.0)
            value_dot = tl.sum(output_gradient_row * value_row, axis=0)
            score_gradient = probability * (value_dot - delta)
            query_gradient_row += score_gradient * scale * key_row
            key_gradient_ptrs = (
                key_gradient
                + (((batch_index * sequence_length + key_position) * kv_heads + kv_head) * HEAD_DIM)
                + head_offsets
            )
            value_gradient_ptrs = (
                value_gradient
                + ((batch_index * sequence_length + key_position) * kv_heads + kv_head) * VALUE_DIM
                + value_offsets
            )
            tl.atomic_add(
                key_gradient_ptrs,
                score_gradient * scale * query_row,
                mask=token_valid & head_mask,
            )
            tl.atomic_add(
                value_gradient_ptrs,
                probability * output_gradient_row,
                mask=token_valid & value_mask,
            )

        tl.store(query_gradient + row * HEAD_DIM + head_offsets, query_gradient_row, mask=head_mask)


class _QSATritonAttention(torch.autograd.Function):
    """Autograd adapter for the single-launch-per-pass Triton prototype."""

    @staticmethod
    def forward(ctx, query, key, value, selected_key_blocks, attention_scale):
        """Launch the online-softmax forward kernel and save backward inputs."""
        batch, sequence_length, query_heads, head_dim = query.shape
        kv_heads = key.shape[2]
        value_dim = value.shape[-1]
        output = torch.empty(
            (batch, sequence_length, query_heads, value_dim),
            dtype=query.dtype,
            device=query.device,
        )
        logsumexp = torch.empty(
            (batch, sequence_length, query_heads), dtype=torch.float32, device=query.device
        )
        block_head_dim = triton.next_power_of_2(head_dim)
        block_value_dim = triton.next_power_of_2(value_dim)
        grid = (batch * sequence_length * query_heads,)
        with torch.cuda.device(query.device):
            _qsa_triton_forward_kernel[grid](
                query,
                key,
                value,
                selected_key_blocks,
                output,
                logsumexp,
                sequence_length,
                query_heads,
                kv_heads,
                selected_key_blocks.shape[-1],
                attention_scale,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *selected_key_blocks.stride(),
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                BLOCK_HEAD_DIM=block_head_dim,
                BLOCK_VALUE_DIM=block_value_dim,
                QSA_BLOCK_SIZE=_QSA_BLOCK_SIZE,
                num_warps=8 if max(head_dim, value_dim) >= 128 else 4,
            )
        ctx.save_for_backward(query, key, value, selected_key_blocks, output, logsumexp)
        ctx.attention_scale = attention_scale
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        """Launch the probability-recomputation Q/K/V backward kernel."""
        query, key, value, selected_key_blocks, output, logsumexp = ctx.saved_tensors
        batch, sequence_length, query_heads, head_dim = query.shape
        kv_heads = key.shape[2]
        value_dim = value.shape[-1]
        output_gradient = output_gradient.contiguous()
        query_gradient = torch.empty_like(query)
        # Shared selected keys/values receive atomic updates. Accumulating into
        # FP32 avoids order-dependent BF16 atomic rounding; autograd converts
        # these tensors to the input dtype at the owning leaf or operation.
        key_gradient = torch.zeros(key.shape, dtype=torch.float32, device=key.device)
        value_gradient = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
        block_head_dim = triton.next_power_of_2(head_dim)
        block_value_dim = triton.next_power_of_2(value_dim)
        grid = (batch * sequence_length * query_heads,)
        with torch.cuda.device(query.device):
            _qsa_triton_backward_kernel[grid](
                query,
                key,
                value,
                selected_key_blocks,
                output,
                output_gradient,
                logsumexp,
                query_gradient,
                key_gradient,
                value_gradient,
                sequence_length,
                query_heads,
                kv_heads,
                selected_key_blocks.shape[-1],
                ctx.attention_scale,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *selected_key_blocks.stride(),
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                BLOCK_HEAD_DIM=block_head_dim,
                BLOCK_VALUE_DIM=block_value_dim,
                QSA_BLOCK_SIZE=_QSA_BLOCK_SIZE,
                num_warps=8 if max(head_dim, value_dim) >= 128 else 4,
            )
        return query_gradient, key_gradient, value_gradient, None, None


def qsa_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    *,
    scale: Optional[float] = None,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Run the experimental single-launch Triton QSA forward and backward.

    The tensor and selection semantics match :func:`qsa_block_sparse_attention`.
    Unlike the indexed-SDPA reference, this prototype reads selected K/V
    tokens indirectly, applies online FP32 softmax, and does not enter Python
    once per query chunk. Backward recomputes probabilities and accumulates
    shared K/V gradients into FP32 buffers before autograd converts them to the
    owning input dtype.

    This path requires CUDA, Triton, dimensions no larger than 256, contiguous
    per-head feature dimensions, and dropout zero. It is a correctness and
    launch-overhead prototype rather than a production kernel: one Triton
    program handles one query head, so grouped-query heads repeat K/V reads,
    and atomics can contend when many queries select the same block. Packed
    sequences, context parallelism, and CUDA graph capture are unsupported.

    By default, index contents are checked for range, causality, and
    uniqueness. That check synchronizes CUDA and sorts the last dimension.
    ``validate_indices=False`` skips only those content checks for indices
    produced by a trusted selector; shape, dtype, device, and attention input
    checks still run.
    """
    if triton is None:
        raise RuntimeError("qsa_triton_attention requires the Triton package")
    if query.device.type != "cuda":
        raise RuntimeError("qsa_triton_attention requires CUDA tensors")
    _validate_qsa_inputs(
        query,
        key,
        value,
        selected_key_blocks,
        query_chunk_size=1,
        validate_indices=validate_indices,
    )
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("qsa_triton_attention requires contiguous head dimensions")
    if query.shape[-1] > 256 or value.shape[-1] > 256:
        raise ValueError("qsa_triton_attention supports query/key and value dimensions up to 256")
    attention_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    return _QSATritonAttention.apply(query, key, value, selected_key_blocks, float(attention_scale))


def _qsa_chunk_positions(
    selected_key_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    key_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gathered key positions and visibility with no sequence-square tensor."""
    batch, chunk_size, _ = selected_key_blocks.shape
    offsets = torch.arange(_QSA_BLOCK_SIZE, device=selected_key_blocks.device)

    selected_positions = (selected_key_blocks[..., None] * _QSA_BLOCK_SIZE + offsets).reshape(
        batch, chunk_size, -1
    )
    complete_mask = (
        (selected_key_blocks >= 0)[..., None]
        .expand(-1, -1, -1, _QSA_BLOCK_SIZE)
        .reshape(batch, chunk_size, -1)
    )

    # A partial current block is always retained. When it becomes complete, it
    # participates only if the indexer selected it as a complete block.
    tail_positions = (query_positions // _QSA_BLOCK_SIZE)[:, None] * _QSA_BLOCK_SIZE + offsets
    tail_mask = ((query_positions + 1) % _QSA_BLOCK_SIZE != 0)[:, None] & (
        tail_positions <= query_positions[:, None]
    )
    positions = torch.cat((selected_positions, tail_positions[None].expand(batch, -1, -1)), dim=-1)
    visible = torch.cat((complete_mask, tail_mask[None].expand(batch, -1, -1)), dim=-1)
    safe_positions = positions.clamp(min=0, max=key_sequence_length - 1)
    return safe_positions, visible


def _qsa_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend one query chunk with explicit FP32 score/value operations."""
    batch, chunk_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    queries_per_kv = num_query_heads // num_kv_heads
    safe_positions, visible = _qsa_chunk_positions(
        selected_key_blocks, query_positions, key.shape[1]
    )

    batch_positions = torch.arange(batch, device=query.device)[:, None, None]
    selected_key = key[batch_positions, safe_positions].float()
    selected_value = value[batch_positions, safe_positions].float()
    grouped_query = query.reshape(batch, chunk_size, num_kv_heads, queries_per_kv, head_dim)
    # CUDA/CPU autocast would otherwise downcast the FP32 einsums to BF16.
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.einsum("bqhgd,bqthd->bqhgt", grouped_query.float(), selected_key)
        logits = logits * scale
        expanded_visible = visible[:, :, None, None, :]
        logits = logits.masked_fill(~expanded_visible, torch.finfo(torch.float32).min)
        probabilities = torch.softmax(logits, dim=-1).masked_fill(~expanded_visible, 0.0)
        output = torch.einsum("bqhgt,bqthv->bqhgv", probabilities, selected_value)
    return output.reshape(batch, chunk_size, num_query_heads, value.shape[-1]).to(query.dtype)


def _qsa_sdpa_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run indexed QSA through PyTorch SDPA for one query chunk."""
    batch, chunk_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    queries_per_kv = num_query_heads // num_kv_heads
    safe_positions, visible = _qsa_chunk_positions(
        selected_key_blocks, query_positions, key.shape[1]
    )
    batch_positions = torch.arange(batch, device=query.device)[:, None, None]
    selected_key = key[batch_positions, safe_positions]
    selected_value = value[batch_positions, safe_positions]
    selected_token_count = safe_positions.shape[-1]
    row_count = batch * chunk_size * num_kv_heads
    query_rows = query.reshape(row_count, queries_per_kv, 1, head_dim)
    key_rows = selected_key.permute(0, 1, 3, 2, 4).reshape(
        row_count, 1, selected_token_count, head_dim
    )
    value_rows = selected_value.permute(0, 1, 3, 2, 4).reshape(
        row_count, 1, selected_token_count, value.shape[-1]
    )

    has_keys = visible.any(dim=-1)
    safe_visible = visible.clone()
    safe_visible[..., 0] |= ~has_keys
    mask_rows = safe_visible[:, :, None, None, None, :].expand(-1, -1, num_kv_heads, -1, -1, -1)
    mask_rows = mask_rows.reshape(row_count, 1, 1, selected_token_count)
    # Autocast would silently change an FP32 request to BF16 SDPA.
    with torch.autocast(device_type=query.device.type, enabled=False):
        output_rows = F.scaled_dot_product_attention(
            query_rows,
            key_rows,
            value_rows,
            attn_mask=mask_rows,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
    output = output_rows.reshape(batch, chunk_size, num_query_heads, value.shape[-1])
    return output * has_keys[:, :, None, None]


def qsa_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    *,
    query_chunk_size: int = 8,
    scale: Optional[float] = None,
    checkpoint_chunks: bool = True,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Compute differentiable QwenAir QSA attention from per-token block indices.

    ``query``, ``key``, and ``value`` have ``[batch, sequence, heads, dim]``
    layout. ``selected_key_blocks`` has ``[batch, sequence, budget]`` layout;
    each entry is the index of a *complete* four-token key block, or ``-1`` for
    padding. The selected blocks are unique for each query token. The current
    incomplete block is always retained, and all future tokens are excluded.
    Query heads are grouped over KV heads, so 24 query and 2 KV heads work
    without repeating KV tensors before the gather. Outputs have query dtype
    and shape ``[batch, sequence, query_heads, value_dim]``.

    The function accumulates scores and values in FP32 and supports BF16/FP32
    inputs. It requires contiguous, unpadded self-attention sequences; packed
    sequences, left padding, context parallelism, dropout, and KV cache are not
    implemented. ``query_chunk_size`` bounds temporary gather size, and
    ``checkpoint_chunks`` recomputes each chunk during backward to reduce
    saved activations. ``validate_indices=False`` skips the synchronizing range,
    causality, and uniqueness checks when indices come from a trusted selector;
    all tensor metadata checks remain enabled. For strict FP32 parity, CUDA TF32
    matmul must be disabled by the caller for the entire forward and backward
    pass. BF16 inputs may run with TF32 enabled, but strict dense-reference
    parity then requires an IEEE test setting. This PyTorch reference is not a
    production sparse kernel.
    """
    _validate_qsa_inputs(
        query,
        key,
        value,
        selected_key_blocks,
        query_chunk_size,
        validate_indices=validate_indices,
    )
    attention_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    outputs = []
    for start in range(0, query.shape[1], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[1])
        query_positions = torch.arange(start, end, device=query.device)
        args = (query[:, start:end], key, value, selected_key_blocks[:, start:end], query_positions)
        if (
            checkpoint_chunks
            and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in (query, key, value))
        ):
            output = checkpoint(
                _qsa_chunk,
                *args,
                attention_scale,
                use_reentrant=False,
            )
        else:
            output = _qsa_chunk(*args, attention_scale)
        outputs.append(output)
    return torch.cat(outputs, dim=1)


def qsa_indexed_sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    *,
    query_chunk_size: int = 8,
    scale: Optional[float] = None,
    checkpoint_chunks: bool = True,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Train with per-token QSA selection using gathered K/V and PyTorch SDPA.

    The input and index contracts match :func:`qsa_block_sparse_attention`.
    Each query attends to at most ``4 * (budget + 1)`` gathered key/value
    tokens, regardless of total sequence length. Checkpointing recomputes the
    gather in backward. PyTorch chooses the SDPA backend; an arbitrary boolean
    selection mask may cause a math fallback even on GPUs with FlashAttention.
    This path is trainable and avoids a sequence-square attention tensor, but
    is not a validated production QSA kernel. Index validation synchronizes the
    GPU and sorts the full selection tensor. ``validate_indices=False`` skips
    only those content checks for a trusted selector. PyTorch must support SDPA
    GQA. CUDA graph capture remains unsupported by this Python chunk loop.
    """
    _validate_qsa_inputs(
        query,
        key,
        value,
        selected_key_blocks,
        query_chunk_size,
        validate_indices=validate_indices,
    )
    attention_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    outputs = []
    for start in range(0, query.shape[1], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[1])
        query_positions = torch.arange(start, end, device=query.device)
        args = (query[:, start:end], key, value, selected_key_blocks[:, start:end], query_positions)
        if (
            checkpoint_chunks
            and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in (query, key, value))
        ):
            output = checkpoint(
                _qsa_sdpa_chunk,
                *args,
                attention_scale,
                use_reentrant=False,
            )
        else:
            output = _qsa_sdpa_chunk(*args, attention_scale)
        outputs.append(output)
    return torch.cat(outputs, dim=1)
