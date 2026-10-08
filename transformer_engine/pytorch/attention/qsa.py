# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from torch.utils.checkpoint import checkpoint


_QSA_BLOCK_SIZE = 4


def _validate_qsa_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    query_chunk_size: int,
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

    # A nonnegative index identifies a *complete* four-token key block visible
    # to this query token. The current incomplete block is handled separately.
    valid = selected_key_blocks >= 0
    if torch.any(selected_key_blocks < -1) or torch.any(
        selected_key_blocks >= math.ceil(seq_len / _QSA_BLOCK_SIZE)
    ):
        raise ValueError("selected key block indices are outside the sequence")
    query_positions = torch.arange(seq_len, device=query.device)[None, :, None]
    if torch.any(
        valid
        & (selected_key_blocks * _QSA_BLOCK_SIZE + _QSA_BLOCK_SIZE - 1 > query_positions)
    ):
        raise ValueError("selected key blocks must be complete, causal blocks or -1")
    if selected_key_blocks.shape[-1] > 1:
        sorted_blocks = selected_key_blocks.sort(dim=-1).values
        duplicate_blocks = (sorted_blocks[..., 1:] == sorted_blocks[..., :-1]) & (
            sorted_blocks[..., 1:] >= 0
        )
        if torch.any(duplicate_blocks):
            raise ValueError("selected key blocks must be unique for each query token")


def _qsa_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend one query chunk to its selected complete blocks and current tail."""
    batch, chunk_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    queries_per_kv = num_query_heads // num_kv_heads
    offsets = torch.arange(_QSA_BLOCK_SIZE, device=query.device)

    selected_positions = (
        selected_key_blocks[..., None] * _QSA_BLOCK_SIZE + offsets
    ).reshape(batch, chunk_size, -1)
    complete_mask = (
        selected_key_blocks >= 0
    )[..., None].expand(-1, -1, -1, _QSA_BLOCK_SIZE).reshape(batch, chunk_size, -1)

    # A partial current block is always retained. When it becomes complete, it
    # participates only if the indexer selected it as a complete block.
    tail_positions = (query_positions // _QSA_BLOCK_SIZE)[:, None] * _QSA_BLOCK_SIZE + offsets
    tail_mask = ((query_positions + 1) % _QSA_BLOCK_SIZE != 0)[:, None] & (
        tail_positions <= query_positions[:, None]
    )
    positions = torch.cat(
        (selected_positions, tail_positions[None].expand(batch, -1, -1)), dim=-1
    )
    visible = torch.cat((complete_mask, tail_mask[None].expand(batch, -1, -1)), dim=-1)

    batch_positions = torch.arange(batch, device=query.device)[:, None, None]
    safe_positions = positions.clamp(min=0, max=key.shape[1] - 1)
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


def qsa_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_key_blocks: torch.Tensor,
    *,
    query_chunk_size: int = 8,
    scale: Optional[float] = None,
    checkpoint_chunks: bool = True,
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
    saved activations. This PyTorch reference is not a production sparse kernel.
    """
    _validate_qsa_inputs(query, key, value, selected_key_blocks, query_chunk_size)
    attention_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    outputs = []
    for start in range(0, query.shape[1], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[1])
        query_positions = torch.arange(start, end, device=query.device)
        args = (query[:, start:end], key, value, selected_key_blocks[:, start:end], query_positions)
        if checkpoint_chunks and torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value)
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
