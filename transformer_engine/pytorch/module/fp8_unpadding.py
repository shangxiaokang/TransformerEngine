# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ..tensor.float8_blockwise_tensor import Float8BlockwiseQTensor


__all__ = ["Fp8Unpadding"]


class _Fp8Unpadding(torch.autograd.Function):
    """functional FP8 unpadding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        in_features = inp.shape[-1]

        total_row = sum(m_splits)
        out_ret = torch.empty([total_row, in_features], dtype=inp.dtype, device=inp.device)

        tex.fused_multi_row_unpadding(inp.view(-1, in_features), out_ret, padded_m_splits, m_splits)

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out_ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = None
        if ctx.requires_dgrad:
            in_features = grad_output.shape[-1]
            total_row = sum(ctx.padded_m_splits)
            if isinstance(grad_output, Float8BlockwiseQTensorBase):
                if (
                    grad_output._is_2D_scaled
                    or grad_output._rowwise_data is None
                    or grad_output._rowwise_scale_inv is None
                ):
                    raise NotImplementedError(
                        "FP8 unpadding backward only supports 1D rowwise FP8 tensors"
                    )
                grad_data = torch.empty(
                    [total_row, in_features],
                    dtype=grad_output._rowwise_data.dtype,
                    device=grad_output.device,
                )
                tex.fused_multi_row_padding(
                    grad_output._rowwise_data.view(-1, in_features),
                    grad_data,
                    ctx.m_splits,
                    ctx.padded_m_splits,
                )
                if grad_output._is_gemm_ready_format():
                    scale_inv = grad_output._rowwise_scale_inv.transpose(-2, -1).contiguous()
                    grad_scale_inv = torch.empty(
                        [total_row, scale_inv.shape[-1]],
                        dtype=scale_inv.dtype,
                        device=scale_inv.device,
                    )
                    tex.fused_multi_row_padding(
                        scale_inv.view(-1, scale_inv.shape[-1]),
                        grad_scale_inv,
                        ctx.m_splits,
                        ctx.padded_m_splits,
                    )
                    grad_scale_inv = grad_scale_inv.transpose(-2, -1).contiguous()
                else:
                    scale_inv = grad_output._rowwise_scale_inv
                    grad_scale_inv = torch.empty(
                        [total_row, scale_inv.shape[-1]],
                        dtype=scale_inv.dtype,
                        device=scale_inv.device,
                    )
                    tex.fused_multi_row_padding(
                        scale_inv.view(-1, scale_inv.shape[-1]),
                        grad_scale_inv,
                        ctx.m_splits,
                        ctx.padded_m_splits,
                    )
                grad_input = Float8BlockwiseQTensor(
                    shape=grad_data.shape,
                    rowwise_data=grad_data,
                    rowwise_scale_inv=grad_scale_inv,
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=grad_output._fp8_dtype,
                    dtype=grad_output.dtype,
                    quantizer=grad_output._quantizer,
                    is_2D_scaled=False,
                    data_format=grad_output._data_format,
                    requires_grad=grad_output.requires_grad,
                )
            else:
                grad_output = grad_output.contiguous()
                grad_input = torch.empty(
                    [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
                )
                tex.fused_multi_row_padding(
                    grad_output.view(-1, in_features), grad_input, ctx.m_splits, ctx.padded_m_splits
                )

        return (grad_input, None, None, None)


class Fp8Unpadding(torch.nn.Module):
    """
    Apply the unpadding for Grouped GEMM input.
    """

    def __init__(
        self,
        num_gemms: int,
        align_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms
        self.align_size = align_size

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> torch.Tensor:
        """Apply the unpadding to the input."""

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        if self.align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16

        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        if m_splits == padded_m_splits:
            return inp

        if torch.is_grad_enabled():
            fn = _Fp8Unpadding.apply
            args = []
        else:
            fn = _Fp8Unpadding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out
