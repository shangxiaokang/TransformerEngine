# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 Padding API"""

from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex

from ..fp8 import FP8GlobalStateManager
from ..jit import no_torch_dynamo


__all__ = ["Fp8Padding", "Fp8PaddingPair"]


class _Fp8Padding(torch.autograd.Function):
    """functional FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible
        in_features = inp.shape[-1]

        # Allocate cast and transpose output tensor
        total_row = sum(padded_m_splits)
        out = torch.empty([total_row, in_features], dtype=inp.dtype, device=inp.device)

        tex.fused_multi_row_padding(inp.view(-1, in_features), out, m_splits, padded_m_splits)

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring

        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            in_features = grad_output.shape[-1]

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )

            tex.fused_multi_row_unpadding(
                grad_output.view(-1, in_features), grad_input, ctx.padded_m_splits, ctx.m_splits
            )

        return (grad_input, None, None, None)


class _Fp8PaddingPair(torch.autograd.Function):
    """functional paired FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp_a: torch.Tensor,
        inp_b: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        in_features_a = inp_a.shape[-1]
        in_features_b = inp_b.shape[-1]

        total_row = sum(padded_m_splits)
        out_a = torch.empty([total_row, in_features_a], dtype=inp_a.dtype, device=inp_a.device)
        out_b = torch.empty([total_row, in_features_b], dtype=inp_b.dtype, device=inp_b.device)

        tex.fused_multi_row_padding_pair(
            inp_a.view(-1, in_features_a),
            inp_b.view(-1, in_features_b),
            out_a,
            out_b,
            m_splits,
            padded_m_splits,
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad_a = inp_a.requires_grad
            ctx.requires_dgrad_b = inp_b.requires_grad

        return out_a, out_b

    @staticmethod
    def backward(ctx, grad_output_a: torch.Tensor, grad_output_b: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input_a = None
        grad_input_b = None

        use_pair_unpadding = (
            ctx.requires_dgrad_a
            and ctx.requires_dgrad_b
            and grad_output_a is not None
            and grad_output_b is not None
        )

        if use_pair_unpadding:
            grad_output_a = grad_output_a.contiguous()
            grad_output_b = grad_output_b.contiguous()
            in_features_a = grad_output_a.shape[-1]
            in_features_b = grad_output_b.shape[-1]
            total_row = sum(ctx.m_splits)
            grad_input_a = torch.empty(
                [total_row, in_features_a], dtype=grad_output_a.dtype, device=grad_output_a.device
            )
            grad_input_b = torch.empty(
                [total_row, in_features_b], dtype=grad_output_b.dtype, device=grad_output_b.device
            )
            tex.fused_multi_row_unpadding_pair(
                grad_output_a.view(-1, in_features_a),
                grad_output_b.view(-1, in_features_b),
                grad_input_a,
                grad_input_b,
                ctx.padded_m_splits,
                ctx.m_splits,
            )
        else:
            if ctx.requires_dgrad_a and grad_output_a is not None:
                grad_output_a = grad_output_a.contiguous()
                in_features_a = grad_output_a.shape[-1]
                total_row = sum(ctx.m_splits)
                grad_input_a = torch.empty(
                    [total_row, in_features_a],
                    dtype=grad_output_a.dtype,
                    device=grad_output_a.device,
                )
                tex.fused_multi_row_unpadding(
                    grad_output_a.view(-1, in_features_a),
                    grad_input_a,
                    ctx.padded_m_splits,
                    ctx.m_splits,
                )
            if ctx.requires_dgrad_b and grad_output_b is not None:
                grad_output_b = grad_output_b.contiguous()
                in_features_b = grad_output_b.shape[-1]
                total_row = sum(ctx.m_splits)
                grad_input_b = torch.empty(
                    [total_row, in_features_b],
                    dtype=grad_output_b.dtype,
                    device=grad_output_b.device,
                )
                tex.fused_multi_row_unpadding(
                    grad_output_b.view(-1, in_features_b),
                    grad_input_b,
                    ctx.padded_m_splits,
                    ctx.m_splits,
                )

        return (grad_input_a, grad_input_b, None, None, None)


class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    align_size : int, optional
                 the alignment size for the input tensor. If not provided, the alignment size will
                 be determined by the FP8 recipe (32 for MXFP8 and 16 for others) in the first
                 forward pass.
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
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Apply the padding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        if self.align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16

        # FP8 padding calculate
        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits:
            return inp, m_splits

        if torch.is_grad_enabled():
            fn = _Fp8Padding.apply
            args = []
        else:
            fn = _Fp8Padding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out, padded_m_splits


class Fp8PaddingPair(torch.nn.Module):
    """
    Apply the padding to two tensors with shared row splits.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simultaneously.
    align_size : int, optional
                 the alignment size for the input tensor. If not provided, the alignment size will
                 be determined by the FP8 recipe (32 for MXFP8 and 16 for others) in the first
                 forward pass.
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
        inp_a: torch.Tensor,
        inp_b: torch.Tensor,
        m_splits: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Apply the padding to paired inputs with identical split rows.

        Parameters
        ----------
        inp_a : torch.Tensor
                First input tensor.
        inp_b : torch.Tensor
                Second input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        if self.align_size is None:
            self.align_size = 32 if FP8GlobalStateManager.get_fp8_recipe().mxfp8() else 16

        padded_m_splits = [
            (m + self.align_size - 1) // self.align_size * self.align_size for m in m_splits
        ]
        # no padding needed
        if m_splits == padded_m_splits:
            return inp_a, inp_b, m_splits

        if torch.is_grad_enabled():
            fn = _Fp8PaddingPair.apply
            args = []
        else:
            fn = _Fp8PaddingPair.forward
            args = [None]

        args += (
            inp_a,
            inp_b,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out_a, out_b = fn(*args)

        return out_a, out_b, padded_m_splits
