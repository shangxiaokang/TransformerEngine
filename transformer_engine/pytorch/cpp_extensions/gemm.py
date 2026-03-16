# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import ctypes
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from ..tensor.quantized_tensor import Quantizer
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ..tensor.utils import is_experimental
from ..experimental.gemm import experimental_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
    "general_grouped_gemm_for_grouped_tensor",
]


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are experimental tensors -> dispatch to quantizers's qgemm implementation
    if is_experimental(A) or is_experimental(B):
        return experimental_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorBase) or isinstance(B, Float8BlockwiseQTensorBase):
        # There is not use_split_accumulator == False
        # implementation for Float8BlockwiseQTensorBase GEMM
        use_split_accumulator = True

        # Check that data format is supported
        if (
            A._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
            or B._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
        ):
            raise RuntimeError("GEMM with Float8BlockwiseQTensor requires GEMM_READY format")

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input


@functools.lru_cache(maxsize=None)
def get_grouped_gemm_setup_workspace_size(num_tensors: int) -> int:
    """Return workspace size for grouped GEMM pointer setup.
    Must match GroupedGemmSetupWorkspace::required_setup_size in cublaslt_grouped_gemm.cu.
    """
    ptr_bytes = ctypes.sizeof(ctypes.c_void_p)
    int_bytes = ctypes.sizeof(ctypes.c_int)
    ptr_size = num_tensors * ptr_bytes
    int_size = num_tensors * int_bytes
    k_ptr_alignment = 16
    # Each pointer array is placed at a 16-byte-aligned offset (matching kPtrAlignment in C++).
    # aligned_ptr_size = round_up(num_tensors * ptr_bytes, 16)
    aligned_ptr_size = ((ptr_size + k_ptr_alignment - 1) // k_ptr_alignment) * k_ptr_alignment
    size = 8 * aligned_ptr_size + 6 * int_size
    alignment = 256
    return ((size + alignment - 1) // alignment) * alignment


def general_grouped_gemm_for_grouped_tensor(
    A,
    B,
    out,
    *,
    layout: str = "TN",
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    bias=None,
    grad: bool = False,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Grouped GEMM using GroupedTensor inputs.

    This uses nvte_grouped_gemm and supports different per-matrix shapes.

    The caller must ensure that GroupedTensor metadata is already compatible with the
    underlying GEMM implementation (e.g., aligned offsets and output metadata layout).
    """
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    if grad:
        raise NotImplementedError("grad is not supported for grouped_tensor GEMM yet.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    is_discrete_out = isinstance(out, list)
    is_discrete_in = isinstance(A, list)
    if is_discrete_in and is_discrete_out:
        raise ValueError("Both A and out are discrete. This is not supported yet.")

    if is_discrete_out:
        # wgrad case.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_out
    elif is_discrete_in:
        # Use-case: forward pass with list of weights.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_in
    else:
        # Use-case: Single Grouped Parameter for Weight/ Weight Grads.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_grouped_tensor

    if is_discrete_out and bias is not None:
        raise ValueError(
            "Bias is not supported when out is a list (discrete_out mode) yet. "
            "Apply bias manually after the GEMM."
        )

    num_tensors = B.num_tensors
    rowwise = B.rowwise_data
    device = rowwise.device if rowwise is not None else B.columnwise_data.device

    if alpha is None:
        alpha = torch.ones(num_tensors, dtype=torch.float32, device=device)
    if beta is None:
        if accumulate:
            beta = torch.ones(num_tensors, dtype=torch.float32, device=device)
        else:
            beta = torch.zeros(num_tensors, dtype=torch.float32, device=device)

    if not alpha.is_cuda or not beta.is_cuda:
        raise ValueError("alpha and beta must be CUDA tensors.")

    workspace_setup = torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )
    workspace_cublas = torch.empty(
        get_cublas_workspace_size_bytes(),
        dtype=torch.uint8,
        device=device,
    )

    sm_count = get_sm_count()
    sm_count = sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))

    return grouped_gemm_impl(
        A,
        transa,
        B,
        transb,
        out,
        bias,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        sm_count,
    )
