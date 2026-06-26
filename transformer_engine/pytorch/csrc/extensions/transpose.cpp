/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include <optional>
#include <vector>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

at::Tensor fp8_transpose(at::Tensor input, DType otype, std::optional<at::Tensor> output) {
  init_extension();

  // Tensor dimensions
  const auto shape = getTensorShape(input);
  std::vector<int64_t> transpose_shape_int64;
  if (shape.size() > 0) {
    transpose_shape_int64.push_back(shape.back());
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      transpose_shape_int64.push_back(shape[i]);
    }
  }
  const size_t M = shape.size() > 0 ? product(shape) / shape.back() : 1;
  const size_t N = shape.size() > 0 ? shape.back() : 1;

  // Output tensor
  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    out = at::empty(transpose_shape_int64, opts);
  }

  // Return immediately if tensor is empty
  if (M == 0 || N == 0) {
    return out;
  }

  // Compute transpose
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M}, otype);
  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

at::Tensor swap_first_dims(at::Tensor tensor, std::optional<at::Tensor> out) {
  init_extension();

  // Make sure input is contiguous
  const auto &input = tensor.contiguous();

  // Allocate output tensor if needed
  if (!out) {
    auto in_shape = getTensorShape(input);
    NVTE_CHECK(in_shape.size() >= 2, "Invalid input tensor dimensions (shape=", in_shape, ")");
    std::vector<int64_t> out_shape_int64(in_shape.begin(), in_shape.end());
    out_shape_int64[0] = static_cast<int64_t>(in_shape[1]);
    out_shape_int64[1] = static_cast<int64_t>(in_shape[0]);
    auto opts = at::TensorOptions().dtype(input.dtype()).device(input.device());
    out = at::empty(out_shape_int64, opts);
  }

  // Launch kernel
  const TensorWrapper te_input = makeTransformerEngineTensor(input);
  TensorWrapper te_output = makeTransformerEngineTensor(*out);
  nvte_swap_first_dims(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return std::move(*out);
}


py::object fp8_blockwise_transpose(py::object tensor, py::object quantizer) {
  init_extension();

  NVTE_CHECK(!tensor.is_none(), "Tensor has not been provided");
  NVTE_CHECK(detail::IsFloat8BlockwiseQuantizers(quantizer.ptr()),
             "Quantizer must be a Float8BlockwiseQuantizer");

  torch::Tensor torch_tensor = py::cast<torch::Tensor>(tensor);
  auto torch_dtype = torch_tensor.scalar_type();
  auto te_dtype = DType::kBFloat16;
  switch (torch_dtype) {
    case c10::ScalarType::Float:
      te_dtype = DType::kFloat32;
      break;
    case c10::ScalarType::Half:
      te_dtype = DType::kFloat16;
      break;
    case c10::ScalarType::BFloat16:
      te_dtype = DType::kBFloat16;
      break;
    default:
      NVTE_ERROR("Unsupported dtype");
  }

  TensorWrapper te_tensor = makeTransformerEngineTensor(tensor, quantizer);
  NVTE_CHECK(nvte_tensor_scaling_mode(te_tensor.data()) == NVTE_BLOCK_SCALING_1D,
             "Only 1D block scaling is supported for fp8 blockwise transpose");

  auto quantizer_cpp = convert_quantizer(quantizer);
  auto my_quantizer = static_cast<Float8BlockQuantizer *>(quantizer_cpp.get());
  TORCH_CHECK(my_quantizer->force_pow_2_scales,
              "Only power-of-2 scaling is supported for fp8 blockwise transpose");

  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(my_quantizer->force_pow_2_scales);
  quant_config.set_amax_epsilon(my_quantizer->amax_epsilon);

  nvte_transpose_blockwise(te_tensor.data(), quant_config, te_dtype,
                           at::cuda::getCurrentCUDAStream());

  return tensor;
}

}  // namespace pytorch
}  // namespace transformer_engine
