/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace pytorch {

std::vector<py::object> bgrad_quantize(const at::Tensor &grad_output, py::handle quantizer) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output tensor
  auto grad_output_torch = grad_output.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());

  // Construct grad bias tensor
  const int64_t bias_size = static_cast<int64_t>(shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Unquantized impl only requires computing grad bias
  if (quantizer.is_none()) {
    if (product(shape) == 0) {
      grad_bias_torch.zero_();
    } else {
      at::sum_out(grad_bias_torch, grad_output_torch.reshape({-1, bias_size}), {0});
    }
    return {py::cast(std::move(grad_bias_torch)), py::cast(std::move(grad_output_torch))};
  }

  // Construct grad input tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);

  // Trivial impl if tensors are empty
  if (product(shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Check if fused kernel is supported
  bool with_fused_kernel = false;
  if (detail::IsFloat8Quantizers(quantizer.ptr())) {
    auto prop = at::cuda::getCurrentDeviceProperties();
    const size_t sm_arch = 10 * prop->major + prop->minor;
    if (sm_arch >= 100) {
      // Fused kernel for dbias + FP8 cast on SM arch 10.0+
      with_fused_kernel = true;
    } else if (quantizer_cpp->rowwise_usage && quantizer_cpp->columnwise_usage) {
      // Fused kernel for dbias + FP8 cast + FP8 transpose
      with_fused_kernel = true;
    }
  } else if (detail::IsMXFP8Quantizers(quantizer.ptr())) {
    // Fused kernel for dbias + MXFP8 quantize
    with_fused_kernel = true;
  }

  // Apply unfused impl if fused kernel is not supported
  if (!with_fused_kernel) {
    at::sum_out(grad_bias_torch, grad_output_torch.reshape({-1, bias_size}), {0});
    quantizer_cpp->quantize(grad_output_nvte, grad_input_nvte);
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Query workspace size
  TensorWrapper workspace_nvte;
  at::Tensor workspace_torch;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  // Allocate workspace
  if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
    workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
    workspace_nvte = makeTransformerEngineTensor(workspace_torch.data_ptr(), workspace_nvte.shape(),
                                                 workspace_nvte.dtype());
  }

  // Launch fused kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

std::tuple<std::vector<at::Tensor>, std::vector<py::object>> split_bgrad_quantize(
    const at::Tensor &tensor, const std::vector<int> &split_sections,
    std::vector<py::handle> quantizer_list) {
  init_extension();

  const size_t num_splits = split_sections.size();
  NVTE_CHECK(quantizer_list.size() == num_splits, "Expected ", num_splits,
             " quantizers, but got ", quantizer_list.size());
  if (num_splits == 0) {
    return {{}, {}};
  }

  auto input_py = tensor.contiguous();
  NVTE_CHECK(input_py.dim() > 0, "Input tensor has 0 dims");
  auto input_dtype = GetTransformerEngineDType(input_py.scalar_type());

  std::vector<int64_t> split_sections_i64;
  split_sections_i64.reserve(num_splits);
  size_t split_total = 0;
  for (const auto split : split_sections) {
    NVTE_CHECK(split >= 0, "Attempted to split tensor with negative split section: ", split);
    split_sections_i64.push_back(static_cast<int64_t>(split));
    split_total += static_cast<size_t>(split);
  }
  NVTE_CHECK(split_total == static_cast<size_t>(input_py.size(0)),
             "Split sections must sum to input dim 0. Got ", split_total,
             " but input dim 0 is ", input_py.size(0));

  auto split_tensors = input_py.split_with_sizes(split_sections_i64, 0);
  auto fallback = [&]() -> std::tuple<std::vector<at::Tensor>, std::vector<py::object>> {
    std::vector<at::Tensor> grad_bias_list;
    std::vector<py::object> quantized_list;
    grad_bias_list.reserve(num_splits);
    quantized_list.reserve(num_splits);
    for (size_t i = 0; i < num_splits; ++i) {
      auto result = bgrad_quantize(split_tensors[i], quantizer_list[i]);
      grad_bias_list.emplace_back(result[0].cast<at::Tensor>());
      quantized_list.emplace_back(std::move(result[1]));
    }
    return {std::move(grad_bias_list), std::move(quantized_list)};
  };

  std::vector<std::unique_ptr<Quantizer>> quantizer_cpp_list;
  quantizer_cpp_list.reserve(num_splits);
  bool with_blockwise_fused_kernel = true;
  Float8BlockQuantizer *first_blockwise_quantizer = nullptr;
  for (size_t i = 0; i < num_splits; ++i) {
    quantizer_cpp_list.push_back(convert_quantizer(quantizer_list[i]));
    if (!detail::IsFloat8BlockwiseQuantizers(quantizer_list[i].ptr())) {
      with_blockwise_fused_kernel = false;
      break;
    }

    auto *blockwise_quantizer = dynamic_cast<Float8BlockQuantizer *>(quantizer_cpp_list[i].get());
    if (blockwise_quantizer == nullptr ||
        blockwise_quantizer->get_scaling_mode() != NVTE_BLOCK_SCALING_1D) {
      with_blockwise_fused_kernel = false;
      break;
    }

    if (i == 0) {
      first_blockwise_quantizer = blockwise_quantizer;
      continue;
    }

    if (blockwise_quantizer->dtype != first_blockwise_quantizer->dtype ||
        blockwise_quantizer->force_pow_2_scales !=
            first_blockwise_quantizer->force_pow_2_scales ||
        blockwise_quantizer->amax_epsilon != first_blockwise_quantizer->amax_epsilon ||
        blockwise_quantizer->all_gather_usage != first_blockwise_quantizer->all_gather_usage ||
        blockwise_quantizer->rowwise_usage != first_blockwise_quantizer->rowwise_usage ||
        blockwise_quantizer->columnwise_usage != first_blockwise_quantizer->columnwise_usage) {
      with_blockwise_fused_kernel = false;
      break;
    }
  }

  if (!with_blockwise_fused_kernel || first_blockwise_quantizer == nullptr) {
    return fallback();
  }

  std::vector<at::Tensor> grad_bias_torch_list;
  std::vector<TensorWrapper> input_cpp_list;
  std::vector<TensorWrapper> grad_bias_cpp_list;
  grad_bias_torch_list.reserve(num_splits);
  input_cpp_list.reserve(num_splits);
  grad_bias_cpp_list.reserve(num_splits);
  for (size_t i = 0; i < num_splits; ++i) {
    input_cpp_list.emplace_back(makeTransformerEngineTensor(split_tensors[i]));
    const int64_t bias_size = split_tensors[i].size(split_tensors[i].dim() - 1);
    auto grad_bias_torch = allocateTorchTensor(bias_size, input_dtype);
    grad_bias_cpp_list.emplace_back(makeTransformerEngineTensor(grad_bias_torch));
    grad_bias_torch_list.emplace_back(std::move(grad_bias_torch));
  }

  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_dbias_list;
  nvte_tensor_input_list.reserve(num_splits);
  nvte_tensor_dbias_list.reserve(num_splits);
  for (size_t i = 0; i < num_splits; ++i) {
    nvte_tensor_input_list.push_back(input_cpp_list[i].data());
    nvte_tensor_dbias_list.push_back(grad_bias_cpp_list[i].data());
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_dbias(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                     nvte_tensor_dbias_list.data(), stream);
  });

  auto quantized_list = split_quantize(input_py, split_sections, quantizer_list);
  return {std::move(grad_bias_torch_list), std::move(quantized_list)};
}

namespace {

std::vector<py::object> dact_dbias(
    void (*dact_dbias_func)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                            cudaStream_t),
    void (*dact_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
    at::Tensor grad_output_torch, at::Tensor act_input_torch, py::handle quantizer_py) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output and activation input tensors
  grad_output_torch = grad_output_torch.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto output_shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());
  act_input_torch = act_input_torch.contiguous();
  const TensorWrapper &act_input_nvte = makeTransformerEngineTensor(act_input_torch);
  const auto input_shape = getTensorShape(act_input_torch);

  // Construct tensors
  auto quantizer_cpp = convert_quantizer(quantizer_py);
  auto [grad_input_nvte, grad_input_py] =
      quantizer_cpp->create_tensor(input_shape, grad_output_dtype);
  const int64_t bias_size = static_cast<int64_t>(input_shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(output_shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Choose implementation
  enum class Impl {
    UNFUSED,
    FUSED_DACT_DBIAS_QUANTIZE,
    FUSED_DACT_AMAX_FP8,
    FUSED_DACT_AMAX_NVFP4
  };
  Impl impl = Impl::UNFUSED;
  if (detail::IsFloat8Quantizers(quantizer_py.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_DBIAS_QUANTIZE;
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer_py.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer *>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax) {
      // Post-RHT amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_DACT_AMAX_NVFP4;
    }
  }

  // Perform compute
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (impl) {
    case Impl::UNFUSED:
      // Unfused dact, dbias, quantize
      {
        auto [temp_nvte, temp_py] =
            NoneQuantizer(py::none()).create_tensor(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = temp_py.cast<at::Tensor>();
        at::sum_out(grad_bias_torch, temp_torch.reshape({-1, bias_size}), {0});
        quantizer_cpp->quantize(temp_nvte, grad_input_nvte);
        break;
      }
    case Impl::FUSED_DACT_DBIAS_QUANTIZE:
      // Fused dact-dbias-quantize kernel
      {
        // Query workspace size
        TensorWrapper workspace_nvte;
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });

        // Allocate workspace
        at::Tensor workspace_torch;
        if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
          workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
          workspace_nvte = makeTransformerEngineTensor(
              workspace_torch.data_ptr(), workspace_nvte.shape(), workspace_nvte.dtype());
        }

        // Launch kernel
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });
        break;
      }
    case Impl::FUSED_DACT_AMAX_FP8:
      // Fused dact-amax kernel, unfused dbias and FP8 quantize
      {
        auto *fp8_quantizer_cpp =
            dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
        NVTE_CHECK(fp8_quantizer_cpp != nullptr,
                   "Invalid quantizer for fused dact-amax kernel impl");
        auto [temp_nvte, temp_py] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = temp_py.cast<at::Tensor>();
        at::sum_out(grad_bias_torch, temp_torch.reshape({-1, bias_size}), {0});
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte);
        break;
      }
    case Impl::FUSED_DACT_AMAX_NVFP4:
      // Fused dact-amax kernel, unfused dbias and NVFP4 quantize
      {
        auto *nvfp4_quantizer_cpp =
            static_cast<NVFP4Quantizer *>(quantizer_cpp.get());  // Already checked cast is valid
        NVTE_CHECK(nvfp4_quantizer_cpp != nullptr,
                   "Invalid quantizer for fused dact-amax kernel impl");
        auto [temp_nvte, temp_py] = nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(
            grad_input_nvte, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = temp_py.cast<at::Tensor>();
        at::sum_out(grad_bias_torch, temp_torch.reshape({-1, bias_size}), {0});
        nvfp4_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte);
        break;
      }
    default:
      NVTE_ERROR("Invalid implementation");
  }

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

}  // namespace

std::vector<py::object> dbias_dgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dgelu, nvte_dgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsilu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsilu, nvte_dsilu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_drelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_drelu, nvte_drelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dqgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dqgelu, nvte_dqgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsrelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsrelu, nvte_dsrelu, grad_output, act_input, quantizer);
}

}  // namespace pytorch
}  // namespace transformer_engine
