/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

void fused_multi_row_padding(at::Tensor input, at::Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list) {
  NVTE_CHECK(input_row_list.size() == padded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal  2.");

  const auto num_tensors = input_row_list.size();
  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<DType> input_type_list;
  void* d_input_ptr = reinterpret_cast<void*>(input.data_ptr());
  void* d_output_ptr = reinterpret_cast<void*>(output.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    // Move the input pointer to the next split.
    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset =
        input_row_list[tensor_id] * input.size(1) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back({input_row_list[tensor_id], static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    // Move the output pointer to the next split.
    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset =
        padded_input_row_list[tensor_id] * output.size(1) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back(
        {padded_input_row_list[tensor_id], static_cast<size_t>(output.size(1))});
  }

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> padded_num_rows_list;
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_list.emplace_back(
        make_tensor(input_dptr_list[i], input_shape_list[i], input_type_list[i]));
    nvte_output_list.emplace_back(
        make_tensor(output_dptr_list[i], output_shape_list[i], input_type_list[i]));
    padded_num_rows_list.emplace_back(padded_input_row_list[i]);
  }

  // Check tensor lists
  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");
  NVTE_CHECK(padded_num_rows_list.size() == nvte_input_list.size() &&
             "Number of input and padded row list must match");

  // Launch TE kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_padding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                       padded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
  });
}

void fused_multi_row_padding_pair(at::Tensor input_a, at::Tensor input_b, at::Tensor output_a,
                                  at::Tensor output_b, std::vector<size_t> input_row_list,
                                  std::vector<size_t> padded_input_row_list) {
  NVTE_CHECK(input_row_list.size() == padded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input_a.dim() == 2, "Dimension of first input must equal 2.");
  NVTE_CHECK(input_b.dim() == 2, "Dimension of second input must equal 2.");
  NVTE_CHECK(output_a.dim() == 2, "Dimension of first output must equal 2.");
  NVTE_CHECK(output_b.dim() == 2, "Dimension of second output must equal 2.");
  NVTE_CHECK(input_a.size(0) == input_b.size(0), "Paired inputs must have matching rows.");
  NVTE_CHECK(output_a.size(0) == output_b.size(0), "Paired outputs must have matching rows.");

  const auto num_tensors = input_row_list.size();
  std::vector<void*> input_a_dptr_list, output_a_dptr_list, input_b_dptr_list, output_b_dptr_list;
  std::vector<std::vector<size_t>> input_a_shape_list, output_a_shape_list, input_b_shape_list,
      output_b_shape_list;
  const DType input_a_type = GetTransformerEngineDType(input_a.scalar_type());
  const DType input_b_type = GetTransformerEngineDType(input_b.scalar_type());

  void* d_input_a_ptr = reinterpret_cast<void*>(input_a.data_ptr());
  void* d_output_a_ptr = reinterpret_cast<void*>(output_a.data_ptr());
  void* d_input_b_ptr = reinterpret_cast<void*>(input_b.data_ptr());
  void* d_output_b_ptr = reinterpret_cast<void*>(output_b.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_a_dptr_list.push_back(d_input_a_ptr);
    output_a_dptr_list.push_back(d_output_a_ptr);
    input_b_dptr_list.push_back(d_input_b_ptr);
    output_b_dptr_list.push_back(d_output_b_ptr);

    char* input_a_char_ptr = reinterpret_cast<char*>(d_input_a_ptr);
    input_a_char_ptr += input_row_list[tensor_id] * input_a.size(1) * input_a.element_size();
    d_input_a_ptr = reinterpret_cast<void*>(input_a_char_ptr);

    char* input_b_char_ptr = reinterpret_cast<char*>(d_input_b_ptr);
    input_b_char_ptr += input_row_list[tensor_id] * input_b.size(1) * input_b.element_size();
    d_input_b_ptr = reinterpret_cast<void*>(input_b_char_ptr);

    input_a_shape_list.push_back(
        {input_row_list[tensor_id], static_cast<size_t>(input_a.size(1))});
    input_b_shape_list.push_back(
        {input_row_list[tensor_id], static_cast<size_t>(input_b.size(1))});

    char* output_a_char_ptr = reinterpret_cast<char*>(d_output_a_ptr);
    output_a_char_ptr +=
        padded_input_row_list[tensor_id] * output_a.size(1) * output_a.element_size();
    d_output_a_ptr = reinterpret_cast<void*>(output_a_char_ptr);

    char* output_b_char_ptr = reinterpret_cast<char*>(d_output_b_ptr);
    output_b_char_ptr +=
        padded_input_row_list[tensor_id] * output_b.size(1) * output_b.element_size();
    d_output_b_ptr = reinterpret_cast<void*>(output_b_char_ptr);

    output_a_shape_list.push_back(
        {padded_input_row_list[tensor_id], static_cast<size_t>(output_a.size(1))});
    output_b_shape_list.push_back(
        {padded_input_row_list[tensor_id], static_cast<size_t>(output_b.size(1))});
  }

  std::vector<NVTETensor> nvte_input_a_list, nvte_output_a_list, nvte_input_b_list,
      nvte_output_b_list;
  std::vector<TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> padded_num_rows_list;
  for (size_t i = 0; i < input_a_dptr_list.size(); ++i) {
    if (input_a_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_a_list.emplace_back(
        make_tensor(input_a_dptr_list[i], input_a_shape_list[i], input_a_type));
    nvte_output_a_list.emplace_back(
        make_tensor(output_a_dptr_list[i], output_a_shape_list[i], input_a_type));
    nvte_input_b_list.emplace_back(
        make_tensor(input_b_dptr_list[i], input_b_shape_list[i], input_b_type));
    nvte_output_b_list.emplace_back(
        make_tensor(output_b_dptr_list[i], output_b_shape_list[i], input_b_type));
    padded_num_rows_list.emplace_back(padded_input_row_list[i]);
  }

  NVTE_CHECK(nvte_output_a_list.size() == nvte_input_a_list.size(),
             "Number of first input and output tensors must match");
  NVTE_CHECK(nvte_output_b_list.size() == nvte_input_b_list.size(),
             "Number of second input and output tensors must match");
  NVTE_CHECK(nvte_input_a_list.size() == nvte_input_b_list.size(),
             "Number of tensor pairs must match");
  NVTE_CHECK(padded_num_rows_list.size() == nvte_input_a_list.size() &&
             "Number of input and padded row list must match");

  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_padding_pair(nvte_input_a_list.size(), nvte_input_a_list.data(),
                            nvte_output_a_list.data(), nvte_input_b_list.data(),
                            nvte_output_b_list.data(), padded_num_rows_list.data(),
                            at::cuda::getCurrentCUDAStream());
  });
}

void fused_multi_row_unpadding(at::Tensor input, at::Tensor output,
                               std::vector<size_t> input_row_list,
                               std::vector<size_t> unpadded_input_row_list) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input_row_list.size() == unpadded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal  2.");

  const auto num_tensors = input_row_list.size();
  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<transformer_engine::DType> input_type_list;
  void* d_input_ptr = reinterpret_cast<void*>(input.data_ptr());
  void* d_output_ptr = reinterpret_cast<void*>(output.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    // Move the input pointer to the next split.
    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset =
        input_row_list[tensor_id] * input.size(1) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back({input_row_list[tensor_id], static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    // Move the output pointer to the next split.
    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset =
        unpadded_input_row_list[tensor_id] * output.size(1) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back(
        {unpadded_input_row_list[tensor_id], static_cast<size_t>(output.size(1))});
  }

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> unpadded_num_rows_list;
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_list.emplace_back(
        make_tensor(input_dptr_list[i], input_shape_list[i], input_type_list[i]));
    nvte_output_list.emplace_back(
        make_tensor(output_dptr_list[i], output_shape_list[i], input_type_list[i]));
    unpadded_num_rows_list.emplace_back(unpadded_input_row_list[i]);
  }

  // Check tensor lists
  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");
  NVTE_CHECK(unpadded_num_rows_list.size() == nvte_input_list.size() &&
             "Number of input and padded row list must match");

  // Launch TE kernel
  nvte_multi_unpadding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                       unpadded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
}

void fused_multi_row_unpadding_pair(at::Tensor input_a, at::Tensor input_b, at::Tensor output_a,
                                    at::Tensor output_b, std::vector<size_t> input_row_list,
                                    std::vector<size_t> unpadded_input_row_list) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input_row_list.size() == unpadded_input_row_list.size(),
             "Number of input row list and unpadded row list must match.");
  NVTE_CHECK(input_a.dim() == 2, "Dimension of first input must equal 2.");
  NVTE_CHECK(input_b.dim() == 2, "Dimension of second input must equal 2.");
  NVTE_CHECK(output_a.dim() == 2, "Dimension of first output must equal 2.");
  NVTE_CHECK(output_b.dim() == 2, "Dimension of second output must equal 2.");
  NVTE_CHECK(input_a.size(0) == input_b.size(0), "Paired inputs must have matching rows.");
  NVTE_CHECK(output_a.size(0) == output_b.size(0), "Paired outputs must have matching rows.");

  const auto num_tensors = input_row_list.size();
  std::vector<void*> input_a_dptr_list, output_a_dptr_list, input_b_dptr_list, output_b_dptr_list;
  std::vector<std::vector<size_t>> input_a_shape_list, output_a_shape_list, input_b_shape_list,
      output_b_shape_list;
  const DType input_a_type = GetTransformerEngineDType(input_a.scalar_type());
  const DType input_b_type = GetTransformerEngineDType(input_b.scalar_type());

  void* d_input_a_ptr = reinterpret_cast<void*>(input_a.data_ptr());
  void* d_output_a_ptr = reinterpret_cast<void*>(output_a.data_ptr());
  void* d_input_b_ptr = reinterpret_cast<void*>(input_b.data_ptr());
  void* d_output_b_ptr = reinterpret_cast<void*>(output_b.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_a_dptr_list.push_back(d_input_a_ptr);
    output_a_dptr_list.push_back(d_output_a_ptr);
    input_b_dptr_list.push_back(d_input_b_ptr);
    output_b_dptr_list.push_back(d_output_b_ptr);

    char* input_a_char_ptr = reinterpret_cast<char*>(d_input_a_ptr);
    input_a_char_ptr += input_row_list[tensor_id] * input_a.size(1) * input_a.element_size();
    d_input_a_ptr = reinterpret_cast<void*>(input_a_char_ptr);

    char* input_b_char_ptr = reinterpret_cast<char*>(d_input_b_ptr);
    input_b_char_ptr += input_row_list[tensor_id] * input_b.size(1) * input_b.element_size();
    d_input_b_ptr = reinterpret_cast<void*>(input_b_char_ptr);

    input_a_shape_list.push_back(
        {input_row_list[tensor_id], static_cast<size_t>(input_a.size(1))});
    input_b_shape_list.push_back(
        {input_row_list[tensor_id], static_cast<size_t>(input_b.size(1))});

    char* output_a_char_ptr = reinterpret_cast<char*>(d_output_a_ptr);
    output_a_char_ptr +=
        unpadded_input_row_list[tensor_id] * output_a.size(1) * output_a.element_size();
    d_output_a_ptr = reinterpret_cast<void*>(output_a_char_ptr);

    char* output_b_char_ptr = reinterpret_cast<char*>(d_output_b_ptr);
    output_b_char_ptr +=
        unpadded_input_row_list[tensor_id] * output_b.size(1) * output_b.element_size();
    d_output_b_ptr = reinterpret_cast<void*>(output_b_char_ptr);

    output_a_shape_list.push_back(
        {unpadded_input_row_list[tensor_id], static_cast<size_t>(output_a.size(1))});
    output_b_shape_list.push_back(
        {unpadded_input_row_list[tensor_id], static_cast<size_t>(output_b.size(1))});
  }

  std::vector<NVTETensor> nvte_input_a_list, nvte_output_a_list, nvte_input_b_list,
      nvte_output_b_list;
  std::vector<TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> unpadded_num_rows_list;
  for (size_t i = 0; i < input_a_dptr_list.size(); ++i) {
    if (input_a_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_a_list.emplace_back(
        make_tensor(input_a_dptr_list[i], input_a_shape_list[i], input_a_type));
    nvte_output_a_list.emplace_back(
        make_tensor(output_a_dptr_list[i], output_a_shape_list[i], input_a_type));
    nvte_input_b_list.emplace_back(
        make_tensor(input_b_dptr_list[i], input_b_shape_list[i], input_b_type));
    nvte_output_b_list.emplace_back(
        make_tensor(output_b_dptr_list[i], output_b_shape_list[i], input_b_type));
    unpadded_num_rows_list.emplace_back(unpadded_input_row_list[i]);
  }

  NVTE_CHECK(nvte_output_a_list.size() == nvte_input_a_list.size(),
             "Number of first input and output tensors must match");
  NVTE_CHECK(nvte_output_b_list.size() == nvte_input_b_list.size(),
             "Number of second input and output tensors must match");
  NVTE_CHECK(nvte_input_a_list.size() == nvte_input_b_list.size(),
             "Number of tensor pairs must match");
  NVTE_CHECK(unpadded_num_rows_list.size() == nvte_input_a_list.size() &&
             "Number of input and unpadded row list must match");

  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_unpadding_pair(nvte_input_a_list.size(), nvte_input_a_list.data(),
                              nvte_output_a_list.data(), nvte_input_b_list.data(),
                              nvte_output_b_list.data(), unpadded_num_rows_list.data(),
                              at::cuda::getCurrentCUDAStream());
  });
}

}  // namespace transformer_engine::pytorch
