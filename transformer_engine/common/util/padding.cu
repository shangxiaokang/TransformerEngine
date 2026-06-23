/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/padding.h>

#include <cfloat>
#include <iostream>
#include <vector>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// Parameters to tune
constexpr int n_warps_per_tile = 4;
constexpr int threads_per_block = THREADS_PER_WARP * n_warps_per_tile;
constexpr int desired_load_store_size = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

struct MultiPaddingArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix heights (padded)
  int padded_num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

struct MultiPaddingPairArgs {
  // (input) Data buffers for first tensor list
  void* input_a_list[kMaxTensorsPerKernel];
  // (output) Data buffers for first tensor list
  void* output_a_list[kMaxTensorsPerKernel];
  // (input) Data buffers for second tensor list
  void* input_b_list[kMaxTensorsPerKernel];
  // (output) Data buffers for second tensor list
  void* output_b_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix heights (padded)
  int padded_num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths for first tensor list
  int row_length_a_list[kMaxTensorsPerKernel];
  // Input matrix widths for second tensor list
  int row_length_b_list[kMaxTensorsPerKernel];
  // Number of tiles for the first tensor in each pair
  int a_num_tiles_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each tensor pair
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensor pairs being processed by kernel
  int num_tensors;
};

template <int nvec, typename Type>
__global__ void __launch_bounds__(threads_per_block) multi_padding_kernel(MultiPaddingArgs args) {
  using Vec = Vec<Type, nvec>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = n_warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec x nvec subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec;

  // Number of nvec x nvec subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const Type* input = reinterpret_cast<const Type*>(args.input_list[tensor_id]);
  Type* output = reinterpret_cast<Type*>(args.output_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int padded_num_rows = args.padded_num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  Type local_zero = static_cast<Type>(0.f);
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec; ++i2) {
      const int row = tile_row + i1 * nvec + i2;
      const int col = tile_col + j1 * nvec;
      Vec local_input;
      Vec local_output;
      local_input.clear();
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec; ++j2) {
        local_output.data.elt[j2] = local_input.data.elt[j2];
      }
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_output.data.elt[j2];
          }
        }
      } else if (row < padded_num_rows) {
        // padding
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_zero;
          }
        }
      }
    }
  }
}

template <int nvec, typename Type>
__global__ void __launch_bounds__(threads_per_block) multi_unpadding_kernel(MultiPaddingArgs args) {
  using Vec = Vec<Type, nvec>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = n_warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec x nvec subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec;

  // Number of nvec x nvec subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const Type* input = reinterpret_cast<const Type*>(args.input_list[tensor_id]);
  Type* output = reinterpret_cast<Type*>(args.output_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  Type local_zero = static_cast<Type>(0.f);
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec; ++i2) {
      const int row = tile_row + i1 * nvec + i2;
      const int col = tile_col + j1 * nvec;
      Vec local_input;
      Vec local_output;
      local_input.clear();
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec; ++j2) {
        local_output.data.elt[j2] = local_input.data.elt[j2];
      }
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_output.data.elt[j2];
          }
        }
      }
    }
  }
}

template <int nvec, typename Type, bool do_padding>
__device__ void multi_padding_pair_copy_tile(const Type* input, Type* output, int num_rows,
                                             int padded_num_rows, int row_length, int tile_id) {
  using Vector = Vec<Type, nvec>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = n_warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;

  // Input tensors are divided into tiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec;

  // Number of nvec x nvec subtiles for each thread to load/store
  constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  Type local_zero = static_cast<Type>(0.f);
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec; ++i2) {
      const int row = tile_row + i1 * nvec + i2;
      const int col = tile_col + j1 * nvec;
      Vector local_input;
      local_input.clear();
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_input.data.elt[j2];
          }
        }
      } else if (do_padding && row < padded_num_rows) {
        for (int j2 = 0; j2 < nvec; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_zero;
          }
        }
      }
    }
  }
}

template <int nvec_a, typename TypeA, int nvec_b, typename TypeB, bool do_padding>
__global__ void __launch_bounds__(threads_per_block)
    multi_padding_pair_kernel(MultiPaddingPairArgs args) {
  const int bid = blockIdx.x;

  // Find tensor pair corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }

  const int local_tile_id = bid - args.block_range[tensor_id];
  const int num_rows = args.num_rows_list[tensor_id];
  const int padded_num_rows = args.padded_num_rows_list[tensor_id];

  if (local_tile_id < args.a_num_tiles_list[tensor_id]) {
    const TypeA* input =
        reinterpret_cast<const TypeA*>(args.input_a_list[tensor_id]);
    TypeA* output = reinterpret_cast<TypeA*>(args.output_a_list[tensor_id]);
    multi_padding_pair_copy_tile<nvec_a, TypeA, do_padding>(
        input, output, num_rows, padded_num_rows, args.row_length_a_list[tensor_id],
        local_tile_id);
  } else {
    const TypeB* input =
        reinterpret_cast<const TypeB*>(args.input_b_list[tensor_id]);
    TypeB* output = reinterpret_cast<TypeB*>(args.output_b_list[tensor_id]);
    multi_padding_pair_copy_tile<nvec_b, TypeB, do_padding>(
        input, output, num_rows, padded_num_rows, args.row_length_b_list[tensor_id],
        local_tile_id - args.a_num_tiles_list[tensor_id]);
  }
}

template <bool do_padding>
void launch_multi_padding_pair_kernel(DType type_a, DType type_b,
                                      const MultiPaddingPairArgs& kernel_args,
                                      cudaStream_t stream) {
  const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
  if (n_blocks == 0) {
    return;
  }
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      type_a, TypeA,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          type_b, TypeB, constexpr int nvec_a = desired_load_store_size / sizeof(TypeA);
          constexpr int nvec_b = desired_load_store_size / sizeof(TypeB);
          multi_padding_pair_kernel<nvec_a, TypeA, nvec_b, TypeB, do_padding>
          <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args);););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

void multi_padding(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                   const std::vector<int> padded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type = input_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_padding_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_padding_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == type, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == type, "Output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(output.data.shape[0] == padded_num_rows_list[tensor_id],
               "output tensor shape does not match padded input shape.");
  }

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec x nvec subtiles
  const int tile_dim_m = THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type);
  const int tile_dim_n = THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type);

  // Add tensors to kernel argument struct
  MultiPaddingArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          type, Type, constexpr int nvec = desired_load_store_size / sizeof(Type);
          const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
          multi_padding_kernel<nvec, Type>
          <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
      NVTE_CHECK_CUDA(cudaGetLastError());
      kernel_args.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const int num_rows = input_list[tensor_id]->data.shape[0];
    const int padded_num_rows = padded_num_rows_list[tensor_id];
    const int row_length = input_list[tensor_id]->data.shape[1];
    const int num_tiles_m = (padded_num_rows + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.padded_num_rows_list[pos] = padded_num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        type, Type, constexpr int nvec = desired_load_store_size / sizeof(Type);
        const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
        multi_padding_kernel<nvec, Type>
        <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void multi_unpadding(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                     const std::vector<int> unpadded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type = input_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_unpadding_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_unpadding_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == type, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == type, "Output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(output.data.shape[0] == unpadded_num_rows_list[tensor_id],
               "output tensor shape does not match padded input shape.");
  }

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec x nvec subtiles
  const int tile_dim_m = THREADS_PER_WARP * desired_load_store_size / typeToSize(type);
  const int tile_dim_n = THREADS_PER_WARP * desired_load_store_size / typeToSize(type);

  // Add tensors to kernel argument struct
  MultiPaddingArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          type, Type, constexpr int nvec = desired_load_store_size / sizeof(Type);
          const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
          multi_unpadding_kernel<nvec, Type>
          <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
      NVTE_CHECK_CUDA(cudaGetLastError());
      kernel_args.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const int num_rows = unpadded_num_rows_list[tensor_id];
    const int row_length = input_list[tensor_id]->data.shape[1];
    const int num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        type, Type, constexpr int nvec = desired_load_store_size / sizeof(Type);
        const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
        multi_unpadding_kernel<nvec, Type>
        <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void multi_padding_pair(const std::vector<Tensor*>& input_a_list,
                        const std::vector<Tensor*>& output_a_list,
                        const std::vector<Tensor*>& input_b_list,
                        const std::vector<Tensor*>& output_b_list,
                        const std::vector<int>& padded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_a_list.size() == input_a_list.size(),
             "Number of first input and output tensors must match");
  NVTE_CHECK(input_b_list.size() == input_a_list.size(),
             "Number of first and second input tensors must match");
  NVTE_CHECK(output_b_list.size() == input_a_list.size(),
             "Number of first input and second output tensors must match");
  NVTE_CHECK(padded_num_rows_list.size() == input_a_list.size(),
             "Number of input and padded row list must match");
  if (input_a_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type_a = input_a_list[0]->data.dtype;
  DType type_b = input_b_list[0]->data.dtype;
  NVTE_CHECK(is_high_precision_dtype(type_a),
             "First tensor list must have FP32, FP16, or BF16 dtype.");
  NVTE_CHECK(is_high_precision_dtype(type_b),
             "Second tensor list must have FP32, FP16, or BF16 dtype.");
  for (size_t tensor_id = 0; tensor_id < input_a_list.size(); ++tensor_id) {
    const auto& input_a = *input_a_list[tensor_id];
    const auto& output_a = *output_a_list[tensor_id];
    const auto& input_b = *input_b_list[tensor_id];
    const auto& output_b = *output_b_list[tensor_id];
    CheckInputTensor(input_a, "multi_padding_pair_input_a_" + std::to_string(tensor_id));
    CheckInputTensor(output_a, "multi_padding_pair_output_a_" + std::to_string(tensor_id));
    CheckInputTensor(input_b, "multi_padding_pair_input_b_" + std::to_string(tensor_id));
    CheckInputTensor(output_b, "multi_padding_pair_output_b_" + std::to_string(tensor_id));

    NVTE_CHECK(input_a.data.dtype == type_a, "First input tensor types do not match.");
    NVTE_CHECK(output_a.data.dtype == type_a, "First output tensor types do not match.");
    NVTE_CHECK(input_b.data.dtype == type_b, "Second input tensor types do not match.");
    NVTE_CHECK(output_b.data.dtype == type_b, "Second output tensor types do not match.");

    NVTE_CHECK(input_a.data.shape.size() == 2, "First input tensor must have 2 dimensions.");
    NVTE_CHECK(output_a.data.shape.size() == 2, "First output tensor must have 2 dimensions.");
    NVTE_CHECK(input_b.data.shape.size() == 2, "Second input tensor must have 2 dimensions.");
    NVTE_CHECK(output_b.data.shape.size() == 2, "Second output tensor must have 2 dimensions.");
    NVTE_CHECK(input_b.data.shape[0] == input_a.data.shape[0],
               "Paired input tensors must have matching rows.");
    NVTE_CHECK(output_a.data.shape[0] == padded_num_rows_list[tensor_id],
               "First output tensor shape does not match padded input shape.");
    NVTE_CHECK(output_b.data.shape[0] == padded_num_rows_list[tensor_id],
               "Second output tensor shape does not match padded input shape.");
  }

  // Input matrices are divided into tiles
  const int tile_dim_m_a =
      THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type_a);
  const int tile_dim_n_a =
      THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type_a);
  const int tile_dim_m_b =
      THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type_b);
  const int tile_dim_n_b =
      THREADS_PER_WARP * desired_load_store_size * 8 / typeToNumBits(type_b);

  // Add tensors to kernel argument struct
  MultiPaddingPairArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_a_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      launch_multi_padding_pair_kernel<true>(type_a, type_b, kernel_args, stream);
      kernel_args.num_tensors = 0;
      kernel_args.block_range[0] = 0;
    }

    // Calculate number of thread blocks needed for tensor pair
    const int num_rows = input_a_list[tensor_id]->data.shape[0];
    const int padded_num_rows = padded_num_rows_list[tensor_id];
    const int row_length_a = input_a_list[tensor_id]->data.shape[1];
    const int row_length_b = input_b_list[tensor_id]->data.shape[1];
    const int num_tiles_m_a = (padded_num_rows + tile_dim_m_a - 1) / tile_dim_m_a;
    const int num_tiles_n_a = (row_length_a + tile_dim_n_a - 1) / tile_dim_n_a;
    const int num_tiles_a = num_tiles_m_a * num_tiles_n_a;
    const int num_tiles_m_b = (padded_num_rows + tile_dim_m_b - 1) / tile_dim_m_b;
    const int num_tiles_n_b = (row_length_b + tile_dim_n_b - 1) / tile_dim_n_b;
    const int num_tiles_b = num_tiles_m_b * num_tiles_n_b;

    // Add tensor pair to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_a_list[pos] = const_cast<void*>(input_a_list[tensor_id]->data.dptr);
    kernel_args.output_a_list[pos] = output_a_list[tensor_id]->data.dptr;
    kernel_args.input_b_list[pos] = const_cast<void*>(input_b_list[tensor_id]->data.dptr);
    kernel_args.output_b_list[pos] = output_b_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.padded_num_rows_list[pos] = padded_num_rows;
    kernel_args.row_length_a_list[pos] = row_length_a;
    kernel_args.row_length_b_list[pos] = row_length_b;
    kernel_args.a_num_tiles_list[pos] = num_tiles_a;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles_a + num_tiles_b;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    launch_multi_padding_pair_kernel<true>(type_a, type_b, kernel_args, stream);
  }
}

void multi_unpadding_pair(const std::vector<Tensor*>& input_a_list,
                          const std::vector<Tensor*>& output_a_list,
                          const std::vector<Tensor*>& input_b_list,
                          const std::vector<Tensor*>& output_b_list,
                          const std::vector<int>& unpadded_num_rows_list,
                          cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_a_list.size() == input_a_list.size(),
             "Number of first input and output tensors must match");
  NVTE_CHECK(input_b_list.size() == input_a_list.size(),
             "Number of first and second input tensors must match");
  NVTE_CHECK(output_b_list.size() == input_a_list.size(),
             "Number of first input and second output tensors must match");
  NVTE_CHECK(unpadded_num_rows_list.size() == input_a_list.size(),
             "Number of input and unpadded row list must match");
  if (input_a_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType type_a = input_a_list[0]->data.dtype;
  DType type_b = input_b_list[0]->data.dtype;
  NVTE_CHECK(is_high_precision_dtype(type_a),
             "First tensor list must have FP32, FP16, or BF16 dtype.");
  NVTE_CHECK(is_high_precision_dtype(type_b),
             "Second tensor list must have FP32, FP16, or BF16 dtype.");
  for (size_t tensor_id = 0; tensor_id < input_a_list.size(); ++tensor_id) {
    const auto& input_a = *input_a_list[tensor_id];
    const auto& output_a = *output_a_list[tensor_id];
    const auto& input_b = *input_b_list[tensor_id];
    const auto& output_b = *output_b_list[tensor_id];
    CheckInputTensor(input_a, "multi_unpadding_pair_input_a_" + std::to_string(tensor_id));
    CheckInputTensor(output_a, "multi_unpadding_pair_output_a_" + std::to_string(tensor_id));
    CheckInputTensor(input_b, "multi_unpadding_pair_input_b_" + std::to_string(tensor_id));
    CheckInputTensor(output_b, "multi_unpadding_pair_output_b_" + std::to_string(tensor_id));

    NVTE_CHECK(input_a.data.dtype == type_a, "First input tensor types do not match.");
    NVTE_CHECK(output_a.data.dtype == type_a, "First output tensor types do not match.");
    NVTE_CHECK(input_b.data.dtype == type_b, "Second input tensor types do not match.");
    NVTE_CHECK(output_b.data.dtype == type_b, "Second output tensor types do not match.");

    NVTE_CHECK(input_a.data.shape.size() == 2, "First input tensor must have 2 dimensions.");
    NVTE_CHECK(output_a.data.shape.size() == 2, "First output tensor must have 2 dimensions.");
    NVTE_CHECK(input_b.data.shape.size() == 2, "Second input tensor must have 2 dimensions.");
    NVTE_CHECK(output_b.data.shape.size() == 2, "Second output tensor must have 2 dimensions.");
    NVTE_CHECK(input_b.data.shape[0] == input_a.data.shape[0],
               "Paired input tensors must have matching rows.");
    NVTE_CHECK(output_a.data.shape[0] == unpadded_num_rows_list[tensor_id],
               "First output tensor shape does not match unpadded input shape.");
    NVTE_CHECK(output_b.data.shape[0] == unpadded_num_rows_list[tensor_id],
               "Second output tensor shape does not match unpadded input shape.");
  }

  // Input matrices are divided into tiles
  const int tile_dim_m_a = THREADS_PER_WARP * desired_load_store_size / typeToSize(type_a);
  const int tile_dim_n_a = THREADS_PER_WARP * desired_load_store_size / typeToSize(type_a);
  const int tile_dim_m_b = THREADS_PER_WARP * desired_load_store_size / typeToSize(type_b);
  const int tile_dim_n_b = THREADS_PER_WARP * desired_load_store_size / typeToSize(type_b);

  // Add tensors to kernel argument struct
  MultiPaddingPairArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_a_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      launch_multi_padding_pair_kernel<false>(type_a, type_b, kernel_args, stream);
      kernel_args.num_tensors = 0;
      kernel_args.block_range[0] = 0;
    }

    // Calculate number of thread blocks needed for tensor pair
    const int num_rows = unpadded_num_rows_list[tensor_id];
    const int padded_num_rows = input_a_list[tensor_id]->data.shape[0];
    const int row_length_a = input_a_list[tensor_id]->data.shape[1];
    const int row_length_b = input_b_list[tensor_id]->data.shape[1];
    const int num_tiles_m_a = (num_rows + tile_dim_m_a - 1) / tile_dim_m_a;
    const int num_tiles_n_a = (row_length_a + tile_dim_n_a - 1) / tile_dim_n_a;
    const int num_tiles_a = num_tiles_m_a * num_tiles_n_a;
    const int num_tiles_m_b = (num_rows + tile_dim_m_b - 1) / tile_dim_m_b;
    const int num_tiles_n_b = (row_length_b + tile_dim_n_b - 1) / tile_dim_n_b;
    const int num_tiles_b = num_tiles_m_b * num_tiles_n_b;

    // Add tensor pair to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_a_list[pos] = const_cast<void*>(input_a_list[tensor_id]->data.dptr);
    kernel_args.output_a_list[pos] = output_a_list[tensor_id]->data.dptr;
    kernel_args.input_b_list[pos] = const_cast<void*>(input_b_list[tensor_id]->data.dptr);
    kernel_args.output_b_list[pos] = output_b_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.padded_num_rows_list[pos] = padded_num_rows;
    kernel_args.row_length_a_list[pos] = row_length_a;
    kernel_args.row_length_b_list[pos] = row_length_b;
    kernel_args.a_num_tiles_list[pos] = num_tiles_a;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles_a + num_tiles_b;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    launch_multi_padding_pair_kernel<false>(type_a, type_b, kernel_args, stream);
  }
}

}  // namespace transformer_engine

void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_padding);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  std::vector<int> padded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
    padded_num_rows_list_.push_back(padded_num_rows_list[i]);
  }
  multi_padding(input_list_, output_list_, padded_num_rows_list_, stream);
}

void nvte_multi_padding_pair(size_t num_tensors, const NVTETensor* input_a_list,
                             NVTETensor* output_a_list, const NVTETensor* input_b_list,
                             NVTETensor* output_b_list, const int* padded_num_rows_list,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_padding_pair);
  using namespace transformer_engine;
  std::vector<Tensor*> input_a_list_, output_a_list_, input_b_list_, output_b_list_;
  std::vector<int> padded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_a_list_.push_back(convertNVTETensorCheck(input_a_list[i]));
    output_a_list_.push_back(convertNVTETensorCheck(output_a_list[i]));
    input_b_list_.push_back(convertNVTETensorCheck(input_b_list[i]));
    output_b_list_.push_back(convertNVTETensorCheck(output_b_list[i]));
    padded_num_rows_list_.push_back(padded_num_rows_list[i]);
  }
  multi_padding_pair(input_a_list_, output_a_list_, input_b_list_, output_b_list_,
                     padded_num_rows_list_, stream);
}

void nvte_multi_unpadding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                          const int* unpadded_num_rows_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_unpadding);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  std::vector<int> unpadded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
    unpadded_num_rows_list_.push_back(unpadded_num_rows_list[i]);
  }
  multi_unpadding(input_list_, output_list_, unpadded_num_rows_list_, stream);
}

void nvte_multi_unpadding_pair(size_t num_tensors, const NVTETensor* input_a_list,
                               NVTETensor* output_a_list, const NVTETensor* input_b_list,
                               NVTETensor* output_b_list, const int* unpadded_num_rows_list,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_unpadding_pair);
  using namespace transformer_engine;
  std::vector<Tensor*> input_a_list_, output_a_list_, input_b_list_, output_b_list_;
  std::vector<int> unpadded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_a_list_.push_back(convertNVTETensorCheck(input_a_list[i]));
    output_a_list_.push_back(convertNVTETensorCheck(output_a_list[i]));
    input_b_list_.push_back(convertNVTETensorCheck(input_b_list[i]));
    output_b_list_.push_back(convertNVTETensorCheck(output_b_list[i]));
    unpadded_num_rows_list_.push_back(unpadded_num_rows_list[i]);
  }
  multi_unpadding_pair(input_a_list_, output_a_list_, input_b_list_, output_b_list_,
                       unpadded_num_rows_list_, stream);
}
