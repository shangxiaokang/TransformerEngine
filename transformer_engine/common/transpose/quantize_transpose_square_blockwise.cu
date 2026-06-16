/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cuda/barrier>
#include <limits>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/util/cuda_runtime.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "transformer_engine/transpose.h"

#if (!defined(__CUDA_MINIMUM_ARCH__) && __CUDA_ARCH__ >= 900) || \
    (defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ >= 900)
#define TMA_HW_SUPPORTED
#endif

namespace transformer_engine {
namespace {

// const values configuration

constexpr size_t kThreadsPerWarp = 32;
#ifdef TMA_HW_SUPPORTED
constexpr size_t BLOCK_TILE_DIM = 128;
constexpr size_t WARP_TILE_DIM_X = 32;
constexpr size_t WARP_TILE_DIM_Y = 64;
constexpr size_t THREAD_TILE_DIM_X = 16;
constexpr size_t THREAD_TILE_DIM_Y = 4;
#else
constexpr size_t BLOCK_TILE_DIM = 128;
constexpr size_t WARP_TILE_DIM_X = 64;
constexpr size_t WARP_TILE_DIM_Y = 32;
constexpr size_t THREAD_TILE_DIM_X = 8;
constexpr size_t THREAD_TILE_DIM_Y = 8;
#endif

#ifdef TMA_HW_SUPPORTED
constexpr size_t NUM_BYTES_PER_BANK = 4;
constexpr size_t NUM_BANKS_PER_SHARED_ELEM = THREAD_TILE_DIM_Y / NUM_BYTES_PER_BANK;
constexpr size_t SHARED_BLOCK_TILE_DIM_Y = BLOCK_TILE_DIM;
constexpr size_t SHARED_BLOCK_TILE_DIM_X_BANKS =
    BLOCK_TILE_DIM / (NUM_BYTES_PER_BANK * NUM_BANKS_PER_SHARED_ELEM);
constexpr size_t NUM_BANKS_Y_IN_WARP = WARP_TILE_DIM_Y / NUM_BYTES_PER_BANK;
#endif
constexpr size_t ELE_PER_THREAD = THREAD_TILE_DIM_X * THREAD_TILE_DIM_Y;
constexpr size_t THREADS_PER_BLOCK = BLOCK_TILE_DIM * BLOCK_TILE_DIM / ELE_PER_THREAD;
constexpr size_t NUM_WARPS_X_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_X;
constexpr size_t NUM_WARPS_Y_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_Y;
constexpr size_t NUM_WARPS_IN_BLOCK = NUM_WARPS_X_IN_BLOCK * NUM_WARPS_Y_IN_BLOCK;

constexpr size_t NUM_THREADS_X_IN_WARP = WARP_TILE_DIM_X / THREAD_TILE_DIM_X;
constexpr size_t NUM_THREADS_Y_IN_WARP = kThreadsPerWarp / NUM_THREADS_X_IN_WARP;

#define MIN(a, b) (a < b ? a : b)

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    block_scaled_cast_transpose_kernel(const IType* const input, OType* const output_c,
                                       OType* const output_t, CType* const tile_scales_inv_c,
                                       CType* const tile_scales_inv_t, const size_t row_length,
                                       const size_t num_rows, const size_t scale_stride_x,
                                       const size_t scale_stride_y, const size_t scale_t_stride_x,
                                       const size_t scale_t_stride_y, const float epsilon,
                                       const __grid_constant__ CUtensorMap tensor_map_output_t,
                                       bool pow_2_scaling, const float* noop_ptr) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = kReturnTranspose ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  // This is ONLY true if the input is a full tile
  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_idx =
      tile_id_y * BLOCK_TILE_DIM * row_length + tile_id_x * BLOCK_TILE_DIM;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

// Step 1: Load a block tile of input data into thread tiles on registers
#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
    thrd_tile_input[i].load_from(input + thread_tile_start_idx + i * row_length);
  }

  // Step 2: calculate block tile amax and scale
  // Calculate thread_tile amax
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be needed,
  // instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (kReturnTranspose) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  OVecCast tmp_output_c;

  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      // Step 3: Store cast output
      CType scale_data = block_tile_scale;

      OType scaled_elt =
          static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
      tmp_output_c.data.elt[j] = scaled_elt;
      // Step 4: do transpose within thread tile
      if constexpr (kReturnTranspose) {
        thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
      }
    }
    tmp_output_c.store_to(output_c + thread_tile_start_idx + i * row_length);
  }

  // Step 4: store transpose into shared memory
  if constexpr (kReturnTranspose) {
#ifdef TMA_HW_SUPPORTED
    __shared__ alignas(128)
        OVecTrans block_tile_trans_shared[SHARED_BLOCK_TILE_DIM_Y][SHARED_BLOCK_TILE_DIM_X_BANKS];
    OType(*block_tile_trans_shared_otype_ptr)[BLOCK_TILE_DIM] =
        reinterpret_cast<OType(*)[BLOCK_TILE_DIM]>(block_tile_trans_shared);

#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      auto warp_id_in_block_x_ = warp_id_in_block_y;
      auto warp_id_in_block_y_ = warp_id_in_block_x;
      int row_idx = warp_id_in_block_y_ * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
                    tid_in_warp_x * THREAD_TILE_DIM_X + i;
      int col_idx =
          warp_id_in_block_x_ * (NUM_BANKS_Y_IN_WARP / NUM_BANKS_PER_SHARED_ELEM) + tid_in_warp_y;
      block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Step 5: store transpose output
    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t*>(&tensor_map_output_t), tile_id_y * BLOCK_TILE_DIM,
          tile_id_x * BLOCK_TILE_DIM,
          reinterpret_cast<uint64_t*>(block_tile_trans_shared_otype_ptr));
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      ptx::cp_async_bulk_wait_group_read<0>();
    }
#else
    // Step 4 Alternative (when TMA is not available, skip writing to shared memory)
    const size_t block_tile_t_start_idx =
        tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
    const size_t warp_tile_t_start_idx =
        block_tile_t_start_idx +
        warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
        warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
    const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                           tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                           tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      thrd_tile_out_trans[i].store_to(output_t + thread_tile_t_start_idx + i * num_rows);
    }
#endif
  }
}

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) block_scaled_cast_transpose_kernel_notaligned(
    const IType* const input, OType* const output_c, OType* const output_t,
    CType* const tile_scales_inv_c, CType* const tile_scales_inv_t, const size_t row_length,
    const size_t num_rows, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    bool pow_2_scaling, const float* noop_ptr) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = kReturnTranspose ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_row_idx = tile_id_y * BLOCK_TILE_DIM;
  const size_t block_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM;
  const size_t block_tile_start_idx =
      block_tile_start_row_idx * row_length + block_tile_start_col_idx;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  // handle non-full tile
  // check for three cases: full thread tile, nonfull thread tile, empty thread tile
  // for empty thread tile, directly write zero to the transposed shared mem buffer
  // for nonfull thread tile, fill zero to thread tile and act as if it's full
  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  const size_t thread_tile_start_col_idx =
      tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
      tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + THREAD_TILE_DIM_Y - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + THREAD_TILE_DIM_X - 1;

  bool full_thrd_tile =
      (thread_tile_end_row_idx < num_rows) && (thread_tile_end_col_idx < row_length);
  bool empty_thrd_tile =
      (thread_tile_start_row_idx >= num_rows) || (thread_tile_start_col_idx >= row_length);
  bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      MIN(THREAD_TILE_DIM_X,
          (MIN(thread_tile_end_col_idx, row_length - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      MIN(THREAD_TILE_DIM_Y,
          (MIN(thread_tile_end_row_idx, num_rows - 1) - thread_tile_start_row_idx + 1));

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

  if (!empty_thrd_tile) {
    // Step 1: Load a block tile of input data into thread tiles on registers
    // Edge case: nonfull thread tile case, will use the partial load function here
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        if (i >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                            thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                          THREAD_TILE_DIM_X);
      }
    }

    // Step 2: calculate block tile amax and scale
    // Calculate thread_tile amax
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
      }
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be needed,
  // instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (kReturnTranspose) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  // Edge case: in the non-full tile case, there are three subcases
  // for full thread tile, it's the same thing here
  // for nonfull thread tile, pay attention when saving tmp_output_c to global
  // memory, cannot vec store_to, but need to elt store to for empty tile,
  // it should not enter this step, skip to Step 4

  // set thrd_tile_out_trans to all zero
  if constexpr (kReturnTranspose) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
      if (i >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        // Step 3: Store cast output
        CType scale_data = block_tile_scale;

        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
        tmp_output_c.data.elt[j] = scaled_elt;
        // Step 4: do transpose within thread tile
        if constexpr (kReturnTranspose) {
          thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
        }
      }
      tmp_output_c.store_to_elts(output_c + thread_tile_start_idx + i * row_length, 0,
                                 thread_tile_ncols);
    }

    if constexpr (kReturnTranspose) {
      const size_t block_tile_t_start_idx =
          tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
      const size_t warp_tile_t_start_idx =
          block_tile_t_start_idx +
          warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
          warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
      const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                             tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                             tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
      for (int i = 0; i < thread_tile_ncols; i++) {
        thrd_tile_out_trans[i].store_to_elts(output_t + thread_tile_t_start_idx + i * num_rows, 0,
                                             thread_tile_nrows);
      }
    }
  }
}

template <typename OutputType>
CUtensorMap get_tensor_map(const SimpleTensor& tensor, size_t global_dim_x, size_t global_dim_y) {
  CUtensorMapDataType dataType;
  if constexpr (std::is_same_v<OutputType, __nv_fp8_e4m3> ||
                std::is_same_v<OutputType, __nv_fp8_e5m2>) {
    dataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    NVTE_CHECK(false, "Invalid Output type (must be FP8).");
  }

  CUtensorMap tensor_map_output_trans{};
  create_2D_tensor_map(tensor_map_output_trans, tensor, global_dim_y, global_dim_x,
                       /*shmemY=*/BLOCK_TILE_DIM, /*shmemX=*/BLOCK_TILE_DIM,
                       /*stride_elems=*/global_dim_x, /*offset_elems=*/0, sizeof(OutputType) * 8);
  return tensor_map_output_trans;
}

constexpr int kMaxTensorsPerSquareBlockwiseKernel = 32;
constexpr int kMaxTensorsPerSquareBlockwiseTmaKernel = 8;

struct MultiSquareBlockwiseQuantizeArgs {
  void* input_list[kMaxTensorsPerSquareBlockwiseKernel];
  void* output_c_list[kMaxTensorsPerSquareBlockwiseKernel];
  void* output_t_list[kMaxTensorsPerSquareBlockwiseKernel];
  void* scale_inv_c_list[kMaxTensorsPerSquareBlockwiseKernel];
  void* scale_inv_t_list[kMaxTensorsPerSquareBlockwiseKernel];
  int row_length_list[kMaxTensorsPerSquareBlockwiseKernel];
  int num_rows_list[kMaxTensorsPerSquareBlockwiseKernel];
  int scale_stride_x_list[kMaxTensorsPerSquareBlockwiseKernel];
  int scale_stride_y_list[kMaxTensorsPerSquareBlockwiseKernel];
  int scale_t_stride_x_list[kMaxTensorsPerSquareBlockwiseKernel];
  int scale_t_stride_y_list[kMaxTensorsPerSquareBlockwiseKernel];
  int block_range[kMaxTensorsPerSquareBlockwiseKernel + 1];
  int num_tensors;
};

struct alignas(64) MultiSquareBlockwiseTmaQuantizeArgs {
  void* input_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  void* output_c_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  void* output_t_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  void* scale_inv_c_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  void* scale_inv_t_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  alignas(64) CUtensorMap tensor_map_output_t_list[kMaxTensorsPerSquareBlockwiseTmaKernel];
  int row_length;
  int num_rows;
  int scale_stride_x;
  int scale_stride_y;
  int scale_t_stride_x;
  int scale_t_stride_y;
  int num_tensors;
};

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__device__ __forceinline__ void block_scaled_cast_transpose_kernel_notaligned_impl(
    const IType* const input, OType* const output_c, OType* const output_t,
    CType* const tile_scales_inv_c, CType* const tile_scales_inv_t, const size_t row_length,
    const size_t num_rows, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    bool pow_2_scaling, const size_t tile_id_x, const size_t tile_id_y,
    CType* const block_tile_amax_shared) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = kReturnTranspose ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  const size_t block_tile_start_row_idx = tile_id_y * BLOCK_TILE_DIM;
  const size_t block_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM;
  const size_t block_tile_start_idx =
      block_tile_start_row_idx * row_length + block_tile_start_col_idx;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  const size_t thread_tile_start_col_idx =
      tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
      tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + THREAD_TILE_DIM_Y - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + THREAD_TILE_DIM_X - 1;

  bool full_thrd_tile =
      (thread_tile_end_row_idx < num_rows) && (thread_tile_end_col_idx < row_length);
  bool empty_thrd_tile =
      (thread_tile_start_row_idx >= num_rows) || (thread_tile_start_col_idx >= row_length);
  bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      MIN(THREAD_TILE_DIM_X,
          (MIN(thread_tile_end_col_idx, row_length - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      MIN(THREAD_TILE_DIM_Y,
          (MIN(thread_tile_end_row_idx, num_rows - 1) - thread_tile_start_row_idx + 1));

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

  if (!empty_thrd_tile) {
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        if (i >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                            thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                          THREAD_TILE_DIM_X);
      }
    }

    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
      }
    }
  }

  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (kReturnTranspose) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

  if constexpr (kReturnTranspose) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
      if (i >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        CType scale_data = block_tile_scale;

        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
        tmp_output_c.data.elt[j] = scaled_elt;
        if constexpr (kReturnTranspose) {
          thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
        }
      }
      tmp_output_c.store_to_elts(output_c + thread_tile_start_idx + i * row_length, 0,
                                 thread_tile_ncols);
    }

    if constexpr (kReturnTranspose) {
      const size_t block_tile_t_start_idx =
          tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
      const size_t warp_tile_t_start_idx =
          block_tile_t_start_idx +
          warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
          warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
      const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                             tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                             tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
      for (int i = 0; i < thread_tile_ncols; i++) {
        thrd_tile_out_trans[i].store_to_elts(output_t + thread_tile_t_start_idx + i * num_rows, 0,
                                             thread_tile_nrows);
      }
    }
  }
}

#ifdef TMA_HW_SUPPORTED
template <typename CType, typename IType, typename OType>
__device__ __forceinline__ void block_scaled_cast_transpose_kernel_tma_full_tile_impl(
    const IType* const input, OType* const output_c, CType* const tile_scales_inv_c,
    CType* const tile_scales_inv_t, const size_t row_length, const size_t num_rows,
    const size_t scale_stride_x, const size_t scale_stride_y, const size_t scale_t_stride_x,
    const size_t scale_t_stride_y, const float epsilon, const CUtensorMap* const tensor_map_output_t,
    bool pow_2_scaling, const size_t tile_id_x, const size_t tile_id_y,
    CType* const block_tile_amax_shared,
    Vec<OType, THREAD_TILE_DIM_Y> (*block_tile_trans_shared)[SHARED_BLOCK_TILE_DIM_X_BANKS]) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  const size_t block_tile_start_idx =
      tile_id_y * BLOCK_TILE_DIM * row_length + tile_id_x * BLOCK_TILE_DIM;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
    thrd_tile_input[i].load_from(input + thread_tile_start_idx + i * row_length);
  }

  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
    }
  }

  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    row_idx = tile_id_x;
    col_idx = tile_id_y;
    tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
  }

  OVecCast tmp_output_c;
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      CType scale_data = block_tile_scale;
      OType scaled_elt =
          static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
      tmp_output_c.data.elt[j] = scaled_elt;
      thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
    }
    tmp_output_c.store_to(output_c + thread_tile_start_idx + i * row_length);
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
    auto warp_id_in_block_x_ = warp_id_in_block_y;
    auto warp_id_in_block_y_ = warp_id_in_block_x;
    int row_idx = warp_id_in_block_y_ * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
                  tid_in_warp_x * THREAD_TILE_DIM_X + i;
    int col_idx =
        warp_id_in_block_x_ * (NUM_BANKS_Y_IN_WARP / NUM_BANKS_PER_SHARED_ELEM) + tid_in_warp_y;
    block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
  }

  OType(*block_tile_trans_shared_otype_ptr)[BLOCK_TILE_DIM] =
      reinterpret_cast<OType(*)[BLOCK_TILE_DIM]>(block_tile_trans_shared);

  ptx::fence_proxy_async_shared_cta();
  __syncthreads();

  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(tensor_map_output_t), tile_id_y * BLOCK_TILE_DIM,
        tile_id_x * BLOCK_TILE_DIM,
        reinterpret_cast<uint64_t*>(block_tile_trans_shared_otype_ptr));
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read<0>();
  }
}
#endif

template <typename InputType, typename OutputType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    multi_block_scaled_square_cast_transpose_tma_kernel(
        const __grid_constant__ MultiSquareBlockwiseTmaQuantizeArgs args, const float epsilon,
        bool pow_2_scaling, const float* noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  const int tensor_id = blockIdx.z;
  if (tensor_id >= args.num_tensors) {
    return;
  }

#ifdef TMA_HW_SUPPORTED
  using OVecTrans = Vec<OutputType, THREAD_TILE_DIM_Y>;

  __shared__ float block_tile_amax_shared[NUM_WARPS_IN_BLOCK];
  __shared__ alignas(128)
      OVecTrans block_tile_trans_shared[SHARED_BLOCK_TILE_DIM_Y][SHARED_BLOCK_TILE_DIM_X_BANKS];

  block_scaled_cast_transpose_kernel_tma_full_tile_impl<float, InputType, OutputType>(
      reinterpret_cast<const InputType*>(args.input_list[tensor_id]),
      reinterpret_cast<OutputType*>(args.output_c_list[tensor_id]),
      reinterpret_cast<float*>(args.scale_inv_c_list[tensor_id]),
      reinterpret_cast<float*>(args.scale_inv_t_list[tensor_id]),
      static_cast<size_t>(args.row_length), static_cast<size_t>(args.num_rows),
      static_cast<size_t>(args.scale_stride_x), static_cast<size_t>(args.scale_stride_y),
      static_cast<size_t>(args.scale_t_stride_x), static_cast<size_t>(args.scale_t_stride_y),
      epsilon, &args.tensor_map_output_t_list[tensor_id], pow_2_scaling,
      static_cast<size_t>(blockIdx.x), static_cast<size_t>(blockIdx.y), block_tile_amax_shared,
      block_tile_trans_shared);
#else
  __shared__ float block_tile_amax_shared[NUM_WARPS_IN_BLOCK];
  block_scaled_cast_transpose_kernel_notaligned_impl<true, float, InputType, OutputType>(
      reinterpret_cast<const InputType*>(args.input_list[tensor_id]),
      reinterpret_cast<OutputType*>(args.output_c_list[tensor_id]),
      reinterpret_cast<OutputType*>(args.output_t_list[tensor_id]),
      reinterpret_cast<float*>(args.scale_inv_c_list[tensor_id]),
      reinterpret_cast<float*>(args.scale_inv_t_list[tensor_id]),
      static_cast<size_t>(args.row_length), static_cast<size_t>(args.num_rows),
      static_cast<size_t>(args.scale_stride_x), static_cast<size_t>(args.scale_stride_y),
      static_cast<size_t>(args.scale_t_stride_x), static_cast<size_t>(args.scale_t_stride_y),
      epsilon, pow_2_scaling, static_cast<size_t>(blockIdx.x), static_cast<size_t>(blockIdx.y),
      block_tile_amax_shared);
#endif
}

template <typename InputType, typename OutputType>
void launch_multi_block_scaled_square_cast_transpose_tma_kernel(
    const MultiSquareBlockwiseTmaQuantizeArgs& kernel_args, const float epsilon,
    const bool pow_2_scaling, const float* noop_ptr, cudaStream_t stream) {
  if (kernel_args.num_tensors == 0) {
    return;
  }

  const dim3 grid(static_cast<unsigned int>(kernel_args.row_length / BLOCK_TILE_DIM),
                  static_cast<unsigned int>(kernel_args.num_rows / BLOCK_TILE_DIM),
                  static_cast<unsigned int>(kernel_args.num_tensors));
  multi_block_scaled_square_cast_transpose_tma_kernel<InputType, OutputType>
      <<<grid, THREADS_PER_BLOCK, 0, stream>>>(kernel_args, epsilon, pow_2_scaling, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    multi_block_scaled_square_cast_transpose_kernel(MultiSquareBlockwiseQuantizeArgs args,
                                                    const float epsilon, bool pow_2_scaling,
                                                    const float* noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  int tensor_id = 0;
  const int bid = blockIdx.x;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }

  const size_t row_length = static_cast<size_t>(args.row_length_list[tensor_id]);
  const size_t num_rows = static_cast<size_t>(args.num_rows_list[tensor_id]);
  const size_t num_tiles_x = DIVUP(row_length, static_cast<size_t>(BLOCK_TILE_DIM));
  const int tile_id = bid - args.block_range[tensor_id];
  const size_t tile_idx_x = static_cast<size_t>(tile_id) % num_tiles_x;
  const size_t tile_idx_y = static_cast<size_t>(tile_id) / num_tiles_x;

  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];
  block_scaled_cast_transpose_kernel_notaligned_impl<kReturnTranspose, CType, IType, OType>(
      reinterpret_cast<const IType*>(args.input_list[tensor_id]),
      reinterpret_cast<OType*>(args.output_c_list[tensor_id]),
      reinterpret_cast<OType*>(args.output_t_list[tensor_id]),
      reinterpret_cast<CType*>(args.scale_inv_c_list[tensor_id]),
      reinterpret_cast<CType*>(args.scale_inv_t_list[tensor_id]), row_length, num_rows,
      static_cast<size_t>(args.scale_stride_x_list[tensor_id]),
      static_cast<size_t>(args.scale_stride_y_list[tensor_id]),
      static_cast<size_t>(args.scale_t_stride_x_list[tensor_id]),
      static_cast<size_t>(args.scale_t_stride_y_list[tensor_id]), epsilon, pow_2_scaling,
      tile_idx_x, tile_idx_y, block_tile_amax_shared);
}

template <bool kReturnTranspose, typename InputType, typename OutputType>
void launch_multi_block_scaled_square_cast_transpose_kernel(
    const MultiSquareBlockwiseQuantizeArgs& kernel_args, const float epsilon,
    const bool pow_2_scaling, const float* noop_ptr, cudaStream_t stream) {
  if (kernel_args.num_tensors == 0) {
    return;
  }

  const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
  multi_block_scaled_square_cast_transpose_kernel<kReturnTranspose, float, InputType, OutputType>
      <<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(kernel_args, epsilon, pow_2_scaling, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

bool try_launch_multi_block_scaled_square_cast_transpose_tma(
    const std::vector<Tensor*>& input_list, std::vector<Tensor*>& output_list,
    const DType input_dtype, const DType output_dtype, const float epsilon,
    const bool pow_2_scale, const float* noop_ptr, cudaStream_t stream) {
  if (cuda::sm_arch(cuda::current_device()) < 90) {
    return false;
  }

  size_t common_row_length = 0;
  size_t common_num_rows = 0;
  size_t common_scale_stride_y = 0;
  size_t common_scale_t_stride_y = 0;
  bool common_shape_initialized = false;

  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = input_list[tensor_id]->data;
    auto& output = output_list[tensor_id]->data;
    auto& output_t = output_list[tensor_id]->columnwise_data;
    auto& scale_inv = output_list[tensor_id]->scale_inv;
    auto& scale_inv_t = output_list[tensor_id]->columnwise_scale_inv;

    if (output_list[tensor_id]->scaling_mode != NVTE_BLOCK_SCALING_2D ||
        input.dtype != input_dtype || output.dtype != output_dtype || input.shape != output.shape ||
        scale_inv.shape.size() != 2 || output_t.shape.size() != input.shape.size() ||
        output.dtype != output_t.dtype || scale_inv_t.shape.size() != 2 || output.dptr == nullptr ||
        output_t.dptr == nullptr || scale_inv.dptr == nullptr || scale_inv_t.dptr == nullptr ||
        !is_aligned_ptr(output_t.dptr, TMA_GMEM_ALIGNMENT)) {
      return false;
    }

    const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
    size_t num_rows = 1;
    size_t num_elements = row_length;
    for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
      num_rows *= input.shape.at(i);
      num_elements *= input.shape.at(i);
    }
    if (num_elements == 0 || row_length % BLOCK_TILE_DIM != 0 ||
        num_rows % BLOCK_TILE_DIM != 0) {
      return false;
    }

    if (output_t.shape.size() > 0) {
      if (output_t.shape[0] != row_length) {
        return false;
      }
      for (size_t i = 1; i < output_t.shape.size(); ++i) {
        if (output_t.shape.at(i) != input.shape.at(i - 1)) {
          return false;
        }
      }
    }

    const size_t scale_stride_y = scale_inv.shape[1];
    const size_t scale_t_stride_y = scale_inv_t.shape[1];

    if (!common_shape_initialized) {
      common_row_length = row_length;
      common_num_rows = num_rows;
      common_scale_stride_y = scale_stride_y;
      common_scale_t_stride_y = scale_t_stride_y;
      common_shape_initialized = true;
    } else if (row_length != common_row_length || num_rows != common_num_rows ||
               scale_stride_y != common_scale_stride_y ||
               scale_t_stride_y != common_scale_t_stride_y) {
      return false;
    }
  }

  if (!common_shape_initialized) {
    return false;
  }

  auto check_int_range = [](size_t value, const char* name) -> int {
    NVTE_CHECK(value <= static_cast<size_t>(std::numeric_limits<int>::max()), name,
               " exceeds int range: ", value);
    return static_cast<int>(value);
  };

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input_dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output_dtype, OutputType,
          MultiSquareBlockwiseTmaQuantizeArgs kernel_args{};
          auto reset_kernel_args = [&]() {
            kernel_args.num_tensors = 0;
            kernel_args.row_length = check_int_range(common_row_length, "Row length");
            kernel_args.num_rows = check_int_range(common_num_rows, "Number of rows");
            kernel_args.scale_stride_x = 1;
            kernel_args.scale_stride_y = check_int_range(common_scale_stride_y, "Scale stride y");
            kernel_args.scale_t_stride_x = 1;
            kernel_args.scale_t_stride_y =
                check_int_range(common_scale_t_stride_y, "Scale_t stride y");
          };
          auto launch_kernel_args = [&]() {
            if (kernel_args.num_tensors == 0) {
              return;
            }
            launch_multi_block_scaled_square_cast_transpose_tma_kernel<InputType, OutputType>(
                kernel_args, epsilon, pow_2_scale, noop_ptr, stream);
            reset_kernel_args();
          };
          reset_kernel_args();

          for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
            if (kernel_args.num_tensors == kMaxTensorsPerSquareBlockwiseTmaKernel) {
              launch_kernel_args();
            }

            const auto& input = input_list[tensor_id]->data;
            auto& output = output_list[tensor_id]->data;
            auto& output_t = output_list[tensor_id]->columnwise_data;
            auto& scale_inv = output_list[tensor_id]->scale_inv;
            auto& scale_inv_t = output_list[tensor_id]->columnwise_scale_inv;

            const int pos = kernel_args.num_tensors;
            kernel_args.input_list[pos] = input.dptr;
            kernel_args.output_c_list[pos] = output.dptr;
            kernel_args.output_t_list[pos] = output_t.dptr;
            kernel_args.scale_inv_c_list[pos] = scale_inv.dptr;
            kernel_args.scale_inv_t_list[pos] = scale_inv_t.dptr;
            kernel_args.tensor_map_output_t_list[pos] =
                get_tensor_map<OutputType>(output_t, common_num_rows, common_row_length);
            ++kernel_args.num_tensors;
          }

          launch_kernel_args();)  // OutputType
  )                              // InputType

  return true;
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_square_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         const bool return_transpose, const bool pow_2_scale,
                                         const SimpleTensor& noop_tensor, cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_square_blockwise);
  checkCuDriverContext(stream);

  NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");
  const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
  }

  NVTE_CHECK(scale_inv.shape.size() == 2, "scale_inv must have 2 dimensions.");

  size_t scale_k = scale_inv.shape[1];

  const size_t scale_stride_x = 1;
  const size_t scale_stride_y = scale_k;

  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  if (return_transpose) {
    NVTE_CHECK(output_t.shape.size() == input.shape.size(),
               "output_t must have same number of dimensions as input.");
    if (output_t.shape.size() > 0) {
      NVTE_CHECK(output_t.shape[0] == row_length, "Wrong dimension 0 of output_t.");
      for (size_t i = 1; i < output_t.shape.size(); ++i) {
        NVTE_CHECK(output_t.shape.at(i) == input.shape.at(i - 1), "Wrong dimension in output_t");
      }
    }
    NVTE_CHECK(output.dtype == output_t.dtype, "output and output_t need to have the same type.");

    NVTE_CHECK(scale_inv_t.shape.size() == 2, "scale_inv_t must have 2 dimensions.");

    scale_t_stride_x = 1;
    scale_t_stride_y = scale_inv_t.shape[1];
  }

  const size_t num_blocks_x = DIVUP(row_length, BLOCK_TILE_DIM);
  const size_t num_blocks_y = DIVUP(num_rows, BLOCK_TILE_DIM);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output.dtype, OutputType,

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, kReturnTranspose,

              dim3 grid(num_blocks_x, num_blocks_y, 1);
              const bool full_tile =
                  row_length % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

              if (full_tile) {
                CUtensorMap tensor_map_output_trans;
                if (return_transpose) {
                  tensor_map_output_trans =
                      get_tensor_map<OutputType>(output_t, num_rows, row_length);
                }
                block_scaled_cast_transpose_kernel<kReturnTranspose, float, InputType, OutputType>
                    <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                        reinterpret_cast<const InputType*>(input.dptr),
                        reinterpret_cast<OutputType*>(output.dptr),
                        reinterpret_cast<OutputType*>(output_t.dptr),
                        reinterpret_cast<float*>(scale_inv.dptr),
                        reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                        scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon,
                        tensor_map_output_trans, pow_2_scale, noop_ptr);
              } else {
                block_scaled_cast_transpose_kernel_notaligned<kReturnTranspose, float, InputType,
                                                              OutputType>
                    <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                        reinterpret_cast<const InputType*>(input.dptr),
                        reinterpret_cast<OutputType*>(output.dptr),
                        reinterpret_cast<OutputType*>(output_t.dptr),
                        reinterpret_cast<float*>(scale_inv.dptr),
                        reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                        scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon,
                        pow_2_scale, noop_ptr);
              }  // full-tile
              )  // return_transpose
          )      // OutputType
      )          // InputType
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_quantize_transpose_square_blockwise(
    const std::vector<Tensor*>& input_list, std::vector<Tensor*>& output_list, const float epsilon,
    const bool return_transpose, const bool pow_2_scale, const SimpleTensor& noop_tensor,
    cudaStream_t stream) {
  NVTE_API_CALL(multi_quantize_transpose_square_blockwise);
  checkCuDriverContext(stream);

  NVTE_CHECK(input_list.size() == output_list.size(),
             "Number of input and output tensors must match.");
  if (input_list.empty()) {
    return;
  }

  const DType input_dtype = input_list[0]->data.dtype;
  DType output_dtype = DType::kNumTypes;
  for (const auto* output : output_list) {
    if (output->data.dptr != nullptr || !output->data.shape.empty()) {
      output_dtype = output->data.dtype;
      break;
    }
  }
  NVTE_CHECK(output_dtype != DType::kNumTypes, "Unable to infer output dtype.");
  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  if (return_transpose &&
      try_launch_multi_block_scaled_square_cast_transpose_tma(input_list, output_list, input_dtype,
                                                              output_dtype, epsilon, pow_2_scale,
                                                              noop_ptr, stream)) {
    return;
  }

  auto check_int_range = [](size_t value, const char* name) -> int {
    NVTE_CHECK(value <= static_cast<size_t>(std::numeric_limits<int>::max()), name,
               " exceeds int range: ", value);
    return static_cast<int>(value);
  };

  auto reset_kernel_args = [](MultiSquareBlockwiseQuantizeArgs& args) {
    args.num_tensors = 0;
    args.block_range[0] = 0;
  };

  MultiSquareBlockwiseQuantizeArgs kernel_args;
  reset_kernel_args(kernel_args);

  auto launch_kernel_args = [&]() {
    if (kernel_args.num_tensors == 0) {
      return;
    }
    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
        input_dtype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output_dtype, OutputType,
            TRANSFORMER_ENGINE_SWITCH_CONDITION(
                return_transpose, kReturnTranspose,
                launch_multi_block_scaled_square_cast_transpose_kernel<kReturnTranspose, InputType,
                                                                       OutputType>(
                    kernel_args, epsilon, pow_2_scale, noop_ptr, stream);)  // kReturnTranspose
        )                                                                  // OutputType
    )                                                                      // InputType
    reset_kernel_args(kernel_args);
  };

  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = input_list[tensor_id]->data;
    auto& output = output_list[tensor_id]->data;
    auto& output_t = output_list[tensor_id]->columnwise_data;
    auto& scale_inv = output_list[tensor_id]->scale_inv;
    auto& scale_inv_t = output_list[tensor_id]->columnwise_scale_inv;

    NVTE_CHECK(output_list[tensor_id]->scaling_mode == NVTE_BLOCK_SCALING_2D,
               "Output tensor ", tensor_id, " must use 2D block scaling.");
    NVTE_CHECK(input.dtype == input_dtype, "Input tensor types do not match.");
    NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");
    NVTE_CHECK(output.dtype == output_dtype, "Output tensor types do not match.");
    NVTE_CHECK(scale_inv.shape.size() == 2, "scale_inv must have 2 dimensions.");

    const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
    size_t num_rows = 1;
    size_t num_elements = row_length;
    for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
      num_rows *= input.shape.at(i);
      num_elements *= input.shape.at(i);
    }
    if (num_elements == 0) {
      continue;
    }

    size_t scale_k = scale_inv.shape[1];
    const size_t scale_stride_x = 1;
    const size_t scale_stride_y = scale_k;
    size_t scale_t_stride_x = 0;
    size_t scale_t_stride_y = 0;

    if (return_transpose) {
      NVTE_CHECK(output_t.shape.size() == input.shape.size(),
                 "output_t must have same number of dimensions as input.");
      if (output_t.shape.size() > 0) {
        NVTE_CHECK(output_t.shape[0] == row_length, "Wrong dimension 0 of output_t.");
        for (size_t i = 1; i < output_t.shape.size(); ++i) {
          NVTE_CHECK(output_t.shape.at(i) == input.shape.at(i - 1),
                     "Wrong dimension in output_t.");
        }
      }
      NVTE_CHECK(output.dtype == output_t.dtype,
                 "output and output_t need to have the same type.");
      NVTE_CHECK(scale_inv_t.shape.size() == 2, "scale_inv_t must have 2 dimensions.");
      scale_t_stride_x = 1;
      scale_t_stride_y = scale_inv_t.shape[1];
    }

    const size_t num_blocks_x = DIVUP(row_length, static_cast<size_t>(BLOCK_TILE_DIM));
    const size_t num_blocks_y = DIVUP(num_rows, static_cast<size_t>(BLOCK_TILE_DIM));
    const size_t num_blocks = num_blocks_x * num_blocks_y;
    if (num_blocks == 0) {
      continue;
    }
    check_int_range(num_blocks, "Number of tiles");

    NVTE_CHECK(output.dptr != nullptr, "Rowwise output data pointer must not be null.");
    NVTE_CHECK(scale_inv.dptr != nullptr, "Rowwise scale inverse pointer must not be null.");
    if (return_transpose) {
      NVTE_CHECK(output_t.dptr != nullptr, "Columnwise output data pointer must not be null.");
      NVTE_CHECK(scale_inv_t.dptr != nullptr,
                 "Columnwise scale inverse pointer must not be null.");
    }

    if (kernel_args.num_tensors == kMaxTensorsPerSquareBlockwiseKernel) {
      launch_kernel_args();
    }

    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input.dptr;
    kernel_args.output_c_list[pos] = output.dptr;
    kernel_args.output_t_list[pos] = return_transpose ? output_t.dptr : nullptr;
    kernel_args.scale_inv_c_list[pos] = scale_inv.dptr;
    kernel_args.scale_inv_t_list[pos] = return_transpose ? scale_inv_t.dptr : nullptr;
    kernel_args.row_length_list[pos] = check_int_range(row_length, "Row length");
    kernel_args.num_rows_list[pos] = check_int_range(num_rows, "Number of rows");
    kernel_args.scale_stride_x_list[pos] = check_int_range(scale_stride_x, "Scale stride x");
    kernel_args.scale_stride_y_list[pos] = check_int_range(scale_stride_y, "Scale stride y");
    kernel_args.scale_t_stride_x_list[pos] = check_int_range(scale_t_stride_x, "Scale_t stride x");
    kernel_args.scale_t_stride_y_list[pos] = check_int_range(scale_t_stride_y, "Scale_t stride y");
    kernel_args.block_range[pos + 1] =
        kernel_args.block_range[pos] + check_int_range(num_blocks, "Number of tiles");
    ++kernel_args.num_tensors;
  }

  launch_kernel_args();
}

}  // namespace transformer_engine::detail

void nvte_multi_quantize_transpose_square_blockwise(size_t num_tensors,
                                                    const NVTETensor* input_list,
                                                    NVTETensor* output_list,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_quantize_transpose_square_blockwise);
  using namespace transformer_engine;

  std::vector<Tensor*> input_list_, output_list_;
  input_list_.reserve(num_tensors);
  output_list_.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(convertNVTETensorCheck(input_list[i]));
    output_list_.push_back(convertNVTETensorCheck(output_list[i]));
  }

  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig*>(quant_config);
  }

  Tensor dummy_tensor;
  Tensor* noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  bool return_transpose = false;
  for (const auto* output : output_list_) {
    return_transpose |= output->has_columnwise_data();
  }

  detail::multi_quantize_transpose_square_blockwise(
      input_list_, output_list_, quant_config_cpp.amax_epsilon, return_transpose,
      quant_config_cpp.force_pow_2_scales, noop_tensor->data, stream);
}
