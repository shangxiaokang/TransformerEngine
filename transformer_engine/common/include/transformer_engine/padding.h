/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file padding.h
 *  \brief Functions handling padding.
 */

#ifndef TRANSFORMER_ENGINE_PADDING_H_
#define TRANSFORMER_ENGINE_PADDING_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Padding multiple tensors.
 *
 *  NOTE: Padding mode only support bottom.
 *
 *  For example, 3x3 matrix pad to 4x3 matrix.
 *
 *  source
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *
 *  destination
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *  | 0 | 0 | 0 |
 *
 *  \param[in]     num_tensors              Number of tensors.
 *  \param[in]     input_list               List of 2D input tensors.
 *  \param[in,out] output_list              List of padded tensors. Dimensions
 *                                          match tensors in input_list.
 *  \param[in]     padded_num_rows_list     List of padded num rows corresponding to input tensors.
 *  \param[in]     stream                   CUDA stream used for the operation.
 */
void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream);

/*! \brief Padding paired tensors with the same row splits.
 *
 *  This is equivalent to calling nvte_multi_padding twice, once for each tensor
 *  list, but supports different dtypes for the two lists and launches one CUDA
 *  kernel. It is intended for MoE expert inputs where hidden states and router
 *  probabilities share row splits but have different feature widths/dtypes.
 *
 *  \param[in]     num_tensors              Number of tensor pairs.
 *  \param[in]     input_a_list             First list of 2D input tensors.
 *  \param[in,out] output_a_list            First list of padded output tensors.
 *  \param[in]     input_b_list             Second list of 2D input tensors.
 *  \param[in,out] output_b_list            Second list of padded output tensors.
 *  \param[in]     padded_num_rows_list     List of padded num rows for each pair.
 *  \param[in]     stream                   CUDA stream used for the operation.
 */
void nvte_multi_padding_pair(size_t num_tensors, const NVTETensor* input_a_list,
                             NVTETensor* output_a_list, const NVTETensor* input_b_list,
                             NVTETensor* output_b_list, const int* padded_num_rows_list,
                             cudaStream_t stream);

/*! \brief Unpadding multiple tensors (reverse operation of padding).
 *
 *  NOTE: Unpadding mode only removes bottom rows.
 *
 *  For example, 4x3 matrix unpad to 3x3 matrix.
 *
 *  source
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *  | 0 | 0 | 0 |
 *
 *  destination
 *  | 1 | 2 | 3 |
 *  | 4 | 5 | 6 |
 *  | 7 | 8 | 9 |
 *
 *  \param[in]     num_tensors               Number of tensors.
 *  \param[in]     input_list                List of 2D padded input tensors.
 *  \param[in,out] output_list               List of unpadded tensors. Dimensions
 *                                           match original unpadded tensors.
 *  \param[in]     unpadded_num_rows_list    List of unpadded num rows corresponding to input tensors.
 *  \param[in]     stream                    CUDA stream used for the operation.
 */
void nvte_multi_unpadding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                          const int* unpadded_num_rows_list, cudaStream_t stream);

/*! \brief Unpadding paired tensors with the same row splits.
 *
 *  This is equivalent to calling nvte_multi_unpadding twice, once for each
 *  tensor list, but supports different dtypes for the two lists and launches one
 *  CUDA kernel.
 *
 *  \param[in]     num_tensors               Number of tensor pairs.
 *  \param[in]     input_a_list              First list of padded input tensors.
 *  \param[in,out] output_a_list             First list of unpadded output tensors.
 *  \param[in]     input_b_list              Second list of padded input tensors.
 *  \param[in,out] output_b_list             Second list of unpadded output tensors.
 *  \param[in]     unpadded_num_rows_list    List of unpadded num rows for each pair.
 *  \param[in]     stream                    CUDA stream used for the operation.
 */
void nvte_multi_unpadding_pair(size_t num_tensors, const NVTETensor* input_a_list,
                               NVTETensor* output_a_list, const NVTETensor* input_b_list,
                               NVTETensor* output_b_list, const int* unpadded_num_rows_list,
                               cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_PADDING_H_
