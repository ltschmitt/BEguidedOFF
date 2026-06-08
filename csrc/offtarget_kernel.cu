/*
 * Main off-target search CUDA kernel.
 *
 * This is a placeholder / scaffold. It demonstrates the kernel signature
 * and shared-memory layout described in the design doc.
 */
#include <cstdint>
#include <cuda_runtime.h>
#include "common.h"
#include "topk.cuh"

#define MAX_GUIDES_PER_BLOCK 128

__global__ void search_offtargets_kernel(
    // Genome data
    const uint32_t* __restrict__ genome_packed,
    const int64_t genome_len_bp,

    // Guide data
    const int64_t* __restrict__ guide_encoded,
    const int n_guides,

    // PAM data
    const uint8_t* __restrict__ pam_masks,
    const uint8_t* __restrict__ pam_lengths,
    const uint8_t* __restrict__ pam_positions,

    // Edit window data
    const uint8_t* __restrict__ edit_win_start,
    const uint8_t* __restrict__ edit_win_end,
    const uint8_t* __restrict__ editor_types,

    // Search parameters
    const int max_mismatches,

    // Output buffers
    int* __restrict__ offtarget_counts,
    float* __restrict__ top_scores,
    int64_t* __restrict__ top_positions,
    int* __restrict__ top_mismatches,

    // Full output mode
    int64_t* __restrict__ all_positions,
    float* __restrict__ all_scores,
    int* __restrict__ all_write_idx,
    const int max_all_per_guide,
    const float score_threshold
) {
    extern __shared__ uint8_t shared_mem[];

    // Layout in shared memory
    int64_t* s_guides = (int64_t*)shared_mem;
    uint8_t* s_pam_masks = (uint8_t*)&s_guides[MAX_GUIDES_PER_BLOCK];
    uint8_t* s_pam_lens  = (uint8_t*)&s_pam_masks[MAX_GUIDES_PER_BLOCK * 6];
    uint8_t* s_edit_data = (uint8_t*)&s_pam_lens[MAX_GUIDES_PER_BLOCK];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load guide data into shared memory
    // (In practice, use cooperative loading by thread blocks)
    if (tid < n_guides) {
        s_guides[tid] = guide_encoded[tid];
        s_pam_lens[tid] = pam_lengths[tid];
        s_edit_data[tid * 3 + 0] = edit_win_start[tid];
        s_edit_data[tid * 3 + 1] = edit_win_end[tid];
        s_edit_data[tid * 3 + 2] = editor_types[tid];
        for (int p = 0; p < 6; ++p) {
            s_pam_masks[tid * 6 + p] = pam_masks[tid * 6 + p];
        }
    }
    __syncthreads();

    // TODO: actual genome scanning, PAM checking, XOR+popcount,
    // edit-window filtering, and top-K / full-output writing.

    // Placeholder to avoid empty kernel warning
    if (gid == 0) {
        offtarget_counts[0] = 0;
    }
}
