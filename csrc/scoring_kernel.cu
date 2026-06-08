/*
 * GPU scoring kernel (placeholder).
 */
#include <cstdint>
#include <cuda_runtime.h>

__global__ void scoring_kernel(
    const float* __restrict__ penalty_matrix,
    const int* __restrict__ mismatch_positions,
    const int* __restrict__ mismatch_counts,
    const char* __restrict__ pam_seqs,
    int n_offtargets,
    float* __restrict__ scores
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_offtargets) return;

    // TODO: vectorized CFD scoring per off-target
    scores[idx] = 0.0f;
}
