/*
 * Top-K tracking utilities for CUDA kernels.
 */
#ifndef OTBE_TOPK_CUH
#define OTBE_TOPK_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Simple top-3 tracker using local registers
// Each thread keeps K best scores and their metadata
struct TopKTracker {
    static constexpr int K = 3;
    float scores[K];
    int64_t positions[K];
    int mismatches[K];

    __device__ TopKTracker() {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scores[i] = -1.0f;
            positions[i] = -1;
            mismatches[i] = -1;
        }
    }

    __device__ void insert(float score, int64_t position, int mm) {
        // Find insertion point (descending order)
        int idx = K;
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            if (score > scores[i]) {
                idx = i;
                break;
            }
        }
        if (idx >= K) return;  // Not in top K

        // Shift down
        #pragma unroll
        for (int i = K - 1; i > idx; --i) {
            scores[i] = scores[i-1];
            positions[i] = positions[i-1];
            mismatches[i] = mismatches[i-1];
        }
        scores[idx] = score;
        positions[idx] = position;
        mismatches[idx] = mm;
    }
};

#endif // OTBE_TOPK_CUH
