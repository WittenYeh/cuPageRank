/*
 * @Autor: Witten Yeh
 * @Date: 2024-11-18 18:22:00
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-11-18 20:07:08
 * @Description: 
 */

#pragma once

#include "helper_pr.cuh"

namespace cupagerank {
namespace common {

__global__ void initialize(
    bool *label, 
    ValueT *delta, 
    ValueT *residual, 
    ValueT *value, 
    const uint64_t vertex_count, 
    const uint64_t *vertexList, 
    ValueT alpha
) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}

};  // namespace common
};  // namespace cupagerank