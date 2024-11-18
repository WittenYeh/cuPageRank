/*
 * @Autor: Witten Yeh
 * @Date: 2024-11-18 18:22:00
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-11-18 18:47:11
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

__global__ void update(
    bool *label, 
    ValueT *delta, 
    ValueT *residual, 
    ValueT *value, 
    const uint64_t vertex_count, 
    const uint64_t *vertexList, 
    ValueT tolerance, 
    ValueT alpha, 
    bool *changed
) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
	}
}


};  // namespace common
};  // namespace cupagerank