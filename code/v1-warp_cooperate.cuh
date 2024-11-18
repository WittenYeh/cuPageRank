/*
 * @Autor: Witten Yeh
 * @Date: 2024-11-18 18:31:56
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-11-18 21:03:40
 * @Description: 
 */

#include "helper_pr.cuh"

namespace cupagerank {
namespace warp_cooperate {

__global__ void kernel_coalesce(
    bool* label, 
    ValueT *delta, 
    ValueT *residual, 
    const uint64_t vertex_count, 
    const uint64_t *vertexList, 
    const EdgeT *edgeList
) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start)
                atomicAdd(&residual[edgeList[i]], delta[warpIdx]);

        label[warpIdx] = false;
    }
}

};  // namespace warp_cooperative
};  // namespace cupagerank