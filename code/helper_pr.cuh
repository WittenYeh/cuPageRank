/*
 * @Autor: Witten Yeh
 * @Date: 2024-11-18 16:31:54
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-11-18 18:32:36
 * @Description: 
 */

#pragma once

#include <cuda.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
#include "helper_cuda.cuh"

#define BLOCK_SIZE 1024
#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

typedef enum {
    BASELINE = 0,
    WARP_LB = 1,
    BLOCK_LB = 2
} impl_type;

typedef uint64_t EdgeT;
typedef float ValueT;