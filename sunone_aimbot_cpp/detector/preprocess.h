#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void cuda_preprocess(const uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    int num_channels,
    cudaStream_t stream);