#pragma once
#ifdef USE_CUDA
#include <cstddef>
#include <cuda_runtime.h>

void launch_hwc_to_chw_norm(
    const float* srcHwc,
    size_t srcStepBytes,
    float* dstChw,
    int width,
    int height,
    cudaStream_t stream
);
#endif
