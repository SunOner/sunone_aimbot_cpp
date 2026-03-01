#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

void launch_hwc_to_chw_norm(
    const cv::cuda::GpuMat& hwcFloat3,
    float* dstChw,
    int width,
    int height,
    cudaStream_t stream
);
#endif