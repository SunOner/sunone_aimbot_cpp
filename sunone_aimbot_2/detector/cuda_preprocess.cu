#ifdef USE_CUDA
#include <cuda_runtime.h>

static __global__ void hwc_to_chw_norm_kernel(
    const float* __restrict__ srcHwc, int srcStepFloats,
    float* __restrict__ dstChw,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int hw = height * width;
    const int idx = y * width + x;

    const float* p = srcHwc + y * srcStepFloats + x * 3;

    dstChw[0 * hw + idx] = p[2]; // R
    dstChw[1 * hw + idx] = p[1]; // G
    dstChw[2 * hw + idx] = p[0]; // B
}

void launch_hwc_to_chw_norm(
    const float* srcHwc,
    size_t srcStepBytes,
    float* dstChw,
    int width,
    int height,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    const int stepFloats = static_cast<int>(srcStepBytes / sizeof(float));

    hwc_to_chw_norm_kernel << <grid, block, 0, stream >> > (
        srcHwc, stepFloats, dstChw, width, height
        );
}
#endif
