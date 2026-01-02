#include "gpu_resource_manager.h"

bool GPUResourceManager::reserveGPUMemory(size_t reservedMemoryMB)
{
    size_t totalMemory, freeMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);

    reservedSize = reservedMemoryMB * 1024 * 1024;

    if (freeMemory < reservedSize) {
        return false;
    }

    cudaMalloc(&reservedBuffer, reservedSize);
    cudaMemset(reservedBuffer, 0, reservedSize);

    return true;
}

bool GPUResourceManager::setGPUExclusiveMode()
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    return true;
}