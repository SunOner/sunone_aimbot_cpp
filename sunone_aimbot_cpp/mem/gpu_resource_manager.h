#pragma once
#include <cuda_runtime.h>

class GPUResourceManager {
public:
    bool reserveGPUMemory(size_t reservedMemoryMB);
    bool setGPUExclusiveMode();
    void releaseReservation();

private:
    void* reservedBuffer = nullptr;
    size_t reservedSize = 0;
};