#pragma once
#include <windows.h>

class CPUAffinityManager {
public:
    bool reserveCPUCores(int numCores);
    bool reserveSystemMemory(size_t reservedMemoryMB);

private:
    DWORD_PTR originalMask;
    void* reservedMemory = nullptr;
};