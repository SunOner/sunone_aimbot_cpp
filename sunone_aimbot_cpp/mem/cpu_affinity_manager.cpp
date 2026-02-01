#include "cpu_affinity_manager.h"

#include <iostream>

bool CPUAffinityManager::reserveCPUCores(int numCores)
{
    DWORD_PTR mask = 0;

    for (int i = 0; i < numCores; i++) {
        mask |= (1ULL << i);
    }

    originalMask = SetThreadAffinityMask(GetCurrentThread(), mask);
    if (originalMask == 0)
    {
        std::cerr << "[CPU] Failed to set thread affinity mask. GetLastError="
                  << GetLastError() << std::endl;
        return false;
    }

    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
    {
        std::cerr << "[CPU] Failed to set process priority. GetLastError="
                  << GetLastError() << std::endl;
    }

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST))
    {
        std::cerr << "[CPU] Failed to set thread priority. GetLastError="
                  << GetLastError() << std::endl;
    }

    return true;
}

bool CPUAffinityManager::reserveSystemMemory(size_t reservedMemoryMB)
{
    size_t reservedSize = reservedMemoryMB * 1024 * 1024;

    reservedMemory = malloc(reservedSize);
    if (reservedMemory)
    {
        memset(reservedMemory, 0, reservedSize);
        return true;
    }
    std::cerr << "[CPU] Failed to reserve system memory: " << reservedMemoryMB << " MB." << std::endl;
    return false;
}
