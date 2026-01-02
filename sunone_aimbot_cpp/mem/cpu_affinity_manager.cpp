#include "cpu_affinity_manager.h"

bool CPUAffinityManager::reserveCPUCores(int numCores)
{
    DWORD_PTR mask = 0;

    for (int i = 0; i < numCores; i++) {
        mask |= (1ULL << i);
    }

    originalMask = SetThreadAffinityMask(GetCurrentThread(), mask);

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    return originalMask != 0;
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
    return false;
}