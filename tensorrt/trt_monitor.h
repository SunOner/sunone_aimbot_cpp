#ifdef USE_CUDA
#pragma once
#include <atomic>
#include <string>
#include <mutex>
#include <map>

struct ProgressPhase {
    std::string name;
    int current;
    int max;
};

inline std::mutex gProgressMutex;
inline std::map<std::string, ProgressPhase> gProgressPhases;
inline std::atomic<bool> gIsTrtExporting = false;

class ImGuiProgressMonitor : public nvinfer1::IProgressMonitor {
public:
    void phaseStart(const char* phaseName, const char* parentPhase, int32_t nbSteps) noexcept override {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        ProgressPhase phase;
        phase.name = phaseName;
        phase.current = 0;
        phase.max = nbSteps;
        gProgressPhases[phaseName] = phase;
        gIsTrtExporting = true;
    }

    bool stepComplete(const char* phaseName, int32_t step) noexcept override {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        gProgressPhases[phaseName].current = step;
        return true;
    }

    void phaseFinish(const char* phaseName) noexcept override {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        gProgressPhases.erase(phaseName);
        if (gProgressPhases.empty())
            gIsTrtExporting = false;
    }
};
#endif