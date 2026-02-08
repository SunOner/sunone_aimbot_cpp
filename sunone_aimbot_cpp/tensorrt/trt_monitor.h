#ifdef USE_CUDA
#pragma once
#include <atomic>
#include <string>
#include <mutex>
#include <map>
#include <chrono>

struct ProgressPhase {
    std::string name;
    int current;
    int max;
};

inline std::mutex gProgressMutex;
inline std::map<std::string, ProgressPhase> gProgressPhases;
inline std::atomic<bool> gIsTrtExporting = false;
inline std::atomic<bool> gTrtExportCancelRequested = false;
inline std::atomic<long long> gTrtExportLastUpdateMs = 0;

inline long long TrtNowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline void TrtExportResetState()
{
    gTrtExportCancelRequested = false;
    gTrtExportLastUpdateMs = TrtNowMs();
}

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
        gTrtExportLastUpdateMs = TrtNowMs();
    }

    bool stepComplete(const char* phaseName, int32_t step) noexcept override {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        gProgressPhases[phaseName].current = step;
        gTrtExportLastUpdateMs = TrtNowMs();
        return !gTrtExportCancelRequested.load();
    }

    void phaseFinish(const char* phaseName) noexcept override {
        std::lock_guard<std::mutex> lock(gProgressMutex);
        gProgressPhases.erase(phaseName);
        if (gProgressPhases.empty())
            gIsTrtExporting = false;
        gTrtExportLastUpdateMs = TrtNowMs();
    }
};
#endif
