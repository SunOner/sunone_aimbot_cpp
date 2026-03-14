#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "AimbotTarget.h"

class MouseThread;
class Game_overlay;

extern Game_overlay* gameOverlayPtr;
extern std::thread gameOverlayThread;
extern std::atomic<bool> gameOverlayShouldExit;

extern std::mutex g_trackerDebugMutex;
extern std::vector<TrackDebugInfo> g_trackerDebugTracks;
extern int g_trackerLockedId;

void mouseThreadFunction(MouseThread& mouseThread);
void gameOverlayRenderLoop();
