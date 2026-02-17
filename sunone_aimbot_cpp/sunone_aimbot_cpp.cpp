#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_cpp.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "Game_overlay.h"
#include "ghub.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "mem/gpu_resource_manager.h"
#include "mem/cpu_affinity_manager.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

#ifdef USE_CUDA
TrtDetector trt_detector;
#endif

DirectMLDetector* dml_detector = nullptr;
MouseThread* globalMouseThread = nullptr;
Config config;

Game_overlay* gameOverlayPtr = nullptr;
std::thread gameOverlayThread;
std::atomic<bool> gameOverlayShouldExit(false);

GhubMouse* gHub = nullptr;
Arduino* arduinoSerial = nullptr;
KmboxNetConnection* kmboxNetSerial = nullptr;
KmboxAConnection* kmboxASerial = nullptr;
MakcuConnection* makcuSerial = nullptr;

std::atomic<bool> detection_resolution_changed(false);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> capture_window_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);

static std::string g_lastIconPath;
static int g_iconImageId = 0;
static std::mutex g_iconMutex;

std::string g_iconLastError;

static int FatalExit(const std::string& message)
{
    std::cerr << message << std::endl;
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return -1;
}

static void HandleThreadCrash(const char* name, const std::exception* ex)
{
    std::cerr << "[Thread] " << name << " crashed: "
              << (ex ? ex->what() : "unknown exception") << std::endl;
    shouldExit = true;
    gameOverlayShouldExit.store(true);
    detectionBuffer.cv.notify_all();
}

template <typename Func>
static std::thread StartThreadGuarded(const char* name, Func func)
{
    return std::thread([name, func]() mutable {
        try
        {
            func();
        }
        catch (const std::exception& e)
        {
            HandleThreadCrash(name, &e);
        }
        catch (...)
        {
            HandleThreadCrash(name, nullptr);
        }
        });
}

static void draw_target_correction_demo_game_overlay(Game_overlay* overlay, float centerX, float centerY)
{
    if (!overlay)
        return;

    const float scale = 4.0f;
    float near_px = config.nearRadius * scale;
    float snap_px = config.snapRadius * scale;
    near_px = std::max(10.0f, near_px);
    snap_px = std::max(6.0f, std::min(snap_px, near_px - 4.0f));

    overlay->AddCircle({ centerX, centerY, near_px }, ARGB(180, 80, 120, 255), 2.0f);
    overlay->AddCircle({ centerX, centerY, snap_px }, ARGB(180, 255, 100, 100), 2.0f);

    static float dist_px = 0.0f;
    static float vel_px = 0.0f;
    static auto last_t = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_t).count();
    last_t = now;
    dt = std::max(0.0, std::min(dt, 0.1));

    if (dist_px <= 0.0f || dist_px > near_px)
        dist_px = near_px;

    double dist_units = dist_px / scale;
    double speed_mult;
    if (dist_units < config.snapRadius)
    {
        speed_mult = config.minSpeedMultiplier * config.snapBoostFactor;
    }
    else if (dist_units < config.nearRadius)
    {
        double t = dist_units / config.nearRadius;
        double crv = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        speed_mult = config.minSpeedMultiplier +
            (config.maxSpeedMultiplier - config.minSpeedMultiplier) * crv;
    }
    else
    {
        double norm = std::max(0.0, std::min(dist_units / config.nearRadius, 1.0));
        speed_mult = config.minSpeedMultiplier +
            (config.maxSpeedMultiplier - config.minSpeedMultiplier) * norm;
    }

    float max_multiplier = std::max(0.1f, config.maxSpeedMultiplier);
    float demo_duration_s = std::max(0.6f, std::min(2.2f / max_multiplier, 3.0f));
    float base_px_s = near_px / demo_duration_s;
    vel_px = base_px_s * static_cast<float>(speed_mult);
    dist_px -= vel_px * static_cast<float>(dt);
    if (dist_px <= 0.0f)
        dist_px = near_px;

    overlay->FillCircle({ centerX - dist_px, centerY, 4.0f }, ARGB(255, 255, 255, 80));
}


void createInputDevices()
{
    if (arduinoSerial)
    {
        delete arduinoSerial;
        arduinoSerial = nullptr;
    }

    if (gHub)
    {
        gHub->mouse_close();
        delete gHub;
        gHub = nullptr;
    }

    if (kmboxNetSerial)
    {
        delete kmboxNetSerial;
        kmboxNetSerial = nullptr;
    }

    if (kmboxASerial)
    {
        delete kmboxASerial;
        kmboxASerial = nullptr;
    }

    if (makcuSerial)
    {
        delete makcuSerial;
        makcuSerial = nullptr;
    }

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        arduinoSerial = new Arduino(config.arduino_port, config.arduino_baudrate);
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0))
        {
            std::cerr << "[Ghub] Error with opening mouse." << std::endl;
            delete gHub;
            gHub = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_NET")
    {
        std::cout << "[Mouse] Using KMBOX_NET input." << std::endl;
        kmboxNetSerial = new KmboxNetConnection(config.kmbox_net_ip, config.kmbox_net_port, config.kmbox_net_uuid);
        if (!kmboxNetSerial->isOpen())
        {
            std::cerr << "[KmboxNet] Error connecting." << std::endl;
            delete kmboxNetSerial; kmboxNetSerial = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_A")
    {
        std::cout << "[Mouse] Using KMBOX_A input." << std::endl;
        if (config.kmbox_a_pidvid.empty())
        {
            std::cerr << "[KmboxA] PIDVID is empty." << std::endl;
            return;
        }
        kmboxASerial = new KmboxAConnection(config.kmbox_a_pidvid);
        if (!kmboxASerial->isOpen())
        {
            std::cerr << "[KmboxA] Error connecting." << std::endl;
            delete kmboxASerial;
            kmboxASerial = nullptr;
        }
    }
    else if (config.input_method == "MAKCU")
    {
        std::cout << "[Mouse] Using MAKCU input." << std::endl;
        makcuSerial = new MakcuConnection(config.makcu_port, config.makcu_baudrate);
        if (!makcuSerial->isOpen())
        {
            std::cerr << "[Makcu] Error connecting." << std::endl;
            delete makcuSerial;
            makcuSerial = nullptr;
        }
    }
    else
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    }
}

void assignInputDevices()
{
    if (globalMouseThread)
    {
        globalMouseThread->setArduinoConnection(arduinoSerial);
        globalMouseThread->setGHubMouse(gHub);
        globalMouseThread->setKmboxAConnection(kmboxASerial);
        globalMouseThread->setKmboxNetConnection(kmboxNetSerial);
        globalMouseThread->setMakcuConnection(makcuSerial);
    }
}

void handleEasyNoRecoil(MouseThread& mouseThread)
{
    if (config.easynorecoil && shooting.load() && zooming.load())
    {
        std::lock_guard<std::mutex> lock(mouseThread.input_method_mutex);
        int recoil_compensation = static_cast<int>(config.easynorecoilstrength);
        
        if (arduinoSerial)
        {
            arduinoSerial->move(0, recoil_compensation);
        }
        else if (gHub)
        {
            gHub->mouse_xy(0, recoil_compensation);
        }
        else if (kmboxNetSerial)
        {
            kmboxNetSerial->move(0, recoil_compensation);
        }
        else if (kmboxASerial)
        {
            kmboxASerial->move(0, recoil_compensation);
        }
        else if (makcuSerial)
        {
            makcuSerial->move(0, recoil_compensation);
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dx = 0;
            input.mi.dy = recoil_compensation;
            input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
            SendInput(1, &input, sizeof(INPUT));
        }
    }
}

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;

    while (!shouldExit)
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        {
            std::unique_lock<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.cv.wait(lock, [&] {
                return detectionBuffer.version > lastVersion || shouldExit;
                }
            );

            if (shouldExit) break;
            boxes = detectionBuffer.boxes;
            classes = detectionBuffer.classes;
            lastVersion = detectionBuffer.version;
        }

        if (input_method_changed.load())
        {
            createInputDevices();
            assignInputDevices();
            input_method_changed.store(false);
        }

        if (detection_resolution_changed.load())
        {
            {
                std::lock_guard<std::mutex> cfgLock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.fovX,
                    config.fovY,
                    config.minSpeedMultiplier,
                    config.maxSpeedMultiplier,
                    config.predictionInterval,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            detection_resolution_changed.store(false);
        }

        AimbotTarget* target = sortTargets(
            boxes,
            classes,
            config.detection_resolution,
            config.detection_resolution,
            config.disable_headshot
        );

        if (target)
        {
            mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
            mouseThread.setTargetDetected(true);

            auto futurePositions = mouseThread.predictFuturePositions(
                target->pivotX,
                target->pivotY,
                config.prediction_futurePositions
            );
            mouseThread.storeFuturePositions(futurePositions);
        }
        else
        {
            mouseThread.clearFuturePositions();
            mouseThread.setTargetDetected(false);
        }

        if (aiming)
        {
            if (target)
            {
                mouseThread.moveMousePivot(target->pivotX, target->pivotY);

                if (config.auto_shoot)
                {
                    mouseThread.pressMouse(*target);
                }
            }
            else
            {
                if (config.auto_shoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }
        else
        {
            if (config.auto_shoot)
            {
                mouseThread.releaseMouse();
            }
        }

        handleEasyNoRecoil(mouseThread);

        mouseThread.checkAndResetPredictions();

        delete target;
    }
}

static void gameOverlayRenderLoop()
{
    const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);

    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    HMONITOR hPrimary = MonitorFromPoint(POINT{ 0,0 }, MONITOR_DEFAULTTOPRIMARY);
    GetMonitorInfo(hPrimary, &mi);
    RECT pr = mi.rcMonitor;
    const int pw = pr.right - pr.left;
    const int ph = pr.bottom - pr.top;

    const int offX = pr.left - vx;
    const int offY = pr.top - vy;

#ifdef USE_CUDA
    static depth_anything::DepthAnythingTrt depthDebugModel;
    static std::string depthDebugModelPath;
    static int depthDebugColormap = -1;
    static int depthDebugImageId = 0;
    static int depthMaskImageId = 0;
    static cv::Mat depthDebugFrame;
    static auto lastDepthUpdate = std::chrono::steady_clock::time_point::min();
    static bool lastDepthInferenceEnabled = true;
#endif
    int lastDetectionVersion = -1;

    while (!gameOverlayShouldExit.load())
    {
        if (!config.game_overlay_enabled)
        {
            if (gameOverlayPtr)
            {
                gameOverlayPtr->Stop();
                delete gameOverlayPtr;
                gameOverlayPtr = nullptr;
                std::lock_guard<std::mutex> lk(g_iconMutex);
                g_lastIconPath.clear();
                g_iconImageId = 0;
                g_iconLastError.clear();
            }
#ifdef USE_CUDA
            depthDebugModel.reset();
            depthDebugModelPath.clear();
            depthDebugColormap = -1;
            depthDebugImageId = 0;
            depthMaskImageId = 0;
            depthDebugFrame.release();
            lastDepthUpdate = std::chrono::steady_clock::time_point::min();
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            continue;
        }

        if (!gameOverlayPtr)
        {
            gameOverlayPtr = new Game_overlay();
            gameOverlayPtr->SetWindowBounds(0, 0, pw, ph);
            gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);
            gameOverlayPtr->Start();
        }
        else if (!gameOverlayPtr->IsRunning())
        {
            gameOverlayPtr->SetWindowBounds(0, 0, pw, ph);
            gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);
            gameOverlayPtr->Start();
        }

        if (!gameOverlayPtr || !gameOverlayPtr->IsRunning())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            continue;
        }

        gameOverlayPtr->SetMaxFPS(config.game_overlay_max_fps > 0 ? (unsigned)config.game_overlay_max_fps : 0);

        const int detRes = config.detection_resolution;
        if (detRes <= 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        int regionW = detRes;
        int regionH = detRes;

        if (regionW > pw) regionW = pw;
        if (regionH > ph) regionH = ph;

        const int baseX = (pw - regionW) / 2;
        const int baseY = (ph - regionH) / 2;

        const float scaleX = detRes > 0 ? (static_cast<float>(regionW) / static_cast<float>(detRes)) : 1.0f;
        const float scaleY = detRes > 0 ? (static_cast<float>(regionH) / static_cast<float>(detRes)) : 1.0f;

        std::vector<cv::Rect> boxesCopy;
        std::vector<int> classesCopy;
        int detectionVersion = lastDetectionVersion;
        {
            std::unique_lock<std::mutex> lk(detectionBuffer.mutex);
            const unsigned fpsCap = (unsigned)config.game_overlay_max_fps;
            const int waitMs = (fpsCap > 0) ? static_cast<int>(std::max(1u, 1000u / fpsCap)) : 8;
            detectionBuffer.cv.wait_for(lk, std::chrono::milliseconds(waitMs), [&] {
                return detectionBuffer.version != lastDetectionVersion || gameOverlayShouldExit.load();
                });
            boxesCopy = detectionBuffer.boxes;
            classesCopy = detectionBuffer.classes;
            detectionVersion = detectionBuffer.version;
        }
        lastDetectionVersion = detectionVersion;

        decltype(globalMouseThread->getFuturePositions()) futurePts;
        if (config.game_overlay_draw_future && globalMouseThread)
            futurePts = globalMouseThread->getFuturePositions();

        if (config.game_overlay_icon_enabled)
        {
            std::lock_guard<std::mutex> lk(g_iconMutex);
            if (config.game_overlay_icon_path != g_lastIconPath)
            {
                if (g_iconImageId != 0)
                {
                    gameOverlayPtr->UnloadImage(g_iconImageId);
                    g_iconImageId = 0;
                }
                g_lastIconPath = config.game_overlay_icon_path;
                std::error_code fsErr;
                std::filesystem::path p;
                try
                {
                    p = std::filesystem::u8path(g_lastIconPath);
                }
                catch (const std::exception&)
                {
                    p = std::filesystem::path(g_lastIconPath);
                }
                const bool hasFile = !g_lastIconPath.empty() && p.has_filename() && std::filesystem::is_regular_file(p, fsErr);
                if (fsErr)
                {
                    g_iconImageId = 0;
                    g_iconLastError = "[GameOverlay] Failed to read icon path: " + g_lastIconPath + " (" + fsErr.message() + ")";
                    std::cerr << g_iconLastError << std::endl;
                }
                else if (hasFile)
                {
                    const std::wstring wpath = p.wstring();
                    g_iconLastError.clear();

                    UINT iw = 0, ih = 0;
                    std::string verr;
                    if (!IsValidImageFile(wpath, iw, ih, verr))
                    {
                        g_iconImageId = 0;
                        g_iconLastError = "[GameOverlay] Invalid image '" + g_lastIconPath + "': " + verr;
                        std::cerr << g_iconLastError << std::endl;
                    }
                    else
                    {
                        try
                        {
                            int id = gameOverlayPtr->LoadImageFromFile(wpath);
                            if (id != 0)
                            {
                                g_iconImageId = id;
                                std::cout << "[GameOverlay] Loaded icon (" << iw << "x" << ih << "): " << g_lastIconPath << std::endl;
                            }
                            else
                            {
                                g_iconImageId = 0;
                                g_iconLastError = "[GameOverlay] Failed to load icon (loader returned 0): " + g_lastIconPath;
                                std::cerr << g_iconLastError << std::endl;
                            }
                        }
                        catch (const std::exception& e)
                        {
                            g_iconImageId = 0;
                            g_iconLastError = std::string("[GameOverlay] Exception while loading icon: ") + e.what();
                            std::cerr << g_iconLastError << std::endl;
                        }
                        catch (...)
                        {
                            g_iconImageId = 0;
                            g_iconLastError = "[GameOverlay] Unknown exception while loading icon.";
                            std::cerr << g_iconLastError << std::endl;
                        }
                    }
                }
                else
                {
                    g_iconImageId = 0;
                    g_iconLastError = "[GameOverlay] Icon file not found: " + g_lastIconPath;
                    std::cerr << g_iconLastError << std::endl;
                }
            }
        }

        gameOverlayPtr->BeginFrame();

#ifdef USE_CUDA
        if (!config.depth_inference_enabled)
        {
            if (lastDepthInferenceEnabled)
            {
                if (gameOverlayPtr)
                {
                    if (depthDebugImageId != 0)
                    {
                        gameOverlayPtr->UnloadImage(depthDebugImageId);
                        depthDebugImageId = 0;
                    }
                    if (depthMaskImageId != 0)
                    {
                        gameOverlayPtr->UnloadImage(depthMaskImageId);
                        depthMaskImageId = 0;
                    }
                }

                depthDebugModel.reset();
                depthDebugModelPath.clear();
                depthDebugColormap = -1;
                depthDebugFrame.release();
                lastDepthUpdate = std::chrono::steady_clock::time_point::min();

                auto& depthMask = depth_anything::GetDepthMaskGenerator();
                depthMask.reset();
            }
            lastDepthInferenceEnabled = false;
        }
        else
        {
            lastDepthInferenceEnabled = true;

            if (config.depth_debug_overlay_enabled)
            {
                cv::Mat frameCopy;
                {
                    std::lock_guard<std::mutex> lk(frameMutex);
                    if (!latestFrame.empty())
                        latestFrame.copyTo(frameCopy);
                }

                if (config.depth_model_path.empty())
                {
                    if (depthDebugModel.ready())
                        depthDebugModel.reset();
                    depthDebugModelPath.clear();
                }
                else if (depthDebugModelPath != config.depth_model_path || !depthDebugModel.ready())
                {
                    if (depthDebugModel.initialize(config.depth_model_path, gLogger))
                    {
                        depthDebugModelPath = config.depth_model_path;
                    }
                }

                if (config.depth_colormap != depthDebugColormap)
                {
                    depthDebugModel.setColormap(config.depth_colormap);
                    depthDebugColormap = config.depth_colormap;
                }

                if (depthDebugModel.ready() && !frameCopy.empty())
                {
                    auto now = std::chrono::steady_clock::now();
                    bool shouldUpdate = depthDebugFrame.empty();
                    if (config.depth_fps <= 0)
                    {
                        shouldUpdate = true;
                    }
                    else if (!shouldUpdate)
                    {
                        auto interval = std::chrono::milliseconds(1000 / config.depth_fps);
                        shouldUpdate = (now - lastDepthUpdate) >= interval;
                    }
                    if (shouldUpdate)
                    {
                        cv::Mat depthFrame = depthDebugModel.predict(frameCopy);
                        if (!depthFrame.empty())
                        {
                            depthDebugFrame = depthFrame;
                            lastDepthUpdate = now;
                        }
                    }
                }

                float depthW = 0.0f;
                float depthH = 0.0f;
                if (!depthDebugFrame.empty())
                {
                    cv::Mat depthBGRA;
                    cv::cvtColor(depthDebugFrame, depthBGRA, cv::COLOR_BGR2BGRA);
                    int newId = gameOverlayPtr->UpdateImageFromBGRA(
                        depthBGRA.data,
                        depthBGRA.cols,
                        depthBGRA.rows,
                        static_cast<int>(depthBGRA.step),
                        depthDebugImageId);
                    if (newId != 0)
                        depthDebugImageId = newId;
                    depthW = static_cast<float>(regionW);
                    depthH = static_cast<float>(regionH);
                }

                float maskW = 0.0f;
                float maskH = 0.0f;
                if (config.depth_mask_enabled)
                {
                    auto& depthMask = depth_anything::GetDepthMaskGenerator();
                    cv::Mat mask = depthMask.getMask();
                    if (!mask.empty())
                    {
                        cv::Mat alpha(mask.size(), CV_8U, cv::Scalar(0));
                        alpha.setTo(config.depth_mask_alpha, mask);

                        std::vector<cv::Mat> channels(4);
                        channels[0] = alpha;
                        channels[1] = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
                        channels[2] = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
                        channels[3] = alpha;

                        cv::Mat maskBGRA;
                        cv::merge(channels, maskBGRA);

                        int newId = gameOverlayPtr->UpdateImageFromBGRA(
                            maskBGRA.data,
                            maskBGRA.cols,
                            maskBGRA.rows,
                            static_cast<int>(maskBGRA.step),
                            depthMaskImageId);
                        if (newId != 0)
                            depthMaskImageId = newId;

                        maskW = static_cast<float>(regionW);
                        maskH = static_cast<float>(regionH);
                    }
                }

                if (depthDebugImageId != 0 || depthMaskImageId != 0)
                {
                    const float pad = 8.0f;
                    float depthX = static_cast<float>(baseX);
                    float depthY = static_cast<float>(baseY);
                    float maskX = depthX;
                    float maskY = depthY;

                    if (depthDebugImageId != 0 && depthW > 0.0f && depthH > 0.0f)
                    {
                        gameOverlayPtr->DrawImage(depthDebugImageId, depthX, depthY, depthW, depthH, 1.0f);
                        gameOverlayPtr->AddRect({ depthX, depthY, depthW, depthH }, ARGB(120, 255, 255, 255), 1.0f);
                    }

                    if (depthMaskImageId != 0 && maskW > 0.0f && maskH > 0.0f)
                    {
                        gameOverlayPtr->DrawImage(depthMaskImageId, maskX, maskY, maskW, maskH, 1.0f);
                        gameOverlayPtr->AddText(maskX + pad, maskY + pad + 18.0f, L"Depth mask", 16.0f, ARGB(220, 255, 255, 255));
                    }
                }
            }
        }
#endif

        // CAPTURE FRAME
        if (config.game_overlay_draw_frame)
        {
            int A = config.game_overlay_frame_a;
            int R = config.game_overlay_frame_r;
            int G = config.game_overlay_frame_g;
            int B = config.game_overlay_frame_b;
            auto clamp255 = [](int& v) { if (v < 0) v = 0; else if (v > 255) v = 255; };
            clamp255(A); clamp255(R); clamp255(G); clamp255(B);
            const uint32_t col =
                (uint32_t(A) << 24) |
                (uint32_t(R) << 16) |
                (uint32_t(G) << 8) |
                uint32_t(B);

            float thickness = config.game_overlay_frame_thickness;
            if (thickness <= 0.f) thickness = 1.f;

            if (config.circle_mask)
            {
                float cx = baseX + regionW * 0.5f;
                float cy = baseY + regionH * 0.5f;
                float radius = std::min(regionW, regionH) * 0.5f;
                gameOverlayPtr->AddCircle({ cx, cy, radius }, col, thickness);
            }
            else
            {
                gameOverlayPtr->AddRect(
                    { static_cast<float>(baseX),
                      static_cast<float>(baseY),
                      static_cast<float>(regionW),
                      static_cast<float>(regionH) },
                    col, thickness);
            }
        }

        // BOXES
        if (config.game_overlay_draw_boxes && !boxesCopy.empty())
        {
            int A = config.game_overlay_box_a;
            int R = config.game_overlay_box_r;
            int G = config.game_overlay_box_g;
            int B = config.game_overlay_box_b;
            auto clamp255 = [](int& v) { if (v < 0) v = 0; else if (v > 255) v = 255; };
            clamp255(A); clamp255(R); clamp255(G); clamp255(B);
            const uint32_t col =
                (uint32_t(A) << 24) |
                (uint32_t(R) << 16) |
                (uint32_t(G) << 8) |
                uint32_t(B);

            float thickness = config.game_overlay_box_thickness;
            if (thickness <= 0.f) thickness = 1.f;

            for (const auto& b : boxesCopy)
            {
                if (b.width <= 0 || b.height <= 0) continue;

                int bx = std::max(0, std::min(b.x, detRes));
                int by = std::max(0, std::min(b.y, detRes));
                int bw = std::max(0, std::min(b.width, detRes - bx));
                int bh = std::max(0, std::min(b.height, detRes - by));
                if (bw == 0 || bh == 0) continue;

                float x = baseX + bx * scaleX;
                float y = baseY + by * scaleY;
                float w = bw * scaleX;
                float h = bh * scaleY;

                if (x + w < baseX || y + h < baseY ||
                    x > baseX + regionW || y > baseY + regionH)
                    continue;

                gameOverlayPtr->AddRect({ x, y, w, h }, col, thickness);
            }
        }

        // FUTURE POINTS
        if (config.game_overlay_draw_future && !futurePts.empty())
        {
            const int total = static_cast<int>(futurePts.size());
            const int baseA = std::max(5, std::min(255, config.game_overlay_box_a));

            for (int i = 0; i < total; ++i)
            {
                float alphaFactor =
                    std::exp(-config.game_overlay_future_alpha_falloff *
                        (static_cast<float>(i) / static_cast<float>(total)));

                int a = static_cast<int>(baseA * alphaFactor);
                if (a < 12) a = 12;

                const uint32_t col =
                    (uint32_t(a) << 24) |
                    (uint32_t(255 - (i * 255 / total)) << 16) |
                    (uint32_t(50) << 8) |
                    (uint32_t(i * 255 / total));

                float px = static_cast<float>(baseX) + static_cast<float>(futurePts[i].first) * scaleX;
                float py = static_cast<float>(baseY) + static_cast<float>(futurePts[i].second) * scaleY;

                if (px < baseX - 40 || py < baseY - 40 ||
                    px > baseX + regionW + 40 || py > baseY + regionH + 40)
                    continue;

                gameOverlayPtr->FillCircle({ px, py, config.game_overlay_future_point_radius }, col);
            }
        }

        // ICONS
        if (config.game_overlay_icon_enabled && g_iconImageId != 0 && !boxesCopy.empty())
        {
            const int iconW = config.game_overlay_icon_width;
            const int iconH = config.game_overlay_icon_height;
            const float offXIcon = config.game_overlay_icon_offset_x;
            const float offYIcon = config.game_overlay_icon_offset_y;
            std::string anchor = config.game_overlay_icon_anchor;
            const int wantedClass = config.game_overlay_icon_class;
            const size_t count = boxesCopy.size();
            for (size_t i = 0; i < count; ++i)
            {
                const auto& b = boxesCopy[i];
                int cls = (i < classesCopy.size()) ? classesCopy[i] : -1;
                // Class filter (-1 = all)
                if (wantedClass >= 0 && cls != wantedClass)
                {
                    continue;
                }

                if (b.width <= 0 || b.height <= 0) continue;
                int bx = std::max(0, std::min(b.x, detRes));
                int by = std::max(0, std::min(b.y, detRes));
                int bw = std::max(0, std::min(b.width, detRes - bx));
                int bh = std::max(0, std::min(b.height, detRes - by));
                if (bw == 0 || bh == 0) continue;

                float boxX = baseX + bx * scaleX;
                float boxY = baseY + by * scaleY;
                float boxW = bw * scaleX;
                float boxH = bh * scaleY;

                float drawX = boxX;
                float drawY = boxY;

                if (anchor == "center")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH / 2.0f - iconH / 2.0f;
                }
                else if (anchor == "top" || anchor == "head")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY - iconH;
                }
                else if (anchor == "bottom")
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH;
                }
                else
                {
                    drawX = boxX + boxW / 2.0f - iconW / 2.0f;
                    drawY = boxY + boxH / 2.0f - iconH / 2.0f;
                }

                drawX += offXIcon;
                drawY += offYIcon;

                gameOverlayPtr->DrawImage(g_iconImageId, drawX, drawY, (float)iconW, (float)iconH, 1.0f);
            }
        }

        if (config.game_overlay_show_target_correction)
        {
            draw_target_correction_demo_game_overlay(
                gameOverlayPtr,
                static_cast<float>(baseX) + regionW * 0.5f,
                static_cast<float>(baseY) + regionH * 0.5f);
        }

        gameOverlayPtr->EndFrame();
    }

    if (gameOverlayPtr)
    {
        gameOverlayPtr->Stop();
        delete gameOverlayPtr;
        gameOverlayPtr = nullptr;
    }
}

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetRandomConsoleTitle();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    {
        wchar_t exePath[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, exePath, MAX_PATH) > 0)
        {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            std::error_code ec;
            std::filesystem::current_path(exeDir, ec);
            if (ec && config.verbose)
            {
                std::cout << "[Config] Failed to set working dir: " << exeDir.u8string()
                          << " (" << ec.message() << ")" << std::endl;
            }
        }
    }

    if (!config.loadConfig())
    {
        std::cerr << "[Config] Error with loading config!" << std::endl;
        std::cin.get();
        return -1;
    }

    CPUAffinityManager cpuManager;

    if (config.cpuCoreReserveCount > 0)
    {
        if (!cpuManager.reserveCPUCores(config.cpuCoreReserveCount))
            return FatalExit("[MAIN] Failed to reserve CPU cores.");
    }

    if (config.systemMemoryReserveMB > 0)
    {
        if (!cpuManager.reserveSystemMemory(config.systemMemoryReserveMB))
            return FatalExit("[MAIN] Failed to reserve system memory.");
    }

    try
    {
#ifdef USE_CUDA
        int cuda_runtime_version = 0;
        cudaError_t runtime_status = cudaRuntimeGetVersion(&cuda_runtime_version);

        if (runtime_status != cudaSuccess)
        {
            std::cerr << "[MAIN] CUDA runtime check failed: " << cudaGetErrorString(runtime_status) << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.verbose)
            std::cout << "[CUDA] Version: " << cuda_runtime_version << std::endl;

        const int required_cuda_version = 13010;
        if (cuda_runtime_version < required_cuda_version)
        {
            int required_major = required_cuda_version / 1000;
            int required_minor = (required_cuda_version % 1000) / 10;
            int runtime_major = cuda_runtime_version / 1000;
            int runtime_minor = (cuda_runtime_version % 1000) / 10;
            std::cerr << "[MAIN] CUDA " << required_major << "." << required_minor
                << " required. Detected " << runtime_major << "." << runtime_minor << "." << std::endl;
            const wchar_t* title = L"CUDA Update Required";
            std::wstring message =
                L"An outdated CUDA version was detected. "
                L"Please update your graphics drivers to the latest version "
                L"and install CUDA 13.1.\n\n"
                L"The program will now attempt to continue.";
            MessageBoxW(nullptr, message.c_str(), title, MB_OK | MB_ICONWARNING);
        }

        GPUResourceManager gpuManager;
        if (config.gpuMemoryReserveMB > 0)
        {
            if (!gpuManager.reserveGPUMemory(config.gpuMemoryReserveMB))
                return FatalExit("[MAIN] Failed to reserve GPU memory.");
        }
        
        if (config.enableGpuExclusiveMode)
        {
            if (!gpuManager.setGPUExclusiveMode())
                return FatalExit("[MAIN] Failed to set GPU exclusive mode.");
        }

        int cuda_devices = 0;
        if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0)
        {
            std::cerr << "[MAIN] CUDA required but no devices found." << std::endl;
            std::cin.get();
            return -1;
        }
#endif
        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"models", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models folder" << std::endl;
            std::cin.get();
            return -1;
        }
        if (!CreateDirectory(L"models\\depth", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models\\depth folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.capture_method == "virtual_camera")
        {
            auto cams = VirtualCameraCapture::GetAvailableVirtualCameras();
            if (!cams.empty())
            {
                if (config.virtual_camera_name == "None" ||
                    std::find(cams.begin(), cams.end(), config.virtual_camera_name) == cams.end())
                {
                    config.virtual_camera_name = cams[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Set virtual_camera_name = " << config.virtual_camera_name << std::endl;
                }
                std::cout << "[MAIN] Virtual cameras loaded: " << cams.size() << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No virtual cameras found" << std::endl;
            }
        }

        std::string modelPath = "models/" + config.ai_model;

        if (!std::filesystem::exists(modelPath))
        {
            std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

            std::vector<std::string> modelFiles = getModelFiles();

            if (!modelFiles.empty())
            {
                config.ai_model = modelFiles[0];
                config.saveConfig();
                std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        createInputDevices();

        MouseThread mouseThread(
            config.detection_resolution,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier,
            arduinoSerial,
            gHub,
            kmboxASerial,
            kmboxNetSerial,
            makcuSerial
        );

        globalMouseThread = &mouseThread;
        assignInputDevices();

        std::vector<std::string> availableModels = getAvailableModels();

        if (!config.ai_model.empty())
        {
            std::string modelPath = "models/" + config.ai_model;
            if (!std::filesystem::exists(modelPath))
            {
                std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

                if (!availableModels.empty())
                {
                    config.ai_model = availableModels[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
                }
                else
                {
                    std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }
        else
        {
            if (!availableModels.empty())
            {
                config.ai_model = availableModels[0];
                config.saveConfig();
                std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No AI models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        std::thread dml_detThread;

        if (config.backend == "DML")
        {
            dml_detector = new DirectMLDetector("models/" + config.ai_model);
            std::cout << "[MAIN] DML detector initialized." << std::endl;
            dml_detThread = StartThreadGuarded("DmlDetector", [] {
                dml_detector->dmlInferenceThread();
                });
        }
#ifdef USE_CUDA
        else
        {
            trt_detector.initialize("models/" + config.ai_model);
        }
#endif

        detection_resolution_changed.store(true);

        std::thread keyThread = StartThreadGuarded("KeyboardListener", [] {
            keyboardListener();
            });
        std::thread capThread = StartThreadGuarded("CaptureThread", [] {
            captureThread(config.detection_resolution, config.detection_resolution);
            });

#ifdef USE_CUDA
        std::thread trt_detThread = StartThreadGuarded("TrtDetector", [] {
            trt_detector.inferenceThread();
            });
#endif
        std::thread mouseMovThread = StartThreadGuarded("MouseThread", [&mouseThread] {
            mouseThreadFunction(mouseThread);
            });
        std::thread overlayThread = StartThreadGuarded("OverlayThread", [] {
            OverlayThread();
            });

        gameOverlayShouldExit.store(false);
        gameOverlayThread = StartThreadGuarded("GameOverlay", [] {
            gameOverlayRenderLoop();
            });

        welcome_message();

        keyThread.join();
        capThread.join();
        if (dml_detThread.joinable())
        {
            dml_detector->shouldExit = true;
            dml_detector->inferenceCV.notify_all();
            dml_detThread.join();
        }

#ifdef USE_CUDA
        trt_detThread.join();
#endif
        mouseMovThread.join();
        overlayThread.join();

        if (arduinoSerial)
        {
            delete arduinoSerial;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
        }

        if (kmboxASerial)
        {
            delete kmboxASerial;
            kmboxASerial = nullptr;
        }

        if (dml_detector)
        {
            delete dml_detector;
            dml_detector = nullptr;
        }

        gameOverlayShouldExit.store(true);
        if (gameOverlayThread.joinable()) gameOverlayThread.join();
        if (gameOverlayPtr)
        {
            gameOverlayPtr->Stop();
            delete gameOverlayPtr;
            gameOverlayPtr = nullptr;
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}
