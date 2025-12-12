#ifndef OVERLAY_H
#define OVERLAY_H

#include <d3d11.h>
#include <vector>
#include <string>
#include <atomic>

#include "sunone_aimbot_cpp.h"

extern ID3D11Device* g_pd3dDevice;
extern ID3D11DeviceContext* g_pd3dDeviceContext;
extern IDXGISwapChain* g_pSwapChain;
extern ID3D11RenderTargetView* g_mainRenderTargetView;
extern HWND g_hwnd;

void OverlayThread();

extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::atomic<bool> capture_window_changed;
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> show_window_changed;

extern std::vector<std::string> key_names;
extern std::vector<const char*> key_names_cstrs;

extern const int BASE_OVERLAY_WIDTH;
extern const int BASE_OVERLAY_HEIGHT;
extern int overlayWidth;
extern int overlayHeight;

#endif // OVERLAY_H