#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

// for browser
#include "shellapi.h"

#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <d3d11.h>
#include <dxgi.h>
#include <sys/stat.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include "config.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "memory_images.h"

static ID3D11Device* g_pd3dDevice = NULL;
static ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
static IDXGISwapChain* g_pSwapChain = NULL;
static ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
static HWND g_hwnd = NULL;

extern Config config;
extern std::mutex configMutex;
extern std::atomic<bool> shouldExit;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();

ID3D11BlendState* g_pBlendState = nullptr;
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int overlayWidth = 680;
int overlayHeight = 480;

// init vars
std::vector<std::string> engine_models;
__int64 prev_ai_model_index;
__int64 current_ai_model_index;
int prev_imgsz_index;
int selected_imgsz;

// export model
std::string tensorrt_path = get_tensorrt_path();
bool tensorrt_path_found = !tensorrt_path.empty();
std::string tensorrt_bin_path;
std::string trtexec_path;
std::vector<std::string> onnx_models;
int onnx_model_index = 0;
std::string output_engine_file;

static std::atomic<bool> is_exporting(false);
static std::mutex export_mutex;
std::string export_status_message;

// Realtime config vars
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> show_window_changed;

// textures
ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

bool InitializeBlendState()
{
    D3D11_BLEND_DESC blendDesc;
    ZeroMemory(&blendDesc, sizeof(blendDesc));

    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    HRESULT hr = g_pd3dDevice->CreateBlendState(&blendDesc, &g_pBlendState);
    if (FAILED(hr))
        return false;

    float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
    g_pd3dDeviceContext->OMSetBlendState(g_pBlendState, blendFactor, 0xffffffff);

    return true;
}

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = overlayWidth;
    sd.BufferDesc.Height = overlayHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 0;
    sd.BufferDesc.RefreshRate.Denominator = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };

    HRESULT res = D3D11CreateDeviceAndSwapChain(NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        createDeviceFlags,
        featureLevelArray,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        &featureLevel,
        &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    if (!InitializeBlendState())
        return false;

    CreateRenderTarget();
    return true;
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer = NULL;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
    if (g_pBlendState) { g_pBlendState->Release(); g_pBlendState = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            RECT rect;
            if (GetClientRect(hWnd, &rect))
            {
                UINT width = rect.right - rect.left;
                UINT height = rect.bottom - rect.top;

                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
        }
        return 0;
    case WM_DESTROY:
        shouldExit = true;
        ::PostQuitMessage(0);
        return 0;
    default:
        return ::DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

void SetupImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ImGui::StyleColorsDark();

    // Other setups
    // load available engine models
    engine_models = getEngineFiles();
    prev_ai_model_index = getModelIndex(engine_models);
    current_ai_model_index = prev_ai_model_index;

    // ai model image sizes
    int model_sizes[] = { 320, 480, 640 };
    const int model_sizes_count = sizeof(model_sizes) / sizeof(model_sizes[0]);

    selected_imgsz = getImageSizeIndex(config.engine_image_size, model_sizes, model_sizes_count);
    prev_imgsz_index = selected_imgsz;
    
    // Load body texture
    int image_width = 0;
    int image_height = 0;

    std::string body_image = std::string(bodyImageBase64_1) + std::string(bodyImageBase64_2) + std::string(bodyImageBase64_3);

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (!ret)
    {
        std::cerr << "[Overlay] Can't load image!" << std::endl;
    }
    else
    {
        bodyImageSize = ImVec2((float)image_width, (float)image_height);
    }
}

bool CreateOverlayWindow()
{
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      _T("Edge"), NULL };
    ::RegisterClassEx(&wc);

    g_hwnd = ::CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED,
        wc.lpszClassName, _T("Chrome"),
        WS_POPUP, 0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);

    if (g_hwnd == NULL)
        return false;
    
    // Opacity checks and init
    if (config.overlay_opacity <= 0)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to less than one, this value is unacceptable." << std::endl;
        std::cin.get();
        return false;
    }

    if (config.overlay_opacity >= 256)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to more than 255, this value is unacceptable." << std::endl;
        std::cin.get();
        return false;
    }

    BYTE opacity = config.overlay_opacity;

    SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);

    if (!CreateDeviceD3D(g_hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

    return true;
}

void OverlayThread()
{
    if (!CreateOverlayWindow())
    {
        std::cerr << "Can't create overlay window" << std::endl;
        return;
    }

    SetupImGui();

    bool show_overlay = false;

    // Real time settings vars
    static int prev_detection_resolution = config.detection_resolution;
    
    bool prev_capture_method = config.duplication_api;
    bool prev_capture_cursor = config.capture_cursor;
    bool prev_capture_borders = config.capture_borders;
    int prev_capture_fps = config.capture_fps;

    bool prev_show_window = config.show_window;
    int prev_opacity = config.overlay_opacity;

    int current_ai_model_index = prev_ai_model_index;
    float prev_confidence_threshold = config.confidence_threshold;
    float prev_nms_threshold = config.nms_threshold;

    bool prev_disable_headshot = config.disable_headshot;
    float prev_body_y_offset = config.body_y_offset;
    bool prev_ignore_third_person = config.ignore_third_person;

    int prev_dpi = config.dpi;
    float prev_sensitivity = config.sensitivity;
    int prev_fovX = config.fovX;
    int prev_fovY = config.fovY;
    float prev_minSpeedMultiplier = config.minSpeedMultiplier;
    float prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
    float prev_predictionInterval = config.predictionInterval;

    bool prev_auto_shoot = config.auto_shoot;
    float prev_bScope_multiplier = config.bScope_multiplier;

    bool prev_show_fps = config.show_fps;
    int prev_window_size = config.window_size;
    int prev_screenshot_delay = config.screenshot_delay;
    bool prev_always_on_top = config.always_on_top;

    // keycodes
    std::vector<std::string> key_names;
    for (const auto& pair : KeyCodes::key_code_map)
    {
        key_names.push_back(pair.first);
    }
    std::sort(key_names.begin(), key_names.end());

    std::vector<const char*> key_names_cstrs;
    key_names_cstrs.reserve(key_names.size());
    for (const auto& name : key_names)
    {
        key_names_cstrs.push_back(name.c_str());
    }

    // input methods
    int input_method_index = 0;
    if (config.input_method == "WIN32")
        input_method_index = 0;
    else if (config.input_method == "GHUB")
        input_method_index = 1;
    else if (config.input_method == "ARDUINO")
        input_method_index = 2;
    else
        input_method_index = 0;

    std::string ghub_version = get_ghub_version();

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (!shouldExit)
    {
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
            {
                shouldExit = true;
                return;
            }
        }

        if (isAnyKeyPressed(config.button_open_overlay) & 0x1)
        {
            show_overlay = !show_overlay;

            if (show_overlay)
            {
                ShowWindow(g_hwnd, SW_SHOW);
                SetForegroundWindow(g_hwnd);
            }
            else
            {
                ShowWindow(g_hwnd, SW_HIDE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        if (show_overlay)
        {
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2((float)overlayWidth, (float)overlayHeight));

            ImGui::Begin("Options", &show_overlay, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            {
                std::lock_guard<std::mutex> lock(configMutex);
                if (ImGui::BeginTabBar("Options tab bar"))
                {
                    // ********************************************* CAPTURE ********************************************
                    if (ImGui::BeginTabItem("Capture"))
                    {
                        ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 720);
                        ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240);
                        ImGui::Checkbox("Duplication API", &config.duplication_api);
                        if (config.duplication_api == false)
                        {
                            ImGui::Checkbox("Capture Borders", &config.capture_borders);
                            ImGui::Checkbox("Capture Cursor", &config.capture_cursor);
                        }

                        ImGui::EndTabItem();
                    }

                    // ********************************************* TARGET *********************************************
                    if (ImGui::BeginTabItem("Target"))
                    {
                        ImGui::Checkbox("Disable Headshot", &config.disable_headshot);

                        ImGui::Separator();

                        ImGui::SliderFloat("Approximate Body Y Offset", &config.body_y_offset, 0.0f, 1.0f, "%.2f");
                        if (bodyTexture)
                        {
                            ImGui::Image((void*)bodyTexture, bodyImageSize);

                            ImVec2 image_pos = ImGui::GetItemRectMin();
                            ImVec2 image_size = ImGui::GetItemRectSize();

                            ImDrawList* draw_list = ImGui::GetWindowDrawList();

                            float normalized_value = (config.body_y_offset - 1.0f) / 1.0f;

                            float line_y = image_pos.y + (1.0f + normalized_value) * image_size.y;

                            ImVec2 line_start = ImVec2(image_pos.x, line_y);
                            ImVec2 line_end = ImVec2(image_pos.x + image_size.x, line_y);

                            draw_list->AddLine(line_start, line_end, IM_COL32(255, 0, 0, 255), 2.0f);
                        }
                        else
                        {
                            ImGui::Text("Image not found!");
                        }
                        ImGui::Text("Note: There is a different value for each game, as the sizes of the player models may vary.");
                        ImGui::Separator();
                        ImGui::Checkbox("Ignore Third Person", &config.ignore_third_person);

                        ImGui::EndTabItem();
                    }

                    // ********************************************** MOUSE *********************************************
                    if (ImGui::BeginTabItem("Mouse"))
                    {
                        ImGui::SliderInt("DPI", &config.dpi, 800, 5000);
                        ImGui::SliderFloat("Sensitivity", &config.sensitivity, 0.1f, 10.0f, "%.1f");
                        ImGui::SliderInt("FOV X", &config.fovX, 60, 120);
                        ImGui::SliderInt("FOV Y", &config.fovY, 40, 100);
                        ImGui::SliderFloat("Min Speed Multiplier", &config.minSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Max Speed Multiplier", &config.maxSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Prediction Interval", &config.predictionInterval, 0.10f, 3.00f, "%.2f");

                        ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
                        ImGui::SliderFloat("bScope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");
                        
                        // INPUT METHODS
                        ImGui::Separator();
                        std::vector<std::string> input_methods = {"WIN32", "GHUB", "ARDUINO" };
                        std::vector<const char*> method_items;
                        method_items.reserve(input_methods.size());
                        for (const auto& item : input_methods)
                        {
                            method_items.push_back(item.c_str());
                        }

                        std::string combo_label = "Mouse Input method";
                        int input_method_index = 0;
                        for (size_t i = 0; i < input_methods.size(); ++i)
                        {
                            if (input_methods[i] == config.input_method)
                            {
                                input_method_index = static_cast<int>(i);
                                break;
                            }
                        }

                        if (ImGui::Combo("Mouse Input Method", &input_method_index, method_items.data(), static_cast<int>(method_items.size())))
                        {
                            std::string new_input_method = input_methods[input_method_index];

                            if (new_input_method != config.input_method)
                            {
                                config.input_method = new_input_method;
                                config.saveConfig("config.ini");
                                input_method_changed.store(true);
                            }
                        }

                        if (config.input_method == "ARDUINO")
                        {
                            std::vector<std::string> port_list;
                            for (int i = 1; i <= 30; ++i)
                            {
                                port_list.push_back("COM" + std::to_string(i));
                            }

                            std::vector<const char*> port_items;
                            port_items.reserve(port_list.size());
                            for (const auto& port : port_list)
                            {
                                port_items.push_back(port.c_str());
                            }

                            int port_index = 0;
                            for (size_t i = 0; i < port_list.size(); ++i)
                            {
                                if (port_list[i] == config.arduino_port)
                                {
                                    port_index = static_cast<int>(i);
                                    break;
                                }
                            }

                            if (ImGui::Combo("Arduino Port", &port_index, port_items.data(), static_cast<int>(port_items.size())))
                            {
                                config.arduino_port = port_list[port_index];
                                config.saveConfig("config.ini");
                                input_method_changed.store(true);
                            }

                            std::vector<int> baud_rate_list = { 9600, 19200, 38400, 57600, 115200 };
                            std::vector<std::string> baud_rate_str_list;
                            for (const auto& rate : baud_rate_list)
                            {
                                baud_rate_str_list.push_back(std::to_string(rate));
                            }

                            std::vector<const char*> baud_rate_items;
                            baud_rate_items.reserve(baud_rate_str_list.size());
                            for (const auto& rate_str : baud_rate_str_list)
                            {
                                baud_rate_items.push_back(rate_str.c_str());
                            }

                            int baud_rate_index = 0;
                            for (size_t i = 0; i < baud_rate_list.size(); ++i)
                            {
                                if (baud_rate_list[i] == config.arduino_baudrate)
                                {
                                    baud_rate_index = static_cast<int>(i);
                                    break;
                                }
                            }

                            if (ImGui::Combo("Arduino Baudrate", &baud_rate_index, baud_rate_items.data(), static_cast<int>(baud_rate_items.size())))
                            {
                                config.arduino_baudrate = baud_rate_list[baud_rate_index];
                                config.saveConfig("config.ini");
                                input_method_changed.store(true);
                            }

                            if (ImGui::Checkbox("Arduino 16-bit Mouse", &config.arduino_16_bit_mouse))
                            {
                                config.saveConfig("config.ini");
                                input_method_changed.store(true);
                            }
                            if (ImGui::Checkbox("Arduino Enable Keys", &config.arduino_enable_keys))
                            {
                                config.saveConfig("config.ini");
                                input_method_changed.store(true);
                            }
                        }
                        else if (config.input_method == "GHUB")
                        {
                            if (ghub_version == "13.1.4")
                            {
                                std::string ghub_version_label = "The correct version of Ghub is installed: " + ghub_version;
                                ImGui::Text(ghub_version_label.c_str());
                            }
                            else
                            {
                                if (ghub_version == "")
                                {
                                    ghub_version = "unknown";
                                }

                                std::string ghub_version_label = "Installed Ghub version: " + ghub_version;
                                ImGui::Text(ghub_version_label.c_str());
                                ImGui::Text("The wrong version of Ghub is installed or the path to Ghub is not set by default.\nDefault system path: C:\\Program Files\\LGHUB");
                                if (ImGui::Button("GHub Docs"))
                                {
                                    ShellExecute(0, 0, L"https://github.com/SunOner/sunone_aimbot_docs/blob/main/tips/ghub.md", 0, 0, SW_SHOW);
                                }
                            }
                        }

                        ImGui::EndTabItem();
                    }

                    // *********************************************** AI ***********************************************
                    if (ImGui::BeginTabItem("AI"))
                    {
                        // ai models
                        std::vector<const char*> models_items;
                        models_items.reserve(engine_models.size());
                        for (const auto& item : engine_models)
                        {
                            models_items.push_back(item.c_str());
                        }

                        if (ImGui::Combo("Model", &current_ai_model_index, models_items.data(), static_cast<int>(models_items.size())))
                        {
                            if (current_ai_model_index != prev_ai_model_index)
                            {
                                config.ai_model = engine_models[current_ai_model_index];
                                detector_model_changed.store(true);
                                prev_ai_model_index = current_ai_model_index;
                                config.saveConfig("config.ini");
                            }
                        }

                        // update models
                        if (ImGui::Button("Update##updare_models_dir"))
                        {
                            engine_models = getEngineFiles();
                        }

                        // model size
                        ImGui::Separator();
                        int model_sizes[] = { 320, 480, 640 };
                        const int model_sizes_count = sizeof(model_sizes) / sizeof(model_sizes[0]);

                        const char* model_sizes_str[model_sizes_count];
                        std::string model_sizes_buffer[model_sizes_count];

                        for (int i = 0; i < model_sizes_count; ++i)
                        {
                            model_sizes_buffer[i] = intToString(model_sizes[i]);
                            model_sizes_str[i] = model_sizes_buffer[i].c_str();
                        }

                        if (ImGui::Combo("Engine Image Size", &selected_imgsz, model_sizes_str, model_sizes_count))
                        {
                            if (selected_imgsz != prev_imgsz_index)
                            {
                                std::cout << "Image size changed to: " << model_sizes[selected_imgsz] << std::endl;

                                config.engine_image_size = model_sizes[selected_imgsz];
                                detector_model_changed.store(true);
                                prev_imgsz_index = selected_imgsz;

                                config.saveConfig("config.ini");
                            }
                        }

                        ImGui::Separator();
                        ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
                        ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.01f, 1.00f, "%.2f");

                        // Export
                        ImGui::Separator();

                        ImGui::Text("Export");

                        if (!tensorrt_path_found)
                        {
                            ImGui::Text("TensorRT path not found. Please enter the path to TensorRT:");

                            static char tensorrt_path_input[260] = "";
                            ImGui::InputText("TensorRT Path", tensorrt_path_input, sizeof(tensorrt_path_input));

                            if (ImGui::Button("Set TensorRT Path"))
                            {
                                tensorrt_path = std::string(tensorrt_path_input);
                                tensorrt_path_found = !tensorrt_path.empty();

                                if (tensorrt_path_found)
                                {
                                    // TODO ?? Save path in config or something..
                                }
                            }
                        }
                        else
                        {
                            tensorrt_bin_path = tensorrt_path + "\\bin";
                            trtexec_path = tensorrt_bin_path + "\\trtexec.exe";

                            bool trtexec_exists = fileExists(trtexec_path);

                            if (!trtexec_exists)
                            {
                                ImGui::Text("trtexec.exe not found in TensorRT bin directory (%s).", tensorrt_bin_path.c_str());
                                ImGui::Text("Please check your TensorRT installation.");
                            }
                            else
                            {
                                onnx_models = getOnnxFiles();

                                if (onnx_models.empty())
                                {
                                    ImGui::Text("No ONNX models found in the /models directory.");
                                }
                                else
                                {
                                    std::vector<const char*> onnx_models_cstrs;
                                    onnx_models_cstrs.reserve(onnx_models.size());
                                    for (const auto& model : onnx_models)
                                    {
                                        onnx_models_cstrs.push_back(model.c_str());
                                    }

                                    ImGui::Combo("ONNX Model", &onnx_model_index, onnx_models_cstrs.data(), static_cast<int>(onnx_models_cstrs.size()));

                                    std::string selected_onnx_model = onnx_models[onnx_model_index];
                                    output_engine_file = replace_extension(selected_onnx_model, ".engine");

                                    ImGui::Text("Output Engine File: %s", output_engine_file.c_str());

                                    if (ImGui::Button("Export"))
                                    {
                                        if (!is_exporting)
                                        {
                                            std::string command = "\"" + trtexec_path + "\" --onnx=models/" + selected_onnx_model + " --saveEngine=models/" + output_engine_file;

                                            // async export
                                            {
                                                std::lock_guard<std::mutex> lock(export_mutex);
                                                export_status_message = "Exporting...";
                                            }

                                            is_exporting = true;
                                            std::thread([command]()
                                                {
                                                    int exit_code = system(command.c_str());
                                                    {
                                                        std::lock_guard<std::mutex> lock(export_mutex);
                                                        if (exit_code == 0)
                                                        {
                                                            export_status_message = "Export completed successfully.";
                                                        }
                                                        else
                                                        {
                                                            export_status_message = "Export failed.";
                                                        }
                                                    }
                                                    is_exporting = false;
                                                }).detach();
                                        }
                                    }

                                    {
                                        std::lock_guard<std::mutex> lock(export_mutex);
                                        ImGui::Text("%s", export_status_message.c_str());
                                    }
                                }
                            }
                        }
                        ImGui::EndTabItem();
                    }

                    // ********************************************* BUTTONS ********************************************
                    if (ImGui::BeginTabItem("Buttons"))
                    {
                        // targeting
                        ImGui::Text("Targeting Buttons");

                        for (size_t i = 0; i < config.button_targeting.size(); )
                        {
                            std::string& current_key_name = config.button_targeting[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Targeting Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_targeting" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_targeting.erase(config.button_targeting.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##targeting"))
                        {
                            config.button_targeting.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        // exit
                        ImGui::Separator();
                        ImGui::Text("Exit Buttons");

                        for (size_t i = 0; i < config.button_exit.size(); )
                        {
                            std::string& current_key_name = config.button_exit[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Exit Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_exit" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_exit.erase(config.button_exit.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##exit"))
                        {
                            config.button_exit.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        // pause
                        ImGui::Separator();
                        ImGui::Text("Pause Buttons");

                        for (size_t i = 0; i < config.button_pause.size(); )
                        {
                            std::string& current_key_name = config.button_pause[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Pause Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_pause" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_pause.erase(config.button_pause.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##pause"))
                        {
                            config.button_pause.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        // reload config
                        ImGui::Separator();
                        ImGui::Text("Reload config Buttons");

                        for (size_t i = 0; i < config.button_reload_config.size(); )
                        {
                            std::string& current_key_name = config.button_reload_config[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Reload config Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_reload_config" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_reload_config.erase(config.button_reload_config.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##reload_config"))
                        {
                            config.button_reload_config.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        // overlay
                        ImGui::Separator();
                        ImGui::Text("Overlay Buttons");

                        for (size_t i = 0; i < config.button_open_overlay.size(); )
                        {
                            std::string& current_key_name = config.button_open_overlay[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Overlay Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_open_overlay" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_open_overlay.erase(config.button_open_overlay.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##overlay"))
                        {
                            config.button_open_overlay.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        ImGui::EndTabItem();
                    }

                    // ********************************************* OVERLAY ********************************************
                    if (ImGui::BeginTabItem("Overlay"))
                    {
                        ImGui::SliderInt("Overlay Opacity", &config.overlay_opacity, 40, 255);

                        ImGui::EndTabItem();
                    }
                    
                    // TODO CUSTOM CLASSES

                    // ********************************************** DEBUG *********************************************
                    if (ImGui::BeginTabItem("Debug"))
                    {
                        ImGui::Checkbox("Show Window", &config.show_window);
                        ImGui::Checkbox("Show FPS", &config.show_fps);
                        //ImGui::InputText("Window Name", &config.window_name[0], config.window_name.capacity() + 1); // TODO
                        ImGui::SliderInt("Window Size", &config.window_size, 10, 350);

                        // screenshot_button
                        ImGui::Separator();
                        ImGui::Text("Screenshot Buttons");

                        for (size_t i = 0; i < config.screenshot_button.size(); )
                        {
                            std::string& current_key_name = config.screenshot_button[i];

                            int current_index = -1;
                            for (size_t k = 0; k < key_names.size(); ++k)
                            {
                                if (key_names[k] == current_key_name)
                                {
                                    current_index = static_cast<int>(k);
                                    break;
                                }
                            }

                            if (current_index == -1)
                            {
                                current_index = 0;
                            }

                            std::string combo_label = "Screenshot Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig("config.ini");
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.screenshot_button.erase(config.screenshot_button.begin() + i);
                                config.saveConfig("config.ini");
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##overlay"))
                        {
                            config.button_open_overlay.push_back("None");
                            config.saveConfig("config.ini");
                        }

                        ImGui::InputInt("Screenshot delay", &config.screenshot_delay, 50, 500);
                        ImGui::Checkbox("Always On Top", &config.always_on_top);

                        // test functions
                        ImGui::Separator();
                        ImGui::Text("Test functions");
                        if (ImGui::Button("Free terminal"))
                        {
                            HideConsole();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Restore terminal"))
                        {
                            ShowConsole();
                        }

                        ImGui::EndTabItem();
                    }

                    // ******************************************* APPLY VARS *******************************************
                    // DETECTION RESOLUTION
                    if (prev_detection_resolution != config.detection_resolution)
                    {
                        detection_resolution_changed.store(true);
                        detector_model_changed.store(true); // reboot vars for visuals
                        prev_detection_resolution = config.detection_resolution;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE METHOD
                    if (prev_capture_method != config.duplication_api)
                    {
                        capture_method_changed.store(true);
                        prev_capture_method = config.duplication_api;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE CURSOR
                    if (prev_capture_cursor != config.capture_cursor && config.duplication_api == false)
                    {
                        capture_cursor_changed.store(true);
                        prev_capture_cursor = config.capture_cursor;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE BORDERS
                    if (prev_capture_borders != config.capture_borders && config.duplication_api == false)
                    {
                        capture_borders_changed.store(true);
                        prev_capture_borders = config.capture_borders;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE_FPS
                    if (prev_capture_fps != config.capture_fps)
                    {
                        capture_fps_changed.store(true);
                        prev_capture_fps = config.capture_fps;
                        config.saveConfig("config.ini");
                    }

                    // DISABLE_HEADSHOT / BODY_Y_OFFSET / IGNORE_THIRD_PERSON
                    if (prev_disable_headshot != config.disable_headshot ||
                        prev_body_y_offset != config.body_y_offset ||
                        prev_ignore_third_person != config.ignore_third_person)
                    {
                        prev_disable_headshot = config.disable_headshot;
                        prev_body_y_offset = config.body_y_offset;
                        prev_ignore_third_person = config.ignore_third_person;
                        config.saveConfig("config.ini");
                    }

                    // DPI / SENSITIVITY / FOVX / FOVY / MINSPEEDMULTIPLIER / MAXSPEEDMULTIPLIER / PREDICTIONINTERVAL
                    if (prev_dpi != config.dpi ||
                        prev_sensitivity != config.sensitivity ||
                        prev_fovX != config.fovX ||
                        prev_fovY != config.fovY ||
                        prev_minSpeedMultiplier != config.minSpeedMultiplier ||
                        prev_maxSpeedMultiplier != config.maxSpeedMultiplier ||
                        prev_predictionInterval != config.predictionInterval)
                    {
                        prev_dpi = config.dpi;
                        prev_sensitivity = config.sensitivity;
                        prev_fovX = config.fovX;
                        prev_fovY = config.fovY;
                        prev_minSpeedMultiplier = config.minSpeedMultiplier;
                        prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
                        prev_predictionInterval = config.predictionInterval;

                        globalMouseThread->updateConfig(
                            config.detection_resolution,
                            config.dpi,
                            config.sensitivity,
                            config.fovX,
                            config.fovY,
                            config.minSpeedMultiplier,
                            config.maxSpeedMultiplier,
                            config.predictionInterval,
                            config.auto_shoot,
                            config.bScope_multiplier);

                        config.saveConfig("config.ini");
                    }

                    // AUTO_SHOOT / BSCOPE_MULTIPLIER
                    if (prev_auto_shoot != config.auto_shoot ||
                        prev_bScope_multiplier != config.bScope_multiplier)
                    {
                        prev_auto_shoot = config.auto_shoot;
                        prev_bScope_multiplier = config.bScope_multiplier;
                        config.saveConfig("config.ini");
                    }

                    // OVERLAY OPACITY
                    if (prev_opacity != config.overlay_opacity)
                    {
                        BYTE opacity = config.overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);
                        config.saveConfig("config.ini");
                    }

                    // CONFIDENCE THERSHOLD / NMS THRESHOLD
                    if (prev_confidence_threshold != config.confidence_threshold ||
                        prev_nms_threshold != config.nms_threshold)
                    {
                        prev_nms_threshold = config.nms_threshold;
                        prev_confidence_threshold = config.confidence_threshold;
                        config.saveConfig("config.ini");
                    }

                    // SHOW WINDOW
                    if (prev_show_window != config.show_window)
                    {
                        show_window_changed.store(true);
                        prev_show_window = config.show_window;
                        config.saveConfig("config.ini");
                    }
                    
                    // ALWAYS_ON_TOP
                    if (prev_always_on_top != config.always_on_top)
                    {
                        // TODO: update window property
                        prev_always_on_top = config.always_on_top;
                    }

                    // SHOW_FPS / WINDOW_SIZE / SCREENSHOT_DELAY
                    if (prev_show_fps != config.show_fps ||
                        prev_window_size != config.window_size ||
                        prev_screenshot_delay != config.screenshot_delay)
                    {
                        prev_show_fps = config.show_fps;
                        prev_window_size = config.window_size;
                        prev_screenshot_delay = config.screenshot_delay;
                        config.saveConfig("config.ini");
                    }

                ImGui::EndTabBar();
                }
            }

            ImGui::End();
            ImGui::Render();

            const float clear_color_with_alpha[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

            HRESULT result = g_pSwapChain->Present(0, 0);

            if (result == DXGI_STATUS_OCCLUDED) // TODO
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    // unload texture
    if (bodyTexture)
    {
        bodyTexture->Release();
        bodyTexture = nullptr;
    }

    CleanupDeviceD3D();
    ::DestroyWindow(g_hwnd);
    ::UnregisterClass(_T("Edge"), GetModuleHandle(NULL));
}

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPTSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    std::thread overlay(OverlayThread);
    overlay.join();
    return 0;
}