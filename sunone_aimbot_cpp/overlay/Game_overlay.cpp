#include "Game_overlay.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <wincodec.h>
#include <dcomp.h>
#include <wrl/client.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <chrono>

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "dcomp.lib")

using Microsoft::WRL::ComPtr;

static void GetVirtualScreen(int& x, int& y, int& w, int& h)
{
    x = GetSystemMetrics(SM_XVIRTUALSCREEN);
    y = GetSystemMetrics(SM_YVIRTUALSCREEN);
    w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
}

static void SetPerMonitorV2DpiAwareness()
{
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32) return;
    using Fn = BOOL(WINAPI*)(HANDLE);
    if (auto p = reinterpret_cast<Fn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext")))
        p(reinterpret_cast<HANDLE>(-4)); // PER_MONITOR_AWARE_V2
}

static D2D1_COLOR_F ToD2DColor(OverlayColor argb)
{
    float a = float((argb >> 24) & 0xFF) / 255.f;
    float r = float((argb >> 16) & 0xFF) / 255.f;
    float g = float((argb >> 8) & 0xFF) / 255.f;
    float b = float((argb >> 0) & 0xFF) / 255.f;
    return D2D1::ColorF(r, g, b, a);
}

struct DrawCmd
{
    enum Type { Line, Rect, RectFilled, Circle, CircleFilled, Text, Image } type;
    OverlayColor color = 0;
    float thickness = 1.0f;
    union
    {
        OverlayLine line;
        OverlayRect rect;
        OverlayCircle circle;
        struct { float x, y, w, h, opacity; int imageId; } image;
        struct { float x, y, size; } textPos;
    };
    std::wstring text;
    std::wstring fontName;
};

struct DrawList
{
    std::vector<DrawCmd> cmds;
};

struct ImageRes
{
    ComPtr<ID2D1Bitmap1> bmp;
    float w = 0.f;
    float h = 0.f;
};

struct Game_overlay::Impl
{
    HINSTANCE hinst = nullptr;
    HWND hwnd = nullptr;
    int winX = 0, winY = 0, winW = 0, winH = 0;
    bool useVirtual = true;

    ComPtr<ID3D11Device>           d3d;
    ComPtr<ID3D11DeviceContext>    d3dCtx;
    ComPtr<IDXGISwapChain1>        swapChain;
    ComPtr<ID3D11RenderTargetView> rtv;

    ComPtr<ID2D1Factory1>          d2dFactory;
    ComPtr<ID2D1Device>            d2dDevice;
    ComPtr<ID2D1DeviceContext>     d2dCtx;
    ComPtr<ID2D1Bitmap1>           d2dTarget;
    ComPtr<ID2D1SolidColorBrush>   brush;

    ComPtr<IDWriteFactory>         dwFactory;
    std::unordered_map<uint64_t, ComPtr<IDWriteTextFormat>> textFormatCache;

    ComPtr<IWICImagingFactory>     wic;

    ComPtr<IDCompositionDevice>    dcompDevice;
    ComPtr<IDCompositionTarget>    dcompTarget;
    ComPtr<IDCompositionVisual>    dcompRoot;

    std::thread thread;
    std::atomic<bool> running{ false };
    std::atomic<bool> visible{ true };
    std::atomic<unsigned> maxFps{ 0 };

    std::mutex pendingMutex;
    DrawList pending;
    std::shared_ptr<DrawList> current;

    std::mutex imgMutex;
    int nextImageId = 1;
    std::unordered_map<int, ImageRes> images;

    void UseVirtualScreen() { useVirtual = true; }
    void SetBounds(int x, int y, int w, int h) { useVirtual = false; winX = x; winY = y; winW = w; winH = h; }

    bool Start();
    void Stop();

    void BeginFrame();
    void EndFrame();
    void AddCmd(const DrawCmd&);

    int  LoadImageFromFile(const std::wstring& path);
    void UnloadImage(int id);
    void DrawImage(int id, float x, float y, float w, float h, float opacity);

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    bool    CreateWindowAndDevices();
    void    DestroyWindowAndDevices();
    void    RenderLoop();
    void    RenderOne();

    HRESULT EnsureWic();
    HRESULT CreateTextFormat(const std::wstring& font, float size, IDWriteTextFormat** out);
    HRESULT CreateTargets();
    void    ReleaseTargets();
};

Game_overlay::Game_overlay() : impl_(new Impl) {}
Game_overlay::~Game_overlay() { Stop(); }

bool Game_overlay::Start() { return impl_->Start(); }
void Game_overlay::Stop() { impl_->Stop(); }
bool Game_overlay::IsRunning() const { return impl_->running.load(); }
void Game_overlay::SetVisible(bool v) { impl_->visible.store(v); }
bool Game_overlay::GetVisible() const { return impl_->visible.load(); }
void Game_overlay::UseVirtualScreen() { impl_->UseVirtualScreen(); }
void Game_overlay::SetWindowBounds(int x, int y, int w, int h) { impl_->SetBounds(x, y, w, h); }
void Game_overlay::SetMaxFPS(unsigned f) { impl_->maxFps.store(f); }

void Game_overlay::BeginFrame() { impl_->BeginFrame(); }
void Game_overlay::EndFrame() { impl_->EndFrame(); }

void Game_overlay::AddLine(const OverlayLine& l, OverlayColor c, float t)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Line; cmd.color = c; cmd.thickness = t; cmd.line = l;
    impl_->AddCmd(cmd);
}
void Game_overlay::AddRect(const OverlayRect& r, OverlayColor c, float t)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Rect; cmd.color = c; cmd.thickness = t; cmd.rect = r;
    impl_->AddCmd(cmd);
}
void Game_overlay::FillRect(const OverlayRect& r, OverlayColor c)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::RectFilled; cmd.color = c; cmd.rect = r;
    impl_->AddCmd(cmd);
}
void Game_overlay::AddCircle(const OverlayCircle& c0, OverlayColor c, float t)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Circle; cmd.color = c; cmd.thickness = t; cmd.circle = c0;
    impl_->AddCmd(cmd);
}
void Game_overlay::FillCircle(const OverlayCircle& c0, OverlayColor c)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::CircleFilled; cmd.color = c; cmd.circle = c0;
    impl_->AddCmd(cmd);
}
void Game_overlay::AddText(float x, float y, const std::wstring& text,
    float sizePx, OverlayColor c, const std::wstring& font)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Text; cmd.color = c; cmd.text = text;
    cmd.textPos = { x, y, sizePx }; cmd.fontName = font;
    impl_->AddCmd(cmd);
}
int  Game_overlay::LoadImageFromFile(const std::wstring& path) { return impl_->LoadImageFromFile(path); }
void Game_overlay::UnloadImage(int id) { impl_->UnloadImage(id); }
void Game_overlay::DrawImage(int id, float x, float y, float w, float h, float op) { impl_->DrawImage(id, x, y, w, h, op); }

bool Game_overlay::Impl::Start()
{
    if (running.load()) return true;
    running = true;
    thread = std::thread([this] {
        SetPerMonitorV2DpiAwareness();
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        hinst = GetModuleHandleW(nullptr);
        if (!CreateWindowAndDevices()) {
            running = false;
            CoUninitialize();
            return;
        }
        RenderLoop();
        DestroyWindowAndDevices();
        CoUninitialize();
        });
    return true;
}

void Game_overlay::Impl::Stop()
{
    if (!running.exchange(false)) return;
    if (hwnd) PostMessageW(hwnd, WM_CLOSE, 0, 0);
    if (thread.joinable()) thread.join();
}

void Game_overlay::Impl::BeginFrame()
{
    std::lock_guard<std::mutex> lk(pendingMutex);
    pending.cmds.clear();
}

void Game_overlay::Impl::EndFrame()
{
    auto dl = std::make_shared<DrawList>();
    {
        std::lock_guard<std::mutex> lk(pendingMutex);
        dl->cmds = pending.cmds;
    }
    std::atomic_store_explicit(&current, dl, std::memory_order_release);
}

void Game_overlay::Impl::AddCmd(const DrawCmd& c)
{
    std::lock_guard<std::mutex> lk(pendingMutex);
    pending.cmds.push_back(c);
}

HRESULT Game_overlay::Impl::EnsureWic()
{
    if (wic) return S_OK;
    return CoCreateInstance(CLSID_WICImagingFactory, nullptr,
        CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic));
}

int Game_overlay::Impl::LoadImageFromFile(const std::wstring& path)
{
    if (FAILED(EnsureWic())) return 0;

    ComPtr<IWICBitmapDecoder> dec;
    if (FAILED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
        WICDecodeMetadataCacheOnLoad, &dec))) return 0;

    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return 0;

    ComPtr<IWICFormatConverter> conv;
    if (FAILED(wic->CreateFormatConverter(&conv))) return 0;

    if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppPBGRA,
        WICBitmapDitherTypeNone, nullptr, 0.f, WICBitmapPaletteTypeCustom)))
        return 0;

    if (!d2dCtx) return 0;

    D2D1_BITMAP_PROPERTIES1 props =
        D2D1::BitmapProperties1(
            D2D1_BITMAP_OPTIONS_NONE,
            D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
                D2D1_ALPHA_MODE_PREMULTIPLIED),
            96.f, 96.f);

    ComPtr<ID2D1Bitmap1> bmp;
    if (FAILED(d2dCtx->CreateBitmapFromWicBitmap(conv.Get(), &props, &bmp))) return 0;

    auto sz = bmp->GetSize();
    ImageRes ir;
    ir.bmp = bmp;
    ir.w = sz.width;
    ir.h = sz.height;

    std::lock_guard<std::mutex> il(imgMutex);
    int id = nextImageId++;
    images.emplace(id, std::move(ir));
    return id;
}

void Game_overlay::Impl::UnloadImage(int id)
{
    std::lock_guard<std::mutex> il(imgMutex);
    images.erase(id);
}

void Game_overlay::Impl::DrawImage(int id, float x, float y, float w, float h, float opacity)
{
    DrawCmd c{};
    c.type = DrawCmd::Image;
    c.image = { x, y, w, h, opacity, id };
    AddCmd(c);
}

LRESULT CALLBACK Game_overlay::Impl::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    Impl* self = reinterpret_cast<Impl*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
    switch (msg)
    {
    case WM_NCCREATE:
    {
        auto cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hWnd, GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        return TRUE;
    }
    case WM_NCHITTEST:     return HTTRANSPARENT;
    case WM_MOUSEACTIVATE: return MA_NOACTIVATE;
    case WM_SIZE:
        if (self && self->swapChain)
        {
            UINT w = LOWORD(lParam), h = HIWORD(lParam);
            if (w > 0 && h > 0)
            {
                self->ReleaseTargets();
                self->swapChain->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
                self->CreateTargets();
                if (self->dcompDevice) self->dcompDevice->Commit();
            }
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

bool Game_overlay::Impl::CreateWindowAndDevices()
{
    if (useVirtual) GetVirtualScreen(winX, winY, winW, winH);
    else if (winW <= 0 || winH <= 0) GetVirtualScreen(winX, winY, winW, winH);

    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = Game_overlay::Impl::WndProc;
    wc.hInstance = hinst;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"GameOverlayDCompWnd";
    wc.hbrBackground = nullptr;
    if (!RegisterClassExW(&wc))
        return false;

    DWORD ex = WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED;
    hwnd = CreateWindowExW(ex, wc.lpszClassName, L"", WS_POPUP,
        winX, winY, winW, winH, nullptr, nullptr, hinst, this);
    if (!hwnd) return false;

    BOOL dwm = FALSE;
    if (SUCCEEDED(DwmIsCompositionEnabled(&dwm)) && dwm)
    {
        MARGINS m = { -1, -1, -1, -1 };
        DwmExtendFrameIntoClientArea(hwnd, &m);
    }

    ShowWindow(hwnd, SW_SHOWNA);
    SetWindowPos(hwnd, HWND_TOPMOST, winX, winY, winW, winH,
        SWP_NOACTIVATE | SWP_SHOWWINDOW);

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL flOut{};
    if (FAILED(D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
        featureLevels, _countof(featureLevels),
        D3D11_SDK_VERSION, &d3d, &flOut, &d3dCtx)))
    {
        return false;
    }

    // DXGI factory
    ComPtr<IDXGIDevice> dxgiDev; d3d.As(&dxgiDev);
    ComPtr<IDXGIAdapter> adapter; dxgiDev->GetAdapter(&adapter);
    ComPtr<IDXGIFactory2> factory2;
    {
        ComPtr<IDXGIFactory> baseFactory;
        adapter->GetParent(IID_PPV_ARGS(&baseFactory));
        baseFactory.As(&factory2);
    }
    if (!factory2) return false;

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = winW;
    scd.Height = winH;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    scd.Scaling = DXGI_SCALING_STRETCH;

    if (FAILED(factory2->CreateSwapChainForComposition(
        d3d.Get(), &scd, nullptr, &swapChain)))
    {
        return false;
    }

    if (FAILED(DCompositionCreateDevice(
        dxgiDev.Get(), IID_PPV_ARGS(&dcompDevice))))
    {
        return false;
    }
    if (FAILED(dcompDevice->CreateTargetForHwnd(hwnd, TRUE, &dcompTarget))) return false;
    if (FAILED(dcompDevice->CreateVisual(&dcompRoot))) return false;
    if (FAILED(dcompRoot->SetContent(swapChain.Get()))) return false;
    if (FAILED(dcompTarget->SetRoot(dcompRoot.Get()))) return false;
    dcompDevice->Commit();

    if (FAILED(CreateTargets())) return false;

    if (FAILED(d2dCtx->CreateSolidColorBrush(
        D2D1::ColorF(1.f, 1.f, 1.f, 1.f), &brush))) return false;
    d2dCtx->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);
    d2dCtx->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE);

    if (FAILED(DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(dwFactory.GetAddressOf()))))
    {
        return false;
    }

    std::atomic_store_explicit(
        &current, std::make_shared<DrawList>(),
        std::memory_order_release);

    return true;
}

void Game_overlay::Impl::DestroyWindowAndDevices()
{
    ReleaseTargets();
    brush.Reset();
    d2dCtx.Reset();
    d2dDevice.Reset();
    d2dFactory.Reset();
    rtv.Reset();
    swapChain.Reset();
    dcompRoot.Reset();
    dcompTarget.Reset();
    dcompDevice.Reset();
    dwFactory.Reset();
    wic.Reset();
    d3dCtx.Reset();
    d3d.Reset();

    if (hwnd) { DestroyWindow(hwnd); hwnd = nullptr; }
    UnregisterClassW(L"GameOverlayDCompWnd", hinst);
}

HRESULT Game_overlay::Impl::CreateTargets()
{
    ReleaseTargets();

    ComPtr<ID3D11Texture2D> bb;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&bb))))
        return E_FAIL;

    if (FAILED(d3d->CreateRenderTargetView(bb.Get(), nullptr, &rtv)))
        return E_FAIL;

    if (!d2dFactory)
    {
        D2D1_FACTORY_OPTIONS opts{};
#if defined(_DEBUG)
        opts.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif
        if (FAILED(D2D1CreateFactory(
            D2D1_FACTORY_TYPE_MULTI_THREADED,
            __uuidof(ID2D1Factory1),
            &opts,
            reinterpret_cast<void**>(d2dFactory.GetAddressOf()))))
            return E_FAIL;
    }

    if (!d2dDevice)
    {
        ComPtr<IDXGIDevice> dxgiDev; d3d.As(&dxgiDev);
        if (FAILED(d2dFactory->CreateDevice(dxgiDev.Get(), &d2dDevice)))
            return E_FAIL;
    }

    if (!d2dCtx)
    {
        if (FAILED(d2dDevice->CreateDeviceContext(
            D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2dCtx)))
            return E_FAIL;
    }

    ComPtr<IDXGISurface> surf;
    if (FAILED(bb.As(&surf))) return E_FAIL;

    D2D1_BITMAP_PROPERTIES1 props =
        D2D1::BitmapProperties1(
            D2D1_BITMAP_OPTIONS_TARGET |
            D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
            D2D1::PixelFormat(
                DXGI_FORMAT_B8G8R8A8_UNORM,
                D2D1_ALPHA_MODE_PREMULTIPLIED),
            96.f, 96.f
        );

    if (FAILED(d2dCtx->CreateBitmapFromDxgiSurface(
        surf.Get(), &props, &d2dTarget)))
        return E_FAIL;

    d2dCtx->SetTarget(d2dTarget.Get());

    return S_OK;
}

void Game_overlay::Impl::ReleaseTargets()
{
    if (d2dCtx) d2dCtx->SetTarget(nullptr);
    d2dTarget.Reset();
    rtv.Reset();
}

HRESULT Game_overlay::Impl::CreateTextFormat(
    const std::wstring& font, float size,
    IDWriteTextFormat** out)
{
    uint64_t key =
        (std::hash<std::wstring>{}(font) ^ (uint64_t(std::lround(size * 100)) << 1));

    auto it = textFormatCache.find(key);
    if (it != textFormatCache.end())
    {
        *out = it->second.Get();
        (*out)->AddRef();
        return S_OK;
    }

    ComPtr<IDWriteTextFormat> fmt;
    if (FAILED(dwFactory->CreateTextFormat(
        font.c_str(), nullptr,
        DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        size, L"en-US", &fmt)))
        return E_FAIL;

    fmt->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    textFormatCache.emplace(key, fmt);
    *out = fmt.Detach();
    return S_OK;
}

void Game_overlay::Impl::RenderLoop()
{
    running = true;
    auto last = std::chrono::high_resolution_clock::now();

    MSG msg{};
    while (running.load())
    {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running.load()) break;

        unsigned cap = maxFps.load();
        if (cap > 0)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto minDelta = std::chrono::microseconds(1'000'000 / cap);
            if (now - last < minDelta) { Sleep(1); continue; }
            last = now;
        }
        RenderOne();
    }
}

void Game_overlay::Impl::RenderOne()
{
    if (!d2dCtx || !swapChain) return;

    if (!visible.load())
    {
        ShowWindow(hwnd, SW_HIDE);
        Sleep(10);
        return;
    }
    else
    {
        ShowWindow(hwnd, SW_SHOWNA);
    }

    d2dCtx->BeginDraw();
    d2dCtx->Clear(D2D1::ColorF(0.f, 0.f, 0.f, 0.f));

    auto dl = std::atomic_load_explicit(&current, std::memory_order_acquire);
    if (dl)
    {
        for (const auto& c : dl->cmds)
        {
            switch (c.type)
            {
            case DrawCmd::Line:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->DrawLine(
                    { c.line.x1, c.line.y1 },
                    { c.line.x2, c.line.y2 },
                    brush.Get(), c.thickness);
                break;

            case DrawCmd::Rect:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->DrawRectangle(
                    D2D1::RectF(c.rect.x, c.rect.y,
                        c.rect.x + c.rect.w, c.rect.y + c.rect.h),
                    brush.Get(), c.thickness);
                break;

            case DrawCmd::RectFilled:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->FillRectangle(
                    D2D1::RectF(c.rect.x, c.rect.y,
                        c.rect.x + c.rect.w, c.rect.y + c.rect.h),
                    brush.Get());
                break;

            case DrawCmd::Circle:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->DrawEllipse(
                    D2D1::Ellipse({ c.circle.cx, c.circle.cy }, c.circle.r, c.circle.r),
                    brush.Get(), c.thickness);
                break;

            case DrawCmd::CircleFilled:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->FillEllipse(
                    D2D1::Ellipse({ c.circle.cx, c.circle.cy }, c.circle.r, c.circle.r),
                    brush.Get());
                break;

            case DrawCmd::Text:
            {
                brush->SetColor(ToD2DColor(c.color));
                ComPtr<IDWriteTextFormat> fmt;
                if (SUCCEEDED(CreateTextFormat(
                    c.fontName.empty() ? L"Segoe UI" : c.fontName,
                    c.textPos.size, &fmt)))
                {
                    D2D1_RECT_F layout = D2D1::RectF(
                        c.textPos.x, c.textPos.y,
                        c.textPos.x + 4000.f, c.textPos.y + 1200.f);
                    d2dCtx->DrawTextW(
                        c.text.c_str(),
                        (UINT32)c.text.size(),
                        fmt.Get(),
                        layout,
                        brush.Get());
                }
            } break;

            case DrawCmd::Image:
            {
                std::lock_guard<std::mutex> lk(imgMutex);
                auto it = images.find(c.image.imageId);
                if (it != images.end() && it->second.bmp)
                {
                    D2D1_RECT_F dst = D2D1::RectF(
                        c.image.x, c.image.y,
                        c.image.x + c.image.w,
                        c.image.y + c.image.h);
                    d2dCtx->DrawBitmap(
                        it->second.bmp.Get(),
                        &dst,
                        c.image.opacity,
                        D2D1_BITMAP_INTERPOLATION_MODE_LINEAR,
                        nullptr);
                }
            } break;
            }
        }
    }

    HRESULT hrEnd = d2dCtx->EndDraw();
    if (FAILED(hrEnd))
    {
        CreateTargets();
    }

    swapChain->Present(0, 0);
}