#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "winrt_capture.h"
#include "sunone_aimbot_cpp.h"
#include "config.h"
#include "other_tools.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

static HMONITOR GetMonitorHandleByIndex(int index)
{
    struct MonitorEnumData
    {
        int targetIndex;
        int currentIndex;
        HMONITOR hMonitor;
    } data{ index, 0, nullptr };

    auto enumCallback = [](HMONITOR monitor, HDC, LPRECT, LPARAM lParam) -> BOOL
        {
            auto& d = *reinterpret_cast<MonitorEnumData*>(lParam);
            if (d.currentIndex == d.targetIndex)
            {
                d.hMonitor = monitor;
                return FALSE;
            }
            d.currentIndex++;
            return TRUE;
        };

    EnumDisplayMonitors(nullptr, nullptr, enumCallback, reinterpret_cast<LPARAM>(&data));
    return data.hMonitor;
}

IGraphicsCaptureItemInterop* GetInteropFactory()
{
    static IGraphicsCaptureItemInterop* s_factory = nullptr;
    if (!s_factory)
    {
        auto factory = winrt::get_activation_factory<
            GraphicsCaptureItem,
            IGraphicsCaptureItemInterop>();
        s_factory = factory.as<IGraphicsCaptureItemInterop>().get();
    }
    return s_factory;
}

WinRTScreenCapture::WinRTScreenCapture(int desiredWidth, int desiredHeight)
    : regionWidth(desiredWidth)
    , regionHeight(desiredHeight)
{
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    winrt::check_hresult(
        D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            0,
            createDeviceFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            d3dDevice.put(),
            nullptr,
            d3dContext.put()
        )
    );

    winrt::com_ptr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())));
    device = CreateDirect3DDevice(dxgiDevice.get());
    if (!device)
    {
        throw std::runtime_error("[WinRTCapture] Failed to create IDirect3DDevice.");
    }

    HMONITOR hMonitor = GetMonitorHandleByIndex(config.monitor_idx);
    if (!hMonitor)
    {
        throw std::runtime_error("[WinRTCapture] Invalid monitor index in config.");
    }

    captureItem = CreateCaptureItemForMonitor(hMonitor);
    if (!captureItem)
    {
        throw std::runtime_error("[WinRTCapture] CreateCaptureItemForMonitor failed.");
    }

    screenWidth = captureItem.Size().Width;
    screenHeight = captureItem.Size().Height;

    regionX = (screenWidth - regionWidth) / 2;
    regionY = (screenHeight - regionHeight) / 2;

    framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        3,
        captureItem.Size()
    );

    session = framePool.CreateCaptureSession(captureItem);

    if (!config.capture_borders)
    {
        session.IsBorderRequired(false);
    }

    if (!config.capture_cursor)
    {
        session.IsCursorCaptureEnabled(false);
    }

    if (!createStagingTextureCPU())
    {
        throw std::runtime_error("[WinRTCapture] createStagingTextureCPU() failed.");
    }
    

    session.StartCapture();
}

WinRTScreenCapture::~WinRTScreenCapture()
{
    session.Close();
    framePool.Close();

    stagingTextureCPU = nullptr;
    sharedTexture = nullptr;
    d3dContext = nullptr;
    d3dDevice = nullptr;
}

bool WinRTScreenCapture::createStagingTextureCPU()
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = regionWidth;
    desc.Height = regionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, stagingTextureCPU.put());
    if (FAILED(hr))
    {
        std::cerr << "[WinRTCapture] CreateTexture2D(staging) failed hr=" << std::hex << hr << std::endl;
        return false;
    }
    return true;
}

cv::Mat WinRTScreenCapture::GetNextFrameCpu()
{
    Direct3D11CaptureFrame lastFrame{ nullptr };
    while (auto tempFrame = framePool.TryGetNextFrame())
    {
        lastFrame = tempFrame;
    }
    if (!lastFrame)
        return cv::Mat();

    auto frameSurface = lastFrame.Surface();
    auto frameTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frameSurface);
    if (!frameTexture)
        return cv::Mat();

    D3D11_BOX sourceRegion;
    sourceRegion.left = regionX;
    sourceRegion.top = regionY;
    sourceRegion.front = 0;
    sourceRegion.right = regionX + regionWidth;
    sourceRegion.bottom = regionY + regionHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        stagingTextureCPU.get(),
        0,
        0, 0, 0,
        frameTexture.get(),
        0,
        &sourceRegion
    );

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hrMap = d3dContext->Map(stagingTextureCPU.get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hrMap))
    {
        std::cerr << "[WinRTCapture] Map stagingTextureCPU failed hr=" << std::hex << hrMap << std::endl;
        return cv::Mat();
    }

    cv::Mat cpuFrame(regionHeight, regionWidth, CV_8UC4);
    for (int y = 0; y < regionHeight; y++)
    {
        unsigned char* dstRow = cpuFrame.ptr<unsigned char>(y);
        unsigned char* srcRow = (unsigned char*)mapped.pData + y * mapped.RowPitch;
        memcpy(dstRow, srcRow, regionWidth * 4);
    }
    d3dContext->Unmap(stagingTextureCPU.get(), 0);

    return cpuFrame;
}

winrt::Windows::Graphics::Capture::GraphicsCaptureItem
WinRTScreenCapture::CreateCaptureItemForMonitor(HMONITOR hMonitor)
{
    auto interopFactory = GetInteropFactory();
    GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interopFactory->CreateForMonitor(
        hMonitor,
        winrt::guid_of<GraphicsCaptureItem>(),
        winrt::put_abi(item)
    );
    if (FAILED(hr))
    {
        throw std::runtime_error("[WinRTCapture] CreateForMonitor failed, HR=" + std::to_string(hr));
    }
    return item;
}

winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice
WinRTScreenCapture::CreateDirect3DDevice(IDXGIDevice* dxgiDevice)
{
    winrt::com_ptr<::IInspectable> inspectable;
    winrt::check_hresult(
        CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, inspectable.put())
    );
    return inspectable.as<IDirect3DDevice>();
}

template<typename T>
winrt::com_ptr<T> WinRTScreenCapture::GetDXGIInterfaceFromObject(
    winrt::Windows::Foundation::IInspectable const& object)
{
    auto dxgiInterfaceAccess = object.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result = nullptr;
    winrt::check_hresult(
        dxgiInterfaceAccess->GetInterface(winrt::guid_of<T>(), result.put_void())
    );
    return result;
}