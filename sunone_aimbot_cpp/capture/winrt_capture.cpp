#include "winrt_capture.h"
#include "config.h"
#include "other_tools.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

extern Config config;

WinRTScreenCapture::WinRTScreenCapture(int desiredWidth, int desiredHeight)
{
    regionWidth = desiredWidth;
    regionHeight = desiredHeight;

    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    winrt::check_hresult(D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        0,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        d3dDevice.put(),
        nullptr,
        d3dContext.put()
    ));

    winrt::com_ptr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())));
    device = CreateDirect3DDevice(dxgiDevice.get());

    if (!device)
    {
        std::cerr << "[Capture] Can't create Direct3DDevice!" << std::endl;
        return;
    }

    HMONITOR hMonitor = GetMonitorHandleByIndex(config.monitor_idx);

    if (!hMonitor)
    {
        std::cerr << "[Capture] Failed to get monitor handle for index: " << config.monitor_idx << std::endl;
        return;
    }

    captureItem = CreateCaptureItemForMonitor(hMonitor);

    if (!captureItem)
    {
        std::cerr << "[Capture] GraphicsCaptureItem not created!" << std::endl;
        return;
    }

    screenWidth = captureItem.Size().Width;
    screenHeight = captureItem.Size().Height;

    regionX = (screenWidth - regionWidth) / 2;
    regionY = (screenHeight - regionHeight) / 2;

    framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
        device,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        1,
        captureItem.Size()
    );

    session = framePool.CreateCaptureSession(captureItem);

    D3D11_TEXTURE2D_DESC sharedTexDesc = {};
    sharedTexDesc.Width = regionWidth;
    sharedTexDesc.Height = regionHeight;
    sharedTexDesc.MipLevels = 1;
    sharedTexDesc.ArraySize = 1;
    sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sharedTexDesc.SampleDesc.Count = 1;
    sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
    sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    sharedTexDesc.CPUAccessFlags = 0;
    sharedTexDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    HRESULT hr = d3dDevice->CreateTexture2D(&sharedTexDesc, nullptr, sharedTexture.put());
    if (FAILED(hr))
    {
        std::cerr << "[Capture] Failed to create shared texture." << std::endl;
        return;
    }

    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, sharedTexture.get(), cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
    {
        std::cerr << "[Capture] Failed to register shared texture with CUDA." << std::endl;
        return;
    }

    cudaStreamCreate(&cudaStream);

    interopInitialized = false;

    if (!config.capture_borders)
    {
        session.IsBorderRequired(false);
    }

    if (!config.capture_cursor)
    {
        session.IsCursorCaptureEnabled(false);
    }

    session.StartCapture();
}

WinRTScreenCapture::~WinRTScreenCapture()
{
    if (cudaResource)
    {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }

    if (cudaStream)
    {
        cudaStreamDestroy(cudaStream);
        cudaStream = nullptr;
    }

    if (session)
    {
        session.Close();
        session = nullptr;
    }

    if (framePool)
    {
        framePool.Close();
        framePool = nullptr;
    }

    stagingTexture = nullptr;
    d3dContext = nullptr;
    d3dDevice = nullptr;
}

cv::cuda::GpuMat WinRTScreenCapture::GetNextFrame()
{
    try
    {
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame frame = nullptr;

        while (true)
        {
            auto tempFrame = framePool.TryGetNextFrame();
            if (!tempFrame)
                break;
            frame = tempFrame;
        }

        if (!frame)
        {
            return cv::cuda::GpuMat();
        }

        auto surface = frame.Surface();

        winrt::com_ptr<ID3D11Texture2D> frameTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surface);

        if (!frameTexture)
        {
            throw std::runtime_error("[Capture] Failed to get ID3D11Texture2D from frame surface.");
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = regionX;
        sourceRegion.top = regionY;
        sourceRegion.front = 0;
        sourceRegion.right = regionX + regionWidth;
        sourceRegion.bottom = regionY + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(sharedTexture.get(), 0, 0, 0, 0, frameTexture.get(), 0, &sourceRegion);

        cudaGraphicsMapResources(1, &cudaResource, cudaStream);

        cudaArray_t cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

        int width = regionWidth;
        int height = regionHeight;
        cv::cuda::GpuMat frameGpu(height, width, CV_8UC4);

        cudaMemcpy2DFromArrayAsync(frameGpu.data, frameGpu.step, cuArray, 0, 0, width * sizeof(uchar4), height,
            cudaMemcpyDeviceToDevice, cudaStream);

        cudaGraphicsUnmapResources(1, &cudaResource, cudaStream);

        cudaStreamSynchronize(cudaStream);

        return frameGpu;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Error in GetNextFrame: " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

winrt::Windows::Graphics::Capture::GraphicsCaptureItem WinRTScreenCapture::CreateCaptureItemForMonitor(HMONITOR hMonitor)
{
    try
    {
        auto interopFactory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
        HRESULT hr = interopFactory->CreateForMonitor(hMonitor, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item));
        if (FAILED(hr))
        {
            throw std::runtime_error("[Capture] Can't create GraphicsCaptureItem for monitor. HRESULT: " + std::to_string(hr));
        }
        return item;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Error in CreateCaptureItemForMonitor: " << e.what() << std::endl;
        return nullptr;
    }
}

winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice WinRTScreenCapture::CreateDirect3DDevice(IDXGIDevice* dxgiDevice)
{
    winrt::com_ptr<::IInspectable> inspectable;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, inspectable.put()));
    return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
}

template<typename T>
winrt::com_ptr<T> WinRTScreenCapture::GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object)
{
    auto access = object.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result;
    winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
    return result;
}