#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include "timeapi.h"
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.System.Threading.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/base.h>
#include <comdef.h>

#include "capture.h"
#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include <other_tools.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

using namespace std;

cv::cuda::GpuMat latestFrameGpu;
cv::Mat latestFrameCpu;

std::mutex frameMutex;

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

// Realtime config vars
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;

// WinRT
class WinRTScreenCapture : public IScreenCapture
{
public:
    WinRTScreenCapture(int desiredWidth, int desiredHeight);
    ~WinRTScreenCapture();
    cv::Mat GetNextFrame();

private:
    winrt::com_ptr<ID3D11Device> d3dDevice;
    winrt::com_ptr<ID3D11DeviceContext> d3dContext;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice device{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem captureItem{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session{ nullptr };
    winrt::com_ptr<ID3D11Texture2D> stagingTexture;

    int screenWidth = 0;
    int screenHeight = 0;
    int regionWidth = 0;
    int regionHeight = 0;

    int regionX = 0;
    int regionY = 0;

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem CreateCaptureItemForMonitor(HMONITOR hMonitor);
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice* dxgiDevice);

    template<typename T>
    winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object);
};

class DuplicationAPIScreenCapture : public IScreenCapture
{
public:
    DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight);
    ~DuplicationAPIScreenCapture();
    cv::Mat GetNextFrame() override;

private:
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    ID3D11Texture2D* stagingTexture = nullptr;
    IDXGIOutput1* output1 = nullptr;

    int screenWidth = 0;
    int screenHeight = 0;
    int regionWidth = 0;
    int regionHeight = 0;
};

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

cv::Mat WinRTScreenCapture::GetNextFrame()
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
            return cv::Mat();
        }

        auto surface = frame.Surface();

        winrt::com_ptr<ID3D11Texture2D> frameTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surface);

        if (!frameTexture)
        {
            throw std::runtime_error("Failed to get ID3D11Texture2D from frame surface.");
        }

        D3D11_TEXTURE2D_DESC desc;
        frameTexture->GetDesc(&desc);


        if (!stagingTexture)
        {
            D3D11_TEXTURE2D_DESC stagingDesc = {};
            stagingDesc.Width = regionWidth;
            stagingDesc.Height = regionHeight;
            stagingDesc.MipLevels = 1;
            stagingDesc.ArraySize = 1;
            stagingDesc.Format = desc.Format;
            stagingDesc.SampleDesc.Count = 1;
            stagingDesc.SampleDesc.Quality = 0;
            stagingDesc.Usage = D3D11_USAGE_STAGING;
            stagingDesc.BindFlags = 0;
            stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            stagingDesc.MiscFlags = 0;

            winrt::check_hresult(d3dDevice->CreateTexture2D(&stagingDesc, nullptr, stagingTexture.put()));
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = regionX;
        sourceRegion.top = regionY;
        sourceRegion.front = 0;
        sourceRegion.right = regionX + regionWidth;
        sourceRegion.bottom = regionY + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(
            stagingTexture.get(),
            0,
            0,
            0,
            0,
            frameTexture.get(),
            0,
            &sourceRegion
        );

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        winrt::check_hresult(d3dContext->Map(
            stagingTexture.get(),
            0,
            D3D11_MAP_READ,
            0,
            &mappedResource
        ));

        cv::Mat screenshot(regionHeight, regionWidth, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
        cv::Mat result;
        screenshot.copyTo(result);

        d3dContext->Unmap(stagingTexture.get(), 0);

        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Error in GetNextFrame: " << e.what() << std::endl;
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
            throw std::runtime_error("Can't create GraphicsCaptureItem for monitor. HRESULT: " + std::to_string(hr));
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

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight)
{
    regionWidth = desiredWidth;
    regionHeight = desiredHeight;

    IDXGIAdapter1* adapter = nullptr;
    IDXGIOutput* output = nullptr;
    IDXGIFactory1* factory = nullptr;

    DXGI_ADAPTER_DESC adapterDesc;
    DXGI_OUTPUT_DESC outputDesc;

    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)))
    {
        std::cerr << "[Capture] error with DXGI factory. Choose another monitor." << std::endl;
        std::cin.get();
    }
    
    if (factory->EnumAdapters1(config.monitor_idx, &adapter) == DXGI_ERROR_NOT_FOUND)
    {
        std::cerr << "[Capture] ERROR with get enum adapters. Choose another monitor." << std::endl;
        std::cin.get();
    }

    adapter->GetDesc(&adapterDesc);

    if (config.verbose)
    {
        std::wcout << L"[Capture] Using device: " << adapterDesc.Description << std::endl;
        std::wcout << L"[Capture] Device shared system memory: " << adapterDesc.SharedSystemMemory << std::endl;
    }

    if (adapter->EnumOutputs(0, &output) == DXGI_ERROR_NOT_FOUND)
    {
        std::cerr << "[Capture] Error with get enum outputs. Choose another monitor." << std::endl;
        std::cin.get();
    }

    output->GetDesc(&outputDesc);
    if (config.verbose)
    {
        std::wcout << L"[Capture] Using monitor: " << outputDesc.DeviceName << std::endl;
    }

    output->QueryInterface(__uuidof(IDXGIOutput1), (void**)(&output1));

    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, featureLevels, 1, D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dContext);
    output1->DuplicateOutput(d3dDevice, &deskDupl);

    screenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    screenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = regionWidth;
    textureDesc.Height = regionHeight;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_STAGING;
    textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d3dDevice->CreateTexture2D(&textureDesc, nullptr, &stagingTexture);

    output->Release();
    adapter->Release();
    factory->Release();
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
    if (stagingTexture) stagingTexture->Release();
    if (deskDupl) deskDupl->Release();
    if (output1) output1->Release();
    if (d3dContext) d3dContext->Release();
    if (d3dDevice) d3dDevice->Release();
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrame()
{
    try
    {
        IDXGIResource* desktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = deskDupl->AcquireNextFrame(100, &frameInfo, &desktopResource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            return cv::Mat();
        }
        if (FAILED(hr)) return cv::Mat();

        ID3D11Texture2D* desktopTexture = nullptr;
        hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(&desktopTexture));
        if (FAILED(hr))
        {
            desktopResource->Release();
            deskDupl->ReleaseFrame();
            return cv::Mat();
        }

        int regionX = (screenWidth - regionWidth) / 2;
        int regionY = (screenHeight - regionHeight) / 2;
        D3D11_BOX sourceRegion;
        sourceRegion.left = regionX;
        sourceRegion.top = regionY;
        sourceRegion.front = 0;
        sourceRegion.right = regionX + regionWidth;
        sourceRegion.bottom = regionY + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(
            stagingTexture,
            0,
            0,
            0,
            0,
            desktopTexture,
            0,
            &sourceRegion
        );

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = d3dContext->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
        if (FAILED(hr))
        {
            desktopTexture->Release();
            desktopResource->Release();
            deskDupl->ReleaseFrame();
            return cv::Mat();
        }

        cv::Mat screenshot(regionHeight, regionWidth, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
        cv::Mat result;
        screenshot.copyTo(result);

        d3dContext->Unmap(stagingTexture, 0);
        desktopTexture->Release();
        desktopResource->Release();
        deskDupl->ReleaseFrame();

        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Error in GetNextFrame: " << e.what() << std::endl;
        return cv::Mat();
    }
}

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        IScreenCapture* capturer = nullptr;

        if (config.duplication_api)
        {
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                cout << "[Capture] Using Duplication API." << endl;
            }
        }
        else
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            capturer = new WinRTScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                cout << "[Capture] Using WinRT." << endl;
            }
        }

        cv::cuda::GpuMat latestFrameGpu;
        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        bool frameLimitingEnabled = false;

        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();

        while (!shouldExit)
        {
            if (capture_fps_changed.load())
            {
                if (config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled)
                    {
                        timeBeginPeriod(1);
                        frameLimitingEnabled = true;
                    }
                    frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
                }
                else
                {
                    if (frameLimitingEnabled)
                    {
                        timeEndPeriod(1);
                        frameLimitingEnabled = false;
                    }
                    frame_duration = std::nullopt;
                }
                capture_fps_changed.store(false);
            }

            if (detection_resolution_changed.load()
                || capture_method_changed.load()
                || capture_cursor_changed.load()
                || capture_borders_changed.load())
            {
                delete capturer;

                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                if (config.duplication_api)
                {
                    capturer = new DuplicationAPIScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                    {
                        cout << "[Capture] Using Duplication API." << endl;
                    }
                }
                else
                {
                    capturer = new WinRTScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                    {
                        cout << "[Capture] Using WinRT." << endl;
                    }
                }

                screenWidth = new_CAPTURE_WIDTH;
                screenHeight = new_CAPTURE_HEIGHT;

                detection_resolution_changed.store(false);
                capture_method_changed.store(false);
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            cv::Mat screenshot = capturer->GetNextFrame();

            if (!screenshot.empty())
            {
                cv::cuda::GpuMat screenshotGpu;
                try
                {
                    screenshotGpu.upload(screenshot);
                }
                catch (const cv::Exception& e)
                {
                    std::cerr << "[Capture]: " << e.what() << std::endl;
                    continue;
                }

                cv::cuda::GpuMat resizedGpu;

                if (config.circle_mask)
                {
                    cv::Mat mask = cv::Mat::zeros(screenshot.size(), CV_8UC1);
                    cv::Point center(mask.cols / 2, mask.rows / 2);
                    int radius = std::min(mask.cols, mask.rows) / 2;
                    cv::circle(mask, center, radius, cv::Scalar(255), -1);

                    cv::cuda::GpuMat maskGpu;
                    maskGpu.upload(mask);

                    cv::cuda::GpuMat maskedImageGpu;
                    screenshotGpu.copyTo(maskedImageGpu, maskGpu);

                    cv::cuda::resize(maskedImageGpu, resizedGpu, cv::Size(config.engine_image_size, config.engine_image_size), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    cv::cuda::resize(screenshotGpu, resizedGpu, cv::Size(config.engine_image_size, config.engine_image_size), 0, 0, cv::INTER_LINEAR);
                }

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameGpu = resizedGpu.clone();
                }

                detector.processFrame(resizedGpu);
                resizedGpu.download(latestFrameCpu);

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameCpu = latestFrameCpu.clone();
                }
                frameCV.notify_one();

                if (config.screenshot_button[0] != "None")
                {
                    bool buttonPressed = isAnyKeyPressed(config.screenshot_button);

                    if (buttonPressed)
                    {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();

                        if (elapsed >= config.screenshot_delay)
                        {
                            cv::Mat resizedCpu;
                            resizedGpu.download(resizedCpu);

                            auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();
                            std::string filename = std::to_string(epoch_time) + ".jpg";

                            cv::imwrite("screenshots/" + filename, resizedCpu);

                            lastSaveTime = now;
                        }
                    }

                    buttonPreviouslyPressed = buttonPressed;
                }

                captureFrameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = currentTime - captureFpsStartTime;

                if (elapsed.count() >= 1.0)
                {
                    captureFps = static_cast<int>(captureFrameCount / elapsed.count());
                    captureFrameCount = 0;
                    captureFpsStartTime = currentTime;
                }
            }

            if (frame_duration.has_value())
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto work_duration = end_time - start_time;

                auto sleep_duration = frame_duration.value() - work_duration;

                if (sleep_duration > std::chrono::duration<double, std::milli>(0))
                {
                    std::this_thread::sleep_for(sleep_duration);
                }

                start_time = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
        {
            timeEndPeriod(1);
        }

        delete capturer;

        if (!config.duplication_api)
        {
            winrt::uninit_apartment();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CaptureThread] Unhandled exception: " << e.what() << std::endl;
    }
}