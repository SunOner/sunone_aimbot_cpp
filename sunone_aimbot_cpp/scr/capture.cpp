#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <chrono>

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

#include "capture.h"
#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

using namespace std;

extern cv::Mat latestFrame;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;

int screenWidth = 0;
int screenHeight = 0;

class ScreenCapture
{
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

public:
    ScreenCapture(int desiredWidth, int desiredHeight)
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
            std::cerr << "Can't create Direct3DDevice!" << std::endl;
            return;
        }

        POINT pt = { 0, 0 };
        HMONITOR hMonitor = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);

        captureItem = CreateCaptureItemForMonitor(hMonitor);
        
        if (!captureItem)
        {
            std::cerr << "GraphicsCaptureItem not created!" << std::endl;
            return;
        }

        screenWidth = captureItem.Size().Width;
        screenHeight = captureItem.Size().Height;

        regionX = (screenWidth - regionWidth) / 2;
        regionY = (screenHeight - regionHeight) / 2;

        framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            device,
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            captureItem.Size()
        );

        session = framePool.CreateCaptureSession(captureItem);
        session.StartCapture();
    }

    ~ScreenCapture()
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

    cv::Mat GetNextFrame()
    {
        auto frame = framePool.TryGetNextFrame();
        if (!frame)
            return cv::Mat();

        auto surface = frame.Surface();

        winrt::com_ptr<ID3D11Texture2D> frameTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surface);

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

private:
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem CreateCaptureItemForMonitor(HMONITOR hMonitor)
    {
        auto interopFactory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
        HRESULT hr = interopFactory->CreateForMonitor(hMonitor, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item));
        if (FAILED(hr))
        {
            std::cerr << "Can't create GraphicsCaptureItem for monitor. HRESULT: " << std::hex << hr << std::endl;
            return nullptr;
        }
        return item;
    }

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice* dxgiDevice)
    {
        winrt::com_ptr<::IInspectable> inspectable;
        winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, inspectable.put()));
        return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    }

    template<typename T>
    winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object)
    {
        auto access = object.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<T> result;
        winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
        return result;
    }
};

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    ScreenCapture capturer(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    
    cv::Mat h_croppedScreenshot;
    bool buttonPreviouslyPressed = false;

    auto lastSaveTime = std::chrono::steady_clock::now();

    while (!shouldExit)
    {
        cv::Mat screenshot = capturer.GetNextFrame();

        if (!screenshot.empty())
        {
            cv::Mat mask = cv::Mat::zeros(screenshot.size(), CV_8UC1);
            cv::Point center(mask.cols / 2, mask.rows / 2);
            int radius = std::min(mask.cols, mask.rows) / 2;
            cv::circle(mask, center, radius, cv::Scalar(255), -1);

            cv::Mat maskedImage;
            screenshot.copyTo(maskedImage, mask);

            cv::Mat resized;
            cv::resize(maskedImage, resized, cv::Size(config.engine_image_size, config.engine_image_size));

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = resized.clone();
            }

            detector.processFrame(resized);
            frameCV.notify_one();

            bool buttonPressed = isAnyKeyPressed(config.screenshot_button) & 0x8000;

            if (buttonPressed)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();

                if (elapsed >= 250)
                {
                    auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                        ).count();
                    std::string filename = std::to_string(epoch_time) + ".jpg";

                    cv::imwrite("screenshots/" + filename, resized);

                    lastSaveTime = now;
                }
            }

            buttonPreviouslyPressed = buttonPressed;
        }
    }
}

void CloseCapture()
{
    winrt::uninit_apartment();
}