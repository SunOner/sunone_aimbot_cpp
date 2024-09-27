#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "capture.h"
#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

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
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    ID3D11Texture2D* stagingTexture = nullptr;
    IDXGIOutput1* output1 = nullptr;

    int screenWidth = 0;
    int screenHeight = 0;
    int regionWidth = 0;
    int regionHeight = 0;

public:
    ScreenCapture(int desiredWidth, int desiredHeight)
    {
        regionWidth = desiredWidth;
        regionHeight = desiredHeight;

        IDXGIFactory1* factory = nullptr;
        IDXGIAdapter* adapter = nullptr;
        IDXGIOutput* output = nullptr;

        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&factory));
        factory->EnumAdapters(0, &adapter);
        adapter->EnumOutputs(0, &output);
        output->QueryInterface(__uuidof(IDXGIOutput1), (void**)(&output1));

        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
        D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, featureLevels, 1,
            D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dContext);

        output1->DuplicateOutput(d3dDevice, &deskDupl);

        DXGI_OUTPUT_DESC outputDesc;
        output->GetDesc(&outputDesc);
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

    ~ScreenCapture()
    {
        if (stagingTexture) stagingTexture->Release();
        if (deskDupl) deskDupl->Release();
        if (output1) output1->Release();
        if (d3dContext) d3dContext->Release();
        if (d3dDevice) d3dDevice->Release();
    }

    cv::Mat captureScreen()
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
};

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    ScreenCapture capturer(CAPTURE_WIDTH, CAPTURE_HEIGHT);
    cv::Mat h_croppedScreenshot;
    bool buttonPreviouslyPressed = false;

    auto lastSaveTime = std::chrono::steady_clock::now();

    while (!shouldExit)
    {
        cv::Mat screenshot = capturer.captureScreen();
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
                    auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string filename = std::to_string(epoch_time) + ".jpg";

                    cv::imwrite("screenshots/" + filename, resized);

                    lastSaveTime = now;
                }
            }

            buttonPreviouslyPressed = buttonPressed;
        }
    }
}