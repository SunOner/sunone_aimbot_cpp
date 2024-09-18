#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "config.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using namespace cv;
using namespace std;

extern Detector detector;
extern Config config;

int screenWidth = 0;
int screenHeight = 0;

Mat cropCenterCPU(const Mat& src, int targetWidth, int targetHeight)
{
    int startX = (src.cols - targetWidth) / 2;
    int startY = (src.rows - targetHeight) / 2;
    return src(Rect(startX, startY, targetWidth, targetHeight)).clone();
}

class ScreenCapture
{
private:
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    ID3D11Texture2D* stagingTexture = nullptr;
    IDXGIOutput1* output1 = nullptr;

public:
    ScreenCapture()
    {
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
        textureDesc.Width = screenWidth;
        textureDesc.Height = screenHeight;
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

    Mat captureScreen()
    {
        IDXGIResource* desktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;

        HRESULT hr = deskDupl->AcquireNextFrame(100, &frameInfo, &desktopResource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            return Mat();
        }
        if (FAILED(hr)) return Mat();

        ID3D11Texture2D* desktopTexture = nullptr;
        hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(&desktopTexture));
        if (FAILED(hr))
        {
            desktopResource->Release();
            deskDupl->ReleaseFrame();
            return Mat();
        }

        d3dContext->CopyResource(stagingTexture, desktopTexture);

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = d3dContext->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
        if (FAILED(hr))
        {
            desktopTexture->Release();
            desktopResource->Release();
            deskDupl->ReleaseFrame();
            return Mat();
        }

        Mat screenshot(screenHeight, screenWidth, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
        Mat result = screenshot.clone();

        d3dContext->Unmap(stagingTexture, 0);
        desktopTexture->Release();
        desktopResource->Release();
        deskDupl->ReleaseFrame();

        return screenshot;
    }
};

extern Mat latestFrame;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    ScreenCapture capturer;
    Mat h_croppedScreenshot;

    while (!shouldExit)
    {
        Mat screenshot = capturer.captureScreen();
        if (!screenshot.empty())
        {
            h_croppedScreenshot = cropCenterCPU(screenshot, CAPTURE_WIDTH, CAPTURE_HEIGHT);

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = h_croppedScreenshot.clone();
            }
            frameCV.notify_one();

            Mat resized;
            cv::resize(h_croppedScreenshot, resized, cv::Size(config.engine_image_size, config.engine_image_size));
            
            detector.processFrame(resized);
        }
    }
}