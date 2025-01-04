#ifndef WINRT_CAPTURE_H
#define WINRT_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#include "capture.h"
#include <mutex>

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

class WinRTScreenCapture : public IScreenCapture
{
public:
    WinRTScreenCapture(int desiredWidth, int desiredHeight);
    ~WinRTScreenCapture();
    cv::cuda::GpuMat GetNextFrame();

private:
    winrt::com_ptr<ID3D11Device> d3dDevice;
    winrt::com_ptr<ID3D11DeviceContext> d3dContext;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice device{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem captureItem{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session{ nullptr };
    winrt::com_ptr<ID3D11Texture2D> stagingTexture;

    bool interopInitialized = false;
    winrt::com_ptr<ID3D11Texture2D> sharedTexture;
    cudaGraphicsResource* cudaResource = nullptr;
    cudaStream_t cudaStream = nullptr;

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

#endif // WINRT_CAPTURE_H