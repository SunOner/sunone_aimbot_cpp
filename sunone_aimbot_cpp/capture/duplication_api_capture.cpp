#include "sunone_aimbot_cpp.h"
#include "duplication_api_capture.h"
#include "config.h"
#include "other_tools.h"
#include "capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

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
    D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, featureLevels, 1,
        D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dContext);
    output1->DuplicateOutput(d3dDevice, &deskDupl);

    screenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    screenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

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

    HRESULT hr = d3dDevice->CreateTexture2D(&sharedTexDesc, nullptr, &sharedTexture);
    if (FAILED(hr))
    {
        std::cerr << "[Capture] Failed to create shared texture." << std::endl;
        return;
    }

    cudaError_t err = cudaGraphicsD3D11RegisterResource(&cudaResource, sharedTexture, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
    {
        std::cerr << "[Capture] Failed to register shared texture with CUDA." << std::endl;
        return;
    }

    cudaStreamCreate(&cudaStream);

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
    if (cudaResource)
    {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }

    if (sharedTexture)
    {
        sharedTexture->Release();
        sharedTexture = nullptr;
    }

    if (cudaStream)
    {
        cudaStreamDestroy(cudaStream);
        cudaStream = nullptr;
    }
}

cv::cuda::GpuMat DuplicationAPIScreenCapture::GetNextFrame()
{
    try
    {
        IDXGIResource* desktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = deskDupl->AcquireNextFrame(100, &frameInfo, &desktopResource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            return cv::cuda::GpuMat();
        }

        if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET)
        {
            capture_method_changed.store(true);
        }

        if (FAILED(hr)) return cv::cuda::GpuMat();

        ID3D11Texture2D* desktopTexture = nullptr;
        hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(&desktopTexture));
        if (FAILED(hr))
        {
            desktopResource->Release();
            deskDupl->ReleaseFrame();
            return cv::cuda::GpuMat();
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = (screenWidth - regionWidth) / 2;
        sourceRegion.top = (screenHeight - regionHeight) / 2;
        sourceRegion.front = 0;
        sourceRegion.right = sourceRegion.left + regionWidth;
        sourceRegion.bottom = sourceRegion.top + regionHeight;
        sourceRegion.back = 1;

        d3dContext->CopySubresourceRegion(sharedTexture, 0, 0, 0, 0, desktopTexture, 0, &sourceRegion);

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

        desktopTexture->Release();
        desktopResource->Release();
        deskDupl->ReleaseFrame();

        return frameGpu;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Error in GetNextFrame: " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}