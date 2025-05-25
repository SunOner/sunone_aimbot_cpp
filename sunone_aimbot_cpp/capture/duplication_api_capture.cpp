#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "duplication_api_capture.h"
#include "sunone_aimbot_cpp.h"
#include "config.h"
#include "other_tools.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

template <typename T>
inline void SafeRelease(T** ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

struct FrameContext
{
    ID3D11Texture2D* texture = nullptr;
};

class DDAManager
{
public:
    DDAManager()
        : m_device(nullptr)
        , m_context(nullptr)
        , m_duplication(nullptr)
        , m_output1(nullptr)
    {
        ZeroMemory(&m_duplDesc, sizeof(m_duplDesc));
    }

    ~DDAManager()
    {
        Release();
    }

    HRESULT Initialize(
        int monitorIndex,
        int captureWidth,
        int captureHeight,
        int& outScreenWidth,
        int& outScreenHeight,
        ID3D11Device** outDevice,
        ID3D11DeviceContext** outContext)
    {
        HRESULT hr = S_OK;

        IDXGIFactory1* factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] CreateDXGIFactory1 failed hr=" << std::hex << hr << std::endl;
            return hr;
        }

        IDXGIAdapter1* adapter = nullptr;
        hr = factory->EnumAdapters1(monitorIndex, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] No adapter with index " << monitorIndex << std::endl;
            factory->Release();
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumAdapters1 failed hr=" << std::hex << hr << std::endl;
            factory->Release();
            return hr;
        }

        IDXGIOutput* output = nullptr;
        hr = adapter->EnumOutputs(0, &output);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] The adapter has no outputs" << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumOutputs returned error hr=" << std::hex << hr << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        {
            D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
            UINT createDeviceFlags = 0;

            hr = D3D11CreateDevice(
                adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                createDeviceFlags,
                featureLevels,
                1,
                D3D11_SDK_VERSION,
                &m_device,
                nullptr,
                &m_context
            );
            if (FAILED(hr))
            {
                std::cerr << "[DDA] D3D11CreateDevice failed hr=" << std::hex << hr << std::endl;
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&m_output1);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] QueryInterface(IDXGIOutput1) failed hr=" << std::hex << hr << std::endl;
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] DuplicateOutput failed hr=" << std::hex << hr << std::endl;
            SafeRelease(&m_output1);
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        m_duplication->GetDesc(&m_duplDesc);

        DXGI_OUTPUT_DESC oDesc{};
        output->GetDesc(&oDesc);
        outScreenWidth = oDesc.DesktopCoordinates.right - oDesc.DesktopCoordinates.left;
        outScreenHeight = oDesc.DesktopCoordinates.bottom - oDesc.DesktopCoordinates.top;

        SafeRelease(&output);
        SafeRelease(&adapter);
        SafeRelease(&factory);

        if (outDevice)  *outDevice = m_device;
        if (outContext) *outContext = m_context;

        return hr;
    }

    HRESULT AcquireFrame(FrameContext& frameCtx, UINT timeout = 100)
    {
        if (!m_duplication) return E_FAIL;
        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        IDXGIResource* resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (FAILED(hr)) return hr;

        if (resource)
        {
            hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&frameCtx.texture);
            resource->Release();
        }
        return hr;
    }

    void ReleaseFrame()
    {
        if (m_duplication)
            m_duplication->ReleaseFrame();
    }

    void Release()
    {
        if (m_duplication)
        {
            m_duplication->ReleaseFrame();
            m_duplication->Release();
            m_duplication = nullptr;
        }
        SafeRelease(&m_output1);
        SafeRelease(&m_context);
        SafeRelease(&m_device);
    }

public:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_context;
    IDXGIOutputDuplication* m_duplication;
    IDXGIOutput1* m_output1;
    DXGI_OUTDUPL_DESC m_duplDesc;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight)
    : d3dDevice(nullptr)
    , d3dContext(nullptr)
    , deskDupl(nullptr)
    , output1(nullptr)
    , sharedTexture(nullptr)
    , stagingTextureCPU(nullptr)
    , screenWidth(0)
    , screenHeight(0)
    , regionWidth(desiredWidth)
    , regionHeight(desiredHeight)
{
    m_ddaManager = std::make_unique<DDAManager>();

    HRESULT hr = m_ddaManager->Initialize(
        config.monitor_idx,
        regionWidth,
        regionHeight,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext
    );
    if (FAILED(hr))
    {
        std::cerr << "[DDA] DDAManager Initialize failed hr=0x" << std::hex << hr << std::endl;
        return;
    }

    createStagingTextureCPU();
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
    if (m_ddaManager)
    {
        m_ddaManager->Release();
        m_ddaManager.reset();
    }
    SafeRelease(&stagingTextureCPU);
    SafeRelease(&sharedTexture);

    d3dDevice = nullptr;
    d3dContext = nullptr;
    deskDupl = nullptr;
    output1 = nullptr;
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrameCpu()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication || !stagingTextureCPU)
        return cv::Mat();

    FrameContext frameCtx;
    HRESULT hr = m_ddaManager->AcquireFrame(frameCtx, 100);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        return cv::Mat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        capture_method_changed.store(true);
        m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }
    else if (FAILED(hr))
    {
        std::cerr << "[DuplicationAPIScreenCapture] AcquireNextFrame (CPU) failed hr=0x"
            << std::hex << hr << std::endl;
        m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    if (!frameCtx.texture)
    {
        m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    D3D11_BOX sourceRegion;
    sourceRegion.left = (screenWidth - regionWidth) / 2;
    sourceRegion.top = (screenHeight - regionHeight) / 2;
    sourceRegion.front = 0;
    sourceRegion.right = sourceRegion.left + regionWidth;
    sourceRegion.bottom = sourceRegion.top + regionHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        stagingTextureCPU,
        0,
        0, 0, 0,
        frameCtx.texture,
        0,
        &sourceRegion
    );

    m_ddaManager->ReleaseFrame();
    frameCtx.texture->Release();

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hrMap = d3dContext->Map(stagingTextureCPU, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hrMap))
    {
        std::cerr << "[DDA] Map stagingTextureCPU failed hr=" << std::hex << hrMap << std::endl;
        return cv::Mat();
    }

    cv::Mat cpuFrame(regionHeight, regionWidth, CV_8UC4);
    for (int y = 0; y < regionHeight; y++)
    {
        unsigned char* dstRow = cpuFrame.ptr<unsigned char>(y);
        unsigned char* srcRow = (unsigned char*)mapped.pData + y * mapped.RowPitch;
        memcpy(dstRow, srcRow, regionWidth * 4);
    }

    d3dContext->Unmap(stagingTextureCPU, 0);
    return cpuFrame;
}

bool DuplicationAPIScreenCapture::createStagingTextureCPU()
{
    if (!d3dDevice) return false;

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
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, &stagingTextureCPU);
    if (FAILED(hr))
    {
        std::cerr << "[DDA] CreateTexture2D(staging) failed hr=" << std::hex << hr << std::endl;
        return false;
    }
    return true;
}