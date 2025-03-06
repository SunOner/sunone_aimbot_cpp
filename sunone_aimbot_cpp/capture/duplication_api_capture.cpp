#include "sunone_aimbot_cpp.h"
#include "duplication_api_capture.h"
#include "config.h"
#include "other_tools.h"
#include "capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

template <typename T>
inline void SafeRelease(T **ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

struct FrameContext
{
    ID3D11Texture2D *texture = nullptr;
    std::vector<DXGI_OUTDUPL_MOVE_RECT> moveRects;
    std::vector<RECT> dirtyRects;
};

class DDAManager
{
public:
    DDAManager()
        : m_device(nullptr), m_context(nullptr), m_duplication(nullptr), m_output1(nullptr), m_sharedTexture(nullptr), m_cudaResource(nullptr), m_cudaStream(nullptr), m_framePool(5) // Pre-allocate 5 frames in the pool
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
        int &outScreenWidth,
        int &outScreenHeight,
        ID3D11Device **outDevice = nullptr,
        ID3D11DeviceContext **outContext = nullptr)
    {
        HRESULT hr = S_OK;

        IDXGIFactory1 *factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&factory);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] Failed to create DXGIFactory1 (hr = " << std::hex << hr << ")." << std::endl;
            return hr;
        }

        IDXGIAdapter1 *adapter = nullptr;
        hr = factory->EnumAdapters1(monitorIndex, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] Not found adapter with index " << monitorIndex << ". Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            factory->Release();
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumAdapters1 return error (hr = " << std::hex << hr << ")." << std::endl;
            factory->Release();
            return hr;
        }

        IDXGIOutput *output = nullptr;
        hr = adapter->EnumOutputs(0, &output);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            std::cerr << "[DDA] The adapter has no outputs (monitors). Error code: DXGI_ERROR_NOT_FOUND." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }
        else if (FAILED(hr))
        {
            std::cerr << "[DDA] EnumOutputs returned an error (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        {
            D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_0};
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
                &m_context);

            if (FAILED(hr))
            {
                std::cerr << "[DDA] Couldn't create D3D11Device (hr = " << std::hex << hr << ")." << std::endl;
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void **)&m_output1);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] QueryInterface on IDXGIOutput1 failed (hr = " << std::hex << hr << ")." << std::endl;
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
            std::cerr << "[DDA] DuplicateOutput failed (hr = " << std::hex << hr << ")." << std::endl;
            SafeRelease(&m_output1);
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        m_duplication->GetDesc(&m_duplDesc);

        DXGI_OUTPUT_DESC outputDesc{};
        output->GetDesc(&outputDesc);
        outScreenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
        outScreenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

        if (config.verbose)
        {
            std::wcout << L"[DDA] Monitor: " << outputDesc.DeviceName
                       << L", Resolution: " << outScreenWidth << L"x" << outScreenHeight << std::endl;
        }

        {
            D3D11_TEXTURE2D_DESC sharedTexDesc = {};
            sharedTexDesc.Width = captureWidth;
            sharedTexDesc.Height = captureHeight;
            sharedTexDesc.MipLevels = 1;
            sharedTexDesc.ArraySize = 1;
            sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            sharedTexDesc.SampleDesc.Count = 1;
            sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
            sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            sharedTexDesc.CPUAccessFlags = 0;
            sharedTexDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

            hr = m_device->CreateTexture2D(&sharedTexDesc, nullptr, &m_sharedTexture);
            if (FAILED(hr))
            {
                std::cerr << "[DDA] Couldn't create sharedTexture (hr = " << std::hex << hr << ")." << std::endl;
                SafeRelease(&m_duplication);
                SafeRelease(&m_output1);
                SafeRelease(&m_context);
                SafeRelease(&m_device);
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        {
            cudaError_t err = cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_sharedTexture, cudaGraphicsRegisterFlagsNone);
            if (err != cudaSuccess)
            {
                std::cerr << "[DDA] Error registering sharedTexture in CUDA: "
                          << cudaGetErrorString(err) << std::endl;
                Release();
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return E_FAIL;
            }

            cudaStreamCreate(&m_cudaStream);
        }

        SafeRelease(&output);
        SafeRelease(&adapter);
        SafeRelease(&factory);

        if (outDevice)
            *outDevice = m_device;
        if (outContext)
            *outContext = m_context;

        return hr;
    }

    HRESULT AcquireFrame(FrameContext &frameCtx, UINT timeout = 100)
    {
        if (!m_duplication)
            return E_FAIL;

        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        IDXGIResource *resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (FAILED(hr))
            return hr;

        if (frameInfo.TotalMetadataBufferSize > 0)
        {
            UINT moveCount = frameInfo.TotalMetadataBufferSize;
            frameCtx.moveRects.resize(moveCount / sizeof(DXGI_OUTDUPL_MOVE_RECT));
            hr = m_duplication->GetFrameMoveRects(moveCount,
                                                  frameCtx.moveRects.data(), &moveCount);

            UINT dirtyCount = frameInfo.TotalMetadataBufferSize - moveCount;
            frameCtx.dirtyRects.resize(dirtyCount / sizeof(RECT));
            hr = m_duplication->GetFrameDirtyRects(dirtyCount,
                                                   frameCtx.dirtyRects.data(), &dirtyCount);
        }

        hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&frameCtx.texture);
        resource->Release();

        return (SUCCEEDED(hr)) ? S_OK : hr;
    }

    void ReleaseFrame()
    {
        if (m_duplication)
            m_duplication->ReleaseFrame();
    }

    HRESULT CopyFromDesktopTexture(ID3D11Texture2D *srcTexture, int fullWidth, int fullHeight, int regionWidth, int regionHeight)
    {
        if (!m_context || !m_sharedTexture)
            return E_FAIL;

        D3D11_BOX sourceRegion;
        sourceRegion.left = (fullWidth - regionWidth) / 2;
        sourceRegion.top = (fullHeight - regionHeight) / 2;
        sourceRegion.front = 0;
        sourceRegion.right = sourceRegion.left + regionWidth;
        sourceRegion.bottom = sourceRegion.top + regionHeight;
        sourceRegion.back = 1;

        m_context->CopySubresourceRegion(
            m_sharedTexture,
            0,
            0, 0, 0,
            srcTexture,
            0,
            &sourceRegion);
        return S_OK;
    }

    cv::cuda::GpuMat CopySharedTextureToCudaMat(int regionWidth, int regionHeight)
    {
        // Map the shared texture to CUDA
        cudaError_t err = cudaGraphicsMapResources(1, &m_cudaResource, m_cudaStream);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsMapResources error: " << cudaGetErrorString(err) << std::endl;
            return cv::cuda::GpuMat();
        }

        // Get the mapped array
        cudaArray_t cuArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cudaResource, 0, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray error: " << cudaGetErrorString(err) << std::endl;
            cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);
            return cv::cuda::GpuMat();
        }

        // Get a frame from the pool or create a new one if needed
        cv::cuda::GpuMat frameGpu;
        if (!m_framePool.empty())
        {
            frameGpu = m_framePool.back();
            m_framePool.pop_back();

            // Ensure the frame has the correct size
            if (frameGpu.rows != regionHeight || frameGpu.cols != regionWidth)
            {
                frameGpu.release();
                frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
            }
        }
        else
        {
            frameGpu = cv::cuda::GpuMat(regionHeight, regionWidth, CV_8UC4);
        }

        // Copy from the CUDA array to the GpuMat
        cudaMemcpy2DFromArrayAsync(
            frameGpu.data, frameGpu.step,
            cuArray, 0, 0,
            regionWidth * 4, regionHeight,
            cudaMemcpyDeviceToDevice, m_cudaStream);

        // Unmap the resource
        cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);

        // Ensure the copy is complete
        cudaStreamSynchronize(m_cudaStream);

        return frameGpu;
    }

    // Return a frame to the pool for reuse
    void RecycleFrame(cv::cuda::GpuMat &frame)
    {
        if (m_framePool.size() < 10) // Limit pool size
        {
            m_framePool.push_back(frame);
        }
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

        if (m_cudaResource)
        {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        SafeRelease(&m_sharedTexture);

        if (m_cudaStream)
        {
            cudaStreamDestroy(m_cudaStream);
            m_cudaStream = nullptr;
        }
    }

public:
    ID3D11Device *m_device;
    ID3D11DeviceContext *m_context;
    IDXGIOutputDuplication *m_duplication;
    IDXGIOutput1 *m_output1;
    DXGI_OUTDUPL_DESC m_duplDesc;

    ID3D11Texture2D *m_sharedTexture;
    cudaGraphicsResource *m_cudaResource;
    cudaStream_t m_cudaStream;
    std::vector<cv::cuda::GpuMat> m_framePool;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight)
    : d3dDevice(nullptr), d3dContext(nullptr), deskDupl(nullptr), stagingTexture(nullptr), output1(nullptr), sharedTexture(nullptr), cudaResource(nullptr), cudaStream(nullptr), regionWidth(desiredWidth), regionHeight(desiredHeight), screenWidth(0), screenHeight(0)
{
    m_ddaManager = std::make_unique<DDAManager>();

    HRESULT hr = m_ddaManager->Initialize(
        config.monitor_idx,
        regionWidth,
        regionHeight,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext);
    if (FAILED(hr))
    {
        std::cerr << "[Capture] Error initializing DuplicationAPIScreenCapture: hr=0x"
                  << std::hex << hr << std::endl;
        return;
    }
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
    if (m_ddaManager)
    {
        m_ddaManager->Release();
        m_ddaManager.reset();
    }
    d3dDevice = nullptr;
    d3dContext = nullptr;
    deskDupl = nullptr;
    stagingTexture = nullptr;
    output1 = nullptr;
    sharedTexture = nullptr;
    cudaResource = nullptr;
    cudaStream = nullptr;
}

cv::cuda::GpuMat DuplicationAPIScreenCapture::GetNextFrame()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication)
        return cv::cuda::GpuMat();

    HRESULT hr = S_OK;
    FrameContext frameCtx;

    hr = m_ddaManager->AcquireFrame(frameCtx, 100);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        return cv::cuda::GpuMat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_REMOVED)
    {
        capture_method_changed.store(true);
        m_ddaManager->ReleaseFrame();
        return cv::cuda::GpuMat();
    }
    else if (FAILED(hr))
    {
        std::cerr << "[Capture] AcquireNextFrame failed (hr=0x" << std::hex << hr << ")" << std::endl;
        m_ddaManager->ReleaseFrame();
        return cv::cuda::GpuMat();
    }

    if (!frameCtx.texture)
    {
        m_ddaManager->ReleaseFrame();
        return cv::cuda::GpuMat();
    }

    m_ddaManager->CopyFromDesktopTexture(frameCtx.texture, screenWidth, screenHeight, regionWidth, regionHeight);

    m_ddaManager->ReleaseFrame();

    cv::cuda::GpuMat frameGpu = m_ddaManager->CopySharedTextureToCudaMat(regionWidth, regionHeight);

    frameCtx.texture->Release();
    frameCtx.texture = nullptr;

    // Recycle the previous frame if we have one
    if (!m_previousFrame.empty())
    {
        m_ddaManager->RecycleFrame(m_previousFrame);
    }

    // Store the current frame for recycling next time
    m_previousFrame = frameGpu;

    return frameGpu;
}