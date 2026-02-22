#ifndef DUPLICATION_API_CAPTURE_H
#define DUPLICATION_API_CAPTURE_H

#include <d3d11.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
struct cudaGraphicsResource;
#endif

#include "capture.h"

class DDAManager;

class DuplicationAPIScreenCapture : public IScreenCapture
{
public:
    DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight, int monitorIndex);
    ~DuplicationAPIScreenCapture();

    cv::Mat GetNextFrameCpu() override;
#ifdef USE_CUDA
    bool GetNextFrameGpu(cv::cuda::GpuMat& gpuFrameBgra);
#endif

private:
    std::unique_ptr<DDAManager> m_ddaManager;

    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    IDXGIOutput1* output1 = nullptr;

    ID3D11Texture2D* sharedTexture = nullptr;

    ID3D11Texture2D* stagingTextureCPU = nullptr;
#ifdef USE_CUDA
    ID3D11Texture2D* interopTextureGPU = nullptr;
    cudaGraphicsResource* cudaInteropResource = nullptr;
    bool cudaInteropReady = false;
#endif

    int screenWidth = 0;
    int screenHeight = 0;
    int regionWidth = 0;
    int regionHeight = 0;

    bool createStagingTextureCPU();
#ifdef USE_CUDA
    bool createCudaInteropTexture();
    void releaseCudaInteropTexture();
#endif
};

#endif // DUPLICATION_API_CAPTURE_H
