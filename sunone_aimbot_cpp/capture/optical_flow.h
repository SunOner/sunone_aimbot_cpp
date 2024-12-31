#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

class OpticalFlow
{
public:
    OpticalFlow();

    void startOpticalFlowThread();
    void stopOpticalFlowThread();

    void enqueueFrame(const cv::cuda::GpuMat& frame);
    void getMotion(int& xShift, int& yShift);
    void drawOpticalFlow(cv::Mat& frame);
    void manageOpticalFlowThread();

private:
    void computeOpticalFlow(const cv::cuda::GpuMat& frame);
    void opticalFlowLoop();
    void preprocessFrame(cv::cuda::GpuMat& frameGray);

    std::thread opticalFlowThread;
    std::atomic<bool> shouldExit;

    std::queue<cv::cuda::GpuMat> frameQueue;
    std::condition_variable frameCV;
    std::mutex frameMutex;

    cv::cuda::GpuMat prevFrameGray;
    cv::cuda::GpuMat flow;
    std::mutex flowMutex;
    int xShift;
    int yShift;

    cv::cuda::GpuMat hintFlow;
    int flowWidth, flowHeight;
    int hintWidth, hintHeight;
    int outputGridSizeValue;
    int hintGridSizeValue;

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> opticalFlow;
};

#endif // OPTICAL_FLOW_H