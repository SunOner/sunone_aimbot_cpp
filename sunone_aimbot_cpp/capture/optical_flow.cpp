#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "optical_flow.h"
#include "sunone_aimbot_cpp.h"

void OpticalFlow::preprocessFrame(cv::cuda::GpuMat& frameGray)
{
    if (frameGray.empty()) return;

    cv::Mat frameCPU;
    frameGray.download(frameCPU);

    if (frameCPU.type() != CV_8UC1)
    {
        cv::cvtColor(frameCPU, frameCPU, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurredFrame, equalizedFrame, thresholdFrame;

    cv::GaussianBlur(frameCPU, blurredFrame, cv::Size(3, 3), 0);
    cv::equalizeHist(blurredFrame, equalizedFrame);
    cv::threshold(equalizedFrame, thresholdFrame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    frameGray.upload(thresholdFrame);
}

OpticalFlow::OpticalFlow() : xShift(0), yShift(0), opticalFlow(nullptr)
{
}

void OpticalFlow::computeOpticalFlow(const cv::cuda::GpuMat& frame)
{
    isFlowValid = false;

    cv::cuda::GpuMat frameGray;
    if (frame.channels() == 3)
    {
        cv::cuda::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    }
    else if (frame.channels() == 1)
    {
        frameGray = frame;
    }
    else
    {
        return;
    }

    preprocessFrame(frameGray);

    static cv::cuda::GpuMat prevStaticCheck;
    static float staticThreshold = config.staticFrameThreshold;

    if (!prevStaticCheck.empty() && frameGray.size() == prevStaticCheck.size())
    {
        cv::cuda::GpuMat diffFrame;
        cv::cuda::absdiff(frameGray, prevStaticCheck, diffFrame);

        cv::Mat diffCPU;
        diffFrame.download(diffCPU);

        float meanDiff = cv::mean(diffCPU)[0];

        if (meanDiff < staticThreshold)
        {
            flow.release();
            prevFrameGray.release();
            return;
        }
    }

    prevStaticCheck = frameGray.clone();

    if (!prevFrameGray.empty() && prevFrameGray.size() != frameGray.size())
    {
        prevFrameGray.release();
    }

    if (!prevFrameGray.empty())
    {
        if (!opticalFlow)
        {
            cv::Size imageSize(frameGray.cols, frameGray.rows);
            auto perfPreset = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_SLOW;
            auto outputGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
            auto hintGridSize = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_4;
            bool enableTemporalHints = true;
            bool enableExternalHints = false;
            bool enableCostBuffer = false;
            int gpuId = 0;

            opticalFlow = cv::cuda::NvidiaOpticalFlow_2_0::create(
                imageSize,
                perfPreset,
                outputGridSize,
                hintGridSize,
                enableTemporalHints,
                enableExternalHints,
                enableCostBuffer,
                gpuId
            );

            outputGridSizeValue = 4;
            hintGridSizeValue = 4;

            flowWidth = (imageSize.width + outputGridSizeValue - 1) / outputGridSizeValue;
            flowHeight = (imageSize.height + outputGridSizeValue - 1) / outputGridSizeValue;

            hintWidth = (imageSize.width + hintGridSizeValue - 1) / hintGridSizeValue;
            hintHeight = (imageSize.height + hintGridSizeValue - 1) / hintGridSizeValue;

            hintFlow.create(hintHeight, hintWidth, CV_16SC2);
            hintFlow.setTo(cv::Scalar::all(0));
        }

        try
        {
            opticalFlow->calc(
                prevFrameGray,
                frameGray,
                flow,
                cv::cuda::Stream::Null(),
                hintFlow
            );
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "Optical Flow Error: " << e.what() << std::endl;
            return;
        }

        isFlowValid = true;
        cv::Mat flowCpu;
        flow.download(flowCpu);

        int width = flowCpu.cols;
        int height = flowCpu.rows;

        double magnitudeScale = width > height ? width / 640.0 : height / 640.0;
        double dynamicThreshold = config.optical_flow_magnitudeThreshold * magnitudeScale;

        double sumAngularVelocityX = 0.0;
        double sumAngularVelocityY = 0.0;
        int validPointsAngular = 0;

        double sumFlowX = 0.0;
        double sumFlowY = 0.0;
        int validPointsFlow = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Point2f flowAtPoint = flowCpu.at<cv::Point2f>(y, x);

                if (std::isfinite(flowAtPoint.x) && std::isfinite(flowAtPoint.y) &&
                    std::abs(flowAtPoint.x) < width &&
                    std::abs(flowAtPoint.y) < height)
                {
                    double flowMagnitude = cv::norm(flowAtPoint);

                    if (flowMagnitude > dynamicThreshold)
                    {
                        double normalizedX = flowAtPoint.x / width;
                        double normalizedY = flowAtPoint.y / height;
                        double angularVelocityX = normalizedX * config.fovX;
                        double angularVelocityY = normalizedY * config.fovY;

                        sumAngularVelocityX += angularVelocityX;
                        sumAngularVelocityY += angularVelocityY;
                        validPointsAngular++;

                        sumFlowX += flowAtPoint.x;
                        sumFlowY += flowAtPoint.y;
                        validPointsFlow++;
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(flowMutex);
            double currentTime = cv::getTickCount() / cv::getTickFrequency();
            double deltaTime = currentTime - prevTime;

            if (validPointsAngular > 0)
            {
                double newAngularVelocityX = sumAngularVelocityX / validPointsAngular;
                double newAngularVelocityY = sumAngularVelocityY / validPointsAngular;

                prevAngularVelocityX = 0.7 * prevAngularVelocityX + 0.3 * newAngularVelocityX;
                prevAngularVelocityY = 0.7 * prevAngularVelocityY + 0.3 * newAngularVelocityY;

                prevAngularVelocityX = std::clamp(prevAngularVelocityX, -10.0, 10.0);
                prevAngularVelocityY = std::clamp(prevAngularVelocityY, -10.0, 10.0);
            }

            if (validPointsFlow > 0)
            {
                double newFlowX = sumFlowX / validPointsFlow;
                double newFlowY = sumFlowY / validPointsFlow;

                prevPixelFlowX = 0.7 * prevPixelFlowX + 0.3 * newFlowX;
                prevPixelFlowY = 0.7 * prevPixelFlowY + 0.3 * newFlowY;

                double maxFlow = 100.0;
                prevPixelFlowX = std::clamp(prevPixelFlowX, -maxFlow, maxFlow);
                prevPixelFlowY = std::clamp(prevPixelFlowY, -maxFlow, maxFlow);
            }

            prevTime = currentTime;
        }
    }

    prevFrameGray = frameGray.clone();
}

void OpticalFlow::getAngularVelocity(double& angularVelocityXOut, double& angularVelocityYOut)
{
    std::lock_guard<std::mutex> lock(flowMutex);
    angularVelocityXOut = prevAngularVelocityX;
    angularVelocityYOut = prevAngularVelocityY;
}

void OpticalFlow::getAngularAcceleration(double& angularAccelerationXOut, double& angularAccelerationYOut)
{
    std::lock_guard<std::mutex> lock(flowMutex);
    angularAccelerationXOut = angularAccelerationX;
    angularAccelerationYOut = angularAccelerationY;
}

std::pair<double, double> OpticalFlow::getAverageGlobalFlow()
{
    std::lock_guard<std::mutex> lock(flowMutex);
    return { prevPixelFlowX, prevPixelFlowY };
}

void OpticalFlow::getMotion(int& xShiftOut, int& yShiftOut)
{
    std::lock_guard<std::mutex> lock(flowMutex);
    xShiftOut = xShift;
    yShiftOut = yShift;
}

bool OpticalFlow::isOpticalFlowValid() const
{
    return isFlowValid;
}

void OpticalFlow::startOpticalFlowThread()
{
    if (opticalFlowThread.joinable())
    {
        stopOpticalFlowThread();
    }

    shouldExit = false;
    opticalFlowThread = std::thread(&OpticalFlow::opticalFlowLoop, this);
}

void OpticalFlow::manageOpticalFlowThread()
{
    if (config.enable_optical_flow && !opticalFlowThread.joinable())
    {
        startOpticalFlowThread();
    }
    else if (!config.enable_optical_flow && opticalFlowThread.joinable())
    {
        stopOpticalFlowThread();
    }
}

void OpticalFlow::stopOpticalFlowThread()
{
    if (opticalFlowThread.joinable())
    {
        shouldExit = true;
        frameCV.notify_all();
        opticalFlowThread.join();
    }
}

void OpticalFlow::opticalFlowLoop()
{
    while (!shouldExit)
    {
        if (config.enable_optical_flow)
        {
            cv::cuda::GpuMat frame;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [&]() { return !frameQueue.empty() || shouldExit; });
                if (shouldExit) break;
                frame = frameQueue.front();
                frameQueue.pop();
            }
            computeOpticalFlow(frame);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void OpticalFlow::enqueueFrame(const cv::cuda::GpuMat& frame)
{
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        frameQueue.push(frame.clone());
    }

    frameCV.notify_one();
}