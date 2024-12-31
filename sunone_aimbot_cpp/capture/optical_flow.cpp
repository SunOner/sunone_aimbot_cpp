#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include "optical_flow.h"
#include "sunone_aimbot_cpp.h"

void OpticalFlow::preprocessFrame(cv::cuda::GpuMat& frameGray)
{
    cv::Mat frameCPU;
    frameGray.download(frameCPU);
    cv::GaussianBlur(frameCPU, frameCPU, cv::Size(3, 3), 0);
    cv::equalizeHist(frameCPU, frameCPU);
    cv::threshold(frameCPU, frameCPU, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    frameGray.upload(frameCPU);
}

OpticalFlow::OpticalFlow() : xShift(0), yShift(0), opticalFlow(nullptr)
{
}

void OpticalFlow::computeOpticalFlow(const cv::cuda::GpuMat& frame)
{
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

        opticalFlow->calc(
            prevFrameGray,
            frameGray,
            flow,
            cv::cuda::Stream::Null(),
            hintFlow
        );

        cv::Mat flowCpu;
        flow.download(flowCpu);

        cv::Mat flowFloat;
        flowCpu.convertTo(flowFloat, CV_32FC2, config.optical_flow_alpha_cpu);

        cv::Mat magnitude;
        cv::Mat flowChannels[2];
        cv::split(flowFloat, flowChannels);
        cv::magnitude(flowChannels[0], flowChannels[1], magnitude);

        double magThreshold = config.optical_flow_magnitudeThreshold;
        cv::Mat validMask;
        cv::threshold(magnitude, validMask, magThreshold, 1.0, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(validMask, validMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(validMask, validMask, cv::MORPH_CLOSE, kernel);

        validMask.convertTo(validMask, CV_8UC1);

        cv::Mat flowXFiltered, flowYFiltered;
        cv::medianBlur(flowChannels[0], flowXFiltered, 3);
        cv::medianBlur(flowChannels[1], flowYFiltered, 3);

        cv::Scalar avgFlowX = cv::mean(flowXFiltered, validMask);
        cv::Scalar avgFlowY = cv::mean(flowYFiltered, validMask);

        {
            std::lock_guard<std::mutex> lock(flowMutex);
            xShift = static_cast<int>(avgFlowX[0]);
            yShift = static_cast<int>(avgFlowY[0]);
        }
    }

    prevFrameGray = frameGray.clone();
}

void OpticalFlow::getMotion(int& xShiftOut, int& yShiftOut)
{
    std::lock_guard<std::mutex> lock(flowMutex);
    xShiftOut = xShift;
    yShiftOut = yShift;
}

void OpticalFlow::drawOpticalFlow(cv::Mat& frame)
{
    if (flow.empty())
        return;

    cv::Mat flowCpu;
    flow.download(flowCpu);

    cv::Mat flowFloat;
    flowCpu.convertTo(flowFloat, CV_32FC2, config.optical_flow_alpha_cpu);

    cv::Mat flowChannels[2];
    cv::split(flowFloat, flowChannels);

    cv::Mat magnitude;
    cv::magnitude(flowChannels[0], flowChannels[1], magnitude);

    float scaleX = static_cast<float>(frame.cols) / flowFloat.cols;
    float scaleY = static_cast<float>(frame.rows) / flowFloat.rows;

    int step = config.draw_optical_flow_steps;
    double magThreshold = config.optical_flow_magnitudeThreshold;

    for (int y = 0; y < flowFloat.rows; y += step)
    {
        for (int x = 0; x < flowFloat.cols; x += step)
        {
            float mag = magnitude.at<float>(y, x);
            if (mag > magThreshold)
            {
                const cv::Point2f& fxy = flowFloat.at<cv::Point2f>(y, x);

                cv::Point2f pt1(x * scaleX, y * scaleY);
                cv::Point2f pt2 = pt1 + cv::Point2f(fxy.x * scaleX, fxy.y * scaleY);

                cv::line(frame, pt1, pt2, cv::Scalar(0, 223, 255), 1);
                cv::circle(frame, pt1, 1, cv::Scalar(0, 223, 255), -1);
            }
        }
    }

    int centerX = frame.cols / 2;
    int centerY = frame.rows / 2;

    cv::Point center(centerX, centerY);
    cv::Point shiftedCenter(centerX + xShift, centerY + yShift);

    cv::line(frame, center, shiftedCenter, cv::Scalar(0, 0, 255), 2);
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