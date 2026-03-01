#ifdef USE_CUDA

#include "depth_utils.h"

namespace depth_anything
{
    std::tuple<cv::Mat, int, int> resize_depth(const cv::Mat& img, int w, int h)
    {
        cv::Mat result;
        int nw;
        int nh;
        float aspectRatio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

        if (aspectRatio >= 1.0f)
        {
            nw = w;
            nh = static_cast<int>(h / aspectRatio);
        }
        else
        {
            nw = static_cast<int>(w * aspectRatio);
            nh = h;
        }

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(nw, nh));

        result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
        cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat out(h, w, CV_8UC3, 0.0);
        cv::Mat re;
        cv::resize(rgb, re, out.size(), 0, 0, cv::INTER_LINEAR);
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

        return std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);
    }
}

#endif