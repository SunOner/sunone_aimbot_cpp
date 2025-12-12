#include "capture_utils.h"

cv::Mat apply_circle_mask(const cv::Mat& input)
{
    // OPTIMIZED: Cache mask to avoid allocation every frame
    static cv::Mat cached_mask;
    static cv::Size cached_size;

    if (cached_size != input.size())
    {
        cached_mask = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::circle(
            cached_mask,
            { cached_mask.cols / 2, cached_mask.rows / 2 },
            std::min(cached_mask.cols, cached_mask.rows) / 2,
            cv::Scalar(255), -1
        );
        cached_size = input.size();
    }

    cv::Mat output;
    input.copyTo(output, cached_mask);
    return output;
}
