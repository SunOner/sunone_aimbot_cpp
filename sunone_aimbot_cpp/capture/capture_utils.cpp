#include "capture_utils.h"

cv::Mat apply_circle_mask(const cv::Mat& input)
{
    static cv::Mat mask;
    static cv::Size cachedSize(0, 0);

    if (mask.empty() || cachedSize != input.size())
    {
        mask = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::circle(
            mask,
            { mask.cols / 2, mask.rows / 2 },
            std::min(mask.cols, mask.rows) / 2,
            cv::Scalar(255), -1
        );
        cachedSize = input.size();
    }

    cv::Mat output;
    input.copyTo(output, mask);
    return output;
}
