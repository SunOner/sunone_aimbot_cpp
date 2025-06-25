#include "capture_utils.h"

cv::Mat apply_circle_mask(const cv::Mat& input)
{
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
    cv::circle(
        mask,
        { mask.cols / 2, mask.rows / 2 },
        std::min(mask.cols, mask.rows) / 2,
        cv::Scalar(255), -1
    );

    cv::Mat output;
    input.copyTo(output, mask);
    return output;
}

#include "capture_utils.h"

void apply_circle_mask_inplace(cv::Mat& image)
{
    static cv::Mat mask;
    static cv::Size last_size = {0, 0};

    if (mask.empty() || image.size() != last_size)
    {
        last_size = image.size();
        mask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::circle(
            mask,
            { mask.cols / 2, mask.rows / 2 },
            std::min(mask.cols, mask.rows) / 2,
            cv::Scalar(255), 
            -1
        );
    }

    image.setTo(cv::Scalar(0, 0, 0), ~mask);
}