#include "calibration.h"

ImageCalibration::ImageCalibration(int width, int height) : width(width), height(height) {}

cv::Mat ImageCalibration::calibrate(const cv::Mat &frame,
                                    const std::vector<cv::Point> &badPixelCoords,
                                    const cv::Mat &nucParams,
                                    const cv::Mat &shutterData) {

    if (frame.empty() || nucParams.empty() || shutterData.empty()) {
        throw std::runtime_error("Input image or parameters are empty!");
    }

    cv::Mat X = frame - shutterData;
    int meanShutter = cv::mean(shutterData)[0];

    cv::Mat nucRaw(height * width, 3, CV_64F);
    for (int i = 0; i < height * width; ++i) {
        nucRaw.at<double>(i, 0) = X.at<uchar>(i);
        nucRaw.at<double>(i, 1) = meanShutter;
        nucRaw.at<double>(i, 2) = 1.0;
    }

    cv::Mat nucParamsConverted;
    nucParams.convertTo(nucParamsConverted, CV_64F, 1.0 / (1 << 11));

    cv::Mat nucResult = nucRaw.mul(nucParamsConverted);
    cv::reduce(nucResult, nucResult, 1, cv::REDUCE_SUM, CV_32S);
    nucResult = cv::max(nucResult, 0);
    nucResult = cv::min(nucResult, 16383);

    cv::Mat reshapedNucResult = nucResult.reshape(1, height);
    cv::Mat bpcResult = correctBadPixels(reshapedNucResult, badPixelCoords);

    std::vector<int> histogram(16384, 0);
    int maxValue = 0, minValue = 16383;

    for (int i = 0; i < bpcResult.total(); ++i) {
        int val = bpcResult.at<int>(i);
        histogram[val]++;
        if (val > maxValue) maxValue = val;
        if (val < minValue) minValue = val;
    }

    applyPlateauCut(histogram, minValue, maxValue);

    std::vector<double> cdf = calculateCDF(histogram);

    cv::Mat outputImage(height, width, CV_8U);
    std::vector<int> inputData(bpcResult.begin<int>(), bpcResult.end<int>());
    mapTo8bit(inputData, outputImage, cdf);

    return outputImage;
}

cv::Mat ImageCalibration::correctBadPixels(cv::Mat &image, const std::vector<cv::Point> &badPixelCoords) {
    for (const auto &p : badPixelCoords) {
        int x = p.x, y = p.y;

        if (x == 0 && y == image.rows - 1) {
            image.at<int>(y, x) = image.at<int>(y - 1, x);
        }
        else if (x == 0) {
            image.at<int>(y, x) = image.at<int>(y + 1, x);
        }
        else {
            image.at<int>(y, x) = image.at<int>(y, x - 1);
        }
    }
    return image;
}

void ImageCalibration::applyPlateauCut(std::vector<int> &histogram, int minValue, int maxValue) {
    int excessCount = 0;
    int diff = maxValue - minValue;
    int plateau = (width * height) / 256;

    for (int &h : histogram) {
        if (h > plateau) {
            excessCount += h - plateau;
            h = plateau;
        }
    }

    int redistributionAmount = excessCount / diff;
    for (int i = minValue; i <= maxValue; ++i) {
        histogram[i] += redistributionAmount;
    }
}

std::vector<double> ImageCalibration::calculateCDF(const std::vector<int> &histogram) {
    std::vector<double> cdf(histogram.size(), 0);
    cdf[0] = histogram[0];

    for (size_t i = 1; i < histogram.size(); ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    double normFactor = cdf.back();
    for (double &c : cdf) {
        c /= normFactor;
    }
    return cdf;
}

void ImageCalibration::mapTo8bit(const std::vector<int> &inputData, cv::Mat &outputImage, const std::vector<double> &cdf) {
    for (size_t i = 0; i < inputData.size(); ++i) {
        int value = static_cast<int>(cdf[inputData[i]] * 255);
        outputImage.at<uchar>(i) = std::min(value, 255);
    }
}
