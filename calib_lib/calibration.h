#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <qobject.h>
#include <vector>
#include <opencv2/opencv.hpp>

class ImageCalibration : public QObject {
    Q_OBJECT
public:
    ImageCalibration(int width = 640, int height = 480);

    void initialize(int newWidth, int newHeight) {
        width = newWidth;
        height = newHeight;
    }

    cv::Mat calibrate(const cv::Mat &frame,
                      const std::vector<cv::Point> &badPixelCoords,
                      const cv::Mat &nucParams,
                      const cv::Mat &shutterData);

private:
    cv::Mat correctBadPixels(cv::Mat &image, const std::vector<cv::Point> &badPixelCoords);
    void applyPlateauCut(std::vector<int> &histogram, int minValue, int maxValue);
    std::vector<double> calculateCDF(const std::vector<int> &histogram);
    void mapTo8bit(const std::vector<int> &inputData, cv::Mat &outputImage, const std::vector<double> &cdf);

    int width, height;
};

#endif // CALIBRATION_H
