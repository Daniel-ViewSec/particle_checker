#ifndef DATANUC_H
#define DATANUC_H
#include <opencv2/core/mat.hpp>
#include <qtypes.h>
#include <qstring.h>
#include <QDateTime>

class DataNUC {
public:
    int temperature = 0;
    QString savePath;

    std::vector<int16_t> permuteNUC = std::vector<int16_t>();

    QVector<cv::Mat> captureHighTempOpenData = QVector<cv::Mat>();
    QVector<cv::Mat> captureHighTempCloseData = QVector<cv::Mat>();
    QVector<cv::Mat> captureLowTempOpenData = QVector<cv::Mat>();
    QVector<cv::Mat> captureLowTempCloseData = QVector<cv::Mat>();
};
#endif // DATANUC_H
