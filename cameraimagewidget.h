#ifndef CAMERAIMAGEWIDGET_H
#define CAMERAIMAGEWIDGET_H

#include <QWidget>
#include <QLabel>
#include <qdatetime.h>
#include "opencv2/core/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/dnn_superres.hpp"

class cameraImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit cameraImageWidget(QWidget *parent = nullptr);
    bool initCamera = false;
    double avg = 0, deltaAvg = 0;

private:
    QPoint mPoint;
    QTimer *timer;
    cv::Mat *currentIamge;
    cv::VideoCapture captureVideo;
    // cv::Mat filterImage;
    std::vector<uchar> *filterTable;
    cv::Mat *lut;
    bool initModel = false, srEnable = false, enableSR = false, fpsEnable = true, agcEnable = false, fpsUpdate = false, enableTC = false;
    QSize *cropCenterSize;
    cv::dnn_superres::DnnSuperResImpl *sr;
    int currentFormat = 0; // 0 = MJPG, 1 = Y16, 2 = U264, 3 = R264, 4 = U422
    int perviewImage = 0;// 0 =og, 1 = Y16 high byte, 2 = Y16 low byte
    bool dataCaptured = false;
    std::vector<int32_t> histogramDisplay;
    int frameCount = 0, fps = 0;
    QTimer *fpsUpdateTimer;
    QTime lastUpdate;
    // AGC
    std::vector<int32_t> histogram;
    std::vector<float> mapping_table, mapping_table_d;
    int max_val_image = 0, min_val_image = 0, update_shutter = 0;
    int pre_max_val = 0, pre_min_val = 0;
    int damped_max_val = 0, damped_min_val = 0;
    int last_min_val = 0, last_max_val = 16383;
    int avg_min_val = 16383, avg_max_val = 0;
    QQueue<double> *avgQueue;

    //Radiometry
    double min_temp = 0, max_temp = 0, center_temp = 0;

    //Test - AGC
    double plateau, max, min, premax, premin;

public slots:
    void paintEvent(QPaintEvent * event);
    void initStreaming(QString cameraName, int newFormat);
    void startStreaming();
    void stopStreaming();
    // void changeSourceMode(int cameraNum);
    void setFilterImage(cv::Mat &input);
    void initSRmodel();
    void setSRenable(bool enable);
    void setTCEnable(bool enable);
    bool getTCenable();
    void setFpsEnable(bool enable);
    bool getFpsEnable();
    void setAgcEnable(bool enable);
    bool getAgcEnable();
    std::vector<int32_t> getHistogram();
    void updateHistograManual();
    cv::Mat cropImageCenter(cv::Mat &input, QSize cropSize);
    cv::Mat superResolution(cv::Mat &input);
    cv::Mat getCaptureImage();
    cv::Mat getCaptureImageNUC();
    void dropFrame(int number);
    void selectPerviewImage(int mode);
    void setInitState();
    QList<QString> getCameraList();

    // utility
    QImage cvMatToQImage(const cv::Mat &mat);
    void remapping16To8(cv::Mat &mat);

    // AGC
    void applyPlateauCut(std::vector<int32_t>& histogram, int& min_value, int& max_value);
    void histogram_cal(std::vector<uint16_t>& input_data, int width, int height, int *max_val, int *min_val);
    void mapTo8bit(const std::vector<uint16_t>& input_data, std::vector<uint8_t>& output_image);

    // NEW AGC
    void applyPlateauCutTest(std::vector<int32_t>& histogram, int& min_value, int& max_value);
    void histogram_cal_test(std::vector<uint16_t>& input_data, int width, int height, int *max_val, int *min_val);
    void mapTo8bitTest(const std::vector<uint16_t>& input_data, std::vector<uint8_t>& output_image);


signals:
    void updateHistogram(std::vector<int32_t> histogram ,double plateau , double max, double min, double premax, double premin);
    void updateOutHistogram(std::vector<int32_t> histogram);

};

#endif // CAMERAIMAGEWIDGET_H
