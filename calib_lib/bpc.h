#ifndef BPC_H
#define BPC_H

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QImage>
#include <QPainter>
#include <QImageReader>
#include <QFileDialog>
#include <QPushButton>
#include <QVBoxLayout>
#include <QObject>
#include <opencv2/opencv.hpp>
#include <QWidget>

class BPC : public QObject {
    Q_OBJECT

public:
    cv::Mat image, orgImg;

    void set_zoomedX(int newzoomedX);
    int get_zoomedX();

    void set_zoomedY(int newzoomedY);
    int get_zoomedY();

    void set_height(int newheight);
    int get_height();

    void set_width(int newwidth);
    int get_width();

    void set_factor(int newfactor);// 設定放大倍率
    int get_factor();

    QVector<QVector<uint8_t>> get_bedpoint();
    void set_bedpoint(QList<QPoint> &pointlist);
    void ini_badpixel();
    void update_badpoixel(QVector<uint8_t> &initMask);
    void clear_badpixel();

    QImage cvMatToQImage(const cv::Mat &mat);
    cv::Mat QImageToCvMat(const QImage &image);
    QVector<uint8_t> convertMatrixToDecimal(const QVector<QVector<uint8_t>>& matrix);
    QVector<QVector<uint8_t>> convertDecimalToMatrix(const QVector<uint8_t>& decimalValues, int columns);

    // write and read data from file
    void read_bpc_bin();
    void read_bpc_bin_by_path(std::string path, std::string file_name);
    void write_bpc_bin(const std::string& filename);

signals:
    void update_bpc(QVector<uint8_t> &updateMask);

private:
    int factor = 50;
    int zoomedX = 0, zoomedY = 0;
    int height,width;
    QVector<QVector<uint8_t>> badPixel;
    int badPixelSize =80*480;
};

#endif // BPC_H
