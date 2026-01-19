#include "cameraimagewidget.h"
#include "opencv2/imgproc/types_c.h"
#include <QTimer>
#include <QPainter>
#include <QQueue>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/dnn_superres.hpp"
#include "../videoInput/videoInput.h"
#define MIN_DIFF 256
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define DAMPING_VALUE 600
#define THERMAL_MAX_VALUE 16383
#define MAPPING_MAX 240
#define MAPPING_MIN 10
#define MAPPING_GAP 20 // 0~100%
#define HISTOGRAM_STEP 4

cameraImageWidget::cameraImageWidget(QWidget *parent) : QWidget(parent) {
    setMouseTracking(true);
    timer = new QTimer(this);
    fpsUpdateTimer = new QTimer(this);
    connect(fpsUpdateTimer, &QTimer::timeout, this, [this](){
        fpsUpdate = true;
    });
    avgQueue = new QQueue<double>();

}

void cameraImageWidget::initStreaming(QString cameraName, int newFormat) {
    // Initialize videoInput
    videoInput VI;
    int cameraNum = 0;

    // List devices
    int numDevices = VI.listDevices();
    // qDebug()<< "Found " << numDevices << " camera(s).";

    for (int i = 0; i < numDevices; i++) {
        std::string name = VI.getDeviceName(i);
        // qDebug() << "Device " << i << ": " << name;
        if(cameraName.toStdString() == name)cameraNum = i;
    }

    // open camera
    captureVideo.open(cameraNum, cv::CAP_DSHOW);
    // captureVideo.open(cameraIndex, cv::CAP_DSHOW);
    // qDebug() << captureVideo.get(cv::VideoCaptureProperties::);
    if (captureVideo.isOpened() == false){
        qDebug() << "Camera can't open";
        return;

    }
    qDebug() << "camera name from cv:" << captureVideo.getBackendName();

    captureVideo.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // check source format
    if (newFormat == 0) {
        captureVideo.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        captureVideo.set(cv::CAP_PROP_CONVERT_RGB, 1);
        captureVideo.set(cv::CAP_PROP_FORMAT, CV_8UC3);
        qDebug() << "Switched to MJPG\n";
    } else if(newFormat == 1 ) {
        captureVideo.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', '1', '6', ' '));
        captureVideo.set(cv::CAP_PROP_CONVERT_RGB, 0);
        captureVideo.set(cv::CAP_PROP_FORMAT, CV_16UC1);
        qDebug() << "Switched to Y16\n";
    } else if(newFormat == 2 ) {
        captureVideo.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
        captureVideo.set(cv::CAP_PROP_CONVERT_RGB, 1);
        captureVideo.set(cv::CAP_PROP_FORMAT, CV_8UC3);
        qDebug() << "Switched to YUV\n";
    } else if(newFormat == 3 ) {
        // captureVideo.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('A', 'V', 'C', '1'));
        // captureVideo.set(cv::CAP_PROP_CONVERT_RGB, 1);
        // // captureVideo.set(cv::CAP_PROP_FORMAT, CV_8UC3);
        qDebug() << "Switched to U264\n";
    } else {
        qDebug() << "Unrecognize format\n";
    }

    // init componet
    mapping_table = std::vector<float>(THERMAL_MAX_VALUE + 1, 0);
    mapping_table_d = std::vector<float>(THERMAL_MAX_VALUE + 1, 0);
    pre_max_val = 0;
    pre_min_val = 0;
    damped_max_val = 0;
    damped_min_val = 0;
    last_min_val = 0;
    last_max_val = THERMAL_MAX_VALUE;
    avg_min_val = 0;
    avg_max_val = THERMAL_MAX_VALUE;
    histogramDisplay = std::vector<int32_t>(THERMAL_MAX_VALUE + 1, 0);
    lut = new cv::Mat();
    currentFormat = newFormat;
    initCamera = true;
    connect(timer, SIGNAL(timeout()), this, SLOT(update()));
    lastUpdate = QTime::currentTime();
    initSRmodel();
}

void cameraImageWidget::startStreaming() {
    if(!initCamera) {
        qDebug() << "Camera didn't init";
        return;
    }
    qDebug() << "Starting Streaming";
    timer->start(1);
    fpsUpdateTimer->start(1000);
}

void cameraImageWidget::stopStreaming() {
    captureVideo.release();
    qDebug() << "Stop Streaming";
    disconnect(timer, SIGNAL(timeout()), this, SLOT(update()));
    timer->stop();
}

void cameraImageWidget::paintEvent(QPaintEvent *) {
    // Read data
    cv::Mat tmpImage;
    cv::Mat gray;
    cv::Mat rgb;
    cv::Mat highByte, lowByte;
    captureVideo.read(tmpImage);

    if (tmpImage.empty() == true){
        qDebug() << "EMPTY!";
        return;
    }

    // FPS lock
    // if (frameCount > 61) return;

    QImage::Format colorMode = QImage::Format_BGR888;
    histogramDisplay.clear();

    // read image for different format
    if(currentFormat == 0 || currentFormat == 2 || currentFormat == 3){
        // qDebug() << "Mat type:" << tmpImage.type();
        if (tmpImage.type() != CV_8UC3) {
            qDebug() << "Image is not CV_8UC3!";
            return;
        }
        rgb = tmpImage.clone();
        cv::cvtColor(tmpImage, gray, cv::COLOR_BGR2GRAY);
    } else if(currentFormat == 1 ) {

        uint16_t max_val = 0;
        uint16_t min_val = 0xFFFF;
        if (tmpImage.type() != CV_16UC1) {
            qDebug() << "Image is not CV_16UC1!";
            return;
        }

        if (tmpImage.cols != IMAGE_WIDTH && tmpImage.rows != IMAGE_HEIGHT) {
            qDebug() << "Image with wrong size!";
            return;
        }
        // cv::medianBlur(tmpImage, tmpImage, 3);
        // histogram
        std::vector<uint16_t> inputFrame(tmpImage.begin<uint16_t>(), tmpImage.end<uint16_t>());
        max_val_image = 0;
        min_val_image = THERMAL_MAX_VALUE;
        // if(agcEnable) {
        //     histogram_cal_test(inputFrame, IMAGE_WIDTH, IMAGE_HEIGHT, &max_val_image, &min_val_image);
        //     std::vector<uint16_t> histogramForm(histogram.size());
        //     std::transform(histogram.begin(), histogram.end(),
        //                    histogramForm.begin(),
        //                    [](int32_t value) { return static_cast<uint16_t>(value); });
        //     histogramDisplay = histogramForm;

        //     applyPlateauCutTest(histogram, min_val_image, max_val_image);
        // } else {
        //     histogram_cal(inputFrame, IMAGE_WIDTH, IMAGE_HEIGHT, &max_val_image, &min_val_image);
        //     std::vector<uint16_t> histogramForm(histogram.size());
        //     std::transform(histogram.begin(), histogram.end(),
        //                    histogramForm.begin(),
        //                    [](int32_t value) { return static_cast<uint16_t>(value); });
        //     histogramDisplay = histogramForm;

        //     applyPlateauCut(histogram, min_val_image, max_val_image);
        // }
        histogram_cal(inputFrame, IMAGE_WIDTH, IMAGE_HEIGHT, &max_val_image, &min_val_image);
        // std::vector<uint16_t> histogramForm(histogram.size());
        // std::transform(histogram.begin(), histogram.end(),
        //                histogramForm.begin(),
        //                [](int32_t value) { return static_cast<uint16_t>(value); });
        histogramDisplay = histogram;

        applyPlateauCut(histogram, min_val_image, max_val_image);
        // gat high image
        highByte = tmpImage / 64;
        highByte.convertTo(highByte, CV_8U);

        // gat low image
        lowByte =  tmpImage & 0xFF;
        lowByte.convertTo(lowByte,CV_8U);

        // avg
        if(fpsUpdate) {
            cv::Scalar gray_mean_value = cv::mean(tmpImage);
            // qDebug() << "image avg:" << gray_mean_value[0];
            avg = gray_mean_value[0];
            avgQueue->append(avg);
            while (avgQueue->size() > 30) {
                avgQueue->pop_front();
            }
            deltaAvg = avgQueue->last() - avgQueue->first();
        }

        // remapping merge
        if(agcEnable) {
            // og AGC
            std::vector<uint8_t> output_image(IMAGE_WIDTH * IMAGE_HEIGHT);
            mapTo8bit(inputFrame, output_image);
            cv::Mat outputMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, output_image.data());
            tmpImage = outputMat.clone();

            // // Convert to float [0,1]
            // cv::Mat img_norm;
            // outputMat.convertTo(img_norm, CV_32F, 1.0 / 255.0);

            // // Avoid log(0) by adding small epsilon
            // const float epsilon = 1e-6f;
            // cv::Mat img_log;
            // cv::log(img_norm + epsilon, img_log);  // log(x)

            // // Equivalent to np.log1p: log(1+x)
            // img_norm += 1.0f;
            // cv::log(img_norm, img_log);
            // // Normalize back to [0,1]
            // cv::normalize(img_log, img_log, 0.0, 1.0, cv::NORM_MINMAX);

            // // Scale to 8-bit
            // cv::Mat img_mapped;
            // img_log.convertTo(img_mapped, CV_8U, 255.0);
            // tmpImage = img_mapped.clone();

        } else {
            //// OG
            remapping16To8(tmpImage);
            //// G-AGC
            // std::vector<uint8_t> output_image(IMAGE_WIDTH * IMAGE_HEIGHT);
            // mapTo8bitTest(inputFrame, output_image);
            // cv::Mat outputMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, output_image.data());
            // tmpImage = outputMat.clone();
        }

        // // test kmean
        // // Flatten image into a 1D vector (float32 for KMeans)
        // cv::Mat samples;
        // tmpImage.convertTo(samples, CV_32F);
        // samples = samples.reshape(1, 640 * 480).clone();
        // // Apply KMeans (3 clusters)
        // int K = 3;
        // cv::Mat labels;
        // cv::Mat centers;
        // cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2);
        // cv::kmeans(samples, K, labels, criteria, 10, cv::KMEANS_RANDOM_CENTERS, centers);

        // // Sort cluster centers (cold → hot)
        // std::vector<std::pair<float, int>> sorted;
        // for (int i = 0; i < centers.rows; i++) {
        //     sorted.push_back({centers.at<float>(i, 0), i});
        // }
        // std::sort(sorted.begin(), sorted.end());

        // // Define colors for cold→mid→hot (BGR)
        // std::vector<cv::Vec3b> clusterColors(K);
        // clusterColors[sorted[0].second] = cv::Vec3b(255, 0, 0);   // Blue (cold)
        // clusterColors[sorted[1].second] = cv::Vec3b(0, 255, 0);   // Green (mid)
        // clusterColors[sorted[2].second] = cv::Vec3b(0, 0, 255);   // Red (hot)

        // // Rebuild segmented image with colors
        // cv::Mat segmented(tmpImage.size(), CV_8UC3);
        // for (int i = 0; i < tmpImage.total(); i++) {
        //     int cluster_id = labels.at<int>(i);
        //     segmented.at<cv::Vec3b>(i) = clusterColors[cluster_id];
        // }

        // // Convert original to 8-bit for display
        // cv::Mat img8u;
        // double minVal, maxVal;
        // cv::minMaxLoc(tmpImage, &minVal, &maxVal);
        // tmpImage.convertTo(img8u, CV_8U, 255.0 / maxVal);

        // // Show results
        // cv::imshow("Original Thermal (8-bit scaled)", img8u);
        // cv::imshow("K-means Segmentation (Cold/Medium/Hot)", segmented);

        // // test -end kmean

        // Radiometry get data
        if(enableTC) {
            min_temp = (min_val_image - 2732) / 10.0;
            max_temp = (max_val_image - 2732) / 10.0;
            center_temp = (inputFrame[240*640 + 320] - 2732) / 10.0;
        }

        // select preview image
        switch(perviewImage) {
        default:
        case 0:
            gray = tmpImage.clone();
            break;
        case 1:
            gray = highByte.clone();
            break;
        case 2:
            gray = lowByte.clone();
            break;
        }
        cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
    } else {
        qDebug() << "Missing type source!!";
        return;
    }

    // up data histogram signal to UI
    emit updateHistogram(histogramDisplay, plateau, max , min, premax, premin);

    // out image histogram
    std::vector<uint16_t> outFrame(gray.begin<uint8_t>(), gray.end<uint8_t>());
    std::vector<int32_t> outHistogramForm(MIN_DIFF);
    for (int x = 0; x < outFrame.size(); ++x) {
        uint16_t pixel_value = outFrame[x];
        if(pixel_value < outHistogramForm.size())
            outHistogramForm[pixel_value]++;
    }
    emit updateOutHistogram(outHistogramForm);


    QImage img = cvMatToQImage(rgb);

    QPixmap pixmap = QPixmap::fromImage(img);
    QPainter painter(this);

    float comprimento = 1.0*width()/pixmap.width();
    float altura = 1.0*height()/pixmap.height();
    float ratio = 0.;

    if (comprimento <= altura)
        ratio = comprimento;
    else
        ratio = altura;

    QSize size = ratio*pixmap.size();
    size.setHeight(size.height()-10);

    QPoint p;
    p.setX(0 + (width()-size.width())/2);
    p.setY(5);
    painter.drawPixmap(QRect(p, size), pixmap.scaled(size, Qt::KeepAspectRatio));

    // init
    QString text = "FPS:" + QString::number(fps);
    QRect textRect;
    QFont font("Arial", 10);
    QFontMetrics fm(font);
    int textWidth = 0, textHeight = 0, x = 0, y = 0;
    painter.setFont(font);

    // draw fps
    if(fpsEnable) {
        textWidth = fm.horizontalAdvance(text);
        textHeight = fm.height();
        x = 10; // to left
        y = 20; // to buttom
        textRect = fm.boundingRect(x, y - textHeight, textWidth, textHeight, Qt::TextSingleLine, text);
        painter.fillRect(textRect, QColor(0,0,0));
        painter.setPen(Qt::white);
        painter.drawText(x, y, text);
    }

    // fps
    if(fpsUpdate) {
        fpsUpdate = false;
        fps = frameCount;
        frameCount = 0;
    }
    frameCount++;

    //radiometry
    if(enableTC && currentFormat == 1) {

        // min
        text = "min:" + QString::number(min_temp, 'f', 2) + "℃";
        textWidth = fm.horizontalAdvance(text);
        textHeight = fm.height();
        x = 10; // to left
        y = (IMAGE_HEIGHT - textHeight); // to buttom
        textRect = fm.boundingRect(x, y - textHeight, textWidth, textHeight, Qt::TextSingleLine, text);
        painter.fillRect(textRect, QColor(0,0,0));
        painter.setPen(Qt::white);
        painter.drawText(x, y, text);

        // max
        text = "max:" + QString::number(max_temp, 'f', 2) + "℃";
        textWidth = fm.horizontalAdvance(text);
        textHeight = fm.height();
        x = (IMAGE_WIDTH - textWidth) / 2;  // to center
        y = (IMAGE_HEIGHT - textHeight); // to buttom
        textRect = fm.boundingRect(x, y - textHeight, textWidth, textHeight, Qt::TextSingleLine, text);
        painter.fillRect(textRect, QColor(0,0,0));
        painter.setPen(Qt::white);
        painter.drawText(x, y, text);

        // center
        text = "center:" + QString::number(center_temp, 'f', 2) + "℃";
        textWidth = fm.horizontalAdvance(text);
        textHeight = fm.height();
        x = (IMAGE_WIDTH - textWidth); // to right
        y = (IMAGE_HEIGHT - textHeight); // to buttom
        textRect = fm.boundingRect(x, y - textHeight, textWidth, textHeight, Qt::TextSingleLine, text);
        painter.fillRect(textRect, QColor(0,0,0));
        painter.setPen(Qt::white);
        painter.drawText(x, y, text);
    }

    // super resolution
    if(enableSR) {
        // Deploy to crop
        cv::Mat cropImg = cropImageCenter(rgb, *cropCenterSize);
        QImage cimg((const unsigned char*)(cropImg.data), cropImg.cols, cropImg.rows, QImage::Format_BGR888);
        QPixmap cpixmap = QPixmap::fromImage(cimg);

        cv::Mat resizeImg;
        if(initModel) {
            resizeImg = superResolution(cropImg);
        } else {
            cv::resize(cropImg, resizeImg, cv::Size(cropImg.cols*4, cropImg.rows*4));
            // painter.drawPoint(5,5);
        }

        if(!resizeImg.empty()) {
            QImage reImg((const unsigned char*)(resizeImg.data), resizeImg.cols, resizeImg.rows, QImage::Format_BGR888);

            QPixmap repixmap = QPixmap::fromImage(reImg);

            int centerX = img.size().width() / 2;
            int centerY = img.size().height() / 2;
            QPoint reP(
                p.x() + (centerX - reImg.width()/2),
                p.y() + (centerY - reImg.height()/2)
                );
            // QPoint reP(0,5);

            painter.drawPixmap(QRect(reP, reImg.size()), repixmap);
            //// show scale verison
            // cv::Mat cresizeImg;
            // cv::resize(cropImg, cresizeImg, cv::Size(cropImg.cols*4, cropImg.rows*4));
            // QImage creImg((const unsigned char*)(cresizeImg.data), cresizeImg.cols, cresizeImg.rows, QImage::Format_BGR888);

            // QPixmap crepixmap = QPixmap::fromImage(creImg);

            // QPoint cp(0, 5);
            // painter.drawPixmap(QRect(cp, creImg.size()), crepixmap);
        }
    }

}

void cameraImageWidget::setFilterImage(cv::Mat &input) {
    if(!input.empty() && input.cols != 0) {
        cv::Size scaleSize(256, 1);
        lut = new cv::Mat(scaleSize, CV_8UC3);
        cv::Mat inputScales = cv::Mat(1, input.rows * input.cols, CV_8UC3, input.data);
        cv::resize(inputScales, *lut, scaleSize);
        qDebug() << "read filter image successful";
        input.release();
    } else {
        lut->release();
        qDebug() << "can't open filter image!!";
    }
}

void cameraImageWidget::initSRmodel() {
    cropCenterSize = new QSize(40,30);
    try {
        sr = new cv::dnn_superres::DnnSuperResImpl();
        sr->readModel("./models/FSRCNN_x4.pb");
        sr->setModel("fsrcnn",4);
        initModel = true;
        // qDebug() << "load model fsrcnn";
    } catch (...) {
        qDebug() << "can't init model";
    }
}

cv::Mat cameraImageWidget::cropImageCenter(cv::Mat &input, QSize cropSize) {
    // crop image center
    cv::Mat croppedImage;
    int ogWidth = input.cols;
    int ogHeight = input.rows;
    if(cropSize.width() > ogWidth || cropSize.height() > ogHeight)
        return croppedImage;

    int centerX = ogWidth / 2;
    int centerY = ogHeight / 2;

    cv::Rect cropROI(
        centerX,
        centerY,
        cropSize.width(),
        cropSize.height()
        );

    croppedImage = input(cropROI).clone();

    return croppedImage;
}

cv::Mat cameraImageWidget::superResolution(cv::Mat &input) {
    cv::Mat srImage;
    if(initModel) {
        sr->upsample(input, srImage);
    } else {
        srImage = input.clone();
    }
    return srImage;
}

void cameraImageWidget::setSRenable(bool enable) {
    enableSR = enable;

}

void cameraImageWidget::setTCEnable(bool enable) {
    enableTC = enable;

}

bool cameraImageWidget::getTCenable() {
    return enableTC;

}

void cameraImageWidget::setFpsEnable(bool enable) {
    fpsEnable = enable;

}

bool cameraImageWidget::getFpsEnable() {
    return fpsEnable;

}

void cameraImageWidget::setAgcEnable(bool enable) {
    agcEnable = enable;

}

bool cameraImageWidget::getAgcEnable() {
    return agcEnable;

}

std::vector<int32_t> cameraImageWidget::getHistogram() {
    return histogramDisplay;
}

void cameraImageWidget::updateHistograManual() {    // Read data
    cv::Mat tmpImage;
    captureVideo.read(tmpImage);

    if (tmpImage.empty() == true){
        qDebug() << "EMPTY!";
        return;
    }
    histogramDisplay.clear();

    if(currentFormat == 1){
        uint16_t max_val = 0;
        uint16_t min_val = 0xFFFF;
        if (tmpImage.type() != CV_16UC1) {
            qDebug() << "Image is not CV_16UC1!";
            return;
        }

        if (tmpImage.cols != IMAGE_WIDTH && tmpImage.rows != IMAGE_HEIGHT) {
            qDebug() << "Image with wrong size!";
            return;
        }

        // histogram
        std::vector<uint16_t> inputFrame(tmpImage.begin<uint16_t>(), tmpImage.end<uint16_t>());
        max_val_image = 0;
        min_val_image = THERMAL_MAX_VALUE;
        histogram_cal(inputFrame, IMAGE_WIDTH, IMAGE_HEIGHT, &max_val_image, &min_val_image);
        // std::vector<uint16_t> histogramForm(histogram.size());
        // std::transform(histogram.begin(), histogram.end(),
        //                histogramForm.begin(),
        //                [](int32_t value) { return static_cast<uint16_t>(value); });
        histogramDisplay = histogram;
    } else {
        return;
    }
    // up data histogram signal to UI
    emit updateHistogram(histogramDisplay, 0, 0, 0, 0, 0);
}

cv::Mat cameraImageWidget::getCaptureImage() {
    cv::Mat tmpImage;
    captureVideo.read(tmpImage);
    // read image for different format
    if(currentFormat == 0 || currentFormat == 2 || currentFormat == 3){
        return tmpImage.clone();
    } else {
        cv::Mat highByte, lowByte, gray;

        uint16_t max_val = 0;
        uint16_t min_val = 0xFFFF;
        if (tmpImage.type() != CV_16UC1) {
            qDebug() << "Image is not CV_16UC1!";
            return tmpImage;
        }

        // histogram
        std::vector<uint16_t> inputFrame(tmpImage.begin<uint16_t>(), tmpImage.end<uint16_t>());
        max_val_image = 0;
        min_val_image = THERMAL_MAX_VALUE;
        histogram_cal(inputFrame, IMAGE_WIDTH, IMAGE_HEIGHT, &max_val_image, &min_val_image);
        std::vector<uint16_t> histogramForm(histogram.size());
        std::transform(histogram.begin(), histogram.end(),
                       histogramForm.begin(),
                       [](int32_t value) { return static_cast<uint16_t>(value); });
        applyPlateauCut(histogram, min_val_image, max_val_image);

        // gat high image
        highByte = tmpImage / 64;
        highByte.convertTo(highByte, CV_8U);

        // gat low image
        lowByte =  tmpImage & 0xFF;
        lowByte.convertTo(lowByte,CV_8U);

        // remapping merge
        if(agcEnable) {
            std::vector<uint8_t> output_image(IMAGE_WIDTH * IMAGE_HEIGHT);
            mapTo8bit(inputFrame, output_image);
            cv::Mat outputMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, output_image.data());
            tmpImage = outputMat.clone();
        }

        // select preview image
        switch(perviewImage) {
        default:
        case 0:
            gray = tmpImage.clone();
            break;
        case 1:
            gray = highByte.clone();
            break;
        case 2:
            gray = lowByte.clone();
            break;
        }
        return gray.clone();

    }
}

cv::Mat cameraImageWidget::getCaptureImageNUC() {
    cv::Mat tmpImage;
    captureVideo.read(tmpImage);
    tmpImage.convertTo(tmpImage, CV_16S);
    return tmpImage.clone();
}

void cameraImageWidget::dropFrame(int number){
    cv::Mat tmpImage;
    for(int var = 0; var < number; var ++){
        captureVideo.read(tmpImage);
    }
}

void cameraImageWidget::selectPerviewImage(int mode) {
    perviewImage = mode;
}

void cameraImageWidget::setInitState() {
    initCamera = false;
}

QList<QString> cameraImageWidget::getCameraList() {

    // Initialize videoInput
    videoInput VI;
    int cameraNum = 0;
    QList<QString> cameraList;

    // List devices
    int numDevices = VI.listDevices();
    // qDebug()<< "Found " << numDevices << " camera(s).";

    for (int i = 0; i < numDevices; i++) {
        std::string name = VI.getDeviceName(i);
        cameraList.append(QString::fromStdString(name));
    }
    return cameraList;
}

// utility
QImage cameraImageWidget::cvMatToQImage(const cv::Mat &mat) {
    if (mat.empty()) {
        return QImage();
    }

    cv::Mat rgba;
    if (mat.channels() == 4) {
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows, rgba.step, QImage::Format_ARGB32).copy();
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, rgba, cv::COLOR_BGR2RGB);
        return QImage(rgba.data, rgba.cols, rgba.rows, rgba.step, QImage::Format_RGB888).copy();
    } else if (mat.channels() == 1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    } else {
        qWarning("cvMatToQImage: Unsupported number of channels");
        return QImage();
    }
}

void cameraImageWidget::remapping16To8(cv::Mat &mat){
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat grayFloat;
    mat.convertTo(grayFloat, CV_32F);
    cv::Mat remappedFloat = (grayFloat - minVal) / (maxVal - minVal); // normalize to 0~1
    remappedFloat *= 255.0;                                            // scale to 0~255

    cv::Mat remapped;
    remappedFloat.convertTo(remapped, CV_8U);
    mat = remapped;
}


// AGC
void cameraImageWidget::applyPlateauCut(std::vector<int32_t>& histogram, int& min_value, int& max_value) {
    double excess_count = 0.0;
    int diff = max_value - min_value;
    int min_value_orig = min_value;
    int max_value_orig = max_value;
    float damping_ratio = 1.0 * DAMPING_VALUE / 1000;

    if (diff < MIN_DIFF) {
        int adjust = (MIN_DIFF - diff + 1) / 2;
        if (min_value - adjust < 0) {
            max_value += (MIN_DIFF - diff - min_value);
            min_value = 0;
        } else if (max_value + adjust > THERMAL_MAX_VALUE) {
            min_value -= (MIN_DIFF - diff - (THERMAL_MAX_VALUE - max_value));
            max_value = THERMAL_MAX_VALUE;
        } else {
            max_value += adjust;
            min_value -= adjust;
        }
        diff = MIN_DIFF;
    }

    plateau = IMAGE_HEIGHT * IMAGE_WIDTH / diff / HISTOGRAM_STEP / HISTOGRAM_STEP;


    for (int i = min_value_orig; i <= max_value_orig; i++)
    {
        if (histogram[i] > plateau )
        {
            excess_count += histogram[i] - plateau;
            histogram[i] = plateau;
        }
    }

    // new
    unsigned int redistribution_amount = excess_count / diff;
    // hist_total += diff * redistribution_amount;
    int cumulative_sum = 0;
    for (unsigned int i = min_value; i <= max_value; i++)
    {
        cumulative_sum += histogram[i] + redistribution_amount; // CDF

        float new_value = (cumulative_sum * 230) / (IMAGE_HEIGHT * IMAGE_WIDTH / HISTOGRAM_STEP / HISTOGRAM_STEP) + 10;
        if (
            (pre_min_val == 0 && pre_max_val == 0) ||
            (pre_max_val - pre_min_val) < 256 ||
            abs(min_value_orig - pre_min_val) > 0.2 * (pre_max_val - pre_min_val) ||
            abs(max_value_orig - pre_max_val) > 0.2 * (pre_max_val - pre_min_val) ||
            update_shutter == 1)
        {
            mapping_table_d[i] = new_value;
        }
        else
        {
            mapping_table_d[i] = mapping_table_d[i] * (1.0 - damping_ratio) + new_value * damping_ratio;
        }
        mapping_table[i] = (mapping_table_d[i] + 0.5);
        if (mapping_table[i] > 240)
        {
            mapping_table_d[i] = 240;
            mapping_table[i] = 240;
        }
    }
    pre_min_val = min_value_orig;
    pre_max_val = max_value_orig;
    update_shutter = 0;

    // check data
    if (min_value > 0) {
        for (int i = 0; i < min_value; i++) {
            mapping_table[i] = 0;
        }
    }
    if (max_value < THERMAL_MAX_VALUE) {
        for (int i = max_value; i < THERMAL_MAX_VALUE; i++) {
            mapping_table[i] = 240;
        }
    }
}

void cameraImageWidget::mapTo8bit(const std::vector<uint16_t>& input_data, std::vector<uint8_t>& output_image) {
    size_t size = input_data.size();
    output_image.resize(size);

    for (size_t i = 0; i < size; ++i) {
        int val = 0;
        if(input_data[i] >= THERMAL_MAX_VALUE) {
            val = THERMAL_MAX_VALUE;
        } else if(input_data[i] <= 0) {
            val = 0;
        } else {
            val = input_data[i];
        }
        uint16_t pixel = mapping_table[val];
        output_image[i] = std::min((int)mapping_table[val], 255);
    }
}

void cameraImageWidget::histogram_cal(std::vector<uint16_t>& input_data, int width, int height, int *max_val, int *min_val) {

    int32_t ori_max = *max_val;
    int32_t ori_min = *min_val;
    histogram = std::vector<int32_t>(THERMAL_MAX_VALUE + 1, 0);

    for (int32_t i = 0; i < height; i += HISTOGRAM_STEP)
    {
        for (int32_t j = 0; j < width; j += HISTOGRAM_STEP)
        {
            int index = i * width + j;
            uint16_t pixel = input_data[index];
            if(pixel > THERMAL_MAX_VALUE) continue;
            histogram[pixel]++;
            ori_min = (pixel < ori_min) ? pixel : ori_min;
            ori_max = (pixel > ori_max) ? pixel : ori_max;
        }
    }

    int32_t total_pixel = (height / HISTOGRAM_STEP) * (width / HISTOGRAM_STEP);
    int32_t percentile = total_pixel * 5 / 1000;   // 0.3%

    int32_t accum = 0;
    // unsigned int min_val_temp = 0, max_val_temp = 16383;
    for (int32_t i = 0; i < THERMAL_MAX_VALUE + 1; ++i)
    {
        accum += histogram[i];
        if (accum >= percentile)
        {
            *min_val = i;
            break;
        }
    }
    accum = 0;
    for (int i = THERMAL_MAX_VALUE; i >= 0; --i)
    {
        accum += histogram[i];
        if (accum >= percentile)
        {
            *max_val = i;
            break;
        }
    }

    const int PERCENTILE_DEADBAND = 8;
    int src_min_val = *min_val;
    int src_max_val = *max_val;

    if (abs((int)*min_val - (int)last_min_val) < PERCENTILE_DEADBAND)
        src_min_val = last_min_val;
    else
        last_min_val = *min_val;

    if (abs((int)*max_val - (int)last_max_val) < PERCENTILE_DEADBAND)
        src_max_val = last_max_val;
    else
        last_max_val = *max_val;

    const float hist_damping = 0.25f;
    const float MAX_JUMP_RATIO = 0.2f;
    float prev_range = damped_max_val - damped_min_val;

    if (damped_min_val == 0 && damped_max_val == 0)
    {
        damped_min_val = src_min_val;
        damped_max_val = src_max_val;
    }
    else if (prev_range < 1 ||
             fabsf((float)src_min_val - damped_min_val) > MAX_JUMP_RATIO * prev_range ||
             fabsf((float)src_max_val - damped_max_val) > MAX_JUMP_RATIO * prev_range)
    {
        damped_min_val = src_min_val;
        damped_max_val = src_max_val;
    }
    else
    {
        damped_min_val = damped_min_val * (1.0f - hist_damping) + src_min_val * hist_damping;
        damped_max_val = damped_max_val * (1.0f - hist_damping) + src_max_val * hist_damping;
    }

    *min_val = (int)damped_min_val;
    *max_val = (int)damped_max_val;
}


// NEW AGC
void cameraImageWidget::applyPlateauCutTest(std::vector<int32_t>& histogram, int& min_value, int& max_value) {
    double excess_count = 0.0;
    int diff = max_value - min_value;
    int min_value_orig = min_value;
    int max_value_orig = max_value;
    float damping_ratio = 1.0 * DAMPING_VALUE / 1000;

    // if (diff < MIN_DIFF) {
    //     int adjust = (MIN_DIFF - diff + 1) / 2;
    //     if (min_value - adjust < 0) {
    //         max_value += (MIN_DIFF - diff - min_value);
    //         min_value = 0;
    //     } else if (max_value + adjust > THERMAL_MAX_VALUE) {
    //         min_value -= (MIN_DIFF - diff - (THERMAL_MAX_VALUE - max_value));
    //         max_value = THERMAL_MAX_VALUE;
    //     } else {
    //         max_value += adjust;
    //         min_value -= adjust;
    //     }
    //     diff = MIN_DIFF;
    // }


    // new
    // hist_total += diff * redistribution_amount;
    int zoon1Total = 0;
    int zoon2Total = 0;
    int median = 0;
    int total = 0;
    plateau = 307200.0 / (diff*0.5 ) / 16;
    for (unsigned int i = min_value; i <= max_value; i++)
    {
        // find median
        total+= histogram[i];
        if(total > 307200 /16/2){
            median = i;
            total = -307200;
        }

        // do cut above plateau
        if (histogram[i] > plateau)
        {
            excess_count += histogram[i] - plateau;
            histogram[i] = plateau;
        }
    }


    int cumulative_sum = 0;
    int zoom1_sum = 0;
    int zoom2_sum = 0;
    int range = abs(MAPPING_MAX - MAPPING_MIN);
    int new_range = 0;
    if(abs(median - avg_min_val) < abs(median - avg_max_val))
        new_range = abs(median - avg_min_val);
    else
        new_range = abs(median - avg_max_val);
    int cut_min = median - new_range;
    int cut_max = median + new_range;
    new_range == 0 ? new_range = 1 : new_range;

    for (unsigned int i = min_value; i <= max_value; i++)
    {
        zoon1Total += histogram[i];
        if(i <= cut_min || i >= cut_max) {
            zoon2Total += histogram[i];
            // zoon1Total++;
        }
        // zoon2Total += histogram[i];

        //     if (histogram[i] > plateau)
        //     {
        //         excess_count += histogram[i] - plateau;
        //         // histogram[i] = plateau;
        //     }
        // }
    }
    unsigned int redistribution_amount = excess_count / diff;
    int total_in_mappingtable = 0;
    for (unsigned int i = min_value; i <= max_value; i++)
    {
        // cumulative_sum += histogram[i] + redistribution_amount; // CDF

        // float new_value = 0;

        float gap = median - (int)i;
        float value_assemble = 0;
        float number_effective = ((float)zoon1Total / 256.0 / (float)zoon2Total);
        float different_effective = (float)diff / 256.0;
        value_assemble =  gap / new_range * number_effective * different_effective;
        float new_value = 0;
        if(i <= cut_min || i >= cut_max) {
            new_value = histogram[i] + redistribution_amount + value_assemble;
        } else {
            // new_value = ((cumulative_sum) * 230) / (307200 / 16) + 10;
            new_value = histogram[i] + redistribution_amount;
        }
        if(i > cut_max) {
            qDebug() << "histogram: " << histogram[i] << " | redistribution:" << redistribution_amount << " | value_assemble:" << value_assemble;
        }
        // if(i <= cut_min || i >= cut_max) {
        //     zoom1_sum += histogram[i];
        //     new_value = (zoom1_sum  * range) / zoon1Total;
        //     // zoom1_sum++;
        // } else {
        //     zoom2_sum += plateau + redistribution_amount;
        //     new_value = (zoom2_sum * range) / (zoon2Total);
        // }
        // if (
        //     (pre_min_val == 0 && pre_max_val == 0) ||
        //     (pre_max_val - pre_min_val) < 256 ||
        //     abs(min_value_orig - pre_min_val) > 0.2 * (pre_max_val - pre_min_val) ||
        //     abs(max_value_orig - pre_max_val) > 0.2 * (pre_max_val - pre_min_val) ||
        //     update_shutter == 1)
        // {
        //     mapping_table_d[i] = new_value;
        // }
        // else
        // {
        //     mapping_table_d[i] = mapping_table_d[i] * (1.0 - damping_ratio) + new_value * damping_ratio;
        // }
        // mapping_table[i] = (mapping_table_d[i] + 0.5);

        // if(i <= cut_min || i >= cut_max) {
        //     mapping_table[i] = MAPPING_MIN + new_value + value_assemble;
        // } else {
        //     mapping_table[i] = MAPPING_MIN + new_value;
        // }
        mapping_table[i] = new_value;

        // if (mapping_table[i] > MAPPING_MAX)
        // {
        //     mapping_table[i] = MAPPING_MAX;
        // }
        // if (mapping_table[i] < MAPPING_MIN)
        // {
        //     mapping_table[i] = MAPPING_MIN;
        // }
        total_in_mappingtable += mapping_table[i];
    }

    int sum_mapping_table = 0;
    for (unsigned int i = min_value; i <= max_value; i++)
    {
        sum_mapping_table += mapping_table[i];
        mapping_table[i] = (sum_mapping_table * 230 / total_in_mappingtable) + 10;
    }
    premax = cut_max;
    premin = cut_min;
    pre_min_val = min_value_orig;
    pre_max_val = max_value_orig;
    update_shutter = 0;

    // check data
    if (min_value > 0) {
        for (int i = 0; i < min_value; i++) {
            mapping_table[i] = MAPPING_MIN;
        }
    }
    if (max_value < THERMAL_MAX_VALUE) {
        for (int i = max_value; i < THERMAL_MAX_VALUE; i++) {
            mapping_table[i] = MAPPING_MAX;
        }
    }
}

void cameraImageWidget::histogram_cal_test(std::vector<uint16_t>& input_data, int width, int height, int *max_val, int *min_val) {

    int32_t ori_max = 0;
    int32_t ori_min = THERMAL_MAX_VALUE;
    int32_t real_max = 0;
    int32_t real_min = THERMAL_MAX_VALUE;
    int32_t sum_max = 0;
    int32_t sum_min = 0;
    // int32_t avg_max = 0;
    // int32_t avg_min = 0;
    // int32_t horizon_max =
    histogram = std::vector<int32_t>(THERMAL_MAX_VALUE + 1, 0);

    for (int32_t i = 0; i < height; i += HISTOGRAM_STEP)
    {
        ori_max = 0;
        ori_min = THERMAL_MAX_VALUE;
        for (int32_t j = 0; j < width; j += HISTOGRAM_STEP)
        {
            int index = i * width + j;
            uint16_t pixel = input_data[index];
            if(pixel > THERMAL_MAX_VALUE) continue;
            histogram[pixel]++;
            ori_min = (pixel < ori_min) ? pixel : ori_min;
            ori_max = (pixel > ori_max) ? pixel : ori_max;
            real_min = (pixel < real_min) ? pixel : real_min;
            real_max = (pixel > real_max) ? pixel : real_max;
        }
        sum_max += ori_max;
        sum_min += ori_min;
    }
    avg_min_val = sum_min / (height / HISTOGRAM_STEP);
    avg_max_val = sum_max / (height / HISTOGRAM_STEP);
    int count = 0;
    int threadhold = 0.01 * 640 * 480;

    int32_t total_pixel = (height / HISTOGRAM_STEP) * (width / HISTOGRAM_STEP);
    int32_t percentile = total_pixel * 5 / 1000;   // 0.3%
    // int32_t percentile = 1;

    int32_t accum = 0;
    // unsigned int min_val_temp = 0, max_val_temp = 16383;
    for (int32_t i = 0; i < THERMAL_MAX_VALUE + 1; ++i)
    {
        accum += histogram[i];
        if (accum >= percentile)
        {
            *min_val = i;
            break;
        }
    }
    accum = 0;
    for (int i = THERMAL_MAX_VALUE; i >= 0; --i)
    {
        accum += histogram[i];
        if (accum >= percentile)
        {
            *max_val = i;
            break;
        }
    }

    const int PERCENTILE_DEADBAND = 8;
    // int src_min_val = *min_val;
    // int src_max_val = *max_val;
    int src_min_val = real_min;
    int src_max_val = real_max;

    // if (abs((int)*min_val - (int)last_min_val) < PERCENTILE_DEADBAND)
    //     src_min_val = last_min_val;
    // else
    //     last_min_val = *min_val;

    // if (abs((int)*max_val - (int)last_max_val) < PERCENTILE_DEADBAND)
    //     src_max_val = last_max_val;
    // else
    //     last_max_val = *max_val;

    const float hist_damping = 0.25f;
    const float MAX_JUMP_RATIO = 0.2f;
    float prev_range = damped_max_val - damped_min_val;

    // if (damped_min_val == 0 && damped_max_val == 0)
    // {
    //     damped_min_val = src_min_val;
    //     damped_max_val = src_max_val;
    // }
    // else if (prev_range < 1 ||
    //          fabsf((float)src_min_val - damped_min_val) > MAX_JUMP_RATIO * prev_range ||
    //          fabsf((float)src_max_val - damped_max_val) > MAX_JUMP_RATIO * prev_range)
    // {
    //     damped_min_val = src_min_val;
    //     damped_max_val = src_max_val;
    // }
    // else
    // {
    //     damped_min_val = damped_min_val * (1.0f - hist_damping) + src_min_val * hist_damping;
    //     damped_max_val = damped_max_val * (1.0f - hist_damping) + src_max_val * hist_damping;
    // }

    damped_max_val = src_max_val;
    damped_min_val = src_min_val;
    max = src_max_val;
    min = src_min_val;

    *min_val = (int)damped_min_val;
    *max_val = (int)damped_max_val;
}

void cameraImageWidget::mapTo8bitTest(const std::vector<uint16_t>& input_data, std::vector<uint8_t>& output_image) {
    size_t size = input_data.size();
    output_image.resize(size);

    for (size_t i = 0; i < size; ++i) {
        int val = 0;
        if(input_data[i] >= THERMAL_MAX_VALUE) {
            val = THERMAL_MAX_VALUE;
        } else if(input_data[i] <= 0) {
            val = 0;
        } else {
            val = input_data[i];
        }
        output_image[i] = std::min((int)mapping_table[val], MAPPING_MAX);
        output_image[i] = (int)mapping_table[val];
    }


    // display mapping table
    int histSize = mapping_table.size(); // 256 bins for intensity range 0-255

    // Normalize to fit into display image
    int hist_w = 512, hist_h = 400;
    double bin_w = (double)hist_w / mapping_table.size();
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    int max_i = max_val_image, min_i = min_val_image;


    // Draw the lines
    // for (int i = min_val_image; i < max_val_image; i++) {
    //     if(mapping_table[i] >= MAPPING_MAX) {
    //         max_i = i;
    //         break;
    //     }
    // }
    // for (int i = max_i; i > -1; i--) {
    //     if(mapping_table[i] <= 0) {
    //         min_i = i;
    //         break;
    //     }
    // }
    double bin_wn = (double)hist_w / abs(max_i - min_i);
    for (int i = min_i; i <= max_i; i++) {
        cv::line(histImage,
                 cv::Point((int)(bin_wn * (i - min_i)), hist_h - mapping_table[i]),
                 cv::Point((int)(bin_wn * (i - min_i)), hist_h - mapping_table[i]),
                 cv::Scalar(255), 2, 8, 0);
    }
    // cv::imshow("mapping_table", histImage);
    // qDebug() << "Max_index:" << max_i << "min_index:" << min_i;
}
