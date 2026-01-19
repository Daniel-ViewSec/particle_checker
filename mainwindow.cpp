#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "CalibrationStep.h"
#include "DataNUC.h"
#include <QFileInfo>
#include <QDir>
#include <QQueue>
#include <QTimer>
#include <QCameraDevice>
#include <QMessageBox>
#include <QStandardItemModel>
#include <QMediaDevices>

QStandardItemModel cameraModle;
QStandardItemModel serialModle;
QString connectCamera, connectSerial;
QCameraDevice cameraDevice;
MainWindow::SourceVideoMode currentSourceVideo = MainWindow::SourceVideoMode::U016;

// state machine
CalibrationStep currentCalibraMode;
bool isFpsEnable = true;

// capture
QVector<cv::Mat>* captureOpenData;
QVector<cv::Mat>* captureCloseData;
DataNUC currentDataNUC;
cv::Mat  shutterDataInput;
cv::Mat captureOpenDataBPC;
cv::Mat captureCloseDataBPC;
CalibrationStep captureSaveMode;
int CAP_UNIT_SIZE = 5;
int CAP_DROP_SIZE = 10;
int captureMode = -1; // -1 - idel, 1 - NUC_data, 2 - NUC_shutter, 3 - BPC, 4 - Radiometry
int captureOnePointCount = 0;
bool captureRawSetting = false , captureShutterSetting = false;
int captureIntervalSetting = 0, captureRoundSetting = 0, captureUnitNumberSetting = 0;
bool autoCalibraStateTemp = false;


// calibration - NUC
constexpr size_t TARGET_LENGTH = 0x1E0000;
constexpr size_t SHUTTERLESS_TARGET_LENGTH = 0x280000;
QByteArray dataBufferNUC;
int dataSizeNUC =4*960*480;
QVector<cv::Mat> decodeData;
std::vector<int16_t> permuteNUC;
QString styleUncalculate = "border-radius: 12px;border:2px solid #737574;";
QString styleCalculated = "border-radius: 12px;border:2px solid #2ec96f;";

// NUC Folder structure definition
int selectHighLow = -1;
QMap<QString, QList<QString>> gainFolders = {
    { "high_gain", { "-30", "0" } },
    { "normal_gain", { "0", "40" } },
    { "low_gain",  { "40", "70" } },
    };
QStringList subFolders = { "30_close", "30_open", "45_close", "45_open" };
CalibrationStep saveCalibraMode;
bool isOnePoint = true;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    uiUpdater = new QTimer(this);
    sendEventTriggerTimer = new QTimer(this);
    nuccla = new NUC_Data();
    bpc_parameter = new BPC();

    // backgound
    connect(sendEventTriggerTimer, &QTimer::timeout, this, &MainWindow::triggerCheckSend);
    connect(uiUpdater, &QTimer::timeout, this , &MainWindow::updateUI);

    // botton
    connect(ui->btnScan, &QPushButton::clicked, this , &MainWindow::scanDevice);
    connect(ui->btnConnect, &QPushButton::clicked, this, &MainWindow::startSerial);
    connect(ui->btnCapture, &QPushButton::clicked, this , [this](){
        if(currentStep == Step::Capture_30){
            selectHighLow = 1;
            triggerCaptureNUC();
        } else if (currentStep == Step::Capture_45) {
            selectHighLow = 0;
            triggerCaptureNUC();
        } else {
            qDebug() << "trigger capture in wrong state";
        }
    });
    connect(ui->btnFinish, &QPushButton::clicked, this, &MainWindow::reset);

    // nuc bpc
    connect(nuccla, &NUC_Data::send_nuc, this, &MainWindow::setDataNUC);
    connect(bpc_parameter, &BPC::update_bpc, this, [this](QVector<uint8_t> &updateMask){
        qDebug() << "load BPC success";
        maskTempBPC.clear();
        maskTempBPC = updateMask;
    });

    // init
    ui->lvSerial->setModel(&serialModle);
    deviceState = new ThermalDeviceState();
    deviceUpdate = new ThermalDeviceState();
    eventRead = new QQueue<SerialEvent>;
    eventSend = new QQueue<SerialEvent>;
    currentStep = Step::Select_COM_Port;
    captureRawSetting = true;
    captureShutterSetting = true;
    captureIntervalSetting = 5;
    captureRoundSetting = 3;
    captureUnitNumberSetting = 5;

    // timer
    sendEventTriggerTimer->start(500);
    uiUpdater->start(100);

    // state
    currentCalibraMode = CalibrationStep::OP_NUC_1;

    // folder
    BPCPath = QDir::currentPath() + "/" + BPC_FOLDER;
    QDir rootDir;

    // Create data save folder
    if (!rootDir.mkpath(BPCPath)) {
        qDebug() << "Failed to create root folder:" << BPCPath;
        QMessageBox msgBox;
        msgBox.setText("Failed to create root folder, plz check again.");
        msgBox.exec();
        return;
    }
    // test();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateUI() {
    switch (currentStep) {
    case Step::Select_COM_Port:
        ui->captureWidget->setHidden(true);
        ui->calculateWidget->setHidden(true);
        ui->resultWidget->setHidden(true);
        ui->connectingWidget->setHidden(true);

        ui->connectionWidget->setHidden(false);
        break;
    case Step::Connectting:
        ui->captureWidget->setHidden(true);
        ui->calculateWidget->setHidden(true);
        ui->resultWidget->setHidden(true);
        ui->connectionWidget->setHidden(true);

        ui->connectingWidget->setHidden(false);
        break;
    case Step::Capture_30:
        ui->connectionWidget->setHidden(true);
        ui->calculateWidget->setHidden(true);
        ui->resultWidget->setHidden(true);
        ui->connectingWidget->setHidden(true);
        ui->labelGuild->setText("Please capture 30°C black body");

        ui->captureWidget->setHidden(false);
        break;
    case Step::Capture_45:
        ui->connectionWidget->setHidden(true);
        ui->calculateWidget->setHidden(true);
        ui->resultWidget->setHidden(true);
        ui->connectingWidget->setHidden(true);
        ui->labelGuild->setText("Please capture 45°C black body");

        ui->captureWidget->setHidden(false);
        break;
    case Step::Calculate_NUC:
        ui->connectionWidget->setHidden(true);
        ui->captureWidget->setHidden(true);
        ui->resultWidget->setHidden(true);
        ui->connectingWidget->setHidden(true);

        ui->calculateWidget->setHidden(false);
        break;
    case Step::Finish:
        ui->connectionWidget->setHidden(true);
        ui->captureWidget->setHidden(true);
        ui->calculateWidget->setHidden(true);
        ui->connectingWidget->setHidden(true);

        ui->resultWidget->setHidden(false);

        break;
    default:
        break;
    }
}

double MainWindow::percentile(std::vector<float>& v, double p01)
{
    if (v.empty()) return 0.0;
    p01 = std::min(1.0, std::max(0.0, p01));
    size_t k = (size_t)std::llround(p01 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return (double)v[k];
}

DetectResult MainWindow::detectDefect(
    const cv::Mat& gray8,
    int border = 30,            // 忽略邊界(避免角落漸暈影響)
    double bgSigma = 45.0,      // 背景模糊尺度(越大越只剩大趨勢)
    double hiPct = 0.999,       // 亮汙點百分位
    double loPct = 0.001,       // 暗汙點百分位
    double margin = 3.0,        // 百分位門檻再加/減一點 margin
    int minArea = 20,           // 最小汙點面積(像素)
    bool debug = false,
    const std::string& debugPrefix = "",
    bool aorc = false
) {
    DetectResult out;

    CV_Assert(gray8.type() == CV_8U);
    cv::Mat gray;
    gray8.convertTo(gray, CV_32F);

    // 背景估計 + 殘差
    cv::Mat bg;
    cv::GaussianBlur(gray, bg, cv::Size(0,0), bgSigma);
    cv::Mat res = gray - bg;

    // ROI 收集殘差值做百分位
    cv::Rect roi(border, border, gray.cols - 2*border, gray.rows - 2*border);
    roi &= cv::Rect(0,0,gray.cols,gray.rows);
    std::vector<float> vals;
    vals.reserve((size_t)roi.area());

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const float* rp = res.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            vals.push_back(rp[x]);
        }
    }

    // 百分位門檻：亮/暗
    double hi = percentile(vals, hiPct);
    double lo = percentile(vals, loPct);
    out.brightThr = hi + margin;
    out.darkThr   = lo - margin;

    cv::Mat brightMask, darkMask;
    cv::compare(res, out.brightThr, brightMask, cv::CMP_GT);
    cv::compare(res, out.darkThr,   darkMask,   cv::CMP_LT);

    // 忽略邊界
    cv::Mat borderMask = cv::Mat::zeros(gray.size(), CV_8U);
    if (roi.width > 0 && roi.height > 0) borderMask(roi).setTo(255);
    cv::bitwise_and(brightMask, borderMask, brightMask);
    cv::bitwise_and(darkMask,   borderMask, darkMask);

    // 去雜訊
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::morphologyEx(brightMask, brightMask, cv::MORPH_OPEN, k3);
    cv::morphologyEx(darkMask,   darkMask,   cv::MORPH_OPEN, k3);

    // 合併亮/暗汙點
    cv::Mat mask;
    cv::bitwise_or(brightMask, darkMask, mask);

    // 連通元件
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    int count = 0;
    std::vector<cv::Rect> boxes;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= minArea) {
            ++count;
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            boxes.emplace_back(x,y,w,h);
        }
    }

    out.componentCount = count;
    out.hasDefect = (count > 0);

    if (debug) {
        cv::Mat vis;
        cv::cvtColor(gray8, vis, cv::COLOR_GRAY2BGR);
        for (auto& r : boxes) cv::rectangle(vis, r, cv::Scalar(0,0,255), 2);

        if(aorc) {
            QImage imageA = ui->cameraPerview->cvMatToQImage(vis);
            QPixmap pixmapA = QPixmap::fromImage(imageA);
            ui->result1->setPixmap(pixmapA);
        } else {
            QImage imageC = ui->cameraPerview->cvMatToQImage(vis);
            QPixmap pixmapC = QPixmap::fromImage(imageC);
            ui->result2->setPixmap(pixmapC);
        }
        cv::imshow("mask", mask);
        cv::imshow("brightMask", brightMask);
        cv::imshow("darkMask", darkMask);
        cv::imshow("vis", vis);
        cv::imwrite("./" + debugPrefix + "_mask.png", mask);
        cv::imwrite("./" + debugPrefix + "_bright.png", brightMask);
        cv::imwrite("./" + debugPrefix + "_dark.png", darkMask);
        cv::imwrite("./" + debugPrefix + "_boxes.png", vis);
    }

    return out;
}

bool MainWindow::isImageFile(QString p)
{
    QFileInfo fileInfo(p);
    if(!fileInfo.isFile()) return false;
    QString ext = fileInfo.suffix();
    return (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" || ext == "tif" || ext == "tiff");
}

void MainWindow::loadBPCFile() {
    // check have BPC data
    QDir bpcDir = QDir();
    bpcDir.setPath(BPCPath);
    QStringList files = bpcDir.entryList(QDir::Files | QDir::NoDotAndDotDot);
    bool foundFlag = false;

    for (const QString &fileName : files) {
        if(fileName.contains(QString::number(deviceState->serialNumber))) {
            foundFlag = true;
            bpc_parameter->read_bpc_bin_by_path(BPCPath.toStdString(), fileName.toStdString());
        }
    }

    if(!foundFlag) {
        qDebug() << "BPC not found!!!";
        reset();
    }

}

void MainWindow::test() {
    bool debug = true;
    QString f = "p1.png";
    int ok = 0, ng = 0;
    cv::Mat img = cv::imread(f.toStdString(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        qDebug() << "Failed to read: " << f << "\n";
    }
    cv::imshow("og", img);
    // 你可以依你的資料再調參數
    // auto r = detectDefect(
    //     img,
    //     /*border*/ 30,
    //     /*bgSigma*/ 45.0,
    //     /*hiPct*/ 0.999,
    //     /*loPct*/ 0.001,
    //     /*margin*/ 1.0,
    //     /*minArea*/ 20,
    //     /*debug*/ debug,
    //     /*debugPrefix*/ "test"
    //     );
    auto r = detectDefect(
        img,
        /*border*/ 30,
        /*bgSigma*/ 10.0,
        /*hiPct*/ 0.999,
        /*loPct*/ 0.001,
        /*margin*/ 1.0,
        /*minArea*/ 20,
        /*debug*/ debug,
        /*debugPrefix*/ "test"
        );

    qDebug() << f.toStdString()
             << " => " << (r.hasDefect ? "DEFECT" : "CLEAN")
             << " (components=" << r.componentCount
             << ", brightThr=" << r.brightThr
             << ", darkThr=" << r.darkThr
             << ")\n";

    if (r.hasDefect) ng++; else ok++;
}

void MainWindow::createFolder() {
    QString rootPath = QDir::currentPath() + "/" + SAVE_FOLDER;
    QDir rootDir;

    // Create data save folder
    if (!rootDir.mkpath(rootPath)) {
        qDebug() << "Failed to create root folder:" << rootPath;
        QMessageBox msgBox;
        msgBox.setText("Failed to create root folder, plz check again.");
        msgBox.exec();
        return;
    }

    QString devicePath = rootPath + "/" + deviceState->deviceNameSN;
    // Create device folder
    if (!rootDir.mkpath(devicePath)) {
        qDebug() << "Failed to create device folder:" << devicePath;
        QMessageBox msgBox;
        msgBox.setText("Failed to create device folder, plz check again.");
        msgBox.exec();
        return;
    }

    currentPath = QDir::currentPath() + "/" + SAVE_FOLDER + "/" + deviceState->deviceNameSN;
}

void MainWindow::reset() {
    closeSerial();
    closeCamera();
    currentPath = "";
    currentStep = Step::Select_COM_Port;

}

// Scan UVC and Serial device
void MainWindow::scanDevice() {
    // Clear list
    cameraModle.clear();
    serialModle.clear();

    qDebug() << "Scan start";

    // Scan COM port
    const auto serialPortInfos = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo &portInfo : serialPortInfos) {
        connectSerial = portInfo.portName();
        serialModle.appendRow(new QStandardItem(portInfo.portName()));
        qDebug() << "combobox value:" << portInfo.portName();
    }

    // end scan animation
    qDebug() << "Scan end";
}


// Serial connection function
void MainWindow::startSerial() {

    if(ui->lvSerial->selectionModel()->selectedIndexes().empty()) {
        QMessageBox msgBox;
        msgBox.setText("No select COM port.");
        msgBox.exec();
        qDebug() << "no select Serial";
        return;
    }
    connectSerial =  ui->lvSerial->selectionModel()->selectedIndexes().first().data().toString();
    const auto serialPortInfos = QSerialPortInfo::availablePorts();
    QSerialPortInfo targetSerialDevice;
    for (const QSerialPortInfo &portInfo : serialPortInfos) {
        qDebug() << portInfo.description();
        if(portInfo.portName() == connectSerial){
            targetSerialDevice = portInfo;
        }
    }

    qDebug() << "\n"
             << "Port:" << targetSerialDevice.portName() << "\n"
             << "Location:" << targetSerialDevice.systemLocation() << "\n"
             << "Description:" << targetSerialDevice.description() << "\n"
             << "Manufacturer:" << targetSerialDevice.manufacturer() << "\n"
             << "Serial number:" << targetSerialDevice.serialNumber() << "\n"
             << "Vendor Identifier:"
             << (targetSerialDevice.hasVendorIdentifier()
                     ? QByteArray::number(targetSerialDevice.vendorIdentifier(), 16)
                     : QByteArray()) << "\n"
             << "Product Identifier:"
             << (targetSerialDevice.hasProductIdentifier()
                     ? QByteArray::number(targetSerialDevice.productIdentifier(), 16)
                     : QByteArray());
    if(!targetSerialDevice.isNull()){
        setSerial(targetSerialDevice);
    } else {
        QMessageBox msgBox;
        msgBox.setText("No COM port can be connect.");
        msgBox.exec();
    }

}

void MainWindow::setSerial(const QSerialPortInfo &serialInfo) {
    bool serialOpenState = false;
    try {
        serial = new QSerialPort;
        serial->setPort(serialInfo);
        serial->setBaudRate(115200);
        serial->open(QIODevice::ReadWrite);
        serialOpenState = true;
    } catch (...) {
        QMessageBox msgBox;
        msgBox.setText("can't connect Serial");
        msgBox.exec();
        return;
    }
    if(serialOpenState) {
        connect(serial, &QSerialPort::readyRead, this, &MainWindow::handleReadyRead);
        connect(serial, &QSerialPort::errorOccurred, this, &MainWindow::handleError);
        // histogramNuc->show();
        // histogramNuc->raise();
        // vtempNuc->show();
        // vtempNuc->raise();
        getDeivceInfo();
        currentStep = Step::Connectting;
    }
}

void MainWindow::closeSerial(){
    try {
        if(serial != nullptr) {
            serial->close();
            serial = nullptr;
        }
    } catch (...) {
        QMessageBox msgBox;
        msgBox.setText("can't close Serial");
        msgBox.exec();
        return;
    }
}

// UVC camera function
void MainWindow::startCamera() {
    connectCamera =  "ThermalCam_" + deviceState->deviceNameSN;
    qDebug() << "try to find device: " << connectCamera;
    const QList<QCameraDevice> availableCameras = QMediaDevices::videoInputs();
    QCameraDevice targetCameraDevice;
    for (int i = 0; i < availableCameras.size(); i++) {

        if(availableCameras[i].description() == connectCamera){
            ui->cameraPerview->initStreaming(connectCamera, (int)currentSourceVideo);
            // ui->cameraPerview->initStreaming(i, (int)currentSourceVideo);
            targetCameraDevice = availableCameras[i];
        }
    }

    if(!targetCameraDevice.isNull()){
        // setCamera(targetCameraDevice);
        ui->cameraPerview->startStreaming();
    } else {
        closeCamera();
        // setCamera(QMediaDevices::defaultVideoInput());
    }
}

void MainWindow::closeCamera() {
    try {
        ui->cameraPerview->stopStreaming();
        ui->cameraPerview->setInitState();
    } catch (...) {
        QMessageBox msgBox;
        msgBox.setText("can't close Camera");
        msgBox.exec();
        return;
    }
}

void MainWindow::updateCameraSource(QString newSource) {
    SourceVideoMode newSourceFormet = stringToFormat(newSource.toStdString());
    if(currentSourceVideo != newSourceFormet) {
        closeCamera();
        qDebug() << "source change !!";
        currentSourceVideo = newSourceFormet;
    } else {
        qDebug() << "source same !!";
    }
    startCamera();
}

// Convert String to Enum
MainWindow::SourceVideoMode MainWindow::stringToFormat(const string& str) {
    auto it = sourceVideoModeFormatMap.find(str);
    return (it != sourceVideoModeFormatMap.end()) ? it->second : SourceVideoMode::UJPG;
}


// serial system
QByteArray MainWindow::buildCommand(QString command, BuildCommandMode action, quint16 value, QString asciiValue) {
    std::string prefix = "VSEC";
    std::string colon = ":";
    std::string get = "GET";
    std::string set = "SET";
    std::string suffix = "@";

    QByteArray data;
    data.append(prefix);
    data.append(colon);
    data.append(command.toStdString());
    data.append(colon);
    switch (action) {
    case BuildCommandMode::GET:
        data.append(get);
        data.append(colon);
        data.append(suffix);
        break;
    case BuildCommandMode::SET_NONE:
        data.append(set);
        data.append(colon);
        data.append(suffix);
        break;
    case BuildCommandMode::SET_UINT8:
        data.append(set);
        data.append(colon);
        data.append(static_cast<char>(value & 0xFF));        // Low byte
        data.append(suffix);
        break;
    case BuildCommandMode::SET_UINT16:
        data.append(set);
        data.append(colon);
        data.append(static_cast<char>(value & 0xFF));        // Low byte
        data.append(static_cast<char>((value >> 8) & 0xFF)); // High byte
        data.append(suffix);
        break;
    case BuildCommandMode::SET_ASCII:
        data.append(set);
        data.append(colon);
        data.append(asciiValue.toStdString());
        data.append(suffix);
        break;
    default:
        break;
    }
    // debug print
    //qDebug() << "Build command:" << data.data();
    //qDebug() << "Build command hex:" << data.toHex();
    return data;
}

void MainWindow::handleReadyRead() {
    // Append the received data to the buffer
    bool isRead16bitInt = false;
    QByteArray buffer;
    quint32 value32;
    quint16 value16;
    int unitSize;
    std::string asciiValue;

    buffer.append(serial->readAll());
    //qDebug() << "data input size:" << buffer.size();

    if(eventRead->size() > 0){

        SerialEvent readEvent = eventRead->first();
        if(buffer.size() == 4 && (QString(buffer.data()) == "FAIL" )) {
            qDebug() << "failed Event:" << QString::number((int)readEvent) <<",  buffer data:"<< QString(buffer.data()) <<", please try again. :(";
            // eventRead->clear();
        } else {
            switch (readEvent) {
            case SerialEvent::GET_SERIAL_NUMBER:

                qDebug() << "get read: GET_SERIAL_NUMBER";
                unitSize = 4;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    memcpy(&value32, unit, 4);
                    // value32 = static_cast<quint32>(unit[3]) | (static_cast<quint32>(unit[2]) << 8)| (static_cast<quint32>(unit[1]) << 16)| (static_cast<quint32>(unit[0]) << 24);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value32;

                    deviceState->serialNumber = value32;
                }
                break;
            case SerialEvent::GET_VER1:
                qDebug() << "get read: GET_VER1";
                asciiValue = buffer.data();
                qDebug() << asciiValue;
                deviceState->fwVersion = QString::fromStdString(asciiValue);
                break;
            case SerialEvent::GET_VER2:
                qDebug() << "get read: GET_VER2";
                asciiValue = buffer.data();
                qDebug() << asciiValue;
                deviceState->companyName = QString::fromStdString(asciiValue);
                break;
            case SerialEvent::GET_3DNR:
                qDebug() << "get read: GET_3DNR";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->noiseReduction = value16 > 0;
                }
                break;
            case SerialEvent::GET_NUCC:
                qDebug() << "get read: GET_NUCC";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->nucControl = value16 > 0;
                }
                break;
            case SerialEvent::GET_BPCC:
                qDebug() << "get read: GET_BPCC";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->bpcControl = value16 > 0;
                }
                break;
            case SerialEvent::GET_VTMP:
                // qDebug() << "get read: GET_VTMP";
                unitSize = 2;
                // To Do : check buffer size > event req
                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]) | (static_cast<quint8>(unit[1]) << 8);

                    deviceState->sensorVTEMP = value16;
                } else {
                    qDebug() << "drop wrong response GET_VTMP";
                }

                if(captureMode == 4) {
                    if(currentCalibraMode == CalibrationStep::CAP_2) {
                        // next step
                        if (!captureRawSetting && !captureShutterSetting)
                            currentCalibraMode = CalibrationStep::CAP_5; // skip both captures
                        else if (captureRawSetting)
                            currentCalibraMode = CalibrationStep::CAP_3; // do open capture
                        else
                            currentCalibraMode = CalibrationStep::CAP_4;
                        checkCaptureFlag();
                    }
                }
                break;
            case SerialEvent::GET_DGFD:
                qDebug() << "get read: GET_DGFD";
                unitSize = 2;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]) | (static_cast<quint8>(unit[1]) << 8);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->sensorDacGfid = value16;
                }
                break;
            case SerialEvent::GET_DGSK:
                qDebug() << "get read: GET_DGSK";
                unitSize = 2;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]) | (static_cast<quint8>(unit[1]) << 8);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->sensorDacGsk = value16;
                }
                break;
            case SerialEvent::GET_CINT:
                qDebug() << "get read: GET_CINT";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->sensorCint = value16;
                }
                break;
            case SerialEvent::GET_TINT:
                qDebug() << "get read: GET_TINT";
                unitSize = 2;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]) | (static_cast<quint8>(unit[1]) << 8);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->sensorTint = value16;
                }
                break;
            case SerialEvent::GET_DDEC:
                qDebug() << "get read: GET_DDEC";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->ddeControl = value16 > 0;
                }
                break;
            case SerialEvent::GET_VOMC:
                qDebug() << "get read: GET_VOMC";
                if( modeTempBuffer.size() + buffer.size() < 4) {
                    qDebug() << "not enough data: "<< buffer.data();
                    modeTempBuffer.setData(buffer);
                    return;
                }
                asciiValue = modeTempBuffer.data() + buffer.data();
                modeTempBuffer.buffer().clear();
                qDebug() << asciiValue;
                deviceState->sourceMode = QString::fromStdString(asciiValue);
                updateCameraSource(QString::fromStdString(asciiValue));
                break;
            case SerialEvent::GET_SUTS:
                qDebug() << "get read: GET_SUTS";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->shutterControl= value16 > 0;
                }
                break;
            case SerialEvent::GET_SUTC:
                qDebug() << "get read: GET_SUTC";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    deviceState->autoShutterControl  = value16 > 0;
                }
                break;
            case SerialEvent::GET_AGCM:
                qDebug() << "get read: GET_AGCM";
                unitSize = 2;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]) | (static_cast<quint8>(unit[1]) << 8);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    // check data in range
                    if(value16 < 2)
                        deviceState->agcMode = value16;
                    else
                        qDebug() << "get agc error, please try again. :(" ;
                }
                break;
            case SerialEvent::GET_GAIN:
                qDebug() << "get read: GET_GAIN";
                unitSize = 1;

                if(buffer.size() >= unitSize) {
                    QByteArray unit = buffer.left(unitSize);
                    buffer.remove(0, unitSize);

                    // Process the unit
                    // Combine bytes: first byte is low, second byte is high
                    value16 = static_cast<quint8>(unit[0]);

                    qDebug() << "Received "<< unitSize << "-byte unit:" << unit.toHex()
                             << "Value:" << value16;
                    // check data in range
                    if(value16 < 4)
                        deviceState->gainMode = value16;
                    else
                        qDebug() << "get gain error, please try again. :(";
                }
                break;
            case SerialEvent::GET_MSNO:
                qDebug() << "get read: GET_MSNO";
                if( modeTempBuffer.size() + buffer.size() < 15) {
                    qDebug() << "Thermal Device:" << "not enough data";
                    modeTempBuffer.setData(buffer);
                    eventRead->prepend(SerialEvent::GET_MSNO);
                    return;
                }
                asciiValue = modeTempBuffer.data() + buffer.data();
                modeTempBuffer.buffer().clear();
                qDebug() << asciiValue;
                deviceState->deviceNameSN = QString::fromStdString(asciiValue);
                break;
            case SerialEvent::SET_3DNR:
                qDebug() << "get read: SET_3DNR";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_NUCC:
                qDebug() << "get read: SET_NUCC";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_BPCC:
                qDebug() << "get read: SET_BPCC";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_DGFD:
                qDebug() << "get read: SET_DGFD";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_DGSK:
                qDebug() << "get read: SET_DGSK";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_CINT:
                qDebug() << "get read: SET_CINT";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_TINT:
                qDebug() << "get read: SET_TINT";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_VOMC:
                qDebug() << "get read: SET_VOMC";
                qDebug() << buffer.data();
                if( modeTempBuffer.size() + buffer.size() < 4) {
                    qDebug() << "not enough data";
                    modeTempBuffer.setData(buffer);
                    return;
                }
                asciiValue = modeTempBuffer.data() + buffer.data();
                modeTempBuffer.buffer().clear();
                qDebug() << asciiValue;

                if(buffer.size() == 4 && asciiValue == "FAIL") {
                    ui->cameraPerview->startStreaming();
                    QMessageBox msgBox;
                    msgBox.setText("failed to switch source failed, please try again. :(");
                    msgBox.exec();
                } else if(buffer.size() == 4 && asciiValue == "DONE") {
                    closeSerial();
                    closeCamera();
                    QMessageBox msgBox;
                    msgBox.setText("switch source successed, please reconnect device");
                    msgBox.exec();
                }

                break;
            case SerialEvent::SET_DDEC:
                qDebug() << "get read: SET_DDEC";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_NUCP:
                qDebug() << "get read: SET_NUCP";
                qDebug() << buffer.data();
                break;
            case SerialEvent::SET_BPCP:
                qDebug() << "get read: SET_BPCP";
                qDebug() << buffer.data();
                break;
            case SerialEvent::SET_SUTS:
                qDebug() << "get read: SET_SUTS";
                qDebug() << buffer.data();

                if(currentCalibraMode == CalibrationStep::CAP_5) {
                    currentCalibraMode = CalibrationStep::CAP_6;
                    // next step
                    if(captureMode == 4) {
                        captureRoundSetting--;
                        if (captureRoundSetting > 0)
                            currentCalibraMode = CalibrationStep::CAP_2; // next round
                    }
                    else
                        currentCalibraMode = CalibrationStep::CAP_7;
                    checkCaptureFlag();
                } else if(currentCalibraMode == CalibrationStep::CAP_3 || currentCalibraMode == CalibrationStep::CAP_4){
                    if(captureMode > 0) {
                        captureDataSet();
                    }
                }
                break;
            case SerialEvent::SET_SUTC:
                qDebug() << "get read: SET_SUTC";
                qDebug() << buffer.data();

                if(currentCalibraMode == CalibrationStep::CAP_1) {
                    currentCalibraMode = CalibrationStep::CAP_2;
                    // next step
                    checkCaptureFlag();
                } else if(currentCalibraMode == CalibrationStep::CAP_6) {
                    currentCalibraMode = CalibrationStep::CAP_7;
                    // next step
                    checkCaptureFlag();
                }
                break;
            case SerialEvent::SET_AGCM:
                qDebug() << "get read: SET_AGCM";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_GAIN:
                qDebug() << "get read: SET_GAIN";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_COLM:
                qDebug() << "get read: SET_COLM";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_FFCS:
                qDebug() << "get read: SET_FFCS";
                qDebug() << buffer.data();

                break;
            case SerialEvent::SET_TPCC:
                qDebug() << "get read: SET_TPCC";
                qDebug() << buffer.data();
                break;
            default:
                break;
            }
        }
        if(!eventRead->isEmpty()) eventRead->pop_front();
        // historySerial.append("data read:");
        // historySerial.append(buffer.data());
    } else {
        qDebug() << "no event data" << buffer.data();
    }

    if(eventSend->size() > 0) {
        // handleReadyWrite();
    } else {
        if(startInitDeviceState){
            startCamera();
            startInitDeviceState = false;
            currentStep = Step::Capture_30;
            createFolder();
            // loadBPCFile();
        }
    }

}

void MainWindow::handleReadyWrite() {
    QByteArray data;

    if(eventSend->size() > 0){
        SerialEvent sendEvent = eventSend->first();
        switch (sendEvent) {
        case SerialEvent::GET_SERIAL_NUMBER:
            data = buildCommand("SSNO", BuildCommandMode::GET, 1);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_SERIAL_NUMBER";
                eventRead->append(SerialEvent::GET_SERIAL_NUMBER);
            }
            break;

        case SerialEvent::GET_VER1:
            data = buildCommand("VER1", BuildCommandMode::GET, 1);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_VER1";
                eventRead->append(SerialEvent::GET_VER1);
            }
            break;

        case SerialEvent::GET_VER2:
            data = buildCommand("VER2", BuildCommandMode::GET, 1);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_VER2";
                eventRead->append(SerialEvent::GET_VER2);
            }
            break;

        case SerialEvent::GET_3DNR:
            data = buildCommand("3DNR", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_3DNR";
                eventRead->append(SerialEvent::GET_3DNR);
            }
            break;
        case SerialEvent::GET_NUCC:
            data = buildCommand("NUCC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_NUCC";
                eventRead->append(SerialEvent::GET_NUCC);
            }
            break;
        case SerialEvent::GET_BPCC:
            data = buildCommand("BPCC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_BPCC";
                eventRead->append(SerialEvent::GET_BPCC);
            }
            break;
        case SerialEvent::GET_VTMP:
            data = buildCommand("VTMP", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_VTMP";
                eventRead->append(SerialEvent::GET_VTMP);
            }
            break;
        case SerialEvent::GET_DGFD:
            data = buildCommand("DGFD", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_DGFD";
                eventRead->append(SerialEvent::GET_DGFD);
            }
            break;
        case SerialEvent::GET_DGSK:
            data = buildCommand("DGSK", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_DGSK";
                eventRead->append(SerialEvent::GET_DGSK);
            }
            break;
        case SerialEvent::GET_CINT:
            data = buildCommand("CINT", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_CINT";
                eventRead->append(SerialEvent::GET_CINT);
            }
            break;
        case SerialEvent::GET_TINT:
            data = buildCommand("TINT", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_TINT";
                eventRead->append(SerialEvent::GET_TINT);
            }
            break;
        case SerialEvent::GET_VOMC:
            data = buildCommand("VOMC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_VOMC";
                eventRead->append(SerialEvent::GET_VOMC);
            }
            break;
        case SerialEvent::GET_DDEC:
            data = buildCommand("DDEC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_DDEC";
                eventRead->append(SerialEvent::GET_DDEC);
            }
            break;
        case SerialEvent::GET_BPCP:
            data = buildCommand("BPCP", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_BPCP";
                eventRead->append(SerialEvent::GET_BPCP);
            }
            break;
        case SerialEvent::GET_NUCP:
            data = buildCommand("NUCP", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_NUCP";
                eventRead->append(SerialEvent::GET_NUCP);
            }
            break;
        case SerialEvent::GET_SUTS:
            data = buildCommand("SUTS", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_SUTS";
                eventRead->append(SerialEvent::GET_SUTS);
            }
            break;
        case SerialEvent::GET_SUTC:
            data = buildCommand("SUTC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_SUTC";
                eventRead->append(SerialEvent::GET_SUTC);
            }
            break;
        case SerialEvent::GET_AGCM:
            data = buildCommand("AGCM", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_AGCM";
                eventRead->append(SerialEvent::GET_AGCM);
            }
            break;
        case SerialEvent::GET_GAIN:
            data = buildCommand("GAIN", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_GAIN";
                eventRead->append(SerialEvent::GET_GAIN);
            }
            break;
        case SerialEvent::GET_COLM:
            data = buildCommand("COLM", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_COLM";
                eventRead->append(SerialEvent::GET_COLM);
            }
            break;
        case SerialEvent::GET_TPCC:
            data = buildCommand("TPCC", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_TPCC";
                eventRead->append(SerialEvent::GET_TPCC);
            }
            break;
        case SerialEvent::GET_MSNO:
            data = buildCommand("MSNO", BuildCommandMode::GET, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: GET_MSNO";
                eventRead->append(SerialEvent::GET_MSNO);
            }
            break;
        case SerialEvent::SET_3DNR:
            data = buildCommand("3DNR", BuildCommandMode::SET_UINT8, deviceUpdate->noiseReduction? 1 : 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_3DNR";
                eventRead->append(SerialEvent::SET_3DNR);
            }
            break;
        case SerialEvent::SET_NUCC:
            data = buildCommand("NUCC", BuildCommandMode::SET_UINT8, deviceUpdate->nucControl? 1 : 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_NUCC";
                eventRead->append(SerialEvent::SET_NUCC);
            }
            break;
        case SerialEvent::SET_BPCC:
            data = buildCommand("BPCC", BuildCommandMode::SET_UINT8, deviceUpdate->bpcControl? 1 : 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_BPCC";
                eventRead->append(SerialEvent::SET_BPCC);
            }
            break;
        case SerialEvent::SET_DGFD:
            data = buildCommand("DGFD", BuildCommandMode::SET_UINT8, deviceUpdate->sensorDacGfid);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_DGFD";
                eventRead->append(SerialEvent::SET_DGFD);
            }
            break;
        case SerialEvent::SET_DGSK:
            data = buildCommand("DGSK", BuildCommandMode::SET_UINT16, deviceUpdate->sensorDacGsk);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_DGSK";
                eventRead->append(SerialEvent::SET_DGSK);
            }
            break;
        case SerialEvent::SET_CINT:
            data = buildCommand("CINT", BuildCommandMode::SET_UINT8, deviceUpdate->sensorCint);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_CINT";
                eventRead->append(SerialEvent::SET_CINT);
            }
            break;
        case SerialEvent::SET_TINT:
            data = buildCommand("TINT", BuildCommandMode::SET_UINT16, deviceUpdate->sensorTint);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_TINT";
                eventRead->append(SerialEvent::SET_TINT);
            }
            break;
        case SerialEvent::SET_VOMC:
            data = buildCommand("VOMC", BuildCommandMode::SET_ASCII, 0, deviceUpdate->sourceMode);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_VOMC";
                eventRead->append(SerialEvent::SET_VOMC);
            }
            break;
        case SerialEvent::SET_DDEC:
            data = buildCommand("DDEC", BuildCommandMode::SET_UINT8, deviceUpdate->ddeControl? 1 : 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_DDEC";
                eventRead->append(SerialEvent::SET_DDEC);
            }
            break;
        case SerialEvent::SET_NUCP:
            data = buildCommand("NUCP", BuildCommandMode::SET_DATA_INT, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_NUCP";
                eventRead->append(SerialEvent::SET_NUCP);
            }
            break;
        case SerialEvent::SET_BPCP:
            data = buildCommand("BPCP", BuildCommandMode::SET_DATA_UINT, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_BPCP";
                eventRead->append(SerialEvent::SET_BPCP);
            }
            break;
        case SerialEvent::SET_SUTS:
            data = buildCommand("SUTS", BuildCommandMode::SET_UINT8, deviceUpdate->shutterControl);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_SUTS";
                eventRead->append(SerialEvent::SET_SUTS);
            }
            break;
        case SerialEvent::SET_SUTC:
            data = buildCommand("SUTC", BuildCommandMode::SET_UINT8, deviceUpdate->autoShutterControl);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_SUTC";
                eventRead->append(SerialEvent::SET_SUTC);
            }
            break;
        case SerialEvent::SET_AGCM:
            data = buildCommand("AGCM", BuildCommandMode::SET_UINT16, deviceUpdate->agcMode);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_AGCM";
                eventRead->append(SerialEvent::SET_AGCM);
            }
            break;
        case SerialEvent::SET_GAIN:
            data = buildCommand("GAIN", BuildCommandMode::SET_UINT8, (int)deviceUpdate->gainMode);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_GAIN";
                eventRead->append(SerialEvent::SET_GAIN);
            }
            break;
        case SerialEvent::SET_CALP:
            data = buildCommand("CALP", BuildCommandMode::SET_NONE, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_CALP";
                eventRead->append(SerialEvent::SET_CALP);
            }
            break;
        case SerialEvent::SET_COLM:
            data = buildCommand("COLM", BuildCommandMode::SET_UINT16, deviceUpdate->colorMode);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_COLM";
                eventRead->append(SerialEvent::SET_COLM);
            }
            break;
        case SerialEvent::SET_FFCS:
            data = buildCommand("FFCS", BuildCommandMode::SET_NONE, 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_FFCS";
                eventRead->append(SerialEvent::SET_FFCS);
            }
            break;
        case SerialEvent::SET_TPCC:
            data = buildCommand("TPCC", BuildCommandMode::SET_UINT8, deviceUpdate->radiometryControl? 1 : 0);
            serial->write(data);
            if(eventRead != NULL) {
                //qDebug() << "set read: SET_TPCC";
                eventRead->append(SerialEvent::SET_TPCC);
            }
            break;
        default:
            break;
        }
        eventSend->pop_front();
    } else {
        //qDebug() << "no command to send";
    }

}

void MainWindow::handleError(QSerialPort::SerialPortError error){
    qDebug() << "ThermalDevice :" << serial->portName();
    qDebug() << "serial error:" << error;
}

void MainWindow::triggerCheckSend() {
    if(eventSend->size() > 0 && eventRead->size() == 0) {
        handleReadyWrite();
    }
}

// serial event
void MainWindow::setNoiseReduction() {
    deviceUpdate->noiseReduction = !deviceState->noiseReduction;
    eventSend->append(SerialEvent::SET_3DNR);
    eventSend->append(SerialEvent::GET_3DNR);
    handleReadyWrite();
}

void MainWindow::setNUCControl() {
    if(deviceState->ddeControl) {
        deviceUpdate->ddeControl = !deviceState->ddeControl;
        eventSend->append(SerialEvent::SET_DDEC);
        eventSend->append(SerialEvent::GET_DDEC);
    }
    deviceUpdate->nucControl = !deviceState->nucControl;
    eventSend->append(SerialEvent::SET_NUCC);
    eventSend->append(SerialEvent::GET_NUCC);
    handleReadyWrite();
}

void MainWindow::setBPCControl() {
    deviceUpdate->bpcControl = !deviceState->bpcControl;
    eventSend->append(SerialEvent::SET_BPCC);
    eventSend->append(SerialEvent::GET_BPCC);
    handleReadyWrite();

}

void MainWindow::setDDEControl() {
    if(!deviceState->nucControl) {
        deviceUpdate->nucControl = !deviceState->nucControl;
        eventSend->append(SerialEvent::SET_NUCC);
        eventSend->append(SerialEvent::GET_NUCC);
    }
    deviceUpdate->ddeControl = !deviceState->ddeControl;
    eventSend->append(SerialEvent::SET_DDEC);
    eventSend->append(SerialEvent::GET_DDEC);
    handleReadyWrite();

}

void MainWindow::setTPCControl() {
    deviceUpdate->radiometryControl = !deviceState->radiometryControl;
    eventSend->append(SerialEvent::SET_TPCC);
    eventSend->append(SerialEvent::GET_TPCC);
    handleReadyWrite();

}

void MainWindow::setAutoCalibration() {
    deviceUpdate->autoShutterControl = !deviceState->autoShutterControl;
    eventSend->append(SerialEvent::SET_SUTC);
    eventSend->append(SerialEvent::GET_SUTC);
    handleReadyWrite();
}

void MainWindow::setAGCMode() {
    eventSend->append(SerialEvent::SET_AGCM);
    eventSend->append(SerialEvent::GET_AGCM);
    handleReadyWrite();

}

void MainWindow::setSourceVideoMode(QString mode) {
    deviceUpdate->sourceMode = mode;
    eventSend->append(SerialEvent::SET_VOMC);
    handleReadyWrite();
}

void MainWindow::getDeivceInfo() {
    startInitDeviceState = true;

    eventSend->append(SerialEvent::GET_SERIAL_NUMBER);
    // source video tyep
    eventSend->append(SerialEvent::GET_VOMC);
    // blue
    // eventSend->append(SerialEvent::GET_3DNR);
    // eventSend->append(SerialEvent::GET_NUCC);
    // eventSend->append(SerialEvent::GET_BPCC);
    // eventSend->append(SerialEvent::GET_DDEC);
    // eventSend->append(SerialEvent::GET_TPCC);
    // // red
    // eventSend->append(SerialEvent::GET_VTMP);
    // eventSend->append(SerialEvent::GET_DGFD);
    // eventSend->append(SerialEvent::GET_DGSK);
    // eventSend->append(SerialEvent::GET_CINT);
    // eventSend->append(SerialEvent::GET_TINT);

    // // AGC mode
    // eventSend->append(SerialEvent::GET_AGCM);

    // // Fillter mode
    // eventSend->append(SerialEvent::GET_COLM);

    // // Gain mode
    // eventSend->append(SerialEvent::GET_GAIN);

    // // shutter status
    // eventSend->append(SerialEvent::GET_SUTS);
    // eventSend->append(SerialEvent::GET_SUTC);

    // // version number
    // eventSend->append(SerialEvent::GET_VER1);
    // eventSend->append(SerialEvent::GET_VER2);

    // Serial number
    eventSend->append(SerialEvent::GET_MSNO);

    handleReadyWrite();
}

void MainWindow::setDeivceSettingDefault() {
    deviceUpdate->nucControl = false;
    deviceUpdate->bpcControl = false;
    deviceUpdate->ddeControl = false;
    deviceUpdate->autoShutterControl = false;
    deviceUpdate->shutterControl = false;

    // image setting
    eventSend->append(SerialEvent::GET_NUCC);
    eventSend->append(SerialEvent::GET_BPCC);
    eventSend->append(SerialEvent::GET_DDEC);

    // shutter status
    eventSend->append(SerialEvent::GET_SUTS);
    eventSend->append(SerialEvent::GET_SUTC);

    // source video tyep
    eventSend->append(SerialEvent::GET_VOMC);
}

void MainWindow::setSensorSetting() {
    // Gain mode
    eventSend->append(SerialEvent::GET_GAIN);
    // Parameter
    eventSend->append(SerialEvent::GET_DGFD);
    eventSend->append(SerialEvent::GET_DGSK);
    eventSend->append(SerialEvent::GET_CINT);
    eventSend->append(SerialEvent::GET_TINT);
}

void MainWindow::setSensorValue() {
    eventSend->append(SerialEvent::SET_DGFD);
    eventSend->append(SerialEvent::GET_DGFD);

    eventSend->append(SerialEvent::SET_DGSK);
    eventSend->append(SerialEvent::GET_DGSK);

    eventSend->append(SerialEvent::SET_CINT);
    eventSend->append(SerialEvent::GET_CINT);

    eventSend->append(SerialEvent::SET_TINT);
    eventSend->append(SerialEvent::GET_TINT);

    handleReadyWrite();
}

void MainWindow::setGainLevel(int gainLevel) {
    // check state change
    deviceUpdate->gainMode = gainLevel;

    eventSend->append(SerialEvent::SET_GAIN);
    eventSend->append(SerialEvent::GET_GAIN);
    eventSend->append(SerialEvent::GET_DGFD);
    eventSend->append(SerialEvent::GET_DGSK);
    eventSend->append(SerialEvent::GET_CINT);
    eventSend->append(SerialEvent::GET_TINT);
    handleReadyWrite();
}

void MainWindow::triggerFFC() {
    eventSend->append(SerialEvent::SET_FFCS);
    handleReadyWrite();
}

// calibration - captrue
void MainWindow::triggerCaptureNUC(){
    // select target
    if(selectHighLow) {
        captureOpenData = &currentDataNUC.captureLowTempOpenData;
        captureCloseData = &currentDataNUC.captureLowTempCloseData;
    } else {
        captureOpenData = &currentDataNUC.captureHighTempOpenData;
        captureCloseData = &currentDataNUC.captureHighTempCloseData;
    }

    if(captureOpenData->count() > 0 || captureCloseData->count() > 0){
        QMessageBox msgBox;
        msgBox.setText("the data already collect");
        msgBox.exec();
    } else {
        captureMode = 1;
        captureSaveMode = currentCalibraMode;
        currentCalibraMode = CalibrationStep::CAP_3;
        checkCaptureFlag();
    }
}

void MainWindow::captureDataSet() {
    bool isClose = currentCalibraMode == CalibrationStep::CAP_4;
    // get five image
    // drop data
    ui->cameraPerview->dropFrame(CAP_DROP_SIZE);
    for (int var = 0; var < CAP_UNIT_SIZE; ++var) {
        // NUC - data
        if (!isClose)
            captureOpenData->append(ui->cameraPerview->getCaptureImageNUC());
        else
            captureCloseData->append(ui->cameraPerview->getCaptureImageNUC());

    }
    if(isOnePoint)captureOnePointCount++;

    // save data to file
    // NUC - data
    // build save path and name
    QString fileName;
    if(selectHighLow) {
        fileName += "30_";
    } else {
        fileName += "45_";
    }

    if(!isClose) {
        fileName += "open";
    } else {
        fileName += "close";
    }

    if(selectHighLow) {
        fileName += "1";
    } else {
        fileName += "2";
    }

    if(!isClose) {
        nuccla->write_raw_to_path(*captureOpenData, currentPath.toStdString(), fileName.toStdString());
    } else {
        nuccla->write_raw_to_path(*captureCloseData, currentPath.toStdString(), fileName.toStdString());
    }

    // next step
    switch (currentCalibraMode) {
    case CalibrationStep::CAP_1:
        currentCalibraMode = CalibrationStep::CAP_2;
        break;
    case CalibrationStep::CAP_2:
        currentCalibraMode = CalibrationStep::CAP_3;
        if(captureMode == 4) {
            if (!captureRawSetting && !captureShutterSetting)
                currentCalibraMode = CalibrationStep::CAP_5; // skip both captures
            else if (captureRawSetting)
                currentCalibraMode = CalibrationStep::CAP_3; // do open capture
            else
                currentCalibraMode = CalibrationStep::CAP_4;
        }
        break;
    case CalibrationStep::CAP_3:
        currentCalibraMode = CalibrationStep::CAP_4;
        if(captureMode == 4) {
            if (captureShutterSetting)
                currentCalibraMode = CalibrationStep::CAP_4; // do close capture
            else
                currentCalibraMode = CalibrationStep::CAP_5; // skip close
            break;
        }
        break;
    case CalibrationStep::CAP_4:
        currentCalibraMode = CalibrationStep::CAP_5;
        break;
    case CalibrationStep::CAP_5:
        currentCalibraMode = CalibrationStep::CAP_6;
        if(captureMode == 4) {
            captureRoundSetting--;
            if (captureRoundSetting > 0)
                currentCalibraMode = CalibrationStep::CAP_2; // next round
        }
        break;
    case CalibrationStep::CAP_6:
        currentCalibraMode = CalibrationStep::CAP_7;
        break;
    case CalibrationStep::CAP_7:
    default:
        currentCalibraMode = captureSaveMode;
        break;
    }
    checkCaptureFlag();
}

void MainWindow::checkCaptureFlag() {

    switch (currentCalibraMode) {
    case CalibrationStep::CAP_1:
        autoCalibraStateTemp = deviceUpdate->autoShutterControl;
        deviceUpdate->autoShutterControl = false;
        eventSend->append(SerialEvent::SET_SUTC);
        break;
    case CalibrationStep::CAP_2:
        eventSend->append(SerialEvent::GET_VTMP);
        break;
    case CalibrationStep::CAP_3:
        deviceUpdate->shutterControl = false;
        eventSend->append(SerialEvent::SET_SUTS);
        break;
    case CalibrationStep::CAP_4:
        deviceUpdate->shutterControl = true;
        eventSend->append(SerialEvent::SET_SUTS);
        break;
    case CalibrationStep::CAP_5:
        deviceUpdate->shutterControl = false;
        eventSend->append(SerialEvent::SET_SUTS);
        break;
    case CalibrationStep::CAP_6:
        deviceUpdate->autoShutterControl = autoCalibraStateTemp;
        eventSend->append(SerialEvent::SET_SUTC);
        break;
    case CalibrationStep::CAP_7:
        endCapture();
    default:
        currentCalibraMode = captureSaveMode;
        break;
    }

}

void MainWindow::endCapture() {
    // if it's for NUC-Preview then open preview window
    captureMode = -1;
    captureOnePointCount = 0;
    if(currentStep == Step::Capture_30) {
        currentStep = Step::Capture_45;
    } else if(currentStep == Step::Capture_45) {
        currentStep = Step::Calculate_NUC;
        calculateParmeterNUC();
    }
}

// calibration - calculate
void MainWindow::calculateParmeterNUC() {
    int progressTarget = CAP_UNIT_SIZE, progressCount = 0;

    // check dataset
    if( currentDataNUC.captureHighTempCloseData.size() < CAP_UNIT_SIZE ||
        currentDataNUC.captureHighTempOpenData.size() < CAP_UNIT_SIZE ||
        currentDataNUC.captureLowTempCloseData.size() < CAP_UNIT_SIZE ||
        currentDataNUC.captureLowTempOpenData.size() < CAP_UNIT_SIZE) {

        QMessageBox msgBox;
        msgBox.setText("capture data not enough");
        msgBox.exec();
        return;
    }

    // start progress
    if(currentCalibraMode == CalibrationStep::NUC_1)
        currentCalibraMode = CalibrationStep::NUC_2;
    else if(currentCalibraMode == CalibrationStep::OP_NUC_1) {
        currentCalibraMode = CalibrationStep::OP_NUC_2;
    }else if(currentCalibraMode == CalibrationStep::RADIOMETRY_1) {
        currentCalibraMode = CalibrationStep::RADIOMETRY_2;
    }
    nuccla->progressPercentage = 0.0;

    // get data
    vector<cv::Mat> raw_30_o, raw_30_c, raw_45_o, raw_45_c;
    for (int i = 0; i < currentDataNUC.captureLowTempOpenData.size(); ++i) {
        raw_30_o.push_back(currentDataNUC.captureLowTempOpenData[i].reshape(1, 640 * 480).clone());
        raw_45_o.push_back(currentDataNUC.captureHighTempOpenData[i].reshape(1, 640 * 480).clone());
        raw_30_c.push_back(currentDataNUC.captureLowTempCloseData[i].reshape(1, 640 * 480).clone());
        raw_45_c.push_back(currentDataNUC.captureHighTempCloseData[i].reshape(1, 640 * 480).clone());
        progressCount++;
        nuccla->progressPercentage = (float)(20 * progressCount / progressTarget);
    }

    // insert data to nuc object
    nuccla->insert_nuc_data(raw_30_o,raw_30_c,raw_45_o,raw_45_c);

    // clear data buffer
    raw_30_o.clear();
    raw_45_o.clear();
    raw_30_c.clear();
    raw_45_c.clear();
    nuccla->calculate_nuc_parameter();
}

void MainWindow::setDataNUC(vector<int16_t> nuc_params, vector<int16_t> a_params, vector<int16_t> c_params) {
    // permuteNUC = nuc_params;

    // build save path and name
    QString fileName = "nuc_param";

    DataNUC* selectTempNUC = nullptr;

    if(isOnePoint) {
        fileName = "nuc_param_one_point";
    }

    //save NUC data
    currentDataNUC.permuteNUC = nuc_params;
    nuccla->write_nuc_bin_to_path(nuc_params, currentPath.toStdString(), fileName.toStdString());

    int rows = 480;
    int cols = 640;

    // CV_16S == int16_t
    cv::Mat matA(rows, cols, CV_16S, a_params.data());
    cv::Mat matC(rows, cols, CV_16S, c_params.data());
    ui->cameraPerview->remapping16To8(matA);
    ui->cameraPerview->remapping16To8(matC);
    // QImage imageA = ui->cameraPerview->cvMatToQImage(matA);
    // QImage imageC = ui->cameraPerview->cvMatToQImage(matC);
    // QPixmap pixmapA = QPixmap::fromImage(imageA);
    // QPixmap pixmapC = QPixmap::fromImage(imageC);
    // ui->result1->setPixmap(pixmapA);
    // ui->result2->setPixmap(pixmapC);

    if(currentCalibraMode == CalibrationStep::NUC_2)
        currentCalibraMode = CalibrationStep::NUC_1;
    else if(currentCalibraMode == CalibrationStep::OP_NUC_2)
        currentCalibraMode = CalibrationStep::OP_NUC_1;
    else if(currentCalibraMode == CalibrationStep::RADIOMETRY_2)
        currentCalibraMode = CalibrationStep::RADIOMETRY_1;

    checkResultPass(matA, matC);
}

void MainWindow::checkResultPass(cv::Mat A, cv::Mat C) {

    bool debug = true;
    QString f = "p1.png";
    int ok = 0, ng = 0;
    cv::Mat img = cv::imread(f.toStdString(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        qDebug() << "Failed to read: " << f << "\n";
    }
    cv::imshow("og", img);
    auto ra = detectDefect(
        C,
        /*border*/ 30,
        /*bgSigma*/ 10.0,
        /*hiPct*/ 0.999,
        /*loPct*/ 0.001,
        /*margin*/ 1.0,
        /*minArea*/ 20,
        /*debug*/ debug,
        /*debugPrefix*/ "test",
        true
        );

    qDebug() << f.toStdString()
             << " => " << (ra.hasDefect ? "DEFECT" : "CLEAN")
             << " (components=" << ra.componentCount
             << ", brightThr=" << ra.brightThr
             << ", darkThr=" << ra.darkThr
             << ")\n";

    if (ra.hasDefect) ng++; else ok++;


    auto r = detectDefect(
        A,
        /*border*/ 30,
        /*bgSigma*/ 10.0,
        /*hiPct*/ 0.999,
        /*loPct*/ 0.001,
        /*margin*/ 1.0,
        /*minArea*/ 20,
        /*debug*/ debug,
        /*debugPrefix*/ "test",
        false
        );

    qDebug() << f.toStdString()
             << " => " << (r.hasDefect ? "DEFECT" : "CLEAN")
             << " (components=" << r.componentCount
             << ", brightThr=" << r.brightThr
             << ", darkThr=" << r.darkThr
             << ")\n";

    if (r.hasDefect) ng++; else ok++;

    if(ng == 0) {
        ui->labelPass->setText("PASS");
        ui->labelPass->setStyleSheet("color: green;");
    } else {
        ui->labelPass->setText("FAIL");
        ui->labelPass->setStyleSheet("color: red;");
    }
    currentStep = Step::Finish;
}

