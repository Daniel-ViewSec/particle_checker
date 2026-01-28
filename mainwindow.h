#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "Step.h"
#include "ThermalDeviceState.h"
#include "BuildCommandMode.h"
#include "SerialEvent.h"
#include "calib_lib/bpc.h"
#include "calib_lib/nuc.h"
#include "opencv2/highgui/highgui.hpp"
#include <QShortcut>
#include <QBuffer>
#include <QSerialPort>
#include <QSerialPortInfo>
#include "cameraimagewidget.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

struct DetectResult {
    bool hasDefect = false;
    int componentCount = 0;
    double brightThr = 0, darkThr = 0;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    enum class SourceVideoMode : int {
        UJPG,
        U016,
        U422,
        // U264,
        R264,
    };
    Q_ENUM(SourceVideoMode);
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void updateUI();
    void createFolder();
    // Scan UVC and Serial device
    void scanDevice();

    // Serial connection function
    void startSerial();
    void setSerial(const QSerialPortInfo &serialInfo);
    void closeSerial();

    // UVC camera function
    void startCamera();
    void closeCamera();
    void updateCameraSource(QString newSource);

    // Convert String to Enum
    SourceVideoMode stringToFormat(const string& str);

    // serial system
    QByteArray buildCommand(QString command, BuildCommandMode action, quint16 value, QString asciiValue = "");

    void handleReadyRead();

    void handleReadyWrite();

    void handleError(QSerialPort::SerialPortError error);

    void triggerCheckSend();

    // serial event
    void setNoiseReduction();

    void setNUCControl();

    void setBPCControl();

    void setDDEControl();

    void setTPCControl();

    void setAutoCalibration();

    void setAGCMode();

    void setSourceVideoMode(QString mode);

    void getDeivceInfo();

    void setDeivceSettingDefault();

    void setSensorSetting();

    void setSensorValue();

    void setGainLevel(int gainLevel);

    void triggerFFC();

    // calubration - captrue
    void triggerCaptureNUC();

    void captureDataSet();

    void checkCaptureFlag();

    void endCapture();

    // calibration - calculate
    void calculateParmeterNUC();
    void setDataNUC(vector<int16_t> nuc_params, vector<int16_t> a_params, vector<int16_t> c_params);
    void checkResultPass(cv::Mat A, cv::Mat C);
    void autoAdjustNUC();

    // utility
    vector<pair<int, int>> find_consecutive_zeros(const vector<int32_t>& data_list, int zero_threshold);
    pair<int, int> find_left_and_right_neighbors(const vector<pair<int, int>>& zero_blocks_results, int target_index, int data_list_length);
    pair<int, int> find_neighbors_for_target(const vector<int32_t>& data_list, int zero_threshold, int target_index);


    void reset();

    DetectResult detectDefect(const cv::Mat& gray8, int border,double bgSigma, double hiPct, double loPct, double margin, int minArea, bool debug, const std::string& debugPrefix, bool aorc);
    void test();
    void loadBPCFile();
    bool isImageFile(QString p);
    double percentile(std::vector<float>& v, double p01);

private:
    Ui::MainWindow *ui;
    QTimer *uiUpdater;
    QTimer *sendEventTriggerTimer;
    Step currentStep = Step::Select_COM_Port;
    QString currentPath;
    QString BPCPath;
    QShortcut *enterShortcut, *enterShortcut2;
    NUC_Data *nuccla;
    BPC *bpc_parameter;
    QVector<uint8_t> maskTempBPC;

    QSerialPort *serial;
    ThermalDeviceState *deviceState;
    ThermalDeviceState *deviceUpdate;
    QQueue<SerialEvent> *eventRead;
    QQueue<SerialEvent> *eventSend;
    bool startInitDeviceState = false;
    QByteArray poolBuffer;
    QBuffer modeTempBuffer;
    unordered_map<string, SourceVideoMode> sourceVideoModeFormatMap = {
       {"UJPG", SourceVideoMode::UJPG},
       {"U016", SourceVideoMode::U016},
       // {"U264", SourceVideoMode::U264},
       {"R264", SourceVideoMode::R264},
       {"U422", SourceVideoMode::U422},
    };

    const QString SAVE_FOLDER = "SAVE_DATA";
    const QString BPC_FOLDER = "SENSOR_DATA";
};
#endif // MAINWINDOW_H
