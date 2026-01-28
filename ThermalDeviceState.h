#ifndef THERMALDEVICESTATE_H
#define THERMALDEVICESTATE_H

#include <qtypes.h>
#include <qstring.h>
class ThermalDeviceState {
public:
    // red zone
    quint32 serialNumber;
    QString fwVersion;
    QString companyName;
    quint16 sensorVTEMP;
    quint16 sensorDacGfid;
    quint16 sensorDacGsk;
    quint16 sensorCint;         //uint8
    quint16 sensorTint;
    QString deviceNameSN;
    bool productionMode;

    // blue zone
    bool noiseReduction;        //uint8
    bool nucControl;            //uint8
    bool bpcControl;            //uint8
    bool ddeControl;            //uint8
    bool radiometryControl;            //uint8

    // shutter control
    bool autoShutterControl;    //uint8
    bool shutterControl;        //uint8

    // Radiometry
    bool tempCaliControl;        //uint8

    // source type
    QString sourceMode;

    // Dynamic Range mode
    quint16 drMode; //0 - LDR, 1 - HDR

    // AGC mode
    quint16 agcMode;

    // GAIN mode
    quint16 gainMode; //0 - Auto, 1 - HighGain, 2 - NormalGain, 3 - LowGain

    // filter color
    quint16 colorMode;


};
#endif // THERMALDEVICESTATE_H
