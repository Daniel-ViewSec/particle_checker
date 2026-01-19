#ifndef CALIBRATIONSTEP_H
#define CALIBRATIONSTEP_H

enum class CalibrationStep : int {
    None,

    BPC_1,  // capture
    BPC_2,  // pick point
    BPC_3,  // save data

    NUC_1,  // preview
    NUC_2,  // caculate data

    OP_NUC_1,  // preview
    OP_NUC_2,  // caculate data

    RADIOMETRY_1,   // preview
    RADIOMETRY_2,   // caculate data

    UP_1,   // upload - ask NUC path
    UP_2,   // upload - switch to R264 and upload
    UP_3,   // upload - switch back to U016

    CAP_1,  // close auto calibration
    CAP_2,  // get vtemp
    CAP_3,  // capture open
    CAP_4,  // capture close
    CAP_5,  // reopen shutter
    CAP_6,  // reopen auto calibration
    CAP_7,  // capture end
};

#endif // CALIBRATIONSTEP_H
