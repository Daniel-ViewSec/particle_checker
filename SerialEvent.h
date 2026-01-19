#ifndef SERIALEVENT_H
#define SERIALEVENT_H
enum class SerialEvent {
    //none
    NONE,

        // get
        GET_SERIAL_NUMBER,
        GET_VER1,
        GET_VER2,
        GET_3DNR,
        GET_NUCC,
        GET_BPCC,
        GET_VTMP,
        GET_DGFD,
        GET_DGSK,
        GET_CINT,
        GET_TINT,
        GET_VOMC,
        GET_DDEC,
        GET_NUCP, // NUC calibration
        GET_BPCP, // BPC calibration
        GET_SUTS, // shutter on off control
        GET_SUTC, // shutter auto control status
        GET_AGCM,
        GET_COLM,
        GET_GAIN,
        GET_TPCC,
        GET_MSNO,

        //set
        SET_3DNR,
        SET_NUCC,
        SET_BPCC,
        SET_DGFD,
        SET_DGSK,
        SET_CINT,
        SET_TINT,
        SET_VOMC,
        SET_DDEC,
        SET_NUCP,
        SET_BPCP,
        SET_SUTS,
        SET_SUTC,
        SET_AGCM,
        SET_GAIN,
        SET_CALP,
        SET_COLM,
        SET_FFCS,
        SET_TPCC,
};

#endif // SERIALEVENT_H
