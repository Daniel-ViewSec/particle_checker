#ifndef BUILDCOMMANDMODE_H
#define BUILDCOMMANDMODE_H

enum class BuildCommandMode : int {
    GET,
    SET_NONE,
    SET_UINT8,
    SET_UINT16,
    SET_ASCII,
    SET_DATA_UINT,
    SET_DATA_INT
};
#endif // BUILDCOMMANDMODE_H
