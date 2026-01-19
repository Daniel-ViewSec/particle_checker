#ifndef SOURCEVIDEOMODE_H
#define SOURCEVIDEOMODE_H
#include <QString>
enum class SourceVideoMode : int {
    UJPG,
    U016,
    U422,
    // U264,
    R264,
};
class SourceVideoModeHelper
{
public:
    static QString toString(SourceVideoMode mode);
    static SourceVideoMode fromString(const QString &str);
};
#endif // SOURCEVIDEOMODE_H
