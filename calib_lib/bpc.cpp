#include "bpc.h"
#include <QDebug>
#include <QtCore/QFile>
#include <fstream>
#include <filesystem> // C++17 需要包含此頭文件

namespace fs = std::filesystem;  // 引用 std::filesystem 的命名空間

void BPC::set_factor(int new_factor) {
    if (new_factor > 0) {
        factor = new_factor;
    }
}

int BPC::get_factor() {
    return factor;
}

void BPC::set_zoomedX(int new_zoomed_X) {
    if (new_zoomed_X >= 0) {
        zoomedX = new_zoomed_X;
    }
}

int BPC::get_zoomedX() {
    return zoomedX;
}

void BPC::set_zoomedY(int new_zoomed_Y) {
    if (new_zoomed_Y >= 0) {
        zoomedY = new_zoomed_Y;
    }
}

int BPC::get_zoomedY() {
    return zoomedY;
}

void BPC::set_height(int new_height) {
    if (new_height >= 0) {
        height = new_height;
    }
}

int BPC::get_height() {
    return height;
}

void BPC::set_width(int new_width) {
    if (new_width >= 0) {
        width = new_width;
    }
}

int BPC::get_width() {
    return width;
}

void BPC::ini_badpixel(){
    badPixel.resize(height);
    for (int i = 0; i < height; ++i) {
        badPixel[i].resize(width);
    }
}

void BPC::set_bedpoint(QList<QPoint> &pointlist) {
    qDebug() << "set_bedpoint \n";
    for (const QPoint &point : pointlist) {
            badPixel[point.y()][point.x()] = badPixel[point.y()][point.x()] ^ 1;
            qDebug() << "badPixel"<<point.y()<<" "<<point.x()<<"" << badPixel[point.y()][point.x()] <<"\n";
        }

}

void BPC::clear_badpixel(){
    for (int x = 0; x < badPixel.length(); ++x) {
        for (int y = 0; y < badPixel[x].length(); ++y) {
            badPixel[x][y] = 0;
        }
    }
}

void BPC::update_badpoixel(QVector<uint8_t> &initMask){
    qDebug() << "update_badpoixel \n";
    qDebug() << "bpc width:" << width << ", height:" << height;

    badPixel = convertDecimalToMatrix(initMask, width);
}

QVector<QVector<uint8_t>> BPC::get_bedpoint(){
    // qDebug() << "get_bedpoint \n";
    return badPixel;
}

QImage BPC::cvMatToQImage(const cv::Mat &mat) {
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

cv::Mat BPC::QImageToCvMat(const QImage &image) {
    if (image.isNull()) {
        qWarning("QImage is empty!");
        return cv::Mat();
    }

    switch (image.format()) {
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    case QImage::Format_RGBA8888: {
        QImage img = image.convertToFormat(QImage::Format_RGBA8888);
        return cv::Mat(img.height(), img.width(), CV_8UC4, (void*)img.bits(), img.bytesPerLine()).clone();
    }
    case QImage::Format_RGB888: {
        QImage img = image.convertToFormat(QImage::Format_RGB888);
        return cv::Mat(img.height(), img.width(), CV_8UC3, (void*)img.bits(), img.bytesPerLine()).clone();
    }
    case QImage::Format_RGB32: {
        QImage img = image.convertToFormat(QImage::Format_RGB888);
        return cv::Mat(img.height(), img.width(), CV_8UC3, (void*)img.bits(), img.bytesPerLine()).clone();
    }
    case QImage::Format_Grayscale8:
        return cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.bits(), image.bytesPerLine()).clone();
    default:
        qWarning("QImageToCvMat: Unsupported QImage format");
        return cv::Mat();
    }
}

QVector<uint8_t> BPC::convertMatrixToDecimal(const QVector<QVector<uint8_t>>& matrix) {
    QVector<uint8_t> decimalValues;

    for (size_t row = 0; row < matrix.size(); ++row) {
        uint8_t decimalValue = 0;
        for (size_t col = 0; col < matrix[row].size(); ++col) {

            if (col % 8 == 0 && col != 0) {
                decimalValues.push_back(decimalValue);
                decimalValue = 0;
            }

            decimalValue = (decimalValue << 1) | matrix[row][col];
        }

        decimalValues.push_back(decimalValue);
    }
    qDebug() << "decimalValues.size" << decimalValues.size() << "\n";
    return decimalValues;
}

QVector<QVector<uint8_t>> BPC::convertDecimalToMatrix(const QVector<uint8_t>& decimalValues, int columns) {
    QVector<QVector<uint8_t>> matrix;

    QVector<uint8_t> row;
    for (int i = 0; i < decimalValues.size(); ++i) {
        uint8_t decimalValue = decimalValues[i];
        QVector<uint8_t> bitValues;

        for (int j = 7; j >= 0; --j) { // Extract bits from MSB to LSB
            bitValues.push_back((decimalValue >> j) & 1);
        }

        row.append(bitValues);

        if (row.size() >= columns) {
            matrix.push_back(row.mid(0, columns)); // Take only required columns
            row = row.mid(columns); // Keep the remaining bits for the next row
        }
    }

    if (!row.isEmpty()) {
        matrix.push_back(row); // Append remaining bits as the last row
    }

    return matrix;
}
void BPC::read_bpc_bin(){
    // 讀取 nuc_param.bin
    std::ifstream bin_file("bpc_param.bin", std::ios::binary);
    if (bin_file.is_open()) {
        // 確定文件大小
        bin_file.seekg(0, std::ios::end);
        size_t file_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);

        // 確保文件大小是 uint8_t 數據的倍數
        if (file_size % sizeof(uint8_t) != 0) {
            std::cerr << "Error: bpc_param.bin file size is not a multiple of uint8_t" << std::endl;
            return;
        }

        // 創建一個足夠大的容器來存儲所有的 uint8_t 數據
        QVector<uint8_t> bpc_params(file_size / sizeof(uint8_t));

        // 讀取數據到 bpc_params
        bin_file.read(reinterpret_cast<char*>(bpc_params.data()), file_size);
        bin_file.close();

        // check size
        if(bpc_params.size() != badPixelSize) {
            std::cout << "read size:bpc_params.size()  error!" << std::endl;
        } else {
            emit update_bpc(bpc_params);
        }
        std::cout << "read finish!" << std::endl;
    } else {
        std::cerr << "Error: Unable to open bpc_param.bin" << std::endl;
    }
}

void BPC::read_bpc_bin_by_path(std::string path, std::string file_name){
    // 讀取 nuc_param.bin
    std::string full_path = path + "/" + file_name;

    if (!fs::exists(full_path)) {
        std::cerr << "File not found: " << full_path << std::endl;
        return;  // return empty Mat
    }

    std::ifstream bin_file(full_path, std::ios::binary);
    if (bin_file.is_open()) {
        // 確定文件大小
        bin_file.seekg(0, std::ios::end);
        size_t file_size = bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);

        // 確保文件大小是 uint8_t 數據的倍數
        if (file_size % sizeof(uint8_t) != 0) {
            std::cerr << "Error: bpc_param.bin file size is not a multiple of uint8_t" << std::endl;
            return;
        }

        // 創建一個足夠大的容器來存儲所有的 uint8_t 數據
        QVector<uint8_t> bpc_params(file_size / sizeof(uint8_t));

        // 讀取數據到 bpc_params
        bin_file.read(reinterpret_cast<char*>(bpc_params.data()), file_size);
        bin_file.close();

        // check size
        if(bpc_params.size() != badPixelSize) {
            std::cout << "read size:bpc_params.size()  error!" << std::endl;
        } else {
            emit update_bpc(bpc_params);
        }
        std::cout << "read finish!" << std::endl;
    } else {
        std::cerr << "Error: Unable to open bpc_param.bin" << std::endl;
    }
}

void BPC::write_bpc_bin(const std::string& filename){
    // Check if file exists
    if (fs::exists(filename)) {
        std::cout << "File exists. Removing: " << filename << std::endl;
        fs::remove(filename);
    }

    QVector<uint8_t> bpc_params = convertMatrixToDecimal(badPixel);
    std::ofstream bin_file(filename, std::ios::binary);
    if (bin_file.is_open()) {
        // 寫入數據到文件

        bin_file.write(reinterpret_cast<const char*>(bpc_params.data()),
                       bpc_params.size() * sizeof(uint8_t));
        bin_file.close();


        // qDebug() << "nuc_param.bin data read successfully!\n";
        // qDebug() << "Number of elements in nuc_param.bin:" << nuc_params.size() << "\n";

        // 發送數據
        std::cout << "save finish!" << std::endl;
    } else {
        std::cerr << "Error: Unable to save nuc_param.bin" << std::endl;
    }
}
