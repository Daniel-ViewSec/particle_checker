#include "nuc.h"
#include <thread>
#include <QtCore/QFile>
#include <fstream>
#include <filesystem> // C++17 需要包含此頭文件
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

namespace fs = filesystem;  // 引用 filesystem 的命名空間
using namespace std;
using namespace Eigen;

NUC_Data::NUC_Data() {
}

// NUC calculation
vector<cv::Mat> NUC_Data::permute_data(const vector<cv::Mat>& raw_o, const vector<cv::Mat>& raw_c) {
    vector<cv::Mat> data;
    int count =0;
    for (size_t i = 0; i < raw_o.size(); i++) {
        for (size_t j = 0; j < raw_c.size(); j++) {
            // 計算 X = raw_o[i] - raw_c[j]
            cv::Mat X = raw_o[i] - raw_c[j];

            // 計算 y 和 z
            double y = cv::mean(raw_c[j])[0];
            double z = cv::mean(raw_o[i])[0] - y;

            // 定義 3 通道的矩陣 (307200, 3)
            cv::Mat d(IMAGE_WIDTH*IMAGE_HEIGHT, 1, CV_16SC3);

            for (int k = 0; k < IMAGE_WIDTH*IMAGE_HEIGHT; k++) {
                // 設定每個像素的值：第一通道為 X，第二通道為 0，第三通道為 z
                int16_t x_value = X.at<int16_t>(k, 0);
                d.at<cv::Vec3s>(k, 0) = cv::Vec3s(x_value, 0, static_cast<int16_t>(z));
            }

            // 將 d 加入 data
            data.push_back(d);
        }
    }

    return data;
}

void NUC_Data::start_nuc_thread(cv::Mat reg_data_clipped) {
    std::thread async_task([this, reg_data_clipped]() {
        // system("NUC_calculator.exe");  // 執行外部 exe
        //     if (fs::exists("nuc_param.bin")) {
        //         read_nuc_bin();
        //     }
        computeLinearRegressionParams(reg_data_clipped);
    });
    async_task.detach();
}

void NUC_Data::insert_nuc_data(vector<cv::Mat> open,vector<cv::Mat> close,vector<cv::Mat> open2,vector<cv::Mat> close2) {
    // 調用 permute_data 並將結果合併
    vector<cv::Mat> data_tmp1 = permute_data(open, close);
    vector<cv::Mat> data_tmp2 = permute_data(open2, close2);

    qDebug() << "tag: " << QDateTime::currentDateTime();
    qDebug() << "data size" << data_tmp1.size();
    qDebug() << "data size" << data_tmp1.size();

    reg_data.insert(reg_data.end(), data_tmp1.begin(), data_tmp1.end());
    reg_data.insert(reg_data.end(), data_tmp2.begin(), data_tmp2.end());
}

void NUC_Data::insert_nuc_data_onetemp(vector<cv::Mat> open,vector<cv::Mat> close) {
    // 調用 permute_data 並將結果合併
    vector<cv::Mat> data_tmp1 = permute_data(open, close);

    qDebug() << "tag: " << QDateTime::currentDateTime();
    qDebug() << "data size" << data_tmp1.size();

    reg_data.insert(reg_data.end(), data_tmp1.begin(), data_tmp1.end());
}

void NUC_Data::calculate_nuc_parameter() {

    qDebug() << "LinearRegression start: " << QDateTime::currentDateTime();
    qDebug() << "tag: " << QDateTime::currentDateTime();
    qDebug() << "reg_data size" << reg_data.size();
    progressPercentage = 23;

    // 將 reg_data 轉換成一個 3D 矩陣，並進行轉置
    int N = reg_data.size();
    int H = reg_data[0].rows;  // 307200
    int W = reg_data[0].cols;  // 1

    // 創建 reg_data_mat (N, H*W, 3)
    cv::Mat reg_data_mat(N, H * W, CV_16SC3);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < H * W; j++) {
            reg_data_mat.at<cv::Vec3s>(i, j) = reg_data[i].at<cv::Vec3s>(j, 0);
        }
    }
    progressPercentage = 26;

    // 轉換為 [H*W, N, 3]
    cv::Mat reg_data_transposed(H * W, N, CV_16SC3);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < H * W; j++) {
            reg_data_transposed.at<cv::Vec3s>(j, i) = reg_data_mat.at<cv::Vec3s>(i, j);
        }
    }
    progressPercentage = 30;

    // 裁剪範圍
    cv::Mat reg_data_clipped = reg_data_transposed.clone();
    cv::min(reg_data_clipped, 16383, reg_data_clipped);
    cv::max(reg_data_clipped, -16383, reg_data_clipped);

    // // 將數據展平並儲存
    // cv::Mat data_to_save = reg_data_clipped.reshape(1, 1);  // 將數據展平
    // data_to_save.convertTo(data_to_save, CV_64F); // 確保是浮點數類型
    // // 將數據保存到文件
    // ofstream output("reg_data.raw", ios::binary);
    // output.write(reinterpret_cast<const char*>(data_to_save.data), data_to_save.total() * data_to_save.elemSize());
    // output.close();

    cout << "Data saved successfully!" << endl;
    start_nuc_thread(reg_data_clipped);

}

vector<Vector3d> NUC_Data::computeLinearRegressionParams(const cv::Mat& reg_data_input) {
    vector<Vector3d> nuc_params;
    vector<uint16_t> parameter;

    // input mat [H*W, N, 3]
    CV_Assert(reg_data_input.type() == CV_16SC3);

    // do linear regression
    for (int n = 0; n < reg_data_input.rows; ++n) { // reg_data_input.rows = H*W

        progressPercentage = ((float)n / (float)reg_data_input.rows * 70.0) + 30.0;
        int imageSize = reg_data_input.cols;
        MatrixXd X(imageSize, 3);
        VectorXd y(imageSize);

        for (int i = 0; i < reg_data_input.cols; ++i) {
            cv::Vec3s pixel_values = reg_data_input.at<cv::Vec3s>(n, i);
            X(i, 0) = pixel_values[0];  // or mat.at<float> if CV_32F
            X(i, 1) = pixel_values[1];
            X(i, 2) = 1.0;
            y(i) = pixel_values[2];

        }
        Vector3d weights = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        nuc_params.push_back(weights);
        parameter.push_back(weights[2]);
    }

    // update last parameter with median filter
    cv::Mat parameterMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16S, parameter.data());
    cv::medianBlur(parameterMat, parameterMat, 3);
    vector<uint16_t> parameterVector(parameterMat.begin<int16_t>(), parameterMat.end<int16_t>());
    for (int i = 0; i < nuc_params.size(); i++) {
        nuc_params[i][2] = parameterVector[i];
    }

    // transform to file save version
    vector<int16_t> final_params;
    vector<int16_t> parameterA;
    vector<int16_t> parameterC;
    final_params.reserve(nuc_params.size() * 3);  // 3 values per row

    for (const auto& vec : nuc_params) {

        int16_t val0 = static_cast<int16_t>(round(vec[0] * 2048.0));
        int16_t val1 = static_cast<int16_t>(round(1.0 * 2048.0));
        int16_t val2 = static_cast<int16_t>(round(vec[2]));

        parameterA.push_back(val0);
        parameterC.push_back(val2);
        final_params.push_back(val0);
        final_params.push_back(val1);
        final_params.push_back(val2);
    }
    emit send_nuc(final_params, parameterA, parameterC);
    reg_data.clear();
    qDebug() << "LinearRegression end: " << QDateTime::currentDateTime();
    progressPercentage = 0;
    return nuc_params;
}

void NUC_Data::clear_unc_data() {
    reg_data.clear();
}

void NUC_Data::clear_tc_data() {
    reg_tc_data.clear();
}

// Radiometry calculation
vector<array<double, 10>> NUC_Data::permute_TC_data(const vector<cv::Mat>& raw_images, const vector<vector<uint16_t>>& vtp_list, double T_fixed) {
    vector<array<double, 10>> data;

    for (size_t i = 0; i < raw_images.size(); i++) {
        const cv::Mat& img = raw_images[i];
        const vector<uint16_t>& v_list = vtp_list[i];

        for (size_t k = 0; k < v_list.size(); k++) {
            double v = static_cast<double>(v_list[k]);
            double x = static_cast<double>(img.at<int16_t>(k));

            array<double, 10> row = {
                pow(x, 3),
                pow(x, 2) * v,
                x * pow(v, 2),
                pow(v, 3),
                pow(x, 2),
                x * v,
                pow(v, 2),
                x,
                v,
                T_fixed
            };
            data.push_back(row);
        }
    }
    return data;
}

cv::Mat NUC_Data::extractUniform64FromROI(const cv::Mat& img)
{
    int roi_size = 240;
    int grid_size = 8;
    int centerX = img.cols / 2;
    int centerY = img.rows / 2;
    int half = roi_size / 2;

    cv::Rect roi(centerX - half, centerY - half, roi_size, roi_size);
    cv::Mat roiImg = img(roi);

    int step = roi_size / grid_size;
    cv::Mat sampled(64, 1, CV_16S);
    int idx = 0;

    for(int i = 0; i < grid_size; i++) {
        for(int j = 0; j < grid_size; j++) {
            int y = i * step + step / 2;
            int x = j * step + step / 2;
            sampled.at<int16_t>(idx++) = roiImg.at<int16_t>(y, x);
        }
    }
    return sampled;
}

void NUC_Data::start_tc_thread(vector<cv::Vec<double, 10>> reg_data_clipped) {
    std::thread async_task([this, reg_data_clipped]() {
        // system("NUC_calculator.exe");  // 執行外部 exe
        //     if (fs::exists("nuc_param.bin")) {
        //         read_nuc_bin();
        //     }
        computeTCParams(reg_data_clipped);
    });
    async_task.detach();
}

void NUC_Data::insert_tc_data(const vector<cv::Mat>& raw_images, const vector<uint16_t>& vtp_values, double target_value)
{
    if(raw_images.size() != vtp_values.size()) {
        qWarning() << "[TC] raw_images size mismatch vtp_values!";
        return;
    }

    for(size_t i = 0; i < raw_images.size(); ++i) {
        cv::Mat raw = raw_images[i];
        cv::Mat pixels;

        // 取 uniform 64 samples (ROI 中心取樣)
        pixels = extractUniform64FromROI(raw);

        double v = static_cast<double>(vtp_values[i]);

        cv::Scalar mean_value = cv::mean(pixels);
        if(v == 0) {
            qDebug() << " U R 0";
            continue;
        }
        qDebug() << "mat_avg" << mean_value[0] << "| vtemp" << v;

        for(int k = 0; k < pixels.rows; ++k) {
            double x = static_cast<double>(pixels.at<int16_t>(k));
            cv::Vec<double, 10> d;
            d[0] = pow(x, 3);
            d[1] = pow(x, 2) * v;
            d[2] = x * pow(v, 2);
            d[3] = pow(v, 3);
            d[4] = pow(x, 2);
            d[5] = x * v;
            d[6] = pow(v, 2);
            d[7] = x;
            d[8] = v;
            d[9] = target_value;
            reg_tc_data.push_back(d);
        }
    }

    // qDebug() << "[TC] Inserted data size:" << reg_tc_data.size();
}

void NUC_Data::calculate_tc_parameter()
{

    start_tc_thread(reg_tc_data);

}

void NUC_Data::computeTCParams(vector<cv::Vec<double, 10>> reg_data_input)
{
    int N = static_cast<int>(reg_data_input.size());
    if (N == 0) {
        qWarning() << "[TC] No data. Abort.";
        return ;
    }

    const int dimX = 9;
    Eigen::MatrixXd X(N, dimX);
    Eigen::VectorXd Z(N);

    for (int i = 0; i < N; ++i) {
        const cv::Vec<double,10>& d = reg_data_input[i];
        for (int j = 0; j < dimX; ++j) {
            X(i, j) = d[j];
        }
        Z(i) = d[9];
    }

    // Solve linear regression: weights (9)


    // 1. Get the number of samples (N) and features (M=9)
    long NN = X.rows();
    long MM = X.cols();

    qDebug() << "X: " << NN << "x" << MM;
    for(int i = 0; i < MM; ++i) {
        qDebug() << X(0, i);
    }

    qDebug() << "Z : " << Z.size();
    qDebug() << "Z(0) : " << Z(0);

    // Construct the Design Matrix A (N x M+1)
    // Eigen::MatrixXd A(NN, MM + 1);
    // A.block(0, 0, NN, MM) = X;
    // A.col(MM).setOnes(); // Add bias term

    // Use ColPivHouseholderQR instead of BDCSVD
    // It is much faster and generates significantly less code
    Eigen::VectorXd tc_params = solveLinearRegressionRobust(X, Z);


    // Eigen::VectorXd tc_params = solveLinearRegression(X, Z);

    // Eigen::VectorXd weights = (X.transpose() * X).ldlt().solve(X.transpose() * Z);

    // Compute intercept (mean residual) to mimic Python LinearRegression intercept
    // Eigen::VectorXd residuals = Z - X * weights;
    // double intercept = residuals.mean();

    // // Build float params vector (a..i, j)
    vector<float> tc_params_float;
    tc_params_float.reserve(9);
    for (int i = 0; i < tc_params.size(); ++i){
        tc_params_float.push_back(static_cast<float>(tc_params[i]));
        qDebug() << i << " : " << tc_params[i];
    }

    // Emit preferred, accurate signal
    emit send_tc(tc_params_float);

    // clear buffer
    reg_tc_data.clear();

    qDebug() << "[TC] computed and emitted params (float size):" << tc_params_float.size();
}

Eigen::VectorXd NUC_Data::solveLinearRegressionRobust(const Eigen::MatrixXd& X_input, const Eigen::VectorXd& Z_input) {
    long N = X_input.rows();
    long M = X_input.cols();

    // 1. Compute Mean and Std Dev for each feature (column)
    Eigen::VectorXd mean = X_input.colwise().mean();
    Eigen::VectorXd std_dev(M);

    for (long i = 0; i < M; ++i) {
        // Compute standard deviation: sqrt( sum((x - mean)^2) / N )
        // Note: Using N or N-1 doesn't matter much for scaling, as long as consistent.
        // We use N here for simplicity.
        double variance = (X_input.col(i).array() - mean(i)).square().sum() / N;
        std_dev(i) = std::sqrt(variance);

        // Avoid division by zero if a column is constant
        if (std_dev(i) < 1e-12) {
            std_dev(i) = 1.0;
        }
    }

    // 2. Normalize the Input Matrix (Z-score normalization)
    Eigen::MatrixXd X_norm(N, M);
    for (long i = 0; i < M; ++i) {
        X_norm.col(i) = (X_input.col(i).array() - mean(i)) / std_dev(i);
    }

    // 3. Construct Design Matrix A_norm for the normalized data
    //    Add a column of 1s for the bias (intercept)
    Eigen::MatrixXd A_norm(N, M + 1);
    A_norm.block(0, 0, N, M) = X_norm;
    A_norm.col(M).setOnes();

    // 4. Solve using ColPivHouseholderQR
    //    This is robust and generates much less code than BDCSVD, fixing the "file too big" error.
    Eigen::VectorXd params_norm = A_norm.colPivHouseholderQr().solve(Z_input);

    // 5. Denormalize the parameters to get back to the original scale
    //    Model: y = w_norm * (x - mean)/std + b_norm
    //           y = (w_norm / std) * x - (w_norm * mean / std) + b_norm
    //
    //    Original Weights w = w_norm / std
    //    Original Bias b = b_norm - sum(w * mean)

    Eigen::VectorXd params_final(M + 1);

    // Calculate weights
    for (long i = 0; i < M; ++i) {
        params_final(i) = params_norm(i) / std_dev(i);
    }

    // Calculate intercept
    double bias_norm = params_norm(M);
    double adjustment = 0.0;
    for (long i = 0; i < M; ++i) {
        adjustment += params_final(i) * mean(i);
    }
    params_final(M) = bias_norm - adjustment;

    return params_final;
}

// read and write file
void NUC_Data::write_tc_bin_to_path(vector<float> tc_float, string path, string file_name)
{
    // Check if file exists
    if (fs::exists(path + "/" + file_name + ".bin")) {
        cout << "File exists. Removing: " << path << "/" + file_name + ".bin" << endl;
        fs::remove(path + "/" + file_name + ".bin");
    }

    ofstream bin_file(path + "/" +file_name + ".bin", ios::binary);
    if (bin_file.is_open()) {
        // 寫入數據到文件
        bin_file.write(reinterpret_cast<const char*>(tc_float.data()),
                       tc_float.size() * sizeof(float));
        bin_file.close();

        // 發送數據
        cout << "save finish!" << endl;
    } else {
        cerr << "Error: Unable to save tc_param_download.bin" << endl;
    }
}

vector<float> NUC_Data::read_tc_bin(string path, string file_name) {
    string full_path = path + "/" + file_name;

    if (!fs::exists(full_path)) {
        cerr << "Error: File does not exist: " << full_path << endl;
        return {};
    }

    ifstream bin_file(full_path, ios::binary);
    if (!bin_file.is_open()) {
        cerr << "Error: Unable to open file: " << full_path << endl;
        return {};
    }

    // Read full file size
    bin_file.seekg(0, ios::end);
    size_t file_size = bin_file.tellg();
    bin_file.seekg(0, ios::beg);

    if (file_size % sizeof(float) != 0) {
        cerr << "Warning: File size is not aligned to sizeof(float)!" << endl;
    }

    size_t element_count = file_size / sizeof(float);
    vector<float> tc_float(element_count);

    // Read binary data
    bin_file.read(reinterpret_cast<char*>(tc_float.data()), file_size);
    bin_file.close();

    cout << "Read finish! Elements: " << tc_float.size() << endl;
    return tc_float;
}

void NUC_Data::write_vtemp_raw(vector<uint16_t> vtemp, string path, string file_name) {
    // Check if file exists
    if (fs::exists(path + "/" + file_name + ".raw")) {
        cout << "File exists. Removing: " << path << "/" + file_name + ".raw" << endl;
        fs::remove(path + "/" + file_name + ".raw");
    }

    ofstream bin_file(path + "/" +file_name + ".raw", ios::binary);
    if (bin_file.is_open()) {
        // 寫入數據到文件
        bin_file.write(reinterpret_cast<const char*>(vtemp.data()),
                       vtemp.size() * sizeof(int16_t));
        bin_file.close();

        // 發送數據
        cout << "save finish!" << endl;
    } else {
        cerr << "Error: Unable to save " << file_name << ".raw" << endl;
    }
}

vector<uint16_t> NUC_Data::read_vtemp_raw(const string &path, const string &file_name) {
    string full_path = path + "/" + file_name;

    if (!fs::exists(full_path)) {
        cerr << "Error: File does not exist: " << full_path << endl;
        return {};
    }

    ifstream bin_file(full_path, ios::binary);
    if (!bin_file.is_open()) {
        cerr << "Error: Unable to open file: " << full_path << endl;
        return {};
    }

    // Get file size
    bin_file.seekg(0, ios::end);
    size_t file_size = bin_file.tellg();
    bin_file.seekg(0, ios::beg);

    // Check size alignment to uint16_t
    if (file_size % sizeof(uint16_t) != 0) {
        cerr << "Warning: File size is not aligned to 2 bytes!" << endl;
    }

    size_t element_count = file_size / sizeof(uint16_t);
    vector<uint16_t> vtemp(element_count);

    // Read the data
    bin_file.read(reinterpret_cast<char*>(vtemp.data()), file_size);
    bin_file.close();

    cout << "Read finish! Elements: " << vtemp.size() << endl;
    return vtemp;
}

vector<int16_t> NUC_Data::read_nuc_bin(string path, string file_name) {
    string full_path = path + "/" + file_name;
    vector<int16_t> nuc_params;
    if (!fs::exists(full_path)) {
        cerr << "File not found: " << full_path << endl;
        return nuc_params;  // return empty Mat
    }

    // 讀取 nuc_param.bin
    ifstream bin_file(full_path, ios::binary);
    if (bin_file.is_open()) {
        // 確定文件大小
        bin_file.seekg(0, ios::end);
        size_t file_size = bin_file.tellg();
        bin_file.seekg(0, ios::beg);

        // 創建一個足夠大的容器來存儲所有的 int16_t 數據
        nuc_params = vector<int16_t>(file_size / sizeof(int16_t));

        // 確保文件大小是 int16_t 數據的倍數
        if (file_size % sizeof(int16_t) != 0) {
            cerr << "Error: nuc_param.bin file size is not a multiple of int16_t" << endl;
            return nuc_params;
        }

        // 讀取數據到 nuc_params
        bin_file.read(reinterpret_cast<char*>(nuc_params.data()), file_size);
        bin_file.close();

        // qDebug() << "nuc_param.bin data read successfully!\n";
        // qDebug() << "Number of elements in nuc_param.bin:" << nuc_params.size() << "\n";



        // 發送數據
        // emit send_nuc(nuc_params);
        cout << "read finish!" << endl;
        return nuc_params;
    } else {
        cerr << "Error: Unable to open nuc_param.bin" << endl;
    }
    return nuc_params;  // return empty Mat
}

cv::Mat NUC_Data::read_raw_file(string path, string file_name, int width, int height) {
    string full_path = path + "/" + file_name;

    if (!fs::exists(full_path)) {
        cerr << "File not found: " << full_path << endl;
        return cv::Mat();  // return empty Mat
    }

    // Open file in binary mode
    ifstream input(full_path, ios::binary);
    if (!input.is_open()) {
        cerr << "Failed to open file: " << full_path << endl;
        return cv::Mat();
    }

    // Prepare flat buffer
    cv::Mat flat_mat(1, width * height, CV_16U);
    input.read(reinterpret_cast<char*>(flat_mat.data), width * height * sizeof(uint16_t));
    input.close();

    // Reshape to original image
    cv::Mat image = flat_mat.reshape(1, height);  // single channel
    // convert type to 16S
    image.convertTo(image, CV_16S);
    return image.clone();  // return a safe copy
}

void NUC_Data::write_nuc_bin(vector<int16_t> nuc_params) {
    // Check if file exists
    if (fs::exists("nuc_param_download.bin")) {
        cout << "File exists. Removing: " << "nuc_param_download.bin" << endl;
        fs::remove("nuc_param_download.bin");
    }

    ofstream bin_file("nuc_param_download.bin", ios::binary);
    if (bin_file.is_open()) {
        // 寫入數據到文件
        bin_file.write(reinterpret_cast<const char*>(nuc_params.data()),
                       nuc_params.size() * sizeof(int16_t));
        bin_file.close();

        // 發送數據
        cout << "save finish!" << endl;
    } else {
        cerr << "Error: Unable to save nuc_param_download.bin" << endl;
    }
}

void NUC_Data::write_nuc_bin_to_path(vector<int16_t> nuc_params, string path, string file_name) {
    // Check if file exists
    if (fs::exists(path + "/" + file_name + ".bin")) {
        cout << "File exists. Removing: " << path << "/" + file_name + ".bin" << endl;
        fs::remove(path + "/" + file_name + ".bin");
    }

    ofstream bin_file(path + "/" +file_name + ".bin", ios::binary);
    if (bin_file.is_open()) {
        // 寫入數據到文件
        bin_file.write(reinterpret_cast<const char*>(nuc_params.data()),
                       nuc_params.size() * sizeof(int16_t));
        bin_file.close();

        // 發送數據
        cout << "save finish!" << endl;
    } else {
        cerr << "Error: Unable to save nuc_param.bin" << endl;
    }
}

void NUC_Data::write_raw_to_path(QVector<cv::Mat> raw, string path, string file_name) {
    int file_count = 0;
    QDir rootDirRaw;
    rootDirRaw.setPath(QString::fromStdString(path));
    if (rootDirRaw.exists()) {
        // read files
        QStringList files = rootDirRaw.entryList(QDir::Files | QDir::NoDotAndDotDot);
        for (const QString &fileName : files) {
            if(fileName.contains(fileName))file_count++;
        }
    }

    for (int i = 0; i < raw.size(); i++) {
        // 將數據展平並儲存
        cv::Mat data_to_save = raw[i].reshape(1, 1);  // 將數據展平
        data_to_save.convertTo(data_to_save, CV_16U); // 確保是浮點數類型

        // // 確認文件是否存在
        // if (fs::exists(path + "/" + file_name + to_string(i) + ".raw")) {
        //     cout << "File exists. Removing: " << path + "/" + file_name + to_string(i) + ".raw"<< endl;
        //     fs::remove(path + "/" + file_name + to_string(i) + ".raw");
        // }

        // 將數據保存到文件
        ofstream output(path + "/" + file_name + to_string(file_count + i) + ".raw", ios::binary);
        output.write(reinterpret_cast<const char*>(data_to_save.data), data_to_save.total() * data_to_save.elemSize());
        output.close();
    }
}

vector<uint8_t> NUC_Data::read_binary_file(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file)
        throw runtime_error("Failed to open file: " + filename);

    return vector<uint8_t>(istreambuf_iterator<char>(file), {});
}

void NUC_Data::write_binary_file(const string& filename, const vector<uint8_t>& data) {
    ofstream file(filename, ios::binary);
    if (!file)
        throw runtime_error("Failed to write file: " + filename);

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

