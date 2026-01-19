#ifndef NUC_H
#define NUC_H
#include <vector>
#include <QFile>
#include <QDataStream>
#include <QProgressDialog>
#include <QApplication>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <QDir>
#include <QString>
#include <Eigen>
using namespace std;
using namespace Eigen;
//nuc's data process and calculate and send nuc parameter
class NUC_Data: public QObject {
    Q_OBJECT
public:
    explicit NUC_Data();
    ~NUC_Data() = default;

    float progressPercentage = 0.0;

    // NUC calculate
    void insert_nuc_data(vector<cv::Mat> open,vector<cv::Mat> close,vector<cv::Mat> open2,vector<cv::Mat> close2);
    void insert_nuc_data_onetemp(vector<cv::Mat> open,vector<cv::Mat> close);
    void calculate_nuc_parameter();
    void clear_unc_data();

    // TC calcylate
    void insert_tc_data(const vector<cv::Mat>& raw_images, const vector<uint16_t>& vtp_values, double target_value);
    void calculate_tc_parameter();
    void clear_tc_data();


    // read and write file
    vector<int16_t> read_nuc_bin(string path, string file_name);
    cv::Mat read_raw_file(string path, string file_name, int width, int height);
    void write_nuc_bin(vector<int16_t> nuc_params);
    void write_nuc_bin_to_path(vector<int16_t> nuc_params, string path, string file_name);
    void write_raw_to_path(QVector<cv::Mat> raw, string path, string file_name);
    vector<uint8_t> read_binary_file(const string& filename);
    void write_binary_file(const string& filename, const vector<uint8_t>& data);
    void write_tc_bin_to_path(vector<float> tc_float, string path, string file_name);
    vector<float> read_tc_bin(string path, string file_name);
    void write_vtemp_raw(vector<uint16_t> vtemp, string path, string file_name);
    vector<uint16_t> read_vtemp_raw(const string &path, const string &file_name);

signals:
    void send_nuc(vector<int16_t> nuc_params, vector<int16_t> a_params, vector<int16_t> c_params);
    void send_tc(vector<float> params);
private:
    // NUC
    vector<cv::Mat> permute_data(const vector<cv::Mat>& raw_o, const vector<cv::Mat> & raw_c);
    void start_nuc_thread(cv::Mat reg_data_clipped);
    vector<Vector3d> computeLinearRegressionParams(const cv::Mat& reg_data);
    vector<cv::Mat> reg_data;

    // TC
    vector<array<double, 10>> permute_TC_data(const vector<cv::Mat>& raw_images, const vector<vector<uint16_t>>& vtp_list, double T_fixed);
    cv::Mat extractUniform64FromROI(const cv::Mat& img);
    void start_tc_thread(vector<cv::Vec<double, 10>> reg_data_clipped);
    void computeTCParams(vector<cv::Vec<double, 10>> reg_data_input);
    Eigen::VectorXd solveLinearRegressionRobust(const Eigen::MatrixXd& X_input, const Eigen::VectorXd& Z_input);
    vector<cv::Vec<double, 10>> reg_tc_data;
};
// save data open and close
struct data {
    vector<cv::Mat> open;
    vector<cv::Mat> close;
};

// TC calculte
struct TC_Sample {
    double x, v, T;
};

#endif // NUC_H
