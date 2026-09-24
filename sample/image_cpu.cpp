#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "elas.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0] << " input_L input_R output_L output_R" << std::endl;
        exit(-1);
    }

    auto infile_left = argv[1];
    auto infile_right = argv[2];
    auto outfile_left = argv[3];
    auto outfile_right = argv[4];

    // load images
    auto image_left = cv::imread(infile_left, cv::IMREAD_GRAYSCALE);
    auto image_right = cv::imread(infile_right, cv::IMREAD_GRAYSCALE);

    // check image size.
    if (image_left.cols <= 0 || image_left.rows <= 0 ||
        image_right.cols <= 0 || image_right.rows <= 0 ||
        image_left.cols != image_right.cols || image_left.rows != image_right.rows) {
        std::cout << "ERROR: Images must be of same size" << std::endl;
        exit(-1);
    }

    cv::Mat disp_left(cv::Size(image_left.cols, image_left.rows), CV_32F);
    cv::Mat disp_right(cv::Size(image_left.cols, image_left.rows), CV_32F);

    // process disparity.
    Elas::parameters param;
    param.postprocess_only_left = false;
    param.add_corners = true;

    Elas elas(image_left.cols, image_left.rows, param);
    elas.process(image_left.data, image_right.data,
                 disp_left.ptr<float>(), disp_right.ptr<float>());

    // write images.
    cv::Mat disp_left_uint16;
    cv::Mat disp_right_uint16;
    disp_left.convertTo(disp_left_uint16, CV_16U, 256);
    disp_right.convertTo(disp_right_uint16, CV_16U, 256);

    // Ensure output directories exist
    fs::path left_path(outfile_left);
    fs::path right_path(outfile_right);
    if (left_path.has_parent_path() && !fs::exists(left_path.parent_path())) {
        fs::create_directories(left_path.parent_path());
    }
    if (right_path.has_parent_path() && !fs::exists(right_path.parent_path())) {
        fs::create_directories(right_path.parent_path());
    }

    cv::imwrite(outfile_left, disp_left_uint16);
    cv::imwrite(outfile_right, disp_right_uint16);

    return 0;
}
