#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "stereo_elas.hpp"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

void execute_process(
    const std::string& indir_left, const std::string& indir_right,
    const std::string& outdir_left, const std::string& outdir_right,
    const int elas_size_width, const int elas_size_height,
    const bool output_csv, const bool estimates_subpixel, const bool disparity_color,
    const bool debug_intermediate_images)
{
    const fs::path leftPath(indir_left);
    const fs::path rightPath(indir_right);
    const fs::path leftOutPath(outdir_left);
    const fs::path rightOutPath(outdir_right);

    // Ensure output directories exist
    if (!fs::exists(leftOutPath)) {
        fs::create_directories(leftOutPath);
    }
    if (!fs::exists(rightOutPath)) {
        fs::create_directories(rightOutPath);
    }

    // get input filenames.
    std::vector<std::string> fileNames;
    try {
        for (const auto& e : boost::make_iterator_range(fs::directory_iterator(leftPath))) {
            if (fs::is_directory(e)) continue;
            std::string entryName = e.path().filename().string();
            fileNames.push_back(entryName);
        }
    } catch (fs::filesystem_error& ex) {
        std::cout << ex.what() << std::endl;
    }

    if (fileNames.empty()) {
        std::cout << "ERROR: Image files not found in left directory." << std::endl;
        exit(-1);
    }

    std::sort(fileNames.begin(), fileNames.end());
    std::cout << "Input file list:" << std::endl;
    for (const auto& name : fileNames) {
        std::cout << name << std::endl;
    }

    // Initialize ELAS
    stereo::StereoElas* stereoElas = stereo::StereoElas::generate(elas_size_width, elas_size_height, estimates_subpixel);

    if (debug_intermediate_images) {
        stereoElas->setDebugIntermediateImages(true);
    }

    // Disparity vectors sized for ELAS output dimensions
    std::vector<float> disparityleft(elas_size_width * elas_size_height);
    std::vector<float> disparityright(elas_size_width * elas_size_height);

    for (const auto& name : fileNames) {
        std::cout << "Processing " << name << std::endl;
        fs::path fileName(name);
        const fs::path leftFilePath = leftPath / fileName;
        const fs::path rightFilePath = rightPath / fileName;

        // load images
        auto imgleft = cv::imread(leftFilePath.string(), cv::IMREAD_COLOR);
        auto imgright = cv::imread(rightFilePath.string(), cv::IMREAD_COLOR);

        // Check if images are loaded
        if (imgleft.empty() || imgright.empty()) {
            std::cerr << "ERROR: Failed to load images!" << std::endl;
            std::cerr << "  Left: " << leftFilePath.string() << std::endl;
            std::cerr << "  Right: " << rightFilePath.string() << std::endl;
            continue;
        }

        // check image size.
        if (imgleft.cols <= 0 || imgleft.rows <= 0 ||
            imgright.cols <= 0 || imgright.rows <= 0 ||
            imgleft.cols != imgright.cols || imgleft.rows != imgright.rows) {
            std::cout << "ERROR: Images must be of same size" << std::endl;
            exit(-1);
        }

        // process disparity.
        cv::Mat d_imgleft = imgleft.clone();
        cv::Mat d_imgright = imgright.clone();

        stereoElas->execute(d_imgleft, d_imgright, disparityleft.data(), disparityright.data());

        // write images (disparity is at ELAS dimensions)
        cv::Mat disparityleft_mat(elas_size_height, elas_size_width, CV_32F, disparityleft.data());
        cv::Mat disparityright_mat(elas_size_height, elas_size_width, CV_32F, disparityright.data());

        const fs::path leftOutFilePath = leftOutPath / fileName;
        const fs::path rightOutFilePath = rightOutPath / fileName;

        if (disparity_color) {
            cv::Mat disparityleft_u8;
            cv::Mat disparityright_u8;
            disparityleft_mat.convertTo(disparityleft_u8, CV_8U);
            disparityright_mat.convertTo(disparityright_u8, CV_8U);

            cv::Mat disparityleft_colormap;
            cv::Mat disparityright_colormap;
            cv::applyColorMap(disparityleft_u8, disparityleft_colormap, cv::COLORMAP_JET);
            cv::applyColorMap(disparityright_u8, disparityright_colormap, cv::COLORMAP_JET);
            cv::imwrite(leftOutFilePath.string(), disparityleft_colormap);
            cv::imwrite(rightOutFilePath.string(), disparityright_colormap);
        } else {
            cv::imwrite(leftOutFilePath.string(), disparityleft_mat);
            cv::imwrite(rightOutFilePath.string(), disparityright_mat);
        }

        // output csv
        if (output_csv) {
            const fs::path leftCsvPath(outdir_left);
            const fs::path rightCsvPath(outdir_right);
            fileName.replace_extension("csv");
            const fs::path leftCsvFilePath = leftCsvPath / fileName;
            const fs::path rightCsvFilePath = rightCsvPath / fileName;
            std::ofstream ofs_l(leftCsvFilePath.string());
            std::ofstream ofs_r(rightCsvFilePath.string());
            ofs_l << "x" << "," << "y" << "," << "disparity" << "," << std::endl;
            ofs_r << "x" << "," << "y" << "," << "disparity" << "," << std::endl;
            for (int y = 0; y < elas_size_height; y++) {
                for (int x = 0; x < elas_size_width; x++) {
                    float disparity_l = disparityleft_mat.at<float>(y, x);
                    float disparity_r = disparityright_mat.at<float>(y, x);
                    if (disparity_l >= 0) ofs_l << x << "," << y << "," << disparity_l << "," << std::endl;
                    if (disparity_r >= 0) ofs_r << x << "," << y << "," << disparity_r << "," << std::endl;
                }
            }
        }
    }
    delete stereoElas;
}

int main(int argc, char** argv) {
    po::positional_options_description pos_options;
    po::options_description io_options("input/output");
    po::variables_map values;

    std::string indir_left;
    std::string indir_right;
    std::string outdir_left;
    std::string outdir_right;
    int elas_size_width;
    int elas_size_height;
    bool output_csv;
    bool estimates_subpixel;
    bool disparity_color = 1;
    bool debug_intermediate_images = 0;

    try {
        pos_options.add("inputDirectoryLeft", 1)
            .add("inputDirectoryRight", 1)
            .add("outputDirectoryLeft", 1)
            .add("outputDirectoryRight", 1);
        io_options.add_options()("help,H", "help message")
            ("inputDirectoryLeft", po::value<std::string>(&indir_left)->required(), "directory to input left rectify image")
            ("inputDirectoryRight", po::value<std::string>(&indir_right)->required(), "directory to input right rectify image")
            ("outputDirectoryLeft", po::value<std::string>(&outdir_left)->required(), "directory to output left elas image")
            ("outputDirectoryRight", po::value<std::string>(&outdir_right)->required(), "directory to output right elas image")
            ("sizeWidth", po::value<int>(&elas_size_width)->default_value(1856), "width size of elas")
            ("sizeHeight", po::value<int>(&elas_size_height)->default_value(384), "height size of elas")
            ("outputCsv", po::value<bool>(&output_csv)->default_value(1), "output disparity as csv file")
            ("subpixel", po::value<bool>(&estimates_subpixel)->default_value(1), "enable subpixel estimates")
            ("disparityColor", po::value<bool>(&disparity_color)->default_value(1), "output of disparity image in color")
            ("debugIntermediateImages", po::value<bool>(&debug_intermediate_images)->default_value(0), "save intermediate images for debugging");

        po::store(po::command_line_parser{argc, argv}
                      .options(io_options)
                      .positional(pos_options)
                      .run(),
            values);

        po::notify(values);
    } catch (po::error& e) {
        std::cout << e.what() << std::endl;
        std::cout << "usage: " << argv[0] << " inputDirectoryLeft inputDirectoryRight outputDirectoryLeft outputDirectoryRight"
                  << " --sizeWidth 1856 --sizeHeight 384 --outputCsv 1 --subpixel 1 --disparityColor 1" << std::endl;
        return 1;
    }

    try {
        execute_process(indir_left, indir_right, outdir_left, outdir_right, elas_size_width, elas_size_height,
                        output_csv, estimates_subpixel, disparity_color, debug_intermediate_images);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
