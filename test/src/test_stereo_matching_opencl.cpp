#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include "util/util.hpp"
#include "stereo_elas.hpp"

// KITTI data paths
namespace kitti {
    const std::string leftDirPath = "data/testimages/kitti/left";
    const std::string rightDirPath = "data/testimages/kitti/right";
    const std::string correctLeftOutputDirPath = "data/testimages/kitti/lidar/left";
    const std::string correctRightOutputDirPath = "data/testimages/kitti/lidar/right";
}

class GlobalEnv : public ::testing::Environment {
public:
    virtual void SetUp() override {
        std::cout << "OpenCL ELAS test environment initialized" << std::endl;
    }
    virtual void TearDown() override {}
};
::testing::Environment* const global_env__ = ::testing::AddGlobalTestEnvironment(new GlobalEnv);

// Accuracy evaluation (checks whether processing completed correctly)
bool evaluateAccuracy(const cv::Mat& output, const cv::Mat& groundTruth,
                     double maxErrorPixel, double maxErrorRate, double maxInvalidRate) {

    cv::Mat outputImg16;
    output.convertTo(outputImg16, CV_16UC1, 256.0);

    cv::Mat correctImg16;
    if (groundTruth.type() != CV_16UC1) {
        groundTruth.convertTo(correctImg16, CV_16UC1);
    } else {
        correctImg16 = groundTruth;
    }

    // Exclude image edges
    const int edgeSize = 40;
    outputImg16.rowRange(0, edgeSize).setTo(cv::Scalar(0));
    outputImg16.rowRange(outputImg16.rows - edgeSize, outputImg16.rows).setTo(cv::Scalar(0));
    outputImg16.colRange(0, edgeSize).setTo(cv::Scalar(0));
    outputImg16.colRange(outputImg16.cols - edgeSize, outputImg16.cols).setTo(cv::Scalar(0));

    auto correctData = reinterpret_cast<uint16_t*>(correctImg16.data);
    auto outputData = reinterpret_cast<uint16_t*>(outputImg16.data);

    size_t totalPixels = outputImg16.cols * outputImg16.rows;
    long errorPixels = 0, invalidPixels = 0, validPixels = 0, nonZeroOutputPixels = 0;

    // First, check whether the output image has valid disparity values
    for (size_t i = 0; i < totalPixels; ++i) {
        if (outputData[i] > 0) {
            nonZeroOutputPixels++;
        }
    }

    // Case where the output image is almost entirely zero (processing may have failed)
    double nonZeroRate = static_cast<double>(nonZeroOutputPixels) / totalPixels;
    if (nonZeroRate < 0.1) {  // Fewer than 10% of the pixels are valid
        std::cout << "  Warning: valid pixel rate of the output is abnormally low: " << nonZeroRate
                  << " (" << (nonZeroRate * 100) << "%)" << std::endl;
        std::cout << "  OpenCL processing may not have run correctly" << std::endl;
        return false;  // Indicates processing failure
    }

    for (size_t i = 0; i < totalPixels; ++i) {
        if (correctData[i] == 0) continue;  // Ground truth is invalid

        validPixels++;
        if (outputData[i] == 0) {
            invalidPixels++;
        } else if (std::abs(outputData[i] - correctData[i]) > maxErrorPixel &&
                  static_cast<double>(std::abs(outputData[i] - correctData[i])) / correctData[i] > 0.05) {
            errorPixels++;
        }
    }

    double invalidRate = static_cast<double>(invalidPixels) / totalPixels;
    double errorRate = validPixels > 0 ? static_cast<double>(errorPixels) / validPixels : 0.0;

    std::cout << "  Accuracy - error rate: " << errorRate << " (threshold: " << maxErrorRate
              << "), invalid rate: " << invalidRate << " (threshold: " << maxInvalidRate
              << "), valid pixel rate: " << nonZeroRate << std::endl;

    EXPECT_LE(invalidRate, maxInvalidRate);
    EXPECT_LE(errorRate, maxErrorRate);

    return true;  // Indicates processing success
}

TEST(stereo_matching_opencl, case_kitti) {
    std::unique_ptr<stereo::StereoElas> stereoElas;
    auto fileNames = Util::getFileList(kitti::leftDirPath);
    ASSERT_GT(fileNames.size(), 0);

    int processedImages = 0;
    int successfulImages = 0;  // Number of images processed successfully

    for (const auto& name : fileNames) {
        SCOPED_TRACE("image file: " + name);

        // Check that the ground-truth images exist
        auto correctLeftPath = Util::pathCat(kitti::correctLeftOutputDirPath, name);
        auto correctRightPath = Util::pathCat(kitti::correctRightOutputDirPath, name);
        cv::Mat correctLeft = cv::imread(correctLeftPath, -1);
        cv::Mat correctRight = cv::imread(correctRightPath, -1);

        if (correctLeft.empty() || correctRight.empty()) {
            continue;  // Skip when no ground-truth image is available
        }

        std::cout << "Processing: " << name << std::endl;

        // Load input images
        auto leftPath = Util::pathCat(kitti::leftDirPath, name);
        auto rightPath = Util::pathCat(kitti::rightDirPath, name);
        cv::Mat leftRaw = cv::imread(leftPath, cv::IMREAD_COLOR);
        cv::Mat rightRaw = cv::imread(rightPath, cv::IMREAD_COLOR);
        cv::Mat leftGray = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(leftRaw.empty()) << "Failed to load left image: " << leftPath;
        ASSERT_FALSE(rightRaw.empty()) << "Failed to load right image: " << rightPath;
        ASSERT_FALSE(leftGray.empty()) << "Failed to load left grayscale image: " << leftPath;

        // Initialize StereoElas (only on the first iteration)
        if (!stereoElas) {
            stereoElas.reset(stereo::StereoElas::generate(leftGray.cols, leftGray.rows));
            ASSERT_NE(stereoElas, nullptr) << "Failed to initialize StereoElas";
        }

        // Run OpenCL ELAS processing
        cv::Mat leftDisp = leftRaw.clone();
        cv::Mat rightDisp = rightRaw.clone();
        std::vector<float> leftOutput(leftRaw.rows * leftRaw.cols);
        std::vector<float> rightOutput(rightRaw.rows * rightRaw.cols);

        std::cout << "  Running OpenCL ELAS processing..." << std::endl;
        stereoElas->execute(leftDisp, rightDisp, leftOutput.data(), rightOutput.data());
        std::cout << "  OpenCL ELAS processing done" << std::endl;

        // Convert the results to Mat format
        cv::Mat leftResult(leftRaw.rows, leftRaw.cols, CV_32F, leftOutput.data());
        cv::Mat rightResult(rightRaw.rows, rightRaw.cols, CV_32F, rightOutput.data());

        // Check sizes
        ASSERT_EQ(leftResult.size(), correctLeft.size());
        ASSERT_EQ(rightResult.size(), correctRight.size());

        // Accuracy evaluation (checks whether processing completed correctly)
        std::cout << "  Left image accuracy:" << std::endl;
        bool leftSuccess = evaluateAccuracy(leftResult, correctLeft, 18 * 256, 0.08, 0.5);

        std::cout << "  Right image accuracy:" << std::endl;
        bool rightSuccess = evaluateAccuracy(rightResult, correctRight, 60 * 256, 0.035, 0.5);

        processedImages++;

        // Count only when both images were processed successfully
        if (leftSuccess && rightSuccess) {
            successfulImages++;
        }
    }

    ASSERT_GT(processedImages, 0) << "No test images were found";

    // Important: check the number of images processed successfully
    std::cout << "Result: " << processedImages << " images, of which " << successfulImages << " processed successfully" << std::endl;

    // Fail the test if not all images were processed successfully
    ASSERT_EQ(successfulImages, processedImages)
        << "OpenCL processing is not working correctly. Succeeded: " << successfulImages
        << "/" << processedImages << ". The OpenCL kernels may have failed to initialize, or the file paths may be wrong.";
}
