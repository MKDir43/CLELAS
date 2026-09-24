#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <gtest/gtest.h>

#include "util/util.hpp"
#include "elas.h"

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
        std::cout << "CPU ELAS test environment initialized" << std::endl;
    }
    virtual void TearDown() override {}
};
::testing::Environment* const global_env__ = ::testing::AddGlobalTestEnvironment(new GlobalEnv);

// Accuracy evaluation
void evaluateAccuracy(const cv::Mat& output, const cv::Mat& groundTruth,
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
    long errorPixels = 0, invalidPixels = 0, validPixels = 0;

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
              << "), invalid rate: " << invalidRate << " (threshold: " << maxInvalidRate << ")" << std::endl;

    EXPECT_LE(invalidRate, maxInvalidRate);
    EXPECT_LE(errorRate, maxErrorRate);
}

TEST(stereo_matching_cpu, case_kitti) {
    std::unique_ptr<Elas> elas;
    auto fileNames = Util::getFileList(kitti::leftDirPath);
    ASSERT_GT(fileNames.size(), 0);

    int processedImages = 0;

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
        cv::Mat leftGray = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
        cv::Mat rightGray = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(leftGray.empty()) << "Failed to load left image: " << leftPath;
        ASSERT_FALSE(rightGray.empty()) << "Failed to load right image: " << rightPath;

        // Initialize ELAS (only on the first iteration)
        if (!elas) {
            Elas::parameters param(Elas::MIDDLEBURY);
            param.postprocess_only_left = false;
            elas = std::make_unique<Elas>(leftGray.cols, leftGray.rows, param);
        }

        // Run ELAS processing
        std::vector<float> leftOutput(leftGray.rows * leftGray.cols);
        std::vector<float> rightOutput(rightGray.rows * rightGray.cols);
        elas->process(leftGray.data, rightGray.data, leftOutput.data(), rightOutput.data());

        // Convert the results to Mat format
        cv::Mat leftResult(leftGray.rows, leftGray.cols, CV_32F, leftOutput.data());
        cv::Mat rightResult(rightGray.rows, rightGray.cols, CV_32F, rightOutput.data());

        // Check sizes
        ASSERT_EQ(leftResult.size(), correctLeft.size());
        ASSERT_EQ(rightResult.size(), correctRight.size());

        // Accuracy evaluation
        std::cout << "  Left image accuracy:" << std::endl;
        evaluateAccuracy(leftResult, correctLeft, 18 * 256, 0.08, 0.5);

        std::cout << "  Right image accuracy:" << std::endl;
        evaluateAccuracy(rightResult, correctRight, 60 * 256, 0.035, 0.5);

        processedImages++;
    }

    ASSERT_GT(processedImages, 0) << "No test images were found";
    std::cout << "Test completed on " << processedImages << " images" << std::endl;
}
