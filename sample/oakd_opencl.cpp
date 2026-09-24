#include <iostream>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>
#include "stereo_elas.hpp"

int main() {
    // Build the OAK-D pipeline
    dai::Pipeline pipeline;

    // Configure the mono cameras (left and right)
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();

    monoLeft->setCamera("left");
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoRight->setCamera("right");
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);

    // Obtain rectified images from the StereoDepth node
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
    stereo->setRectifyEdgeFillColor(0);  // Fill edges with black

    // Configure XLinkOut
    auto xoutLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutRight = pipeline.create<dai::node::XLinkOut>();
    xoutLeft->setStreamName("rectifiedLeft");
    xoutRight->setStreamName("rectifiedRight");

    // Link nodes
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->rectifiedLeft.link(xoutLeft->input);
    stereo->rectifiedRight.link(xoutRight->input);

    // Connect the device
    dai::Device device(pipeline);
    auto leftQueue = device.getOutputQueue("rectifiedLeft", 4, false);
    auto rightQueue = device.getOutputQueue("rectifiedRight", 4, false);

    // Initialize ELAS
    const int width = 1280;
    const int height = 720;
    stereo::StereoElas* stereoElas = stereo::StereoElas::generate(width, height, true);
    std::vector<float> disparityLeft(width * height);
    std::vector<float> disparityRight(width * height);

    // Variables for saving
    cv::Mat saveLeft, saveRight, saveDispColor, saveDispRaw;
    bool showNotify = false;
    int notifyCounter = 0;
    std::string notifyMsg;
    cv::Scalar notifyColor;

    std::cout << "Press 'c' to save images, 'q' to quit" << std::endl;

    while (true) {
        auto leftFrame = leftQueue->get<dai::ImgFrame>();
        auto rightFrame = rightQueue->get<dai::ImgFrame>();

        cv::Mat leftMat = leftFrame->getCvFrame();
        cv::Mat rightMat = rightFrame->getCvFrame();

        // Compute disparity
        cv::Mat leftColor, rightColor;
        cv::cvtColor(leftMat, leftColor, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rightMat, rightColor, cv::COLOR_GRAY2BGR);

        stereoElas->execute(leftColor, rightColor, disparityLeft.data(), disparityRight.data());

        // Visualize disparity
        cv::Mat dispMat(height, width, CV_32F, disparityLeft.data());
        cv::Mat dispU8;
        dispMat.convertTo(dispU8, CV_8U);
        cv::Mat dispColor;
        cv::applyColorMap(dispU8, dispColor, cv::COLORMAP_JET);

        // Copy for saving
        saveLeft = leftMat.clone();
        saveRight = rightMat.clone();
        saveDispColor = dispColor.clone();
        saveDispRaw = dispMat.clone();

        // Create the display image
        cv::Mat displayDisp = dispColor.clone();

        // Show notification
        if (showNotify && notifyCounter > 0) {
            cv::putText(displayDisp, notifyMsg, cv::Point(displayDisp.cols/2 - 100, displayDisp.rows/2),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, notifyColor, 4);
            notifyCounter--;
            if (notifyCounter == 0) showNotify = false;
        }

        cv::imshow("Left", leftMat);
        cv::imshow("Disparity", displayDisp);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == 'c') {
            // Save images
            bool ok1 = cv::imwrite("left_image.png", saveLeft);
            bool ok2 = cv::imwrite("right_image.png", saveRight);
            bool ok3 = cv::imwrite("disparity_color.png", saveDispColor);
            bool ok4 = cv::imwrite("disparity_raw.png", saveDispRaw);

            showNotify = true;
            notifyCounter = 15;  // Show for about 0.5 s (assuming 30 fps)
            if (ok1 && ok2 && ok3 && ok4) {
                notifyMsg = "SAVED!";
                notifyColor = cv::Scalar(0, 255, 0);
            } else {
                notifyMsg = "SAVE FAILED!";
                notifyColor = cv::Scalar(0, 0, 255);
            }
        }
    }

    delete stereoElas;
    return 0;
}
