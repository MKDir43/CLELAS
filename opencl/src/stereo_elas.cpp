#include "stereo_elas.hpp"
#include "elas_opencl.h"

stereo::StereoElas::StereoElas(int width, int height, bool estimates_subpixel)
      : stereo::BaseStereo::BaseStereo(width, height), debug_intermediate_images_(false)
{
    clelas::Parameters param(clelas::setting::ROBOTICS);
    elas = new clelas::Processor(width, height, width, param, estimates_subpixel);
}

stereo::StereoElas::~StereoElas()
{
    delete elas;
}

void stereo::StereoElas::execute(cv::Mat left, cv::Mat right, float* dleft, float* dright)
{
    // resize input image to cv::Size(width,height)
    cv::resize(left, leftResize, cv::Size(width,height), cv::INTER_LINEAR);
    cv::resize(right, rightResize, cv::Size(width,height), cv::INTER_LINEAR);

    // bgr to gray convert
    cv::cvtColor(leftResize, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightResize, rightGray, cv::COLOR_BGR2GRAY);

    // process with elas
    elas->process(leftGray.data, rightGray.data, dleft, dright);

    // Save intermediate images if debug mode is enabled
    if (debug_intermediate_images_) {
        elas->saveIntermediateImages("debug");
    }
}

stereo::StereoElas* stereo::StereoElas::generate(int width, int height, bool estimates_subpixel)
{
    stereo::StereoElas* stereoElas = new stereo::StereoElas(width, height, estimates_subpixel);
    return stereoElas;
}
