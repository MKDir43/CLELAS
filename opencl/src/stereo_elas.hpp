#pragma once

#include <opencv2/opencv.hpp>
#include "base_stereo.hpp"
#include "elas_opencl.h"

namespace stereo
{

class StereoElas : public BaseStereo
{
private:
    clelas::Processor* elas;
    StereoElas(int width, int height, bool estimates_subpixel);
    cv::Mat leftResize, rightResize;
    cv::Mat leftGray, rightGray;
    bool debug_intermediate_images_;

public:
    ~StereoElas();
    void execute(cv::Mat left, cv::Mat right, float* dleft, float* dright);
    static StereoElas* generate(int width, int height, bool estimates_subpixel=0);
    void setDebugIntermediateImages(bool enable) { debug_intermediate_images_ = enable; }
};
} // namespace stereo
