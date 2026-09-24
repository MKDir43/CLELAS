#ifndef BASE_STEREO_H
#define BASE_STEREO_H

#include <opencv2/opencv.hpp>

namespace stereo
{

class BaseStereo
{
protected:
    int width;
    int height;

public:
    BaseStereo(int width_, int height_) : width{width_}, height{height_} {}
    virtual void execute(cv::Mat left, cv::Mat right, float* dleft, float* dright) = 0;
};
} // namespace stereo

#endif
