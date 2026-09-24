#ifndef CLELAS_OPENCL_MEDIAN_FILTER_KERNEL_H
#define CLELAS_OPENCL_MEDIAN_FILTER_KERNEL_H

#include "runtime.h"

#include <cstdint>

namespace opencl {

// Median filter of the post-processing (Elas::median of the CPU version):
// horizontal 7-pixel median, then vertical 7-pixel median, of the valid disparities.
class MedianFilterKernel {
public:
    MedianFilterKernel(const Runtime& runtime, int32_t width, int32_t height);

    // Filters a disparity map (float[width * height]) in place.
    // With subsampling, the map is width/2 x height/2.
    void run(const cl::Buffer& disparity, bool subsampling);

private:
    const Runtime& runtime_;
    cl::Kernel horizontal_;
    cl::Kernel vertical_;
    cl::Buffer temp_;  // float[width * height], result of the horizontal step

    int32_t width_;
    int32_t height_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_MEDIAN_FILTER_KERNEL_H
