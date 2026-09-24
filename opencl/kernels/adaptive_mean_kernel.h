#ifndef CLELAS_OPENCL_ADAPTIVE_MEAN_KERNEL_H
#define CLELAS_OPENCL_ADAPTIVE_MEAN_KERNEL_H

#include "runtime.h"

#include <cstdint>

namespace opencl {

// Adaptive mean filter of the post-processing (Elas::adaptiveMean of the CPU version):
// a bilateral filter, horizontal pass then vertical pass.
class AdaptiveMeanKernel {
public:
    AdaptiveMeanKernel(const Runtime& runtime, int32_t width, int32_t height);

    // Filters a disparity map (float[width * height]) in place.
    // With subsampling, the map is width/2 x height/2 and the filter 4 pixels wide.
    void run(const cl::Buffer& disparity, bool subsampling);

private:
    const Runtime& runtime_;
    cl::Kernel prepare_;
    cl::Kernel horizontal_;
    cl::Kernel vertical_;
    cl::Buffer copy_;  // float[width * height], input with invalid disparities set to -10
    cl::Buffer tmp_;   // float[width * height], result of the horizontal pass

    int32_t width_;
    int32_t height_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_ADAPTIVE_MEAN_KERNEL_H
