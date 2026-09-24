#ifndef CLELAS_OPENCL_LR_CONSISTENCY_CHECK_KERNEL_H
#define CLELAS_OPENCL_LR_CONSISTENCY_CHECK_KERNEL_H

#include "runtime.h"

#include <cstdint>

namespace opencl {

// Invalidates disparities that disagree between the left and right maps.
class LRConsistencyCheckKernel {
public:
    explicit LRConsistencyCheckKernel(const Runtime& runtime);

    // input_left/right -> output_left/right: float disparity maps
    void run(const cl::Buffer& input_left, const cl::Buffer& input_right,
             const cl::Buffer& output_left, const cl::Buffer& output_right,
             int32_t width, int32_t height, int32_t pitch, int32_t lr_threshold,
             bool subsampling, bool estimates_subpixel);

private:
    const Runtime& runtime_;
    cl::Kernel kernel_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_LR_CONSISTENCY_CHECK_KERNEL_H
