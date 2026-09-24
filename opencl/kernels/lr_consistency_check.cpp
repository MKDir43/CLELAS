#include "lr_consistency_check_kernel.h"

namespace opencl {

LRConsistencyCheckKernel::LRConsistencyCheckKernel(const Runtime& runtime)
    : runtime_(runtime)
{
    kernel_ = cl::Kernel(runtime.buildProgram("lr_consistency_check.cl"), "leftRightConsistencyCheck");
}

void LRConsistencyCheckKernel::run(const cl::Buffer& input_left, const cl::Buffer& input_right,
                                   const cl::Buffer& output_left, const cl::Buffer& output_right,
                                   int32_t width, int32_t height, int32_t pitch, int32_t lr_threshold,
                                   bool subsampling, bool estimates_subpixel)
{
    kernel_.setArg(0, input_left);
    kernel_.setArg(1, input_right);
    kernel_.setArg(2, output_left);
    kernel_.setArg(3, output_right);
    kernel_.setArg(4, width);
    kernel_.setArg(5, height);
    kernel_.setArg(6, pitch);
    kernel_.setArg(7, lr_threshold);
    kernel_.setArg(8, int(subsampling));
    kernel_.setArg(9, int(estimates_subpixel));
    runtime_.run(kernel_);
}

} // namespace opencl
