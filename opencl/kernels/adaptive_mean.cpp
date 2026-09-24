#include "adaptive_mean_kernel.h"

namespace opencl {

AdaptiveMeanKernel::AdaptiveMeanKernel(const Runtime& runtime, int32_t width, int32_t height)
    : runtime_(runtime), width_(width), height_(height)
{
    const cl::Program program = runtime.buildProgram("adaptive_mean.cl");
    prepare_ = cl::Kernel(program, "adaptiveMeanPrepare");
    horizontal_ = cl::Kernel(program, "adaptiveMeanHorizontal");
    vertical_ = cl::Kernel(program, "adaptiveMeanVertical");

    const size_t bytes = sizeof(float) * width * height;
    copy_ = cl::Buffer(runtime.context(), CL_MEM_READ_WRITE, bytes);
    tmp_ = cl::Buffer(runtime.context(), CL_MEM_READ_WRITE, bytes);
}

void AdaptiveMeanKernel::run(const cl::Buffer& disparity, bool subsampling)
{
    KERNEL_LOG_START("adaptiveMean", " (%dx%d)\n", width_, height_);

    const int32_t d_width = subsampling ? width_ / 2 : width_;
    const int32_t d_height = subsampling ? height_ / 2 : height_;
    const int32_t taps = subsampling ? 4 : 8;

    prepare_.setArg(0, disparity);
    prepare_.setArg(1, copy_);
    prepare_.setArg(2, tmp_);
    prepare_.setArg(3, d_width * d_height);
    runtime_.run(prepare_);

    horizontal_.setArg(0, copy_);
    horizontal_.setArg(1, tmp_);
    horizontal_.setArg(2, d_width);
    horizontal_.setArg(3, d_height);
    horizontal_.setArg(4, taps);
    runtime_.run(horizontal_);

    vertical_.setArg(0, tmp_);
    vertical_.setArg(1, disparity);
    vertical_.setArg(2, d_width);
    vertical_.setArg(3, d_height);
    vertical_.setArg(4, taps);
    runtime_.run(vertical_);

    KERNEL_LOG_END("adaptiveMean");
}

} // namespace opencl
