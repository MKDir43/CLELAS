#include "median_filter_kernel.h"

namespace opencl {

MedianFilterKernel::MedianFilterKernel(const Runtime& runtime, int32_t width, int32_t height)
    : runtime_(runtime), width_(width), height_(height)
{
    const cl::Program program = runtime.buildProgram("median_filter.cl");
    horizontal_ = cl::Kernel(program, "medianHorizontal");
    vertical_ = cl::Kernel(program, "medianVertical");

    temp_ = cl::Buffer(runtime.context(), CL_MEM_READ_WRITE, sizeof(float) * width * height);
}

void MedianFilterKernel::run(const cl::Buffer& disparity, bool subsampling)
{
    KERNEL_LOG_START("median", " (%dx%d)\n", width_, height_);

    const int32_t d_width = subsampling ? width_ / 2 : width_;
    const int32_t d_height = subsampling ? height_ / 2 : height_;

    // The border of the first step stays 0, as in the CPU version.
    const cl::CommandQueue& queue = runtime_.queue();
    queue.enqueueFillBuffer(temp_, 0.0f, 0, sizeof(float) * d_width * d_height);
    queue.finish();

    horizontal_.setArg(0, disparity);
    horizontal_.setArg(1, temp_);
    horizontal_.setArg(2, d_width);
    horizontal_.setArg(3, d_height);
    runtime_.run(horizontal_);

    vertical_.setArg(0, temp_);
    vertical_.setArg(1, disparity);
    vertical_.setArg(2, d_width);
    vertical_.setArg(3, d_height);
    runtime_.run(vertical_);

    KERNEL_LOG_END("median");
}

} // namespace opencl
