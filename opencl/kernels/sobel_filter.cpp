#include "sobel_filter_kernel.h"

namespace opencl {

SobelFilterKernel::SobelFilterKernel(const Runtime& runtime)
    : runtime_(runtime)
{
    const cl::Program program = runtime.buildProgram("sobel_filter.cl");
    sobel_du_ = cl::Kernel(program, "sobelDu3x3");
    sobel_dv_ = cl::Kernel(program, "sobelDv3x3");
}

void SobelFilterKernel::run(const cl::Buffer& image, DeviceSobel& sobel,
                            int32_t width, int32_t height, int32_t pitch)
{
    runFilter(sobel_du_, image, sobel.h, width, height, pitch);
    runFilter(sobel_dv_, image, sobel.v, width, height, pitch);
}

void SobelFilterKernel::runFilter(cl::Kernel& kernel, const cl::Buffer& input, const cl::Buffer& output,
                                  int32_t width, int32_t height, int32_t pitch)
{
    kernel.setArg(0, input);
    kernel.setArg(1, output);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, pitch);
    runtime_.run(kernel);
}

} // namespace opencl
