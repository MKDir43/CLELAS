#ifndef CLELAS_OPENCL_SOBEL_FILTER_KERNEL_H
#define CLELAS_OPENCL_SOBEL_FILTER_KERNEL_H

#include "device_buffers.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// 3x3 Sobel filters giving the horizontal and vertical image gradients.
class SobelFilterKernel {
public:
    explicit SobelFilterKernel(const Runtime& runtime);

    // image: uint8_t[pitch * height] -> sobel.h, sobel.v (rows `width` apart)
    void run(const cl::Buffer& image, DeviceSobel& sobel,
             int32_t width, int32_t height, int32_t pitch);

private:
    void runFilter(cl::Kernel& kernel, const cl::Buffer& input, const cl::Buffer& output,
                   int32_t width, int32_t height, int32_t pitch);

    const Runtime& runtime_;
    cl::Kernel sobel_du_;
    cl::Kernel sobel_dv_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_SOBEL_FILTER_KERNEL_H
