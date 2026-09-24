#ifndef CLELAS_OPENCL_COMPUTE_DISPARITY_KERNEL_H
#define CLELAS_OPENCL_COMPUTE_DISPARITY_KERNEL_H

#include "device_buffers.h"
#include "elas_parameter.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Dense disparity matching guided by the triangle planes and the disparity grid.
class ComputeDisparityKernel {
public:
    explicit ComputeDisparityKernel(const Runtime& runtime);

    // -> disparity: float[width * height]
    // The kernel works from the viewpoint of the image being matched: for the
    // right image, pass the right Sobel response as sobel_left, the left one as
    // sobel_right, and set right_image.
    void run(const DeviceSobel& sobel_left, const DeviceSobel& sobel_right,
             const DeviceSupportPoints& supports, int32_t support_num,
             const DeviceTriangles& triangles, int32_t triangle_num,
             const DeviceGrid& grid, const cl::Buffer& tri_map,
             const cl::Buffer& disparity, int32_t width, int32_t height,
             const clelas::Parameters& param, bool estimates_subpixel, bool right_image);

private:
    const Runtime& runtime_;
    cl::Kernel kernel_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_COMPUTE_DISPARITY_KERNEL_H
