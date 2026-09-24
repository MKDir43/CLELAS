#ifndef CLELAS_OPENCL_DISPARITY_PLANES_KERNEL_H
#define CLELAS_OPENCL_DISPARITY_PLANES_KERNEL_H

#include "device_buffers.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Fits a disparity plane through the three corners of every triangle.
class DisparityPlanesKernel {
public:
    explicit DisparityPlanesKernel(const Runtime& runtime);

    // triangles.c1..c3 -> triangles.t1a..t1c (left image), triangles.t2a..t2c (right image)
    void run(const DeviceSupportPoints& supports, DeviceTriangles& triangles, int32_t triangle_num);

private:
    const Runtime& runtime_;
    cl::Kernel kernel_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_DISPARITY_PLANES_KERNEL_H
