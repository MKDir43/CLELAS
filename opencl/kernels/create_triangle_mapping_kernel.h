#ifndef CLELAS_OPENCL_CREATE_TRIANGLE_MAPPING_KERNEL_H
#define CLELAS_OPENCL_CREATE_TRIANGLE_MAPPING_KERNEL_H

#include "device_buffers.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Rasterizes the triangles into a per-pixel triangle map.
class CreateTriangleMappingKernel {
public:
    explicit CreateTriangleMappingKernel(const Runtime& runtime);

    // triangles -> tri_map: uint16_t[width * height], triangle index + 1 (0 = no triangle)
    void run(const DeviceSupportPoints& supports, int32_t support_num,
             const DeviceTriangles& triangles, int32_t triangle_num,
             const cl::Buffer& tri_map, int32_t width, int32_t height,
             bool subsampling, bool right_image);

private:
    const Runtime& runtime_;
    cl::Kernel kernel_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_CREATE_TRIANGLE_MAPPING_KERNEL_H
