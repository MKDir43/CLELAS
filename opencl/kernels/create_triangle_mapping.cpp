#include "create_triangle_mapping_kernel.h"

namespace opencl {

CreateTriangleMappingKernel::CreateTriangleMappingKernel(const Runtime& runtime)
    : runtime_(runtime)
{
    kernel_ = cl::Kernel(runtime.buildProgram("create_triangle_mapping.cl"), "createTriangleMapping");
}

void CreateTriangleMappingKernel::run(const DeviceSupportPoints& supports, int32_t support_num,
                                      const DeviceTriangles& triangles, int32_t triangle_num,
                                      const cl::Buffer& tri_map, int32_t width, int32_t height,
                                      bool subsampling, bool right_image)
{
    const cl::CommandQueue& queue = runtime_.queue();
    queue.enqueueFillBuffer(tri_map, uint16_t(0), 0,
                            static_cast<size_t>(width) * height * sizeof(uint16_t));
    queue.finish();

    kernel_.setArg(0, supports.u);
    kernel_.setArg(1, supports.v);
    kernel_.setArg(2, supports.d);
    kernel_.setArg(3, support_num);
    kernel_.setArg(4, triangles.c1);
    kernel_.setArg(5, triangles.c2);
    kernel_.setArg(6, triangles.c3);
    kernel_.setArg(7, triangle_num);
    kernel_.setArg(8, tri_map);
    kernel_.setArg(9, width);
    kernel_.setArg(10, height);
    kernel_.setArg(11, int(subsampling));
    kernel_.setArg(12, int(right_image));
    runtime_.run(kernel_);
}

} // namespace opencl
