#include "disparity_planes_kernel.h"

namespace opencl {

DisparityPlanesKernel::DisparityPlanesKernel(const Runtime& runtime)
    : runtime_(runtime)
{
    kernel_ = cl::Kernel(runtime.buildProgram("disparity_planes.cl"), "computeDisparityPlanes");
}

void DisparityPlanesKernel::run(const DeviceSupportPoints& supports, DeviceTriangles& triangles,
                                int32_t triangle_num)
{
    kernel_.setArg(0, supports.u);
    kernel_.setArg(1, supports.v);
    kernel_.setArg(2, supports.d);
    kernel_.setArg(3, triangles.c1);
    kernel_.setArg(4, triangles.c2);
    kernel_.setArg(5, triangles.c3);
    kernel_.setArg(6, triangles.t1a);
    kernel_.setArg(7, triangles.t1b);
    kernel_.setArg(8, triangles.t1c);
    kernel_.setArg(9, triangles.t2a);
    kernel_.setArg(10, triangles.t2b);
    kernel_.setArg(11, triangles.t2c);
    kernel_.setArg(12, triangle_num);
    runtime_.run(kernel_);
}

} // namespace opencl
