#include "compute_disparity_kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace opencl {

ComputeDisparityKernel::ComputeDisparityKernel(const Runtime& runtime)
    : runtime_(runtime)
{
    kernel_ = cl::Kernel(runtime.buildProgram("compute_disparity.cl"), "computeDisparity");
}

void ComputeDisparityKernel::run(const DeviceSobel& sobel_left, const DeviceSobel& sobel_right,
                                 const DeviceSupportPoints& supports, int32_t support_num,
                                 const DeviceTriangles& triangles, int32_t triangle_num,
                                 const DeviceGrid& grid, const cl::Buffer& tri_map,
                                 const cl::Buffer& disparity, int32_t width, int32_t height,
                                 const clelas::Parameters& param, bool estimates_subpixel, bool right_image)
{
    KERNEL_LOG_START("computeDisparity", " (%dx%d, disp=%d-%d)\n", width, height, param.disp_min, param.disp_max);

    const int32_t grid_width = static_cast<int32_t>(
        std::ceil(static_cast<float>(width) / static_cast<float>(param.grid_size)));
    const int32_t grid_height = static_cast<int32_t>(
        std::ceil(static_cast<float>(height) / static_cast<float>(param.grid_size)));
    const int32_t disp_num = param.disp_max + 1;
    // Disparity search radius around the plane prior (same as the CPU version)
    const int32_t plane_radius = static_cast<int32_t>(
        std::max(std::ceil(param.sigma * param.sradius), 2.0f));

    // Prior array (same formula as the CPU version)
    // P[delta_d] = (-log(gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(gamma))/beta
    const float two_sigma_squared = 2.0f * param.sigma * param.sigma;
    std::vector<int32_t> prior(disp_num);
    for (int32_t delta_d = 0; delta_d < disp_num; delta_d++) {
        prior[delta_d] = static_cast<int32_t>(
            (-std::log(param.gamma + std::exp(-delta_d * delta_d / two_sigma_squared)) + std::log(param.gamma)) / param.beta);
    }
    const cl::Buffer prior_buffer(runtime_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int32_t) * prior.size(), prior.data());

    kernel_.setArg(0, supports.u);
    kernel_.setArg(1, supports.v);
    kernel_.setArg(2, supports.d);
    kernel_.setArg(3, support_num);
    kernel_.setArg(4, triangles.c1);
    kernel_.setArg(5, triangles.c2);
    kernel_.setArg(6, triangles.c3);
    kernel_.setArg(7, triangles.t1a);
    kernel_.setArg(8, triangles.t1b);
    kernel_.setArg(9, triangles.t1c);
    kernel_.setArg(10, triangles.t2a);
    kernel_.setArg(11, triangles.t2b);
    kernel_.setArg(12, triangles.t2c);
    kernel_.setArg(13, triangle_num);
    kernel_.setArg(14, grid.data);
    kernel_.setArg(15, grid.nums);
    kernel_.setArg(16, grid_width);
    kernel_.setArg(17, grid_height);
    kernel_.setArg(18, sobel_left.h);
    kernel_.setArg(19, sobel_right.h);
    kernel_.setArg(20, sobel_left.v);
    kernel_.setArg(21, sobel_right.v);
    kernel_.setArg(22, tri_map);
    kernel_.setArg(23, disparity);
    kernel_.setArg(24, width);
    kernel_.setArg(25, height);
    kernel_.setArg(26, param.disp_min);
    kernel_.setArg(27, param.disp_max);
    kernel_.setArg(28, disp_num);
    kernel_.setArg(29, param.match_texture);
    kernel_.setArg(30, param.grid_size);
    kernel_.setArg(31, int(param.subsampling));
    kernel_.setArg(32, int(estimates_subpixel));
    kernel_.setArg(33, plane_radius);
    kernel_.setArg(34, int(right_image));
    kernel_.setArg(35, prior_buffer);
    runtime_.run(kernel_);

    KERNEL_LOG_END("computeDisparity");
}

} // namespace opencl
