#ifndef CLELAS_OPENCL_CREATE_GRID_KERNEL_H
#define CLELAS_OPENCL_CREATE_GRID_KERNEL_H

#include "device_buffers.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Collects the disparities of nearby support points into each grid cell.
class CreateGridKernel {
public:
    CreateGridKernel(const Runtime& runtime, int32_t grid_width, int32_t grid_height, int32_t disp_max);

    // support points -> grid.data, grid.nums
    void run(const DeviceSupportPoints& supports, int32_t support_num, DeviceGrid& grid,
             int32_t grid_size, bool right_image);

private:
    const Runtime& runtime_;
    cl::Kernel calculate_valid_;
    cl::Kernel compute_grid_;
    cl::Buffer valids_;  // uint8_t[cells * (disp_max + 1)], disparities present in each cell

    int32_t grid_width_;
    int32_t grid_height_;
    int32_t disp_max_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_CREATE_GRID_KERNEL_H
