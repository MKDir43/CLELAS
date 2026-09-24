#include "create_grid_kernel.h"

namespace opencl {

CreateGridKernel::CreateGridKernel(const Runtime& runtime,
                                   int32_t grid_width, int32_t grid_height, int32_t disp_max)
    : runtime_(runtime), grid_width_(grid_width), grid_height_(grid_height), disp_max_(disp_max)
{
    const cl::Program program = runtime.buildProgram("create_grid.cl");
    calculate_valid_ = cl::Kernel(program, "calculateValidGrid");
    compute_grid_ = cl::Kernel(program, "computeGrid");

    const size_t cells = static_cast<size_t>(grid_width) * grid_height;
    valids_ = cl::Buffer(runtime.context(), CL_MEM_READ_WRITE,
                         sizeof(uint8_t) * cells * (disp_max + 1));
}

void CreateGridKernel::run(const DeviceSupportPoints& supports, int32_t support_num, DeviceGrid& grid,
                           int32_t grid_size, bool right_image)
{
    const size_t cells = static_cast<size_t>(grid_width_) * grid_height_;
    const cl::CommandQueue& queue = runtime_.queue();

    queue.enqueueFillBuffer(grid.nums, int32_t(0), 0, cells * sizeof(int32_t));
    queue.enqueueFillBuffer(valids_, uint8_t(0), 0, cells * (disp_max_ + 1) * sizeof(uint8_t));
    queue.finish();

    // Step 1: mark the disparities present in each cell
    calculate_valid_.setArg(0, supports.u);
    calculate_valid_.setArg(1, supports.v);
    calculate_valid_.setArg(2, supports.d);
    calculate_valid_.setArg(3, support_num);
    calculate_valid_.setArg(4, valids_);
    calculate_valid_.setArg(5, grid_width_);
    calculate_valid_.setArg(6, grid_height_);
    calculate_valid_.setArg(7, grid_size);
    calculate_valid_.setArg(8, disp_max_);
    calculate_valid_.setArg(9, int(right_image));
    runtime_.run(calculate_valid_);

    // Step 2: compact the marked disparities into a candidate list per cell
    compute_grid_.setArg(0, grid.data);
    compute_grid_.setArg(1, grid.nums);
    compute_grid_.setArg(2, valids_);
    compute_grid_.setArg(3, grid_width_);
    compute_grid_.setArg(4, grid_height_);
    compute_grid_.setArg(5, disp_max_);
    runtime_.run(compute_grid_);
}

} // namespace opencl
