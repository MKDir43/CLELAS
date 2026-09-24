#ifndef CLELAS_OPENCL_DEVICE_BUFFERS_H
#define CLELAS_OPENCL_DEVICE_BUFFERS_H

#include <CL/opencl.hpp>

namespace opencl {

// Device buffers passed between the pipeline stages. cl::Buffer is reference
// counted, so these structs can be copied freely. The element type and length
// of each buffer is noted next to it.

// Output of the Sobel filter stage (same layout as the CPU version).
struct DeviceSobel {
    cl::Buffer h;  // uint8_t[width * height], horizontal gradient du
    cl::Buffer v;  // uint8_t[width * height], vertical gradient dv
};

// Support points as a list (support_max entries each).
struct DeviceSupportPoints {
    cl::Buffer u;  // int32_t, column
    cl::Buffer v;  // int32_t, row
    cl::Buffer d;  // int32_t, disparity
};

// Delaunay triangles (triangle_max entries each).
struct DeviceTriangles {
    cl::Buffer c1, c2, c3;     // int32_t, support point indices of the corners
    cl::Buffer t1a, t1b, t1c;  // float, disparity plane d = a*u + b*v + c (left image)
    cl::Buffer t2a, t2b, t2c;  // float, disparity plane d = a*u + b*v + c (right image)
};

// Disparity candidates per grid cell (grid_width * grid_height cells).
struct DeviceGrid {
    cl::Buffer data;  // int32_t[cells * (disp_max + 1)], candidate disparities of each cell
    cl::Buffer nums;  // int32_t[cells], number of candidates in each cell
};

} // namespace opencl

#endif // CLELAS_OPENCL_DEVICE_BUFFERS_H
