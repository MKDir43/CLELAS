#ifndef CLELAS_ELAS_RESOURCE_H
#define CLELAS_ELAS_RESOURCE_H

// Internal header of the OpenCL library; not part of the public API.

#include "elas_parameter.h"

#include "adaptive_mean_kernel.h"
#include "compute_disparity_kernel.h"
#include "correct_supports_kernel.h"
#include "create_grid_kernel.h"
#include "create_triangle_mapping_kernel.h"
#include "device_buffers.h"
#include "disparity_planes_kernel.h"
#include "lr_consistency_check_kernel.h"
#include "make_disparity_candidate_kernel.h"
#include "median_filter_kernel.h"
#include "runtime.h"
#include "sobel_filter_kernel.h"

#include <cstdint>
#include <vector>

namespace clelas {

// Sizes derived from the image size and the parameters.
struct Dimensions {
    Dimensions(int32_t width, int32_t height, int32_t pitch, const Parameters& param);

    int32_t width;
    int32_t height;
    int32_t pitch;
    int32_t candi_step;    // spacing of the support point candidate grid
    int32_t candi_width;
    int32_t candi_height;
    int32_t grid_width;    // disparity grid cells
    int32_t grid_height;
    int32_t support_max;   // upper bound of support points
    int32_t triangle_max;  // upper bound of Delaunay triangles
};

// Support points as a list (support_max entries each).
struct HostSupportPoints {
    std::vector<int32_t> u;
    std::vector<int32_t> v;
    std::vector<int32_t> d;
};

// Corner support point indices of the Delaunay triangles (triangle_max entries each).
struct HostTriangleCorners {
    std::vector<int32_t> c1;
    std::vector<int32_t> c2;
    std::vector<int32_t> c3;
};

// Host buffers used by the CPU stages of the pipeline.
struct HostResources {
    explicit HostResources(const Dimensions& dims);

    std::vector<int16_t> candidates;  // candi_width * candi_height (-1 = no candidate)
    HostSupportPoints supports;
    HostTriangleCorners tri_left;
    HostTriangleCorners tri_right;
    std::vector<float> dtemp_left;    // width * height, used by post-processing
    std::vector<float> dtemp_right;
};

// OpenCL runtime, pipeline stages and the device buffers passed between them.
struct OpenCLResources {
    OpenCLResources(const Dimensions& dims, const Parameters& param);

    // The stages keep a reference to the runtime, so it is declared first.
    opencl::Runtime runtime;

    opencl::SobelFilterKernel sobel_filter;
    opencl::DisparityCandidateKernel disparity_candidate;
    opencl::CorrectSupportsKernel correct_supports;
    opencl::DisparityPlanesKernel disparity_planes;
    opencl::CreateGridKernel create_grid;
    opencl::CreateTriangleMappingKernel triangle_mapping;
    opencl::ComputeDisparityKernel compute_disparity;
    opencl::LRConsistencyCheckKernel lr_consistency_check;
    opencl::AdaptiveMeanKernel adaptive_mean;
    opencl::MedianFilterKernel median_filter;

    cl::Buffer image_left;                    // uint8_t[pitch * height]
    cl::Buffer image_right;
    opencl::DeviceSobel sobel_left;
    opencl::DeviceSobel sobel_right;
    cl::Buffer candidates;                    // int16_t[candi_width * candi_height] (-1 = none)
    cl::Buffer thin_candidates;               // int16_t[candi_width * candi_height]
    opencl::DeviceSupportPoints supports;
    opencl::DeviceTriangles tri_left;
    opencl::DeviceTriangles tri_right;
    opencl::DeviceGrid grid_left;
    opencl::DeviceGrid grid_right;
    cl::Buffer tri_map_left;                  // uint16_t[width * height]
    cl::Buffer tri_map_right;
    cl::Buffer dtemp_left;                    // float[width * height], dense matching result
    cl::Buffer dtemp_right;
    cl::Buffer dtemp2_left;                   // float[width * height], after the L/R check and post-processing
    cl::Buffer dtemp2_right;
    cl::Buffer disp_left;                     // float[pitch * height], final disparity
    cl::Buffer disp_right;
};

} // namespace clelas

#endif // CLELAS_ELAS_RESOURCE_H
