#ifndef CLELAS_OPENCL_MAKE_DISPARITY_CANDIDATE_KERNEL_H
#define CLELAS_OPENCL_MAKE_DISPARITY_CANDIDATE_KERNEL_H

#include "device_buffers.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Matches support point candidates on a regular grid in two passes:
// a texture check followed by disparity matching.
class DisparityCandidateKernel {
public:
    DisparityCandidateKernel(const Runtime& runtime, int32_t width, int32_t height,
                             int32_t candi_width, int32_t candi_height, int32_t candi_step);

    // Sobel responses of both images -> candidates: int16_t[candi_width * candi_height]
    // (-1 = no candidate). A candidate is kept only if it passes the left/right
    // consistency check (lr_threshold).
    void run(const DeviceSobel& sobel_left, const DeviceSobel& sobel_right,
             const cl::Buffer& candidates,
             int32_t disp_min, int32_t disp_max,
             int32_t support_texture, float support_threshold, int32_t lr_threshold);

private:
    const Runtime& runtime_;
    cl::Kernel texture_check_;
    cl::Kernel disparity_match_;
    cl::Buffer texture_mask_;  // int32_t[candi_width * candi_height], textured candidates

    int32_t width_;
    int32_t height_;
    int32_t candi_width_;
    int32_t candi_height_;
    int32_t candi_step_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_MAKE_DISPARITY_CANDIDATE_KERNEL_H
