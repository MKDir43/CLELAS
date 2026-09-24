#include "make_disparity_candidate_kernel.h"

namespace opencl {

DisparityCandidateKernel::DisparityCandidateKernel(
    const Runtime& runtime, int32_t width, int32_t height,
    int32_t candi_width, int32_t candi_height, int32_t candi_step)
    : runtime_(runtime),
      width_(width), height_(height),
      candi_width_(candi_width), candi_height_(candi_height), candi_step_(candi_step)
{
    const cl::Program program = runtime.buildProgram("make_disparity_candidate.cl");
    texture_check_ = cl::Kernel(program, "textureCheckPass");
    disparity_match_ = cl::Kernel(program, "disparityMatchPass");

    texture_mask_ = cl::Buffer(runtime.context(), CL_MEM_READ_WRITE,
                               sizeof(int32_t) * candi_width * candi_height);
}

void DisparityCandidateKernel::run(const DeviceSobel& sobel_left, const DeviceSobel& sobel_right,
                                   const cl::Buffer& candidates,
                                   int32_t disp_min, int32_t disp_max,
                                   int32_t support_texture, float support_threshold, int32_t lr_threshold)
{
    const size_t count = static_cast<size_t>(candi_width_) * candi_height_;
    const cl::CommandQueue& queue = runtime_.queue();

    // Initialize the buffers on the host (initializing memory inside the kernel causes stalls)
    queue.enqueueFillBuffer(candidates, int16_t(-1), 0, count * sizeof(int16_t));
    queue.enqueueFillBuffer(texture_mask_, int32_t(0), 0, count * sizeof(int32_t));
    queue.finish();

    // Pass 1: texture check
    texture_check_.setArg(0, sobel_left.h);
    texture_check_.setArg(1, sobel_left.v);
    texture_check_.setArg(2, width_);
    texture_check_.setArg(3, height_);
    texture_check_.setArg(4, candi_width_);
    texture_check_.setArg(5, candi_height_);
    texture_check_.setArg(6, candi_step_);
    texture_check_.setArg(7, support_texture);
    texture_check_.setArg(8, texture_mask_);
    runtime_.run(texture_check_);

    // Pass 2: disparity matching with the left/right consistency check
    disparity_match_.setArg(0, sobel_left.h);
    disparity_match_.setArg(1, sobel_left.v);
    disparity_match_.setArg(2, sobel_right.h);
    disparity_match_.setArg(3, sobel_right.v);
    disparity_match_.setArg(4, width_);
    disparity_match_.setArg(5, height_);
    disparity_match_.setArg(6, candidates);
    disparity_match_.setArg(7, candi_width_);
    disparity_match_.setArg(8, candi_height_);
    disparity_match_.setArg(9, candi_step_);
    disparity_match_.setArg(10, disp_min);
    disparity_match_.setArg(11, disp_max);
    disparity_match_.setArg(12, support_texture);
    disparity_match_.setArg(13, support_threshold);
    disparity_match_.setArg(14, lr_threshold);
    disparity_match_.setArg(15, texture_mask_);
    runtime_.run(disparity_match_);
}

} // namespace opencl
