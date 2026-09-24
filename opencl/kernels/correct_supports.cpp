#include "correct_supports_kernel.h"

namespace opencl {

namespace {

// Same as the call of removeRedundantSupportPoints in the CPU version.
const int32_t REDUN_MAX_DIST = 5;
const int32_t REDUN_THRESHOLD = 1;

} // namespace

CorrectSupportsKernel::CorrectSupportsKernel(const Runtime& runtime,
                                             int32_t candi_width, int32_t candi_height)
    : runtime_(runtime), candi_width_(candi_width), candi_height_(candi_height)
{
    const cl::Program program = runtime.buildProgram("correct_supports.cl");
    remove_inconsistent_ = cl::Kernel(program, "removeInconsistentSupportPoints");
    remove_redundant_ = cl::Kernel(program, "removeRedundantSupportPoints");
}

void CorrectSupportsKernel::run(const cl::Buffer& candidates, const cl::Buffer& thin_candidates,
                                const clelas::Parameters& param)
{
    remove_inconsistent_.setArg(0, candidates);
    remove_inconsistent_.setArg(1, thin_candidates);
    remove_inconsistent_.setArg(2, candi_width_);
    remove_inconsistent_.setArg(3, candi_height_);
    remove_inconsistent_.setArg(4, param.incon_window_width);
    remove_inconsistent_.setArg(5, param.incon_window_height);
    remove_inconsistent_.setArg(6, param.incon_threshold);
    remove_inconsistent_.setArg(7, param.incon_min_support);
    runtime_.run(remove_inconsistent_);

    // Remove support points on straight lines, vertically then horizontally
    // (each pass has to finish before the next one starts).
    remove_redundant_.setArg(0, thin_candidates);
    remove_redundant_.setArg(1, candi_width_);
    remove_redundant_.setArg(2, candi_height_);
    remove_redundant_.setArg(3, REDUN_MAX_DIST);
    remove_redundant_.setArg(4, REDUN_THRESHOLD);
    for (const int32_t vertical : {1, 0}) {
        remove_redundant_.setArg(5, vertical);
        runtime_.run(remove_redundant_);
    }
}

} // namespace opencl
