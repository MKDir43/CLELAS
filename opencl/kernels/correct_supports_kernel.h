#ifndef CLELAS_OPENCL_CORRECT_SUPPORTS_KERNEL_H
#define CLELAS_OPENCL_CORRECT_SUPPORTS_KERNEL_H

#include "device_buffers.h"
#include "elas_parameter.h"
#include "runtime.h"

#include <cstdint>

namespace opencl {

// Thins out the support point candidates on the device, as the CPU version
// does on the host: removes inconsistent candidates, then redundant ones.
class CorrectSupportsKernel {
public:
    CorrectSupportsKernel(const Runtime& runtime, int32_t candi_width, int32_t candi_height);

    // candidates -> thin_candidates: int16_t[candi_width * candi_height] (-1 = removed)
    void run(const cl::Buffer& candidates, const cl::Buffer& thin_candidates,
             const clelas::Parameters& param);

private:
    const Runtime& runtime_;
    cl::Kernel remove_inconsistent_;
    cl::Kernel remove_redundant_;
    int32_t candi_width_;
    int32_t candi_height_;
};

} // namespace opencl

#endif // CLELAS_OPENCL_CORRECT_SUPPORTS_KERNEL_H
