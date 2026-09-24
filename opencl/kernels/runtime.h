#ifndef CLELAS_OPENCL_RUNTIME_H
#define CLELAS_OPENCL_RUNTIME_H

#include <CL/opencl.hpp>

#include <cstddef>
#include <cstdio>
#include <string>

// Kernel execution log (enabled with the DEBUG_KERNEL CMake option)
#ifdef DEBUG_KERNEL
#define KERNEL_LOG_START(name, ...) fprintf(stderr, "[KERNEL] " name ": start" __VA_ARGS__)
#define KERNEL_LOG_END(name) fprintf(stderr, "[KERNEL] " name ": done\n")
#else
#define KERNEL_LOG_START(name, ...) ((void)0)
#define KERNEL_LOG_END(name) ((void)0)
#endif

namespace opencl {

// Work sizes used for every kernel (Global=512, Local=64). All kernels iterate
// over their domain with a 1D strided loop, so this fixed range covers any size.
constexpr size_t DEFAULT_GLOBAL_WORK_SIZE = 512;
constexpr size_t DEFAULT_LOCAL_WORK_SIZE = 64;

// OpenCL context, device and in-order command queue shared by all stages.
class Runtime {
public:
    // Uses the first platform, preferring a GPU and falling back to a CPU device.
    Runtime();

    // Loads a kernel source file and builds it for the device.
    // Throws std::runtime_error with the build log when the build fails.
    cl::Program buildProgram(const std::string& cl_file_name) const;

    // Enqueues a 1D kernel with the default work sizes and waits for completion.
    void run(const cl::Kernel& kernel) const;

    const cl::Context& context() const { return context_; }
    const cl::CommandQueue& queue() const { return queue_; }

private:
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
};

// Formats an OpenCL error as "<API call> failed: <error name> (<code>)".
std::string describe(const cl::Error& error);

} // namespace opencl

#endif // CLELAS_OPENCL_RUNTIME_H
