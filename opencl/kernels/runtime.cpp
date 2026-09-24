#include "runtime.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace opencl {

namespace {

const char* errorName(cl_int error)
{
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        default: return "unknown OpenCL error";
    }
}

// Searches the kernel file in "kernels/" relative to the working directory,
// then in the build directory the kernels were copied to (CLELAS_KERNEL_DIR).
std::string loadKernelSource(const std::string& name)
{
    std::vector<std::string> paths = {"kernels/" + name};
#ifdef CLELAS_KERNEL_DIR
    paths.push_back(std::string(CLELAS_KERNEL_DIR) + "/" + name);
#endif

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary);
        if (file) {
            std::ostringstream source;
            source << file.rdbuf();
            return source.str();
        }
    }
    throw std::runtime_error("Failed to open OpenCL kernel file: " + name);
}

} // namespace

Runtime::Runtime()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platform found");
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL GPU or CPU device found");
    }

    device_ = devices[0];
    context_ = cl::Context(device_);
    queue_ = cl::CommandQueue(context_, device_);
}

cl::Program Runtime::buildProgram(const std::string& cl_file_name) const
{
    // Correctly rounded division gives the same float results as the CPU version.
    // Devices that do not support it keep the default (up to 2.5 ULP) division.
    std::string options;
    if (device_.getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
        options = "-cl-fp32-correctly-rounded-divide-sqrt";
    }

    cl::Program program(context_, loadKernelSource(cl_file_name));
    try {
        program.build({device_}, options.c_str());
    } catch (const cl::BuildError& e) {
        std::string log;
        for (const auto& entry : e.getBuildLog()) {
            log += entry.second;
        }
        throw std::runtime_error("Failed to build " + cl_file_name + ":\n" + log);
    }
    return program;
}

void Runtime::run(const cl::Kernel& kernel) const
{
    queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                cl::NDRange(DEFAULT_GLOBAL_WORK_SIZE),
                                cl::NDRange(DEFAULT_LOCAL_WORK_SIZE));
    queue_.finish();
}

std::string describe(const cl::Error& error)
{
    return std::string(error.what()) + " failed: " + errorName(error.err()) +
           " (" + std::to_string(error.err()) + ")";
}

} // namespace opencl
