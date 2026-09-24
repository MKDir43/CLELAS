// save_intermediate_images.cpp : saves the Sobel images for debugging
#include "elas_opencl.h"
#include "elas_resource.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void clelas::Processor::saveIntermediateImages(const char* prefix)
{
    std::cout << "Saving intermediate images with prefix: " << prefix << std::endl;

    const struct {
        const cl::Buffer& buffer;
        const char* suffix;
    } images[] = {
        {opencl_->sobel_left.h, "_hsobel_left.png"},
        {opencl_->sobel_right.h, "_hsobel_right.png"},
        {opencl_->sobel_left.v, "_vsobel_left.png"},
        {opencl_->sobel_right.v, "_vsobel_right.png"},
    };

    const size_t buffer_size = static_cast<size_t>(width) * height;
    std::vector<uint8_t> pixels(buffer_size);
    try {
        for (const auto& image : images) {
            opencl_->runtime.queue().enqueueReadBuffer(image.buffer, CL_TRUE, 0, buffer_size, pixels.data());
            cv::imwrite(std::string(prefix) + image.suffix, cv::Mat(height, width, CV_8U, pixels.data()));
        }
    } catch (const cl::Error& e) {
        throw std::runtime_error("OpenCL error: " + opencl::describe(e));
    }

    std::cout << "Intermediate images saved successfully" << std::endl;
}
