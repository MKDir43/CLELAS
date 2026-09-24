#include "elas_resource.h"

#include <cmath>

namespace clelas
{

Dimensions::Dimensions(int32_t width, int32_t height, int32_t pitch, const Parameters& param)
    : width(width), height(height), pitch(pitch)
{
    candi_step = param.subsampling
                     ? param.candidate_stepsize + (param.candidate_stepsize % 2)
                     : param.candidate_stepsize;
    candi_width = (width - 1) / candi_step + 1;
    candi_height = (height - 1) / candi_step + 1;
    grid_width = static_cast<int32_t>(
        std::ceil(static_cast<float>(width) / static_cast<float>(param.grid_size)));
    grid_height = static_cast<int32_t>(
        std::ceil(static_cast<float>(height) / static_cast<float>(param.grid_size)));
    support_max = width * height;
    triangle_max = (width - 1) * (height - 1) * 2;
}

HostResources::HostResources(const Dimensions& dims)
    : candidates(static_cast<size_t>(dims.candi_width) * dims.candi_height),
      dtemp_left(static_cast<size_t>(dims.width) * dims.height),
      dtemp_right(static_cast<size_t>(dims.width) * dims.height)
{
    for (auto* list : {&supports.u, &supports.v, &supports.d}) {
        list->resize(dims.support_max);
    }
    for (auto* tri : {&tri_left, &tri_right}) {
        tri->c1.resize(dims.triangle_max);
        tri->c2.resize(dims.triangle_max);
        tri->c3.resize(dims.triangle_max);
    }
}

OpenCLResources::OpenCLResources(const Dimensions& dims, const Parameters& param)
    : sobel_filter(runtime),
      disparity_candidate(runtime, dims.width, dims.height,
                          dims.candi_width, dims.candi_height, dims.candi_step),
      correct_supports(runtime, dims.candi_width, dims.candi_height),
      disparity_planes(runtime),
      create_grid(runtime, dims.grid_width, dims.grid_height, param.disp_max),
      triangle_mapping(runtime),
      compute_disparity(runtime),
      lr_consistency_check(runtime),
      adaptive_mean(runtime, dims.width, dims.height),
      median_filter(runtime, dims.width, dims.height)
{
    auto buffer = [this](size_t bytes) {
        return cl::Buffer(runtime.context(), CL_MEM_READ_WRITE, bytes);
    };

    const size_t image = static_cast<size_t>(dims.pitch) * dims.height;
    const size_t pixels = static_cast<size_t>(dims.width) * dims.height;
    const size_t candi = static_cast<size_t>(dims.candi_width) * dims.candi_height;
    const size_t cells = static_cast<size_t>(dims.grid_width) * dims.grid_height;

    image_left = buffer(sizeof(uint8_t) * image);
    image_right = buffer(sizeof(uint8_t) * image);

    for (auto* sobel : {&sobel_left, &sobel_right}) {
        sobel->h = buffer(sizeof(uint8_t) * image);
        sobel->v = buffer(sizeof(uint8_t) * image);
    }

    candidates = buffer(sizeof(int16_t) * candi);
    thin_candidates = buffer(sizeof(int16_t) * candi);

    supports.u = buffer(sizeof(int32_t) * dims.support_max);
    supports.v = buffer(sizeof(int32_t) * dims.support_max);
    supports.d = buffer(sizeof(int32_t) * dims.support_max);

    for (auto* tri : {&tri_left, &tri_right}) {
        for (auto* corner : {&tri->c1, &tri->c2, &tri->c3}) {
            *corner = buffer(sizeof(int32_t) * dims.triangle_max);
        }
        for (auto* plane : {&tri->t1a, &tri->t1b, &tri->t1c, &tri->t2a, &tri->t2b, &tri->t2c}) {
            *plane = buffer(sizeof(float) * dims.triangle_max);
        }
    }

    for (auto* grid : {&grid_left, &grid_right}) {
        grid->data = buffer(sizeof(int32_t) * cells * (param.disp_max + 1));
        grid->nums = buffer(sizeof(int32_t) * cells);
    }

    tri_map_left = buffer(sizeof(uint16_t) * pixels);
    tri_map_right = buffer(sizeof(uint16_t) * pixels);
    dtemp_left = buffer(sizeof(float) * pixels);
    dtemp_right = buffer(sizeof(float) * pixels);
    dtemp2_left = buffer(sizeof(float) * pixels);
    dtemp2_right = buffer(sizeof(float) * pixels);
    disp_left = buffer(sizeof(float) * image);
    disp_right = buffer(sizeof(float) * image);
}

} // namespace clelas
