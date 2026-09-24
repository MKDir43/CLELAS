/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of CLELAS, which is derived from libelas.
libelas authors: Andreas Geiger
Modified by MKDir43, 2025-2026 (restructuring and OpenCL port).

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include "elas_opencl.h"
#include "elas_resource.h"

#include <climits>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <math.h>
#include <unistd.h>
#include "triangle.h"

// ----- Profiling Function -----
struct ProfTimePoints {
    std::chrono::steady_clock::time_point upload_image;
    std::chrono::steady_clock::time_point descriptor;
    std::chrono::steady_clock::time_point make_candidates;
    std::chrono::steady_clock::time_point download_candidates;
    std::chrono::steady_clock::time_point correct_candidates;
    std::chrono::steady_clock::time_point convert_candidates;
    std::chrono::steady_clock::time_point triangulate;
    std::chrono::steady_clock::time_point upload_triangles;
    std::chrono::steady_clock::time_point plane;
    std::chrono::steady_clock::time_point grid;
    std::chrono::steady_clock::time_point mapping;
    std::chrono::steady_clock::time_point disparity;
    std::chrono::steady_clock::time_point consistency;
    std::chrono::steady_clock::time_point postprocess;
    std::chrono::steady_clock::time_point download_disparity;
    std::chrono::steady_clock::time_point finish;
};

inline void prof_time(std::chrono::steady_clock::time_point& time)
{
#ifdef PROFILE
    time = std::chrono::steady_clock::now();
#endif // PROFILE
}

inline void output_times(ProfTimePoints& tp) {
#ifdef PROFILE
    auto time_upload_image      = std::chrono::duration_cast<std::chrono::microseconds>(tp.descriptor            - tp.upload_image     ).count();
    auto time_descriptor            = std::chrono::duration_cast<std::chrono::microseconds>(tp.make_candidates       - tp.descriptor           ).count();
    auto time_make_candidates       = std::chrono::duration_cast<std::chrono::microseconds>(tp.download_candidates - tp.make_candidates      ).count();
    auto time_download_candidates = std::chrono::duration_cast<std::chrono::microseconds>(tp.correct_candidates    - tp.download_candidates).count();
    auto time_correct_candidates    = std::chrono::duration_cast<std::chrono::microseconds>(tp.convert_candidates    - tp.correct_candidates   ).count();
    auto time_convert_candidates    = std::chrono::duration_cast<std::chrono::microseconds>(tp.triangulate           - tp.convert_candidates   ).count();
    auto time_triangulate           = std::chrono::duration_cast<std::chrono::microseconds>(tp.upload_triangles  - tp.triangulate          ).count();
    auto time_upload_triangles  = std::chrono::duration_cast<std::chrono::microseconds>(tp.plane                 - tp.upload_triangles ).count();
    auto time_plane                 = std::chrono::duration_cast<std::chrono::microseconds>(tp.grid                  - tp.plane                ).count();
    auto time_grid                  = std::chrono::duration_cast<std::chrono::microseconds>(tp.mapping               - tp.grid                 ).count();
    auto time_mapping               = std::chrono::duration_cast<std::chrono::microseconds>(tp.disparity             - tp.mapping              ).count();
    auto time_disparity             = std::chrono::duration_cast<std::chrono::microseconds>(tp.consistency           - tp.disparity            ).count();
    auto time_consistency           = std::chrono::duration_cast<std::chrono::microseconds>(tp.postprocess         - tp.consistency          ).count();
    auto time_postprocess           = std::chrono::duration_cast<std::chrono::microseconds>(tp.download_disparity  - tp.postprocess          ).count();
    auto time_download_disparity  = std::chrono::duration_cast<std::chrono::microseconds>(tp.finish                - tp.download_disparity ).count();
    auto time_total  = std::chrono::duration_cast<std::chrono::microseconds>(tp.finish                - tp.upload_image ).count();

    std::cout << "upload_image:"      << (double)time_upload_image      / 1000.0 << std::endl;
    std::cout << "descriptor:"             << (double)time_descriptor            / 1000.0 << std::endl;
    std::cout << "disparity_candidates:"   << (double)time_make_candidates       / 1000.0 << std::endl;
    std::cout << "download_candidates:" << (double)time_download_candidates / 1000.0 << std::endl;
    std::cout << "correct_candidates:"     << (double)time_correct_candidates    / 1000.0 << std::endl;
    std::cout << "convert_candidates:"     << (double)time_convert_candidates    / 1000.0 << std::endl;
    std::cout << "delaunay_triangulation:" << (double)time_triangulate           / 1000.0 << std::endl;
    std::cout << "upload_triangles:"  << (double)time_upload_triangles  / 1000.0 << std::endl;
    std::cout << "disparity_planes:"       << (double)time_plane                 / 1000.0 << std::endl;
    std::cout << "grid:"                   << (double)time_grid                  / 1000.0 << std::endl;
    std::cout << "triangle_mapping:"       << (double)time_mapping               / 1000.0 << std::endl;
    std::cout << "disparity:"              << (double)time_disparity             / 1000.0 << std::endl;
    std::cout << "lr_consistency_check:"   << (double)time_consistency           / 1000.0 << std::endl;
    std::cout << "postprocess:"            << (double)time_postprocess           / 1000.0 << std::endl;
    std::cout << "download_disparity:"  << (double)time_download_disparity  / 1000.0 << std::endl;
    std::cout << "total_time:"  << (double)time_total  / 1000.0 << std::endl;
#endif // PROFILE
}

// ----- Main Function -----

namespace {

[[noreturn]] void rethrow(const cl::Error& e)
{
    throw std::runtime_error("OpenCL error: " + opencl::describe(e));
}

void uploadSupportPoints(const cl::CommandQueue& queue, const clelas::HostSupportPoints& host,
                         const opencl::DeviceSupportPoints& device, int32_t support_num)
{
    const size_t bytes = sizeof(int32_t) * support_num;
    queue.enqueueWriteBuffer(device.u, CL_TRUE, 0, bytes, host.u.data());
    queue.enqueueWriteBuffer(device.v, CL_TRUE, 0, bytes, host.v.data());
    queue.enqueueWriteBuffer(device.d, CL_TRUE, 0, bytes, host.d.data());
}

void uploadTriangleCorners(const cl::CommandQueue& queue, const clelas::HostTriangleCorners& host,
                           const opencl::DeviceTriangles& device, int32_t triangle_num)
{
    const size_t bytes = sizeof(int32_t) * triangle_num;
    queue.enqueueWriteBuffer(device.c1, CL_TRUE, 0, bytes, host.c1.data());
    queue.enqueueWriteBuffer(device.c2, CL_TRUE, 0, bytes, host.c2.data());
    queue.enqueueWriteBuffer(device.c3, CL_TRUE, 0, bytes, host.c3.data());
}

// Reads a width x height disparity map whose rows are `pitch` apart, on the
// device and on the host.
void downloadDisparity(const cl::CommandQueue& queue, const cl::Buffer& device, float* host,
                       int32_t width, int32_t height, int32_t pitch)
{
    if (host == nullptr) {
        return;
    }
    const size_t pitch_bytes = sizeof(float) * pitch;
    const std::array<size_t, 3> origin = {0, 0, 0};
    const std::array<size_t, 3> region = {sizeof(float) * width, static_cast<size_t>(height), 1};
    queue.enqueueReadBufferRect(device, CL_TRUE, origin, origin, region,
                                pitch_bytes, 0, pitch_bytes, 0, host);
}

} // namespace

clelas::Processor::Processor(int32_t w, int32_t h, int32_t p,
                             const Parameters& param, bool estimates_subpixel)
    : width(w), height(h), pitch(p), param(param), estimates_subpixel(estimates_subpixel)
{
    const Dimensions dims(width, height, pitch, param);
    try {
        host_ = std::make_unique<HostResources>(dims);
        opencl_ = std::make_unique<OpenCLResources>(dims, param);
    } catch (const cl::Error& e) {
        rethrow(e);
    }
}

clelas::Processor::~Processor() = default;

int32_t clelas::Processor::process(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right)
{
    try {
        return processFrame(image_left, image_right, disp_left, disp_right);
    } catch (const cl::Error& e) {
        rethrow(e);
    }
}

int32_t clelas::Processor::processFrame(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right)
{
    ProfTimePoints tp;
    const Dimensions dims(width, height, pitch, param);
    HostResources& host = *host_;
    OpenCLResources& ocl = *opencl_;
    const cl::CommandQueue& queue = ocl.runtime.queue();

    const size_t image_bytes = sizeof(uint8_t) * pitch * height;
    const size_t candi_bytes = sizeof(int16_t) * host.candidates.size();
    // Disparity map size: with subsampling width/2 x height/2 with rows width/2 apart
    // (as in the CPU version), otherwise width x height with rows `pitch` apart.
    const int32_t d_width = param.subsampling ? width / 2 : width;
    const int32_t d_height = param.subsampling ? height / 2 : height;
    const int32_t d_pitch = param.subsampling ? d_width : pitch;
    const size_t map_bytes = sizeof(float) * d_width * d_height;

    // Upload the stereo images
    prof_time(tp.upload_image);
    queue.enqueueWriteBuffer(ocl.image_left, CL_TRUE, 0, image_bytes, image_left);
    queue.enqueueWriteBuffer(ocl.image_right, CL_TRUE, 0, image_bytes, image_right);

    // Create Descriptor
    prof_time(tp.descriptor);
    ocl.sobel_filter.run(ocl.image_left, ocl.sobel_left, width, height, pitch);
    ocl.sobel_filter.run(ocl.image_right, ocl.sobel_right, width, height, pitch);

    // Support Matches
    prof_time(tp.make_candidates);
    ocl.disparity_candidate.run(ocl.sobel_left, ocl.sobel_right, ocl.candidates,
                                param.disp_min, param.disp_max,
                                param.support_texture, param.support_threshold,
                                param.supp_lr_threshold);

    // Download the candidates, unless they are thinned out on the device first
    prof_time(tp.download_candidates);
    const bool thin_on_device = param.support_thinning && param.gpu_thinning;
    if (!thin_on_device) {
        queue.enqueueReadBuffer(ocl.candidates, CL_TRUE, 0, candi_bytes, host.candidates.data());
    }

    // Correct SupportPoint Candidates
    prof_time(tp.correct_candidates);
    if (thin_on_device) {
        ocl.correct_supports.run(ocl.candidates, ocl.thin_candidates, param);
        queue.enqueueReadBuffer(ocl.thin_candidates, CL_TRUE, 0, candi_bytes, host.candidates.data());
    } else if (param.support_thinning) {
        correctSupportPointCandidates(host.candidates.data(), dims.candi_width, dims.candi_height);
    }

    prof_time(tp.convert_candidates);
    const int32_t support_num = convertCandidatesToSupportPoints(
        host.candidates.data(), dims.candi_width, dims.candi_height, dims.candi_step, host.supports);

    if (support_num < 3) {
        std::cerr << "ERROR: Not enough support points (need at least 3, got " << support_num << ")" << std::endl;
        return -1;
    }

    // Delaunay Triangulation
    prof_time(tp.triangulate);
    const int32_t tri_num_left = computeDelaunayTriangulation(
        host.supports, support_num, host.tri_left, dims.triangle_max, false);
    if (tri_num_left < 1) return -1;

    const int32_t tri_num_right = computeDelaunayTriangulation(
        host.supports, support_num, host.tri_right, dims.triangle_max, true);
    if (tri_num_right < 1) return -1;

    // Upload the support points and triangle corners
    prof_time(tp.upload_triangles);
    uploadSupportPoints(queue, host.supports, ocl.supports, support_num);
    uploadTriangleCorners(queue, host.tri_left, ocl.tri_left, tri_num_left);
    uploadTriangleCorners(queue, host.tri_right, ocl.tri_right, tri_num_right);

    // Disparity Planes
    prof_time(tp.plane);
    ocl.disparity_planes.run(ocl.supports, ocl.tri_left, tri_num_left);
    ocl.disparity_planes.run(ocl.supports, ocl.tri_right, tri_num_right);

    // Grid
    prof_time(tp.grid);
    ocl.create_grid.run(ocl.supports, support_num, ocl.grid_left, param.grid_size, false);
    ocl.create_grid.run(ocl.supports, support_num, ocl.grid_right, param.grid_size, true);

    // TriangleMapping
    prof_time(tp.mapping);
    ocl.triangle_mapping.run(ocl.supports, support_num, ocl.tri_left, tri_num_left,
                             ocl.tri_map_left, width, height, param.subsampling, false);
    ocl.triangle_mapping.run(ocl.supports, support_num, ocl.tri_right, tri_num_right,
                             ocl.tri_map_right, width, height, param.subsampling, true);

    // Matching
    prof_time(tp.disparity);
    ocl.compute_disparity.run(ocl.sobel_left, ocl.sobel_right, ocl.supports, support_num,
                              ocl.tri_left, tri_num_left, ocl.grid_left, ocl.tri_map_left,
                              ocl.dtemp_left, width, height, param, estimates_subpixel, false);
    ocl.compute_disparity.run(ocl.sobel_right, ocl.sobel_left, ocl.supports, support_num,
                              ocl.tri_right, tri_num_right, ocl.grid_right, ocl.tri_map_right,
                              ocl.dtemp_right, width, height, param, estimates_subpixel, true);

    // L/R Consistency Check.
    prof_time(tp.consistency);
    ocl.lr_consistency_check.run(ocl.dtemp_left, ocl.dtemp_right, ocl.dtemp2_left, ocl.dtemp2_right,
                                 d_width, d_height, d_width, param.disp_lr_threshold,
                                 param.subsampling, estimates_subpixel);

    // Post-processing (same as the CPU version): remove small segments and
    // interpolate gaps on the host, then the optional filters on the device
    prof_time(tp.postprocess);
    const bool postprocess_right = !param.postprocess_only_left;
    queue.enqueueReadBuffer(ocl.dtemp2_left, CL_TRUE, 0, map_bytes, host.dtemp_left.data());
    if (postprocess_right) {
        queue.enqueueReadBuffer(ocl.dtemp2_right, CL_TRUE, 0, map_bytes, host.dtemp_right.data());
    }

    removeSmallSegments(host.dtemp_left.data());
    gapInterpolation(host.dtemp_left.data());
    if (postprocess_right) {
        removeSmallSegments(host.dtemp_right.data());
        gapInterpolation(host.dtemp_right.data());
    }

    queue.enqueueWriteBuffer(ocl.dtemp2_left, CL_TRUE, 0, map_bytes, host.dtemp_left.data());
    if (postprocess_right) {
        queue.enqueueWriteBuffer(ocl.dtemp2_right, CL_TRUE, 0, map_bytes, host.dtemp_right.data());
    }

    if (param.filter_adaptive_mean) {
        ocl.adaptive_mean.run(ocl.dtemp2_left, param.subsampling);
        if (postprocess_right) {
            ocl.adaptive_mean.run(ocl.dtemp2_right, param.subsampling);
        }
    }
    if (param.filter_median) {
        ocl.median_filter.run(ocl.dtemp2_left, param.subsampling);
        if (postprocess_right) {
            ocl.median_filter.run(ocl.dtemp2_right, param.subsampling);
        }
    }

    // dtemp2 has rows `d_width` apart, the final disparity maps `d_pitch` apart
    const size_t row_bytes = sizeof(float) * d_width;
    const std::array<size_t, 3> origin = {0, 0, 0};
    const std::array<size_t, 3> region = {row_bytes, static_cast<size_t>(d_height), 1};
    queue.enqueueCopyBufferRect(ocl.dtemp2_left, ocl.disp_left, origin, origin, region,
                                row_bytes, 0, sizeof(float) * d_pitch, 0);
    queue.enqueueCopyBufferRect(ocl.dtemp2_right, ocl.disp_right, origin, origin, region,
                                row_bytes, 0, sizeof(float) * d_pitch, 0);
    queue.finish();

    // Download the disparity maps
    prof_time(tp.download_disparity);
    downloadDisparity(queue, ocl.disp_left, disp_left, d_width, d_height, d_pitch);
    downloadDisparity(queue, ocl.disp_right, disp_right, d_width, d_height, d_pitch);
    queue.finish();

    prof_time(tp.finish);
    output_times(tp);

    return 0;
}

// -------------------------------------------------------------------------------------------------- 
// computeSupportMatches functions
// -------------------------------------------------------------------------------------------------- 
void clelas::Processor::removeInconsistentSupportPoints(
    int16_t* D_can, const int32_t D_can_width, const int32_t D_can_height) {
  
    // for all valid support points do
    for (int32_t u_can=0; u_can<D_can_width; u_can++) {
        for (int32_t v_can=0; v_can<D_can_height; v_can++) {
            int16_t d_can = D_can[D_can_width*v_can+u_can];
            if (d_can>=0) {
              
                // compute number of other points supporting the current point
                int32_t support = 0;
                for (int32_t u_can_2=u_can-param.incon_window_width; u_can_2<=u_can+param.incon_window_width; u_can_2++) {
                    for (int32_t v_can_2=v_can-param.incon_window_height; v_can_2<=v_can+param.incon_window_height; v_can_2++) {
                        if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
                            int16_t d_can_2 = D_can[D_can_width*v_can_2+u_can_2];
                            if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
                                support++;
                        }
                    }
                }
                
                // invalidate support point if number of supporting points is too low
                if (support<param.incon_min_support)
                    D_can[D_can_width*v_can+u_can] = -1;
            }
        }
    }
}

void clelas::Processor::removeRedundantSupportPoints(
        int16_t* D_can,int32_t D_can_width, int32_t D_can_height,
        int32_t redun_max_dist, int32_t redun_threshold, 
        bool vertical)
{
    // parameters
    int32_t redun_dir_u[2] = {0,0};
    int32_t redun_dir_v[2] = {0,0};
    if (vertical) {
        redun_dir_v[0] = -1;
        redun_dir_v[1] = +1;
    } else {
        redun_dir_u[0] = -1;
        redun_dir_u[1] = +1;
    }
      
    // for all valid support points do
    for (int32_t u_can=0; u_can<D_can_width; u_can++) {
        for (int32_t v_can=0; v_can<D_can_height; v_can++) {
            int16_t d_can = D_can[D_can_width*v_can+u_can];
            if (d_can>=0) {
              
                // check all directions for redundancy
                bool redundant = true;
                for (int32_t i=0; i<2; i++) {
                  
                    // search for support
                    int32_t u_can_2 = u_can;
                    int32_t v_can_2 = v_can;
                    int16_t d_can_2;
                    bool support = false;
                    for (int32_t j=0; j<redun_max_dist; j++) {
                        u_can_2 += redun_dir_u[i];
                        v_can_2 += redun_dir_v[i];
                        if (u_can_2<0 || v_can_2<0 || u_can_2>=D_can_width || v_can_2>=D_can_height)
                            break;
                        d_can_2 = D_can[D_can_width*v_can_2+u_can_2];
                        if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
                            support = true;
                            break;
                        }
                    }
                    
                    // if we have no support => point is not redundant
                    if (!support) {
                        redundant = false;
                        break;
                    }
                }
                       
                // invalidate support point if it is redundant
                if (redundant)
                    D_can[D_can_width*v_can+u_can] = -1;
            }
        }
    }
}

void clelas::Processor::correctSupportPointCandidates(
        int16_t* candidates, int32_t candi_width, int32_t candi_height)
{
    // remove inconsistent support points
    removeInconsistentSupportPoints(candidates, candi_width, candi_height);
    
    // remove support points on straight lines, since they are redundant
    // this reduces the number of triangles a little bit and hence speeds up
    // the triangulation process
    removeRedundantSupportPoints(candidates, candi_width, candi_height, 5, 1, true);
    removeRedundantSupportPoints(candidates, candi_width, candi_height, 5, 1, false);
}

// -------------------------------------------------------------------------------------------------- 
// convertCandidatesToSupportPoints functions
// -------------------------------------------------------------------------------------------------- 
int32_t clelas::Processor::addCornerSupportPoints(
        int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, int32_t sup_num) {
    // list of border points
    const int32_t corner_num = 4;
    const int32_t border_us[corner_num] = { 0, 0, width-1, width-1 };
    const int32_t border_vs[corner_num] = { 0, height-1, 0, height-1 };
    int32_t border_ds[corner_num] = { 0, 0, 0, 0 };

    // find closest d
    for (int32_t i=0; i<corner_num; i++) {
        int32_t best_dist = std::numeric_limits<int32_t>::max();
        for (int32_t j=0; j<sup_num; j++) {
            int32_t du = border_us[i]-sup_us[j];
            int32_t dv = border_vs[i]-sup_vs[j];
            int32_t curr_dist = du*du+dv*dv;
            if (curr_dist<best_dist) {
                best_dist = curr_dist;
                border_ds[i] = sup_ds[j];
            }
        }
    }

    // add border points to support points
    for (int32_t i=0; i<corner_num; i++) {
        sup_us[sup_num] = border_us[i];
        sup_vs[sup_num] = border_vs[i];
        sup_ds[sup_num] = border_ds[i];
        sup_num++;
    }
    
    // for right image
    sup_us[sup_num] = border_us[2]+border_ds[2];
    sup_vs[sup_num] = border_vs[2];
    sup_ds[sup_num] = border_ds[2];
    sup_num++;

    sup_us[sup_num] = border_us[3]+border_ds[3];
    sup_vs[sup_num] = border_vs[3];
    sup_ds[sup_num] = border_ds[3];
    sup_num++;
    
    return sup_num;
}

int32_t clelas::Processor::convertCandidatesToSupportPoints(
        const int16_t* candidates, int32_t candi_width, int32_t candi_height, int32_t candi_step,
        HostSupportPoints& supports)
{
    // move support points from image representation into a vector representation
    int32_t sup_num=0;
    for (int32_t u_can=decltype(candi_width)(1); u_can<candi_width; u_can++) {
        for (int32_t v_can=decltype(candi_height)(1); v_can<candi_height; v_can++) {
            if (candidates[candi_width*v_can + u_can]>=0){
                supports.u[sup_num] = u_can*candi_step;
                supports.v[sup_num] = v_can*candi_step;
                supports.d[sup_num] = candidates[candi_width*v_can+u_can];
                sup_num++;
            }
        }
    }

    // if flag is set, add support points in image corners
    // with the same disparity as the nearest neighbor support point
    if (param.add_corners)
    {
        sup_num = addCornerSupportPoints(
            supports.u.data(), supports.v.data(), supports.d.data(), sup_num);
    }

    // return support point vector
    return sup_num; 
}

// -------------------------------------------------------------------------------------------------- 
// computeDelaunayTriangulation functions
// -------------------------------------------------------------------------------------------------- 
int32_t clelas::Processor::computeDelaunayTriangulation(
        const HostSupportPoints& supports, const size_t sup_num,
        HostTriangleCorners& tri, const size_t triangle_max,
        const bool right_image)
{
    // input/output structure for triangulation
    struct triangulateio in, out;

    // inputs
    in.numberofpoints = sup_num;
    in.pointlist = static_cast<float*>(malloc(in.numberofpoints*2*sizeof(float)) );
    size_t k = 0;
    if (!right_image) {
        for (int32_t i=0; i<sup_num; i++) {
            in.pointlist[k++] = supports.u[i];
            in.pointlist[k++] = supports.v[i];
        }
    } else {
        for (int32_t i=0; i<sup_num; i++) {
            in.pointlist[k++] = supports.u[i]-supports.d[i];
            in.pointlist[k++] = supports.v[i];
        }
    }

    in.numberofpointattributes = 0;
    in.pointattributelist      = NULL;
    in.pointmarkerlist         = NULL;
    in.numberofsegments        = 0;
    in.numberofholes           = 0;
    in.numberofregions         = 0;
    in.regionlist              = NULL;
    
    // output
    out.pointlist              = NULL;
    out.pointattributelist     = NULL;
    out.pointmarkerlist        = NULL;
    out.trianglelist           = NULL;
    out.triangleattributelist  = NULL;
    out.neighborlist           = NULL;
    out.segmentlist            = NULL;
    out.segmentmarkerlist      = NULL;
    out.edgelist               = NULL;
    out.edgemarkerlist         = NULL;

    // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
    char parameters[] = "zQB";
    triangulate(parameters, &in, &out, NULL);
    
    if (triangle_max < out.numberoftriangles) {
        return -1;
    }

    // put resulting triangles into vector tri
    for (auto i=decltype(out.numberoftriangles)(0); i<out.numberoftriangles; i++) {
        size_t t = i*3; 

        int32_t sup_c1 = out.trianglelist[t];
        int32_t sup_c2 = out.trianglelist[t+1];
        int32_t sup_c3 = out.trianglelist[t+2];
        
        if(!(0<=sup_c1 && sup_c1<sup_num && 
             0<=sup_c2 && sup_c2<sup_num && 
             0<=sup_c3 && sup_c3<sup_num)) {
            return -1;
        }

        tri.c1[i] = sup_c1;
        tri.c2[i] = sup_c2;
        tri.c3[i] = sup_c3;
    }
    
    // free memory used for triangulation
    free(in.pointlist);
    free(out.pointlist);
    free(out.trianglelist);
    
    // return triangles
    return out.numberoftriangles;
}

// -------------------------------------------------------------------------------------------------- 
// post-processing functions (same as the CPU version)
// -------------------------------------------------------------------------------------------------- 
void clelas::Processor::removeSmallSegments(float* D)
{
    // get disparity image dimensions
    int32_t D_width        = width;
    int32_t D_height       = height;
    int32_t D_speckle_size = param.speckle_size;

    if (param.subsampling) {
        D_width        = width/2;
        D_height       = height/2;
        D_speckle_size = sqrt((float)param.speckle_size)*2;
    }

    // allocate memory on heap for dynamic programming arrays
    int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
    int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
    int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
    int32_t seg_list_count;
    int32_t seg_list_curr;
    int32_t u_neighbor[4];
    int32_t v_neighbor[4];
    int32_t u_seg_curr;
    int32_t v_seg_curr;

    // declare loop variables
    int32_t addr_start, addr_curr, addr_neighbor;

    // for all pixels do
    for (int32_t u=0; u<D_width; u++) {
        for (int32_t v=0; v<D_height; v++) {

            // get address of first pixel in this segment
            addr_start = D_width*v+u;

            // if this pixel has not already been processed
            if (*(D_done+addr_start)==0) {

                // init segment list (add first element
                // and set it to be the next element to check)
                *(seg_list_u+0) = u;
                *(seg_list_v+0) = v;
                seg_list_count  = 1;
                seg_list_curr   = 0;

                // add neighboring segments as long as there
                // are none-processed pixels in the seg_list;
                // none-processed means: seg_list_curr<seg_list_count
                while (seg_list_curr<seg_list_count) {
                    // get current position from seg_list
                    u_seg_curr = *(seg_list_u+seg_list_curr);
                    v_seg_curr = *(seg_list_v+seg_list_curr);

                    // get address of current pixel in this segment
                    addr_curr = D_width*v_seg_curr+u_seg_curr;

                    // fill list with neighbor positions
                    u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
                    u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
                    u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
                    u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;

                    // for all neighbors do
                    for (int32_t i=0; i<4; i++) {

                        // check if neighbor is inside image
                        if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {

                            // get neighbor pixel address
                            addr_neighbor = D_width*v_neighbor[i]+u_neighbor[i];

                            // check if neighbor has not been added yet and if it is valid
                            if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

                                // is the neighbor similar to the current pixel
                                // (=belonging to the current segment)
                                if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {

                                    // add neighbor coordinates to segment list
                                    *(seg_list_u+seg_list_count) = u_neighbor[i];
                                    *(seg_list_v+seg_list_count) = v_neighbor[i];
                                    seg_list_count++;            

                                    // set neighbor pixel in I_done to "done"
                                    // (otherwise a pixel may be added 2 times to the list, as
                                    //  neighbor of one pixel and as neighbor of another pixel)
                                    *(D_done+addr_neighbor) = 1;
                                }
                            }

                        } 
                    }

                    // set current pixel in seg_list to "done"
                    seg_list_curr++;

                    // set current pixel in I_done to "done"
                    *(D_done+addr_curr) = 1;

                } // end: while (seg_list_curr<seg_list_count)

                // if segment NOT large enough => invalidate pixels
                if (seg_list_count<D_speckle_size) {
                    // for all pixels in current segment invalidate pixels
                    for (int32_t i=0; i<seg_list_count; i++) {
                        addr_curr = D_width*(*(seg_list_v+i))+(*(seg_list_u+i));
                        *(D+addr_curr) = -10;
                    }
                }
            } // end: if (*(I_done+addr_start)==0)

        }
    }

    // free memory
    free(D_done);
    free(seg_list_u);
    free(seg_list_v);
}

void clelas::Processor::gapInterpolation(float* D) {
  
    // get disparity image dimensions
    int32_t D_width          = width;
    int32_t D_height         = height;
    int32_t D_ipol_gap_width = param.ipol_gap_width;
    if (param.subsampling) {
            D_width          = width/2;
            D_height         = height/2;
            D_ipol_gap_width = param.ipol_gap_width/2+1;
    }

    // discontinuity threshold
    float discon_threshold = 3.0;

    // declare loop variables
    int32_t count,addr,v_first,v_last,u_first,u_last;
    float   d1,d2,d_ipol;

    // 1. Row-wise:
    // for each row do
    for (int32_t v=0; v<D_height; v++) {

        // init counter
        count = 0;

        // for each element of the row do
        for (int32_t u=0; u<D_width; u++) {

            // get address of this location
            addr = D_width*v+u;

            // if disparity valid
            if (*(D+addr)>=0) {

                // check if speckle is small enough
                if (count>=1 && count<=D_ipol_gap_width) {

                    // first and last value for interpolation
                    u_first = u-count;
                    u_last  = u-1;

                    // if value in range
                    if (u_first>0 && u_last<D_width-1) {

                        // compute mean disparity
                        d1 = *(D+D_width*v+(u_first-1) );
                        d2 = *(D+D_width*v+(u_last+1) );
                        if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
                        else                              d_ipol = std::min(d1,d2);

                        // set all values to d_ipol
                        for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
                                *(D+D_width*v+u_curr) = d_ipol;
                    }

                }

                // reset counter
                count = 0;

                // otherwise increment counter
            } else {
                count++;
            }
        }

        // if full size disp map requested
        if (param.add_corners) {

            // extrapolate to the left
            for (int32_t u=0; u<D_width; u++) {

                // get address of this location
                addr = D_width*v+u;

                // if disparity valid
                if (*(D+addr)>=0) {
                    for (int32_t u2=std::max(u-D_ipol_gap_width,0); u2<u; u2++)
                        *(D+D_width*v+u2) = *(D+addr);
                    break;
                }
            }

            // extrapolate to the right
            for (int32_t u=D_width-1; u>=0; u--) {

                // get address of this location
                addr = D_width*v+u;

                // if disparity valid
                if (*(D+addr)>=0) {
                    for (int32_t u2=u; u2<=std::min(u+D_ipol_gap_width,D_width-1); u2++)
                        *(D+D_width*v+u2) = *(D+addr);
                    break;
                }
            }
        }
    }

    // 2. Column-wise:
    // for each column do
    for (int32_t u=0; u<D_width; u++) {

        // init counter
        count = 0;

        // for each element of the column do
        for (int32_t v=0; v<D_height; v++) {

            // get address of this location
            addr = D_width*v+u;

            // if disparity valid
            if (*(D+addr)>=0) {

                // check if gap is small enough
                if (count>=1 && count<=D_ipol_gap_width) {

                    // first and last value for interpolation
                    v_first = v-count;
                    v_last  = v-1;

                    // if value in range
                    if (v_first>0 && v_last<D_height-1) {

                        // compute mean disparity
                        d1 = *(D+D_width*(v_first-1)+u);
                        d2 = *(D+D_width*(v_last+1)+u);
                        if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
                        else                              d_ipol = std::min(d1,d2);

                        // set all values to d_ipol
                        for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
                            *(D+D_width*v_curr+u) = d_ipol;
                    }

                }

                // reset counter
                count = 0;

                // otherwise increment counter
            } else {
                count++;
            }
        }

        // added extrapolation to top and bottom since bottom rows sometimes stay unlabeled...
        // DS 5/12/2014

        // if full size disp map requested
        if (param.add_corners) {

            // extrapolate towards top
            for (int32_t v=0; v<D_height; v++) {

                // get address of this location
                addr = D_width*v+u;

                // if disparity valid
                if (*(D+addr)>=0) {
                    for (int32_t v2=std::max(v-D_ipol_gap_width,0); v2<v; v2++)
                        *(D+D_width*v2+u) = *(D+addr);
                    break;
                }
            }

            // extrapolate towards the bottom
            for (int32_t v=D_height-1; v>=0; v--) {

                // get address of this location
                addr = D_width*v+u;

                // if disparity valid
                if (*(D+addr)>=0) {
                    for (int32_t v2=v; v2<=std::min(v+D_ipol_gap_width,D_height-1); v2++)
                            *(D+D_width*v2+u) = *(D+addr);
                    break;
                }
            }
        }
    }
}
