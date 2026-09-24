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

#include "elas.h"

#include <climits>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <math.h>
#include "triangle.h"

void Elas::process(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right)
{
    // Descriptor
#ifdef PROFILE
    auto descriptor_start = std::chrono::system_clock::now();
#endif // PROFILE
    auto* hsobel_left = (uint8_t*)malloc(width*height*sizeof(uint8_t) );
    auto* vsobel_left = (uint8_t*)malloc(width*height*sizeof(uint8_t) );
    auto* hsobel_right = (uint8_t*)malloc(width*height*sizeof(uint8_t) );
    auto* vsobel_right = (uint8_t*)malloc(width*height*sizeof(uint8_t) );

    sobel3x3(image_left, hsobel_left, vsobel_left, width, height);
    sobel3x3(image_right, hsobel_right, vsobel_right, width, height);

    // Support Matches
#ifdef PROFILE
    auto disparity_candidates_start = std::chrono::system_clock::now();
#endif // PROFILE
    size_t support_max = width*height;
    auto* support_us = (int32_t*)malloc(support_max*sizeof(int32_t) );
    auto* support_vs = (int32_t*)malloc(support_max*sizeof(int32_t) );
    auto* support_ds = (int32_t*)malloc(support_max*sizeof(int32_t) );

    auto support_num = computeSupportMatches(
        hsobel_left, hsobel_right, 
        vsobel_left, vsobel_right,
        support_us, support_vs, support_ds, support_max);
    if (support_num<3) {
        std::cout << "ERROR: Need at least 3 support points!" << std::endl;
        return;
    }

    // Delaunay Triangulation
#ifdef PROFILE
    auto delaunay_triangulation_start = std::chrono::system_clock::now();
#endif // PROFILE
    size_t triangle_max = width*height;
    auto* triangle_as_left = (int32_t*)malloc(triangle_max*sizeof(int32_t) );
    auto* triangle_bs_left = (int32_t*)malloc(triangle_max*sizeof(int32_t) );
    auto* triangle_cs_left = (int32_t*)malloc(triangle_max*sizeof(int32_t) );

    auto* triangle_as_right = (int32_t*)malloc(triangle_max*sizeof(int32_t) );
    auto* triangle_bs_right = (int32_t*)malloc(triangle_max*sizeof(int32_t) );
    auto* triangle_cs_right = (int32_t*)malloc(triangle_max*sizeof(int32_t) );
    
    const int32_t tri_num_left = computeDelaunayTriangulation(
         support_us, support_vs, support_ds, support_num, 
         triangle_as_left, triangle_bs_left, triangle_cs_left,
         false);
    const int32_t tri_num_right = computeDelaunayTriangulation(
         support_us, support_vs, support_ds, support_num, 
         triangle_as_right, triangle_bs_right, triangle_cs_right,
         true);

    // Disparity Planes
#ifdef PROFILE
    auto disparity_planes_start = std::chrono::system_clock::now();
#endif // PROFILE
    auto* triangle_t1as_left = (float*)malloc(tri_num_left*sizeof(float) );
    auto* triangle_t1bs_left = (float*)malloc(tri_num_left*sizeof(float) );
    auto* triangle_t1cs_left = (float*)malloc(tri_num_left*sizeof(float) );
    auto* triangle_t2as_left = (float*)malloc(tri_num_left*sizeof(float) );
    auto* triangle_t2bs_left = (float*)malloc(tri_num_left*sizeof(float) );
    auto* triangle_t2cs_left = (float*)malloc(tri_num_left*sizeof(float) );

    auto* triangle_t1as_right = (float*)malloc(tri_num_right*sizeof(float) );
    auto* triangle_t1bs_right = (float*)malloc(tri_num_right*sizeof(float) );
    auto* triangle_t1cs_right = (float*)malloc(tri_num_right*sizeof(float) );
    auto* triangle_t2as_right = (float*)malloc(tri_num_right*sizeof(float) );
    auto* triangle_t2bs_right = (float*)malloc(tri_num_right*sizeof(float) );
    auto* triangle_t2cs_right = (float*)malloc(tri_num_right*sizeof(float) );
    
    computeDisparityPlanes(
        support_us, support_vs, support_ds, support_num, 
        triangle_as_left, triangle_bs_left, triangle_cs_left, tri_num_left,
        triangle_t1as_left, triangle_t1bs_left, triangle_t1cs_left,
        triangle_t2as_left, triangle_t2bs_left, triangle_t2cs_left,
        false);

    computeDisparityPlanes(
        support_us, support_vs, support_ds, support_num,
        triangle_as_right, triangle_bs_right, triangle_cs_right, tri_num_right,
        triangle_t1as_right, triangle_t1bs_right, triangle_t1cs_right,
        triangle_t2as_right, triangle_t2bs_right, triangle_t2cs_right,
        true);

    // Grid
#ifdef PROFILE
    auto grid_start = std::chrono::system_clock::now();
#endif // PROFILE
    auto grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
    auto grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
    auto* grid_data_left = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width, sizeof(int32_t) );
    auto* grid_nums_left = (int32_t*)calloc(grid_height*grid_width, sizeof(int32_t) );
    auto* grid_data_right = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width, sizeof(int32_t) );
    auto* grid_nums_right = (int32_t*)calloc(grid_height*grid_width, sizeof(int32_t) );

    createGrid(
        support_us, support_vs, support_ds, support_num,
        grid_data_left, grid_nums_left, grid_width, grid_height, false);
    createGrid(
        support_us, support_vs, support_ds, support_num,
        grid_data_right, grid_nums_right, grid_width, grid_height, true);

    // TriangleMapping
#ifdef PROFILE
    auto triangle_mapping_start = std::chrono::system_clock::now();
#endif // PROFILE
    auto* tri_map_left = (uint16_t*)malloc(width*height*sizeof(uint16_t) );
    auto* tri_map_right = (uint16_t*)malloc(width*height*sizeof(uint16_t) );

    createTriangleMapping(
        support_us, support_vs, support_ds, support_num,
        triangle_as_left, triangle_bs_left, triangle_cs_left,
        triangle_t1as_left, triangle_t1bs_left, triangle_t1cs_left, 
        triangle_t2as_left, triangle_t2bs_left, triangle_t2cs_left, tri_num_left,
        tri_map_left, false);

    createTriangleMapping(
        support_us, support_vs, support_ds, support_num,
        triangle_as_right, triangle_bs_right, triangle_cs_right,
        triangle_t1as_right, triangle_t1bs_right, triangle_t1cs_right, 
        triangle_t2as_right, triangle_t2bs_right, triangle_t2cs_right, tri_num_right,
        tri_map_right, true);

    // Matching
#ifdef PROFILE
    auto disparity_start = std::chrono::system_clock::now();
#endif // PROFILE
    computeDisparity(
        support_us, support_vs, support_ds, support_num,
        triangle_as_left, triangle_bs_left, triangle_cs_left,
        triangle_t1as_left, triangle_t1bs_left, triangle_t1cs_left, 
        triangle_t2as_left, triangle_t2bs_left, triangle_t2cs_left, tri_num_left,
        grid_data_left, grid_nums_left, grid_width, grid_height,
        hsobel_left, hsobel_right, 
        vsobel_left, vsobel_right,
        tri_map_left, 
        disp_left, 
        false);

    computeDisparity(
        support_us, support_vs, support_ds, support_num,
        triangle_as_right, triangle_bs_right, triangle_cs_right, 
        triangle_t1as_right, triangle_t1bs_right, triangle_t1cs_right,
        triangle_t2as_right, triangle_t2bs_right, triangle_t2cs_right, tri_num_right,
        grid_data_right, grid_nums_right, grid_width, grid_height,
        hsobel_right, hsobel_left,
        vsobel_right, vsobel_left,
        tri_map_right, 
        disp_right, 
        true);
    
    // L/R Consistency Check.
#ifdef PROFILE
    auto lr_consistency_check_start = std::chrono::system_clock::now();
#endif // PROFILE
    leftRightConsistencyCheck(disp_left, disp_right);
#ifdef PROFILE
    auto lr_consistency_check_end = std::chrono::system_clock::now();
#endif // PROFILE

    // Post-processing (same as libelas)
    removeSmallSegments(disp_left);
    if (!param.postprocess_only_left)
        removeSmallSegments(disp_right);

    gapInterpolation(disp_left);
    if (!param.postprocess_only_left)
        gapInterpolation(disp_right);

    if (param.filter_adaptive_mean) {
        adaptiveMean(disp_left);
        if (!param.postprocess_only_left)
            adaptiveMean(disp_right);
    }

    if (param.filter_median) {
        median(disp_left);
        if (!param.postprocess_only_left)
            median(disp_right);
    }
#ifdef PROFILE
    auto postprocess_end = std::chrono::system_clock::now();
#endif // PROFILE

    // show execution times.
#ifdef PROFILE
    auto descriptor_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_candidates_start - descriptor_start).count();
    auto disparity_candidates_time = std::chrono::duration_cast<std::chrono::microseconds>(delaunay_triangulation_start - disparity_candidates_start).count();
    auto delaunay_triangulation_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_planes_start - delaunay_triangulation_start).count();
    auto disparity_planes_time = std::chrono::duration_cast<std::chrono::microseconds>(grid_start - disparity_planes_start).count();
    auto grid_time = std::chrono::duration_cast<std::chrono::microseconds>(triangle_mapping_start - grid_start).count();
    auto triangle_mapping_time = std::chrono::duration_cast<std::chrono::microseconds>(disparity_start - triangle_mapping_start).count();
    auto disparity_time = std::chrono::duration_cast<std::chrono::microseconds>(lr_consistency_check_start - disparity_start).count();
    auto lr_consistency_check_time = std::chrono::duration_cast<std::chrono::microseconds>(lr_consistency_check_end - lr_consistency_check_start).count();
    auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - lr_consistency_check_end).count();

    std::cout << "descriptor:"             << (double)descriptor_time/1000.0 << std::endl;
    std::cout << "disparity_candidates:"   << (double)disparity_candidates_time/1000.0 << std::endl;
    std::cout << "delaunay_triangulation:" << (double)delaunay_triangulation_time/1000.0 << std::endl;
    std::cout << "disparity_planes:"       << (double)disparity_planes_time/1000.0 << std::endl;
    std::cout << "grid:"                   << (double)grid_time/1000.0 << std::endl;
    std::cout << "triangle_mapping:"       << (double)triangle_mapping_time/1000.0 << std::endl;
    std::cout << "disparity:"              << (double)disparity_time/1000.0 << std::endl;
    std::cout << "lr_consistency_check:"   << (double)lr_consistency_check_time/1000.0 << std::endl;
    std::cout << "postprocess:"            << (double)postprocess_time/1000.0 << std::endl;
#endif // PROFILE

    free(hsobel_left);
    free(vsobel_left);
    free(hsobel_right);
    free(vsobel_right);

    free(support_us);
    free(support_vs);
    free(support_ds);

    free(triangle_t1as_left);
    free(triangle_t1bs_left);
    free(triangle_t1cs_left);
    free(triangle_t2as_left);
    free(triangle_t2bs_left);
    free(triangle_t2cs_left);

    free(triangle_t1as_right);
    free(triangle_t1bs_right);
    free(triangle_t1cs_right);
    free(triangle_t2as_right);
    free(triangle_t2bs_right);
    free(triangle_t2cs_right);

    free(grid_data_left);
    free(grid_nums_left);
    free(grid_data_right);
    free(grid_nums_right);
}

// -------------------------------------------------------------------------------------------------- 
// Descriptor functions
// -------------------------------------------------------------------------------------------------- 
void Elas::sobel3x3(const uint8_t *in, uint8_t *out_v, uint8_t *out_h, int w, int h) {
    const int32_t filter_v[3][3] = {{1,0,-1}, 
                                    {2,0,-2}, 
                                    {1,0,-1}};
    const int32_t filter_h[3][3] = {{1,2,1}, 
                                    {0,0,0}, 
                                    {-1,-2,-1}};
    
    for (int32_t v = 1; v < h - 1; v++) {
        for (int32_t u = 1; u < w - 1; u++) {
            out_v[u + v * w] = filter3x3(in, filter_v, u, v, w);
            out_h[u + v * w] = filter3x3(in, filter_h, u, v, w);
        }
    }
}

// -------------------------------------------------------------------------------------------------- 
// computeSupportMatches functions
// -------------------------------------------------------------------------------------------------- 
void Elas::removeInconsistentSupportPoints(
    int16_t* D_can, const int32_t D_can_width, const int32_t D_can_height) {
  
    // for all valid support points do
    for (int32_t u_can=0; u_can<D_can_width; u_can++) {
        for (int32_t v_can=0; v_can<D_can_height; v_can++) {
            int16_t d_can = D_can[D_can_width*v_can+u_can];
            if (d_can>=0) {
              
                // compute number of other points supporting the current point
                int32_t support = 0;
                for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
                    for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
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

void Elas::removeRedundantSupportPoints(
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

int32_t Elas::addCornerSupportPoints(
        int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, int32_t sup_num) {
    // list of border points
    const int32_t corner_num = 4;
    const int32_t border_us[corner_num] = { 0, 0, width-1, width-1 };
    const int32_t border_vs[corner_num] = { 0, height-1, 0, height-1 };
    int32_t border_ds[corner_num] = { 0, 0, 0, 0};

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

int16_t Elas::computeMatchingDisparity(
        const int32_t u, const int32_t v,
        const uint8_t* hsleft, const uint8_t* hsright,
        const uint8_t* vsleft, const uint8_t* vsright,
        const bool right_image)
{
    const int32_t u_step      = 6;
    const int32_t v_step      = 2;
    const int32_t disp_offset = 2;
    
    // check if we are inside the image region
    if (!(u>disp_offset + u_step && u < width - disp_offset - u_step - 1 && 
                    v > disp_offset+v_step && v <= height - disp_offset - v_step - 1)) {
        return -1;
    }

    // we require at least some texture
    int32_t sum = sumAbsoluteErrorAtDescriptor(
                    hsleft+width*v+u, 
                    vsleft+width*v+u,
                    128,
                    width);
    if (sum<param.support_texture) return -1;
    
    // best match
    int16_t min_1_E = std::numeric_limits<int16_t>::max();
    int16_t min_1_d = -1;
    int16_t min_2_E = std::numeric_limits<int16_t>::max();
    int16_t min_2_d = -1;
  
    int32_t disp_min_valid = std::max(0, param.disp_min);
    int32_t disp_max_valid = !right_image ? 
                       std::min(u-disp_offset-u_step-1, param.disp_max) : 
                       std::min(width-u-disp_offset-u_step-1, param.disp_max);
   
    // get valid disparity range
    if (disp_max_valid-disp_min_valid<10) return -1;
  
    // for all disparities do
    for (auto d = disp_min_valid;  d <= disp_max_valid; d++) {
        int32_t u_warp = !right_image ? u-d : u+d;
        int32_t ul_offset = width*(v-v_step)-u_step;
        int32_t ur_offset = width*(v-v_step)+u_step;
        int32_t dl_offset = width*(v+v_step)-u_step;
        int32_t dr_offset = width*(v+v_step)+u_step;

        // compute match energy at this disparity
        sum = 0;
        sum += sumAbsoluteErrorAtDescriptor(
                hsleft+ul_offset+u, 
                vsleft+ul_offset+u, 
                hsright+ul_offset+u_warp, 
                vsright+ul_offset+u_warp,
                width);
        sum += sumAbsoluteErrorAtDescriptor(
                hsleft+ur_offset+u, 
                vsleft+ur_offset+u, 
                hsright+ur_offset+u_warp, 
                vsright+ur_offset+u_warp,
                width);
        sum += sumAbsoluteErrorAtDescriptor(
                hsleft+dl_offset+u, 
                vsleft+dl_offset+u, 
                hsright+dl_offset+u_warp, 
                vsright+dl_offset+u_warp,
                width);
        sum += sumAbsoluteErrorAtDescriptor(
                hsleft+dr_offset+u, 
                vsleft+dr_offset+u, 
                hsright+dr_offset+u_warp, 
                vsright+dr_offset+u_warp,
                width);

        // best + second best match
        if (sum<min_1_E) {
          min_2_E = min_1_E;
          min_2_d = min_1_d;
          min_1_E = sum;
          min_1_d = d;
        } else if (sum<min_2_E) {
          min_2_E = sum;
          min_2_d = d;
        }
    }
  
    // check if best and second best match are available and if matching ratio is sufficient
    if (min_1_d>=0 && min_2_d>=0 && \
                    (float)min_1_E<param.support_threshold*(float)min_2_E)
        return min_1_d;
    else
        return -1;
}

int32_t Elas::computeSupportMatches(
        const uint8_t* hsleft, const uint8_t* hsright, 
        const uint8_t* vsleft, const uint8_t* vsright,
        int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, const size_t sup_max)
{
    // be sure that at half resolution we only need data from every second line!
    const int32_t candi_step = param.subsampling ? \
        param.candidate_stepsize + (param.candidate_stepsize%2) : param.candidate_stepsize;

    // create matrix for saving disparity candidates
    const int32_t candi_width  = (width+candi_step-1)/candi_step;
    const int32_t candi_height = (height+candi_step-1)/candi_step;
    const auto candi_size = candi_width*candi_height;
    auto* candies = static_cast<int16_t*>(malloc(candi_size*sizeof(int16_t)) );
    for(auto i=decltype(candi_size)(0); i<candi_size; i++){
        candies[i] = -1;
    }

    // loop variables
    int32_t u;
    int32_t v;
    int16_t d,d2;

    // for all point candidates in image 1 do
    for (auto u_can=decltype(candi_width)(1); u_can<candi_width-1; u_can++) {
        u = u_can*candi_step;
        for (auto v_can=decltype(candi_height)(1); v_can<candi_height-1; v_can++) {
            v = v_can*candi_step;

            // find forwards
            d = computeMatchingDisparity(
                            u, v, 
                            hsleft, hsright,
                            vsleft, vsright,
                            false);

            if (d>=0){
                // find backwards
                d2 = computeMatchingDisparity(
                                u-d, v,
                                hsright, hsleft, 
                                vsright, vsleft, 
                                true);
                if (d2>=0 && abs(d-d2)<=param.lr_threshold)
                    candies[candi_width*v_can + u_can] = d;
            }
        }
    }

    // remove inconsistent support points
    removeInconsistentSupportPoints(candies, candi_width, candi_height);

    // remove support points on straight lines, since they are redundant
    // this reduces the number of triangles a little bit and hence speeds up
    // the triangulation process
    removeRedundantSupportPoints(candies, candi_width, candi_height, 5, 1, true);
    removeRedundantSupportPoints(candies, candi_width, candi_height, 5, 1, false);

    // move support points from image representation into a vector representation
    int32_t sup_num=0;
    for (int32_t u_can=decltype(candi_width)(1); u_can<candi_width; u_can++) {
        for (int32_t v_can=decltype(candi_height)(1); v_can<candi_height; v_can++) {
            if (candies[candi_width*v_can + u_can]>=0){
                sup_us[sup_num] = u_can*candi_step;
                sup_vs[sup_num] = v_can*candi_step;
                sup_ds[sup_num] = candies[candi_width*v_can+u_can];
                sup_num++;
            }
        }
    }

    // if flag is set, add support points in image corners
    // with the same disparity as the nearest neighbor support point
    if (param.add_corners)
    {
        sup_num = addCornerSupportPoints(
            sup_us, sup_vs, sup_ds, sup_num);
    }

    // return support point vector
    free(candies);
    return sup_num; 
}

// -------------------------------------------------------------------------------------------------- 
// computeDelaunayTriangulation functions
// -------------------------------------------------------------------------------------------------- 
int32_t Elas::computeDelaunayTriangulation(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, const size_t sup_num, 
        int32_t* tri_as, int32_t* tri_bs, int32_t* tri_cs,
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
            in.pointlist[k++] = sup_us[i];
            in.pointlist[k++] = sup_vs[i];
        }
    } else {
        for (int32_t i=0; i<sup_num; i++) {
            in.pointlist[k++] = sup_us[i]-sup_ds[i];
            in.pointlist[k++] = sup_vs[i];
        }
    }

    in.numberofpointattributes = 0;
    in.pointattributelist      = NULL;
    in.pointmarkerlist         = NULL;
    in.numberofsegments        = 0;
    in.numberofholes           = 0;
    in.numberofregions         = 0;
    in.regionlist              = NULL;
    
    // outputs
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
    
    // put resulting triangles into vector tri
    for (auto i=decltype(out.numberoftriangles)(0); i<out.numberoftriangles; i++) {
        size_t t = i*3; 
        tri_as[i] = out.trianglelist[t];
        tri_bs[i] = out.trianglelist[t+1];
        tri_cs[i] = out.trianglelist[t+2];
    }
    
    // free memory used for triangulation
    free(in.pointlist);
    free(out.pointlist);
    free(out.trianglelist);
    
    // return triangles
    return out.numberoftriangles;
}

// -------------------------------------------------------------------------------------------------- 
// computeDisparityPlanes functions
// -------------------------------------------------------------------------------------------------- 
void Elas::computeDisparityPlanes (
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, const size_t sup_num,
        const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs, const size_t tri_num,
        float* tri_t1as, float* tri_t1bs, float* tri_t1cs, 
        float* tri_t2as, float* tri_t2bs, float* tri_t2cs, 
        const bool right_image)
{
    // for all triangles do
    for (int32_t i=0; i<tri_num; i++) {
      
        // get triangle corner indices
        int32_t c1 = tri_as[i];
        int32_t c2 = tri_bs[i];
        int32_t c3 = tri_cs[i];
        
        // compute matrix A for linear system of left triangle
        // a, b, d
        int32_t a_u = sup_us[c1]; 
        int32_t a_v = sup_vs[c1]; 
        int32_t a_d = sup_ds[c1];
        
        int32_t b_u = sup_us[c2]; 
        int32_t b_v = sup_vs[c2]; 
        int32_t b_d = sup_ds[c2];
        
        int32_t c_u = sup_us[c3]; 
        int32_t c_v = sup_vs[c3]; 
        int32_t c_d = sup_ds[c3];
        
        int32_t det_inv = a_u*b_v+a_v*c_u+b_u*c_v-b_v*c_u-a_v*b_u-a_u*c_v;

        if(det_inv!=0) {
            tri_t1as[i] = (1/(float)det_inv)*((b_v-c_v)*a_d+(-a_v+c_v)*b_d+(a_v-b_v)*c_d);
            tri_t1bs[i] = (1/(float)det_inv)*((-b_u+c_u)*a_d+(a_u-c_u)*b_d+(-a_u+b_u)*c_d);
            tri_t1cs[i] = (1/(float)det_inv)*((b_u*c_v-b_v*c_u)*a_d+(-a_u*c_v+a_v*c_u)*b_d+(a_u*b_v-a_v*b_u)*c_d);

        } else {
            tri_t1as[i] = 0;
            tri_t1bs[i] = 0;
            tri_t1cs[i] = 0;
        }

        // compute matrix A for linear system of right triangle
        a_u = sup_us[c1]-sup_ds[c1]; 
        b_u = sup_us[c2]-sup_ds[c2]; 
        c_u = sup_us[c3]-sup_ds[c3]; 
 
        det_inv = a_u*b_v+a_v*c_u+b_u*c_v-b_v*c_u-a_v*b_u-a_u*c_v;

        if(det_inv!=0) {
            tri_t2as[i] = (1/(float)det_inv)*((b_v-c_v)*a_d+(-a_v+c_v)*b_d+(a_v-b_v)*c_d);
            tri_t2bs[i] = (1/(float)det_inv)*((-b_u+c_u)*a_d+(a_u-c_u)*b_d+(-a_u+b_u)*c_d);
            tri_t2cs[i] = (1/(float)det_inv)*((b_u*c_v-b_v*c_u)*a_d+(-a_u*c_v+a_v*c_u)*b_d+(a_u*b_v-a_v*b_u)*c_d);

        } else {
            tri_t2as[i] = 0;
            tri_t2bs[i] = 0;
            tri_t2cs[i] = 0;
        }
    }  
}

// -------------------------------------------------------------------------------------------------- 
// createGrid functions
// -------------------------------------------------------------------------------------------------- 
void Elas::createGrid(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, const size_t sup_num,
        int32_t* disparity_grid, int32_t* grid_nums, 
        const int32_t grid_width, const int32_t grid_height, const bool right_image)
{
    const auto disp_num = param.disp_max+1;

    // allocate temporary memory
    auto* disp_valids = static_cast<int32_t*>(calloc(disp_num*grid_height*grid_width, sizeof(int32_t)) );
    
    // for all support points do
    for (auto i=decltype(sup_num)(0); i<sup_num; i++) {
        // compute disparity range to fill for this support point.
        auto x_curr = sup_us[i];
        auto y_curr = sup_vs[i];
        auto d_curr = sup_ds[i];
        int32_t d_min  = std::max(d_curr-1, 0);
        int32_t d_max  = std::min(d_curr+1, param.disp_max);

        int32_t grid_x;
        int32_t grid_y;
        if(!right_image) {
            grid_x = std::floor(static_cast<float>(x_curr/param.grid_size));
            grid_y = std::floor(static_cast<float>(y_curr)/static_cast<float>(param.grid_size) );

        } else {
            grid_x = std::floor(static_cast<float>(x_curr-d_curr)/static_cast<float>(param.grid_size) );
            grid_y = std::floor(static_cast<float>(y_curr)/static_cast<float>(param.grid_size) );
        }

        // fill disparity grid helper
        if (grid_x>=0 && grid_x<grid_width &&
                grid_y>=0 && grid_y<grid_height) {
            for (auto d=d_min; d<=d_max; d++) {
                // point may potentially lay outside (corner points)
                disp_valids[grid_width*grid_height*d+grid_width*grid_y+grid_x] = 1;
            }
        }
    }

    // for all grid positions create disparity grid
    for (auto x=decltype(grid_width)(1); x<grid_width-1; x++) {
        for (auto y=decltype(grid_height)(1); y<grid_height-1; y++) {
            // start with second value (first is reserved for count)
            int32_t curr_ind = 0;
            
            // for all disparities do
            for (auto d=decltype(param.disp_max)(0); d<=param.disp_max; d++) {
                // create diffuse temporary grid.
                int32_t disp_offset = grid_width*grid_height*d;
                int32_t valid = \
                    disp_valids[disp_offset+grid_width*(y-1)+(x-1)] | \
                    disp_valids[disp_offset+grid_width*(y-1)+(x+0)] | \
                    disp_valids[disp_offset+grid_width*(y-1)+(x+1)] | \
                    disp_valids[disp_offset+grid_width*(y+0)+(x-1)] | \
                    disp_valids[disp_offset+grid_width*(y+0)+(x+0)] | \
                    disp_valids[disp_offset+grid_width*(y+0)+(x+1)] | \
                    disp_valids[disp_offset+grid_width*(y+1)+(x-1)] | \
                    disp_valids[disp_offset+grid_width*(y+1)+(x+0)] | \
                    disp_valids[disp_offset+grid_width*(y+1)+(x+1)];

                // if yes => add this disparity to current cell
                if (valid>0) {
                    disparity_grid[grid_width*grid_height*curr_ind+grid_width*y+x] = d;
                    curr_ind++;
                }
            }
            
            // finally set number of indices
            grid_nums[y*grid_width+x] = curr_ind;
        }
    }
    // release temporary memory
    free(disp_valids);
}

// -------------------------------------------------------------------------------------------------- 
// createTriangleMapping functions
// -------------------------------------------------------------------------------------------------- 
void Elas::createTriangleMapping(
    const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, const size_t sup_num, 
    const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs,
    const float* tri_t1as, const float* tri_t1bs, const float* tri_t1cs,
    const float* tri_t2as, const float* tri_t2bs, const float* tri_t2cs, const size_t tri_num,
    uint16_t* tri_map, const bool right_image)
{
    // for all triangles do
    for (auto i=decltype(tri_num)(0); i<tri_num; i++) {
        // get plane parameters
        float plane_a;
        float plane_b;
        float plane_c;
        float plane_d;

        if (!right_image) {
            plane_a = tri_t1as[i];
            plane_b = tri_t1bs[i];
            plane_c = tri_t1cs[i];
            plane_d = tri_t2as[i];
        } else {
            plane_a = tri_t2as[i];
            plane_b = tri_t2bs[i];
            plane_c = tri_t2cs[i];
            plane_d = tri_t1as[i];
        }
        
        // triangle corners
        int32_t a = tri_as[i];
        int32_t b = tri_bs[i];
        int32_t c = tri_cs[i];
  
        // sort triangle corners wrt. v (ascending)
        float tri_u[3];
        if (!right_image) {
            tri_u[0] = sup_us[a];
            tri_u[1] = sup_us[b];
            tri_u[2] = sup_us[c];
        } else {
            tri_u[0] = sup_us[a]-sup_ds[a];
            tri_u[1] = sup_us[b]-sup_ds[b];
            tri_u[2] = sup_us[c]-sup_ds[c];
        }

        float tri_v[3] = {
            static_cast<float>(sup_vs[a]), 
            static_cast<float>(sup_vs[b]), 
            static_cast<float>(sup_vs[c])};
        
        for (uint32_t j=0; j<3; j++) {
            for (uint32_t k=0; k<j; k++) {
                if (tri_v[k]>tri_v[j]) {
                    float tri_u_temp = tri_u[j]; 
                    tri_u[j] = tri_u[k]; 
                    tri_u[k] = tri_u_temp;

                    float tri_v_temp = tri_v[j]; 
                    tri_v[j] = tri_v[k]; 
                    tri_v[k] = tri_v_temp;
                }
            }
        }
        
        // rename corners
        float A_u = tri_u[0]; 
        float A_v = tri_v[0];
        
        float B_u = tri_u[1]; 
        float B_v = tri_v[1];
        
        float C_u = tri_u[2];
        float C_v = tri_v[2];
        
        // compute straight lines connecting triangle corners
        float AB_a = 0; 
        float AC_a = 0; 
        float BC_a = 0;

        if ((int32_t)(A_v)!=(int32_t)(B_v)) {
            AB_a = (A_u-B_u)/(A_v-B_v);
        }
        if ((int32_t)(A_v)!=(int32_t)(C_v)) {
            AC_a = (A_u-C_u)/(A_v-C_v);
        }
        if ((int32_t)(B_v)!=(int32_t)(C_v)) {
            BC_a = (B_u-C_u)/(B_v-C_v);
        }

        float AB_b = A_u-AB_a*A_v;
        float AC_b = A_u-AC_a*A_v;
        float BC_b = B_u-BC_a*B_v;
        
        // a plane is only valid if itself and its projection
        // into the other image is not too much slanted
        bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

        // first part (triangle corner A->B)
        if ((int32_t)(A_v)!=(int32_t)(B_v)) {
            for (int32_t v=std::max((int32_t)A_v,0); v<std::min((int32_t)B_v, height); v++){
                if (!param.subsampling || v%2==0) {
                    const float u_1 = AC_a*(float)v+AC_b;
                    const float u_2 = AB_a*(float)v+AB_b;
                    const int32_t u_min = static_cast<int32_t>(std::ceil(std::min<float>(u_1,u_2)));
                    const int32_t u_max = static_cast<int32_t>(std::floor(std::max<float>(u_1,u_2)));
                    for (int32_t u=std::max(u_min,0); u<=std::min(u_max,width-1); u++) {
                        if (!param.subsampling || u%2==0) {
                            tri_map[width*v+u] = i+1;
                        }
                    }
                }
            }
        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_v)!=(int32_t)(C_v)) {
            for (int32_t v=std::max((int32_t)B_v,0); v<std::min((int32_t)C_v, height); v++){
                if (!param.subsampling || v%2==0) {
                    const float u_1 = AC_a*(float)v+AC_b;
                    const float u_2 = BC_a*(float)v+BC_b;
                    const int32_t u_min = static_cast<int32_t>(std::ceil(std::min<float>(u_1,u_2)));
                    const int32_t u_max = static_cast<int32_t>(std::floor(std::max<float>(u_1,u_2)));
                    for (int32_t u=std::max(u_min,0); u<=std::min(u_max,width-1); u++) {
                        if (!param.subsampling || u%2==0) {
                            tri_map[width*v+u] = i+1;
                        }
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------- 
// computeDisparity functions
// -------------------------------------------------------------------------------------------------- 
inline void Elas::findMatch(
        const int32_t u, const int32_t v,
        const float plane_a, const float plane_b, const float plane_c,
        const int32_t* disparity_grid, const int32_t* grid_nums, 
        const int32_t grid_width, const int32_t grid_height,
        const uint8_t* hsleft, const uint8_t* hsright, 
        const uint8_t* vsleft, const uint8_t* vsright, 
        const int32_t *P, const int32_t plane_radius,
        float* disp,
        const bool valid, const bool right_image)
{
    // get image width and height
    const int32_t disp_num = param.disp_max+1;
    const int32_t window_size = 2;

    // check if u is ok
    if (u<window_size || u>=width-window_size) {
        return;
    }

    // compute line start address
    const size_t line_offset = width * std::max(std::min(v, height-(window_size+1)), window_size);

    // does this patch have enough texture?
    const auto sum = sumAbsoluteErrorAtDescriptor(
                    hsleft+line_offset + u, 
                    vsleft+line_offset + u,
                    128,
                    width);
    if (sum<param.match_texture) return;

    // compute disparity, min disparity and max disparity of plane prior
    const auto d_plane     = static_cast<int32_t>(plane_a*static_cast<float>(u)+plane_b*static_cast<float>(v)+plane_c);
    const auto d_plane_min = std::max(d_plane-plane_radius, 0);
    const auto d_plane_max = std::min(d_plane+plane_radius, disp_num-1);

    // get grid pointer
    const auto grid_x = static_cast<int32_t>(floor(static_cast<float>(u)/static_cast<float>(param.grid_size)) );
    const auto grid_y = static_cast<int32_t>(floor(static_cast<float>(v)/static_cast<float>(param.grid_size)) );
    const auto grid_num  = grid_nums[grid_width*grid_y+grid_x];
    
    // loop variables
    int32_t d_curr;
    int32_t u_warp;
    int32_t val;

    int32_t min_val = std::numeric_limits<int32_t>::max();
    int32_t min_d   = -1;

    // left image
    if (!right_image) { 
        for (auto i=decltype(grid_num)(0); i<grid_num; i++) {
            d_curr = disparity_grid[grid_width*grid_height*i+grid_width*grid_y+grid_x];

            if (d_curr<d_plane_min || d_curr>d_plane_max) {
                u_warp = u-d_curr;
                if (u_warp<window_size || u_warp>=width-window_size){
                    continue;
                }

                val = 0;
                val += sumAbsoluteErrorAtDescriptor(
                    hsleft + line_offset + u,
                    vsleft + line_offset + u,
                    hsright + line_offset + u_warp,
                    vsright + line_offset + u_warp,
                    width);

                if (val<min_val) {
                    min_val = val;
                    min_d   = d_curr;
                }
            }
        }

        for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
            u_warp = u-d_curr;
            if (u_warp<window_size || u_warp>=width-window_size) {
                continue;
            }

            val = valid ? P[abs(d_curr-d_plane)] : 0;
            val += sumAbsoluteErrorAtDescriptor(
                hsleft + line_offset + u,
                vsleft + line_offset + u,
                hsright + line_offset + u_warp,
                vsright + line_offset + u_warp,
                width);

            if (val<min_val) {
              min_val = val;
              min_d   = d_curr;
            }
        }
      
    // right image
    } else {
        for (int32_t i=0; i<grid_num; i++) {
            d_curr = disparity_grid[grid_width*grid_height*i+grid_width*grid_y+grid_x];

            if (d_curr<d_plane_min || d_curr>d_plane_max) {
                u_warp = u+d_curr;
                if (u_warp<window_size || u_warp>=width-window_size) {
                    continue;
                }

                val = 0;
                val += sumAbsoluteErrorAtDescriptor(
                    hsleft + line_offset + u,
                    vsleft + line_offset + u,
                    hsright + line_offset + u_warp,
                    vsright + line_offset + u_warp,
                    width);

                if (val<min_val) {
                    min_val = val;
                    min_d   = d_curr;
                }
            }
        }
        for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
            u_warp = u+d_curr;
            if (u_warp<window_size || u_warp>=width-window_size){
                continue;
            }

            val = valid ? P[abs(d_curr-d_plane)] : 0;
            val += sumAbsoluteErrorAtDescriptor(
                hsleft + line_offset + u,
                vsleft + line_offset + u,
                hsright + line_offset + u_warp,
                vsright + line_offset + u_warp,
                width);

            if (val<min_val) {
              min_val = val;
              min_d   = d_curr;
            }
        }
    }

    // set disparity value
    int32_t d_u;
    int32_t d_v;
    int32_t d_width;
    if (!param.subsampling) {
        d_u = u;
        d_v = v;
        d_width = width;
    } else {
        d_u = u/2;
        d_v = v/2;
        d_width = width/2;
    }

    if (min_d>=0){
        disp[d_width*d_v+d_u] = min_d; // MAP value (min neg-Log probability)
    } else {
        disp[d_width*d_v+d_u] = -1;    // invalid disparity
    }
}

void Elas::computeDisparity(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, const size_t sup_num,
        const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs,
        const float* tri_t1as, const float* tri_t1bs, const float* tri_t1cs,
        const float* tri_t2as, const float* tri_t2bs, const float* tri_t2cs, const size_t tri_num,
        const int32_t* disparity_grid, const int32_t* grid_nums, const int32_t grid_width, const int32_t grid_height,
        const uint8_t* hsleft, const uint8_t* hsright, const uint8_t* vsleft, const uint8_t* vsright, 
        const uint16_t* tri_map, float* disp, const bool right_image)
{
    const int32_t disp_num  = param.disp_max+1;
    const int32_t window_size = 2;
    
    // disparity image dimensions: with subsampling, only pixels with even u and v
    // are computed, into a width/2 x height/2 map (as in libelas)
    const int32_t d_width  = param.subsampling ? width/2 : width;
    const int32_t d_height = param.subsampling ? height/2 : height;

    // init disparity image to -10
    size_t pixel_num = static_cast<size_t>(d_width)*d_height;
    for (auto i=decltype(pixel_num)(0); i<pixel_num; i++) disp[i] = -10;
    
    // pre-compute prior 
    float two_sigma_squared = 2*param.sigma*param.sigma;
    auto plane_radius = static_cast<int32_t>(std::max(static_cast<float>(ceil(param.sigma*param.sradius)),static_cast<float>(2.0)) );

    auto* P = static_cast<int32_t*>(malloc(disp_num*sizeof(int32_t)) );
    for (auto delta_d=decltype(disp_num)(0); delta_d<disp_num; delta_d++) {
        P[delta_d] = static_cast<int32_t>((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
    }
    
    // for all triangles do
    for(auto v=decltype(height)(0); v<height; v++){
        for(auto u=decltype(width)(0); u<width; u++){
            if (param.subsampling && (u%2!=0 || v%2!=0 || u/2>=d_width || v/2>=d_height)) {
                continue;
            }
            const int32_t d_u = param.subsampling ? u/2 : u;
            const int32_t d_v = param.subsampling ? v/2 : v;

            // get plane parameters
            auto i = tri_map[width*v+u];
            if(i==0) {
                disp[d_width*d_v+d_u] = -1;
                continue;
            } else {
                i--;
            }

            float plane_a;
            float plane_b;
            float plane_c;
            float plane_d;

            if (!right_image) {
                plane_a = tri_t1as[i];
                plane_b = tri_t1bs[i];
                plane_c = tri_t1cs[i];
                plane_d = tri_t2as[i];
            } else {
                plane_a = tri_t2as[i];
                plane_b = tri_t2bs[i];
                plane_c = tri_t2cs[i];
                plane_d = tri_t1as[i];
            }

            // a plane is only valid if itself and its projection
            // into the other image is not too much slanted
            bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;
                
            findMatch(
                u, v, 
                plane_a, plane_b, plane_c,
                disparity_grid, grid_nums,
                grid_width, grid_height,
                hsleft, hsright, vsleft, vsright, 
                P, plane_radius,
                disp,
                valid, right_image);
        }
    }

    delete[] P;
}

// -------------------------------------------------------------------------------------------------- 
// leftRightConsistencyCheck functions
// -------------------------------------------------------------------------------------------------- 
void Elas::leftRightConsistencyCheck(float* disp_left, float* disp_right) {
    
    // get disparity image dimensions
    int32_t D_width  = width;
    int32_t D_height = height;
    if (param.subsampling) {
        D_width  = width/2;
        D_height = height/2;
    }
    
    // make a copy of both images
    auto* disp_left_copy = (float*)malloc(D_width*D_height*sizeof(float));
    auto* disp_right_copy = (float*)malloc(D_width*D_height*sizeof(float));
    memcpy(disp_left_copy,disp_left,D_width*D_height*sizeof(float));
    memcpy(disp_right_copy,disp_right,D_width*D_height*sizeof(float));

    // loop variables
    uint32_t addr,addr_warp;
    float    u_warp_1,u_warp_2,d1,d2;
    
    // for all image points do
    for (int32_t v=0; v<D_height; v++) {
        for (int32_t u=0; u<D_width; u++) {
            
            // compute address (u,v) and disparity value
            d1       = disp_left_copy[D_width*v+u];
            d2       = disp_right_copy[D_width*v+u];
            if (param.subsampling) {
              u_warp_1 = (float)u-d1/2;
              u_warp_2 = (float)u+d2/2;
            } else {
              u_warp_1 = (float)u-d1;
              u_warp_2 = (float)u+d2;
            }
            
            // check if left disparity is valid
            if (d1>=0 && u_warp_1>=0 && u_warp_1<D_width) {       
                // if check failed
                if (fabs(disp_right_copy[D_width*v+(int32_t)u_warp_1]-d1)>param.lr_threshold){
                    disp_left[D_width*v+u] = -10;
                }
            // set invalid
            } else {
                disp_left[D_width*v+u] = -10;
            }

            // check if right disparity is valid
            if (d2>=0 && u_warp_2>=0 && u_warp_2<D_width) {       
                // if check failed
                if (fabs(disp_left_copy[D_width*v+(int32_t)u_warp_2]-d2)>param.lr_threshold){
                    disp_right[D_width*v+u] = -10;
                }
            // set invalid
            } else {
                disp_right[D_width*v+u] = -10;
            }
        }
    }
    
    // release memory
    free(disp_left_copy);
    free(disp_right_copy);
}

// -------------------------------------------------------------------------------------------------- 
// post-processing functions (same as libelas)
// -------------------------------------------------------------------------------------------------- 
void Elas::removeSmallSegments(float* D)
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

void Elas::gapInterpolation(float* D) {
  
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

namespace {

// |x| as computed by libelas: its absolute mask is _mm_set1_ps(0x7FFFFFFF), which is
// the float 2147483648.0f (bits 0x4F000000), so the AND keeps only some exponent bits.
// The weight 4-|x| of the bilateral filter then becomes about 4 for |x| < 2,
// 2 for 2 <= |x| < 8 and 0 above.
inline float libelasAbs(float x)
{
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits &= 0x4F000000u;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

// One step of the bilateral filter of libelas, with the same float operations as its
// SSE code (lanes k and k+4 are added first). val holds `taps` (4 or 8) disparities
// in ring buffer order, val_curr is the disparity at the center.
// Returns whether the filtered disparity d is valid.
inline bool bilateralMean(const float* val, int32_t taps, float val_curr, float& d)
{
    float weight[4];
    float factor[4];
    for (int32_t k=0; k<4; k++) {
        float weight_1 = 4.0f-libelasAbs(val[k]-val_curr);
        weight_1 = (0.0f>weight_1) ? 0.0f : weight_1;
        const float factor_1 = val[k]*weight_1;
        if (taps==8) {
            float weight_2 = 4.0f-libelasAbs(val[k+4]-val_curr);
            weight_2 = (0.0f>weight_2) ? 0.0f : weight_2;
            const float factor_2 = val[k+4]*weight_2;
            weight[k] = weight_1+weight_2;
            factor[k] = factor_1+factor_2;
        } else {
            weight[k] = weight_1;
            factor[k] = factor_1;
        }
    }
    const float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
    const float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];
    if (weight_sum>0) {
        d = factor_sum/weight_sum;
        return d>=0;
    }
    return false;
}

// Inserts temp into the sorted values vals[0..count-1] (insertion sort as in libelas).
inline void insertSorted(float* vals, int32_t count, float temp)
{
    int32_t i = count-1;
    while (i>=0 && vals[i]>temp) {
        vals[i+1] = vals[i];
        i--;
    }
    vals[i+1] = temp;
}

} // namespace

void Elas::adaptiveMean(float* D) {

    // get disparity image dimensions
    int32_t D_width  = width;
    int32_t D_height = height;
    if (param.subsampling) {
        D_width  = width/2;
        D_height = height/2;
    }
    const size_t pixel_num = static_cast<size_t>(D_width)*D_height;

    // set invalid disparities to -10 (this makes the bilateral weights of all
    // valid disparities to 0 in this region). libelas leaves D_tmp of valid
    // pixels uninitialized, here it starts as a copy of the input.
    std::vector<float> D_copy(D, D+pixel_num);
    for (auto& d : D_copy) {
        if (d<0) d = -10;
    }
    std::vector<float> D_tmp = D_copy;

    // when doing subsampling: 4 pixel bilateral filter width, full resolution: 8 pixel
    const int32_t taps   = param.subsampling ? 4 : 8;
    const int32_t center = taps/2-1;
    float val[8];
    float d;

    // horizontal filter
    for (int32_t v=3; v<D_height-3; v++) {
        for (int32_t u=0; u<taps-1; u++)
            val[u] = D_copy[D_width*v+u];
        for (int32_t u=taps-1; u<D_width; u++) {
            const float val_curr = D_copy[D_width*v+(u-center)];
            val[u%taps] = D_copy[D_width*v+u];
            if (bilateralMean(val, taps, val_curr, d))
                D_tmp[D_width*v+(u-center)] = d;
        }
    }

    // vertical filter
    for (int32_t u=3; u<D_width-3; u++) {
        for (int32_t v=0; v<taps-1; v++)
            val[v] = D_tmp[D_width*v+u];
        for (int32_t v=taps-1; v<D_height; v++) {
            const float val_curr = D_tmp[D_width*(v-center)+u];
            val[v%taps] = D_tmp[D_width*v+u];
            if (bilateralMean(val, taps, val_curr, d))
                D[D_width*(v-center)+u] = d;
        }
    }
}

void Elas::median(float* D) {

    // get disparity image dimensions
    int32_t D_width  = width;
    int32_t D_height = height;
    if (param.subsampling) {
        D_width  = width/2;
        D_height = height/2;
    }

    // temporary memory
    std::vector<float> D_temp(static_cast<size_t>(D_width)*D_height, 0.0f);

    const int32_t window_size = 3;
    float vals[window_size*2+1];

    // first step: horizontal median filter
    for (int32_t u=window_size; u<D_width-window_size; u++) {
        for (int32_t v=window_size; v<D_height-window_size; v++) {
            if (D[D_width*v+u]>=0) {
                int32_t j = 0;
                for (int32_t u2=u-window_size; u2<=u+window_size; u2++)
                    insertSorted(vals, j++, D[D_width*v+u2]);
                D_temp[D_width*v+u] = vals[window_size];
            } else {
                D_temp[D_width*v+u] = D[D_width*v+u];
            }
        }
    }

    // second step: vertical median filter
    for (int32_t u=window_size; u<D_width-window_size; u++) {
        for (int32_t v=window_size; v<D_height-window_size; v++) {
            if (D[D_width*v+u]>=0) {
                int32_t j = 0;
                for (int32_t v2=v-window_size; v2<=v+window_size; v2++)
                    insertSorted(vals, j++, D_temp[D_width*v2+u]);
                D[D_width*v+u] = vals[window_size];
            }
        }
    }
}
