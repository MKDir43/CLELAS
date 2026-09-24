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

#ifndef __ELAS_H__
#define __ELAS_H__

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>

// define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

class Elas {
  
public:
  
    enum setting {ROBOTICS,MIDDLEBURY};
    
    // parameter settings
    struct parameters {
      int32_t disp_min;               // min disparity
      int32_t disp_max;               // max disparity
      float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
      int32_t support_texture;        // min texture for support points
      int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
      int32_t incon_window_size;      // window size of inconsistent support point check
      int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
      int32_t incon_min_support;      // minimum number of consistent support points
      bool    add_corners;            // add support points at image corners with nearest neighbor disparities
      int32_t grid_size;              // size of neighborhood for additional support point extrapolation
      float   beta;                   // image likelihood parameter
      float   gamma;                  // prior constant
      float   sigma;                  // prior sigma
      float   sradius;                // prior sigma radius
      int32_t match_texture;          // min texture for dense matching
      int32_t lr_threshold;           // disparity threshold for left/right consistency check
      float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
      int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
      int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
      bool    filter_median;          // optional median filter (approximated)
      bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
      bool    postprocess_only_left;  // saves time by not postprocessing the right image
      bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
                                      // note: for this option D1 and D2 must be passed with size
                                      //       width/2 x height/2 (rounded towards zero)
      
      // constructor
      parameters (setting s=ROBOTICS) {
        
        // default settings in a robotics environment
        // (do not produce results in half-occluded areas
        //  and are a bit more robust towards lighting etc.)
        if (s==ROBOTICS) {
          disp_min              = 0;
          disp_max              = 255;
          support_threshold     = 0.85;
          support_texture       = 10;
          candidate_stepsize    = 5;
          incon_window_size     = 5;
          incon_threshold       = 5;
          incon_min_support     = 5;
          add_corners           = 0;
          grid_size             = 20;
          beta                  = 0.02;
          gamma                 = 3;
          sigma                 = 1;
          sradius               = 2;
          match_texture         = 1;
          lr_threshold          = 2;
          speckle_sim_threshold = 1;
          speckle_size          = 200;
          ipol_gap_width        = 3;
          filter_median         = 0;
          filter_adaptive_mean  = 1;
          postprocess_only_left = 1;
          subsampling           = 0;
          
        // default settings for middlebury benchmark
        // (interpolate all missing disparities)
        } else {
          disp_min              = 0;
          disp_max              = 255;
          support_threshold     = 0.95;
          support_texture       = 10;
          candidate_stepsize    = 5;
          incon_window_size     = 5;
          incon_threshold       = 5;
          incon_min_support     = 5;
          add_corners           = 1;
          grid_size             = 20;
          beta                  = 0.02;
          gamma                 = 5;
          sigma                 = 1;
          sradius               = 3;
          match_texture         = 0;
          lr_threshold          = 2;
          speckle_sim_threshold = 1;
          speckle_size          = 200;
          ipol_gap_width        = 5000;
          filter_median         = 1;
          filter_adaptive_mean  = 0;
          postprocess_only_left = 0;
          subsampling           = 0;
        }
      }
    };

    // constructor, input: parameters  
    Elas (int32_t w, int32_t h, parameters param) : width(w), height(h), param(param) {}

    // deconstructor
    ~Elas () {}
    
    // matching function
    void process(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right);
  
public:
    // parameter set
    int32_t width;
    int32_t height;
    parameters param;

    // Descriptor
    inline uint8_t filter3x3(const uint8_t *in, const int32_t f[3][3], const int32_t &u, const int32_t &v, const int32_t &w)
    {
        int32_t tmp = 0;
        for (int32_t i = 0; i < 3; i++) {
            for (int32_t j = 0; j < 3; j++) {
                tmp += in[(u + j - 1) + (v + i - 1) * w] * f[i][j];
            }
        }
        tmp >>= 2;
        tmp = tmp > 127 ? 127 : tmp < -128 ? -128 : tmp;
        tmp += 128;
        return static_cast<uint8_t>(tmp);
    }
    void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );

    // Support Matches
    inline int32_t sumAbsoluteErrorAtDescriptor(
            const uint8_t* hsobel, const uint8_t* vsobel, const int32_t val,
            const size_t stride)
    {
        int32_t sum = 0;
        
        // horizonal filter
        sum += abs( static_cast<int32_t>(*(hsobel-(stride*2))) - val);
        sum += abs( static_cast<int32_t>(*(hsobel-(stride-2))) - val);
        sum += abs( static_cast<int32_t>(*(hsobel-(stride)))   - val);
        sum += abs( static_cast<int32_t>(*(hsobel-(stride+2))) - val);
        sum += abs( static_cast<int32_t>(*(hsobel-1))          - val);
        sum += abs( static_cast<int32_t>(*(hsobel-0))          - val);
    
        sum += abs( static_cast<int32_t>(*(hsobel+0))          - val);
        sum += abs( static_cast<int32_t>(*(hsobel+1))          - val);
        sum += abs( static_cast<int32_t>(*(hsobel+(stride-2))) - val);
        sum += abs( static_cast<int32_t>(*(hsobel+(stride)))   - val);
        sum += abs( static_cast<int32_t>(*(hsobel+(stride+2))) - val);
        sum += abs( static_cast<int32_t>(*(hsobel+(stride*2))) - val);
        
        // vertial filter
        sum += abs( static_cast<int32_t>(*(vsobel-stride))     - val);
        sum += abs( static_cast<int32_t>(*(vsobel-1))          - val);
    
        sum += abs( static_cast<int32_t>(*(vsobel+1))          - val);
        sum += abs( static_cast<int32_t>(*(vsobel+stride))     - val);
    
        return sum;
    }
    
    inline int32_t sumAbsoluteErrorAtDescriptor(
            const uint8_t* hsleft, const uint8_t* vsleft,
            const uint8_t* hsright, const uint8_t* vsright,
            const size_t stride)
    {
        int32_t sum = 0;
        
        // horizonal filter
        sum += abs( static_cast<int32_t>(*(hsleft-(stride*2))) - static_cast<int32_t>(*(hsright-(stride*2))) );
        sum += abs( static_cast<int32_t>(*(hsleft-(stride-2))) - static_cast<int32_t>(*(hsright-(stride-2))) );
        sum += abs( static_cast<int32_t>(*(hsleft-(stride)))   - static_cast<int32_t>(*(hsright-(stride))) );
        sum += abs( static_cast<int32_t>(*(hsleft-(stride+2))) - static_cast<int32_t>(*(hsright-(stride+2))) );
        sum += abs( static_cast<int32_t>(*(hsleft-1))          - static_cast<int32_t>(*(hsright-1)) );
        sum += abs( static_cast<int32_t>(*(hsleft-0))          - static_cast<int32_t>(*(hsright-0)) );
    
        sum += abs( static_cast<int32_t>(*(hsleft+0))          - static_cast<int32_t>(*(hsright+0)) );
        sum += abs( static_cast<int32_t>(*(hsleft+1))          - static_cast<int32_t>(*(hsright+1)) );
        sum += abs( static_cast<int32_t>(*(hsleft+(stride-2))) - static_cast<int32_t>(*(hsright+(stride-2))) );
        sum += abs( static_cast<int32_t>(*(hsleft+(stride)))   - static_cast<int32_t>(*(hsright+(stride))) );
        sum += abs( static_cast<int32_t>(*(hsleft+(stride+2))) - static_cast<int32_t>(*(hsright+(stride+2))) );
        sum += abs( static_cast<int32_t>(*(hsleft+(stride*2))) - static_cast<int32_t>(*(hsright+(stride*2))) );
        
        // vertial filter
        sum += abs( static_cast<int32_t>(*(vsleft-stride))     - static_cast<int32_t>(*(vsright-stride)) );
        sum += abs( static_cast<int32_t>(*(vsleft-1))          - static_cast<int32_t>(*(vsright-1)) );
    
        sum += abs( static_cast<int32_t>(*(vsleft+1))          - static_cast<int32_t>(*(vsright+1)) );
        sum += abs( static_cast<int32_t>(*(vsleft+stride))     - static_cast<int32_t>(*(vsright+stride)) );
        return sum;
    }

    void removeInconsistentSupportPoints(
        int16_t* D_can,int32_t D_can_width,int32_t D_can_height);
    void removeRedundantSupportPoints(
        int16_t* D_can,int32_t D_can_width,int32_t D_can_height, 
        int32_t redun_max_dist, int32_t redun_threshold, bool vertical);
    int32_t addCornerSupportPoints(int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, int32_t sup_num);
    int16_t computeMatchingDisparity(
        int32_t u, int32_t v,
        const uint8_t* hsleft, const uint8_t* hsright,
        const uint8_t* vsleft, const uint8_t* vsright,
        bool right_image);
    int32_t computeSupportMatches(
        const uint8_t* hsleft, const uint8_t* hsright, const uint8_t* vsleft, const uint8_t* vsright, 
        int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, size_t sup_max);

    // Delaunay Triangulation
    int32_t computeDelaunayTriangulation(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, size_t sup_num, 
        int32_t* tri_as, int32_t* tri_bs, int32_t* tri_cs, bool right_image);
    
    // Disparity Planes
    void computeDisparityPlanes (
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, size_t sup_num,
        const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs, size_t tri_num,
        float* tri_t1as, float* tri_t1bs, float* tri_t1cs, 
        float* tri_t2as, float* tri_t2bs, float* tri_t2cs, 
        bool right_image);

    // Grid
    void createGrid(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, size_t sup_num,
        int32_t* disparity_grid, int32_t* grid_nums, 
        int32_t grid_width, int32_t grid_height, bool right_image);

    // TriangleMapping
    void createTriangleMapping(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, size_t sup_num, 
        const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs,
        const float* tri_t1as, const float* tri_t1bs, const float* tri_t1cs,
        const float* tri_t2as, const float* tri_t2bs, const float* tri_t2cs, size_t tri_num,
        uint16_t* tri_map, bool right_image);

    // Matching
    inline void findMatch(
        const int32_t u, const int32_t v,
        float plane_a, float plane_b, float plane_c,
        const int32_t* disparity_grid, const int32_t* grid_nums, 
        int32_t grid_width, int32_t grid_height,
        const uint8_t* hsleft, const uint8_t* hsright, 
        const uint8_t* vsleft, const uint8_t* vsright, 
        const int32_t *P, int32_t plane_radius,
        float* disp,
        const bool valid, bool right_image);
    void computeDisparity(
        const int32_t* sup_us, const int32_t* sup_vs, const int32_t* sup_ds, size_t sup_num,
        const int32_t* tri_as, const int32_t* tri_bs, const int32_t* tri_cs,
        const float* tri_t1as, const float* tri_t1bs, const float* tri_t1cs,
        const float* tri_t2as, const float* tri_t2bs, const float* tri_t2cs, size_t tri_num,
        const int32_t* disparity_grid, const int32_t* grid_nums, int32_t grid_width, int32_t grid_height,
        const uint8_t* hsleft, const uint8_t* hsright, const uint8_t* vsleft, const uint8_t* vsright, 
        const uint16_t* tri_map, float* disp, bool right_image);

    // L/R consistency check
    void leftRightConsistencyCheck(float* disp_left, float* disp_right);

    // Post-processing (same as libelas)
    void removeSmallSegments(float* D);
    void gapInterpolation(float* D);
    void adaptiveMean(float* D);
    void median(float* D);
};

#endif
