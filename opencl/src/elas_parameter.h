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

#ifndef CLELAS_ELAS_PARAMETER_H
#define CLELAS_ELAS_PARAMETER_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

namespace clelas {

// parameter presets (same as the CPU version).
enum setting {ROBOTICS,MIDDLEBURY};

struct Parameters {
    // disparity
    int32_t disp_min;               // min disparity
    int32_t disp_max;               // max disparity
    // candidates
    float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
    int32_t support_texture;        // min texture for support points
    int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
    // correct candidiates
    bool    support_thinning;       // delete inconsistent and redundant support points
    bool    gpu_thinning;           // thin out the support points on the device instead of the host
    int32_t incon_window_width;     // window width of inconsistent support point check
    int32_t incon_window_height;    // window height of inconsistent support point check
    int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
    int32_t incon_min_support;      // minimum number of consistent support points
    bool    add_corners;            // add support points at image corners with nearest neighbor disparities
    // grid_size
    int32_t grid_size;              // size of neighborhood for additional support point extrapolation
    // cost
    float   beta;                   // image likelihood parameter
    float   gamma;                  // prior constant
    float   sigma;                  // prior sigma
    float   sradius;                // prior sigma radius (search radius around the plane: max(ceil(sigma*sradius), 2))
    int32_t match_texture;          // min texture for dense matching
    // others
    int32_t supp_lr_threshold;      // supportpoint disparity threshold for left/right consistency check
    int32_t disp_lr_threshold;      // image disparity threshold for left/right consistency check
    bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
    // post-processing (same as the CPU version)
    float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
    int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
    int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
    bool    filter_median;          // optional median filter (approximated)
    bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
    bool    postprocess_only_left;  // saves time by not postprocessing the right image

    // constructor
    Parameters (setting s=ROBOTICS) {
        // default settings in a robotics environment
        if (s==ROBOTICS) {
            disp_min              = 0;
            disp_max              = 255;
            support_threshold     = 0.85;
            support_texture       = 10;
            candidate_stepsize    = 5;
            support_thinning      = 1;
            gpu_thinning          = 0;
            incon_window_width    = 5;
            incon_window_height   = 5;
            incon_threshold       = 5;
            incon_min_support     = 5;
            add_corners           = 0;
            grid_size             = 20;
            beta                  = 0.02;
            gamma                 = 3;
            sigma                 = 1;
            sradius               = 2;
            match_texture         = 1;
            supp_lr_threshold     = 2;
            disp_lr_threshold     = 2;
            subsampling           = 0;
            speckle_sim_threshold = 1;
            speckle_size          = 200;
            ipol_gap_width        = 3;
            filter_median         = 0;
            filter_adaptive_mean  = 1;
            postprocess_only_left = 1;

        // default settings for middlebury benchmark
        } else {
            disp_min              = 0;
            disp_max              = 255;
            support_threshold     = 0.95;
            support_texture       = 10;
            candidate_stepsize    = 5;
            support_thinning      = 1;
            gpu_thinning          = 1;
            incon_window_width    = 5;
            incon_window_height   = 5;
            incon_threshold       = 5;
            incon_min_support     = 5;
            add_corners           = 1;
            grid_size             = 20;
            beta                  = 0.02;
            gamma                 = 5;
            sigma                 = 1;
            sradius               = 3;
            match_texture         = 0;
            supp_lr_threshold     = 2;
            disp_lr_threshold     = 2;
            subsampling           = 0;
            speckle_sim_threshold = 1;
            speckle_size          = 200;
            ipol_gap_width        = 5000;
            filter_median         = 1;
            filter_adaptive_mean  = 0;
            postprocess_only_left = 0;
        }
    }
};


}
#endif // CLELAS_ELAS_PARAMETER_H
