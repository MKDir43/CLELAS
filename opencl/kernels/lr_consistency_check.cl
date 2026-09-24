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

// lr_consistency_check.cl : OpenCL 1.2 kernel for left-right consistency check
// 1D strided loop + barrier for indirect memory access

__kernel void leftRightConsistencyCheck(
    __global const float* input_left, __global const float* input_right,
    __global float* output_left, __global float* output_right,
    int width, int height, int pitch, int lr_threshold, int subsampling, int estimates_subpixel)
{
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;

        // Indirect access: compute the index into input_right from the value of input_left[idx]
        float d_left = input_left[idx];
        float d_right = input_right[idx];

        barrier(CLK_GLOBAL_MEM_FENCE);

        float u_warp_left, u_warp_right;
        if (subsampling) {
            u_warp_left = (float)u - d_left / 2.0f;
            u_warp_right = (float)u + d_right / 2.0f;
        } else {
            u_warp_left = (float)u - d_left;
            u_warp_right = (float)u + d_right;
        }

        // left check
        if (d_left >= 0 && u_warp_left >= 0 && u_warp_left < width) {
            if (fabs(input_right[width * v + (int)u_warp_left] - d_left) > lr_threshold) {
                d_left = -10.0f;
            }
        } else {
            d_left = -10.0f;
        }
        // right check
        if (d_right >= 0 && u_warp_right >= 0 && u_warp_right < width) {
            if (fabs(input_left[width * v + (int)u_warp_right] - d_right) > lr_threshold) {
                d_right = -10.0f;
            }
        } else {
            d_right = -10.0f;
        }

        output_left[pitch * v + u] = d_left;
        output_right[pitch * v + u] = d_right;
    }
}
