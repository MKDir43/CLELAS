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

// sobel_filter.cl : OpenCL 1.2 kernels for the 3x3 Sobel filters
// 1D strided loop, no tiling
//
// Same filters as Elas::sobel3x3 of the CPU version: the response is divided
// by 4, clamped to [-128, 127] and offset by 128. Border pixels are set to 128.

// Horizontal gradient du (responds to vertical edges)
// Kernel:    1  0 -1
//            2  0 -2
//            1  0 -1
__kernel void sobelDu3x3(
    __global const uchar* input_image,
    __global uchar* output_image,
    int width, int height, int pitch)
{
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int gx = idx % width;
        int gy = idx / width;

        if (gx < 1 || gx >= width - 1 || gy < 1 || gy >= height - 1) {
            output_image[width * gy + gx] = 128;
            continue;
        }

        int p00 = input_image[pitch * (gy - 1) + (gx - 1)];
        int p02 = input_image[pitch * (gy - 1) + (gx + 1)];
        int p10 = input_image[pitch * gy + (gx - 1)];
        int p12 = input_image[pitch * gy + (gx + 1)];
        int p20 = input_image[pitch * (gy + 1) + (gx - 1)];
        int p22 = input_image[pitch * (gy + 1) + (gx + 1)];

        int du = p00 - p02 + 2 * p10 - 2 * p12 + p20 - p22;

        du >>= 2;
        du = du > 127 ? 127 : (du < -128 ? -128 : du);
        du += 128;

        output_image[width * gy + gx] = (uchar)du;
    }
}

// Vertical gradient dv (responds to horizontal edges)
// Kernel:    1  2  1
//            0  0  0
//           -1 -2 -1
__kernel void sobelDv3x3(
    __global const uchar* input_image,
    __global uchar* output_image,
    int width, int height, int pitch)
{
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int gx = idx % width;
        int gy = idx / width;

        if (gx < 1 || gx >= width - 1 || gy < 1 || gy >= height - 1) {
            output_image[width * gy + gx] = 128;
            continue;
        }

        int p00 = input_image[pitch * (gy - 1) + (gx - 1)];
        int p01 = input_image[pitch * (gy - 1) + gx];
        int p02 = input_image[pitch * (gy - 1) + (gx + 1)];
        int p20 = input_image[pitch * (gy + 1) + (gx - 1)];
        int p21 = input_image[pitch * (gy + 1) + gx];
        int p22 = input_image[pitch * (gy + 1) + (gx + 1)];

        int dv = p00 + 2 * p01 + p02 - p20 - 2 * p21 - p22;

        dv >>= 2;
        dv = dv > 127 ? 127 : (dv < -128 ? -128 : dv);
        dv += 128;

        output_image[width * gy + gx] = (uchar)dv;
    }
}
