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

// median_filter.cl : OpenCL 1.2 kernels for the median filter
// (Elas::median of the CPU version, which follows libelas)
// 1D strided loop

#define MEDIAN_WINDOW_SIZE 3

// Inserts temp into the sorted values vals[0..count-1] (insertion sort as in libelas).
inline void insertSorted(float* vals, int count, float temp)
{
    int i = count - 1;
    while (i >= 0 && vals[i] > temp) {
        vals[i + 1] = vals[i];
        i--;
    }
    vals[i + 1] = temp;
}

// First step: horizontal median of the valid disparities (D -> D_temp).
// D_temp has to be cleared to 0 beforehand, as the CPU version does.
__kernel void medianHorizontal(__global const float* D, __global float* D_temp, int width, int height)
{
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;
        if (u < MEDIAN_WINDOW_SIZE || u >= width - MEDIAN_WINDOW_SIZE ||
            v < MEDIAN_WINDOW_SIZE || v >= height - MEDIAN_WINDOW_SIZE) {
            continue;
        }

        if (D[idx] >= 0.0f) {
            float vals[2 * MEDIAN_WINDOW_SIZE + 1];
            for (int j = 0; j <= 2 * MEDIAN_WINDOW_SIZE; j++) {
                insertSorted(vals, j, D[width * v + (u - MEDIAN_WINDOW_SIZE + j)]);
            }
            D_temp[idx] = vals[MEDIAN_WINDOW_SIZE];
        } else {
            D_temp[idx] = D[idx];
        }
    }
}

// Second step: vertical median of the first step, for the valid disparities (D_temp -> D).
__kernel void medianVertical(__global const float* D_temp, __global float* D, int width, int height)
{
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;
        if (u < MEDIAN_WINDOW_SIZE || u >= width - MEDIAN_WINDOW_SIZE ||
            v < MEDIAN_WINDOW_SIZE || v >= height - MEDIAN_WINDOW_SIZE) {
            continue;
        }

        if (D[idx] >= 0.0f) {
            float vals[2 * MEDIAN_WINDOW_SIZE + 1];
            for (int j = 0; j <= 2 * MEDIAN_WINDOW_SIZE; j++) {
                insertSorted(vals, j, D_temp[width * (v - MEDIAN_WINDOW_SIZE + j) + u]);
            }
            D[idx] = vals[MEDIAN_WINDOW_SIZE];
        }
    }
}
