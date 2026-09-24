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

// adaptive_mean.cl : OpenCL 1.2 kernels for the adaptive mean filter
// (Elas::adaptiveMean of the CPU version, which follows libelas)
// 1D strided loop

// No fused multiply-add, so that the result matches the CPU version bit for bit.
#pragma OPENCL FP_CONTRACT OFF

// |x| as computed by libelas: its absolute mask is the float 2147483648.0f
// (bits 0x4F000000), so the AND keeps only some exponent bits.
inline float libelasAbs(float x)
{
    return as_float(as_uint(x) & 0x4F000000u);
}

// One step of the bilateral filter, with the same float operations as the CPU
// version (lanes k and k+4 of its SSE code are added first). val holds `taps`
// (4 or 8) disparities in ring buffer order, val_curr is the disparity at the center.
// Returns whether the filtered disparity *d is valid.
inline int bilateralMean(const float* val, int taps, float val_curr, float* d)
{
    float weight[4];
    float factor[4];
    for (int k = 0; k < 4; k++) {
        float weight_1 = 4.0f - libelasAbs(val[k] - val_curr);
        weight_1 = (0.0f > weight_1) ? 0.0f : weight_1;
        float factor_1 = val[k] * weight_1;
        if (taps == 8) {
            float weight_2 = 4.0f - libelasAbs(val[k + 4] - val_curr);
            weight_2 = (0.0f > weight_2) ? 0.0f : weight_2;
            float factor_2 = val[k + 4] * weight_2;
            weight[k] = weight_1 + weight_2;
            factor[k] = factor_1 + factor_2;
        } else {
            weight[k] = weight_1;
            factor[k] = factor_1;
        }
    }
    float weight_sum = weight[0] + weight[1] + weight[2] + weight[3];
    float factor_sum = factor[0] + factor[1] + factor[2] + factor[3];
    if (weight_sum > 0.0f) {
        *d = factor_sum / weight_sum;
        return *d >= 0.0f;
    }
    return 0;
}

// Sets invalid disparities to -10 (this makes the bilateral weights of all valid
// disparities to 0 in this region). Both work buffers start as this copy.
__kernel void adaptiveMeanPrepare(
    __global const float* D, __global float* D_copy, __global float* D_tmp, int total)
{
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        float d = D[idx] < 0.0f ? -10.0f : D[idx];
        D_copy[idx] = d;
        D_tmp[idx] = d;
    }
}

// Horizontal pass (D_copy -> D_tmp). The window of the CPU version ending at
// column u holds the columns u-taps+1..u and its result goes to column u-center.
__kernel void adaptiveMeanHorizontal(
    __global const float* D_copy, __global float* D_tmp, int width, int height, int taps)
{
    int center = taps / 2 - 1;
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;
        if (v < 3 || v >= height - 3 || u < taps - 1) {
            continue;
        }

        __global const float* row = D_copy + width * v;
        float val[8];
        for (int k = 0; k < taps; k++) {
            val[k] = row[u - (u - k) % taps];
        }
        float d;
        if (bilateralMean(val, taps, row[u - center], &d)) {
            D_tmp[width * v + (u - center)] = d;
        }
    }
}

// Vertical pass (D_tmp -> D), the same along the columns.
__kernel void adaptiveMeanVertical(
    __global const float* D_tmp, __global float* D, int width, int height, int taps)
{
    int center = taps / 2 - 1;
    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;
        if (u < 3 || u >= width - 3 || v < taps - 1) {
            continue;
        }

        float val[8];
        for (int k = 0; k < taps; k++) {
            val[k] = D_tmp[width * (v - (v - k) % taps) + u];
        }
        float d;
        if (bilateralMean(val, taps, D_tmp[width * (v - center) + u], &d)) {
            D[width * (v - center) + u] = d;
        }
    }
}
