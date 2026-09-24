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

// compute_disparity.cl : OpenCL 1.2 kernel for disparity computation
// 1D strided loop + barrier for indirect memory access

// No fused multiply-add, so that the plane prior matches the CPU version bit for bit.
#pragma OPENCL FP_CONTRACT OFF

inline float computeDisparityWithSubpixel(
    int min_disparity_energy, int prev_disparity_energy, int next_disparity_energy)
{
    float subpixel = 0.0f;
    int denominator;
    if (next_disparity_energy < prev_disparity_energy) {
        denominator = min_disparity_energy - prev_disparity_energy;
        if (denominator != 0) {
            subpixel = 0.5f * ((float)(next_disparity_energy - prev_disparity_energy) /
                               (float)denominator);
        }
    } else {
        denominator = min_disparity_energy - next_disparity_energy;
        if (denominator != 0) {
            subpixel = 0.5f * ((float)(next_disparity_energy - prev_disparity_energy) /
                               (float)denominator);
        }
    }
    return subpixel;
}

inline int sumAbsoluteErrorAtDescriptor(
    __global const uchar* hsobel1, __global const uchar* vsobel1, int h_stride1, int v_stride1,
    __global const uchar* hsobel2, __global const uchar* vsobel2, int h_stride2, int v_stride2)
{
    int sum = 0;
    sum += abs((int)hsobel1[-(h_stride1*2)] - (int)hsobel2[-(h_stride2*2)]);
    sum += abs((int)hsobel1[-(h_stride1-2)] - (int)hsobel2[-(h_stride2-2)]);
    sum += abs((int)hsobel1[-(h_stride1)]   - (int)hsobel2[-(h_stride2)]);
    sum += abs((int)hsobel1[-(h_stride1+2)] - (int)hsobel2[-(h_stride2+2)]);
    sum += abs((int)hsobel1[-1]             - (int)hsobel2[-1]);
    sum += abs((int)hsobel1[0]              - (int)hsobel2[0]);
    sum += abs((int)hsobel1[0]              - (int)hsobel2[0]);
    sum += abs((int)hsobel1[1]              - (int)hsobel2[1]);
    sum += abs((int)hsobel1[(h_stride1-2)]  - (int)hsobel2[(h_stride2-2)]);
    sum += abs((int)hsobel1[(h_stride1)]    - (int)hsobel2[(h_stride2)]);
    sum += abs((int)hsobel1[(h_stride1+2)]  - (int)hsobel2[(h_stride2+2)]);
    sum += abs((int)hsobel1[(h_stride1*2)]  - (int)hsobel2[(h_stride2*2)]);
    sum += abs((int)vsobel1[-v_stride1] - (int)vsobel2[-v_stride2]);
    sum += abs((int)vsobel1[-1]         - (int)vsobel2[-1]);
    sum += abs((int)vsobel1[1]          - (int)vsobel2[1]);
    sum += abs((int)vsobel1[v_stride1]  - (int)vsobel2[v_stride2]);
    return sum;
}

// Texture of the descriptor at hsobel/vsobel: sum of |element - val| over 16 elements
// (same as the texture check of Elas::findMatch in the CPU version).
inline int textureAtDescriptor(
    __global const uchar* hsobel, __global const uchar* vsobel, int val, int stride)
{
    int sum = 0;
    sum += abs((int)hsobel[-(stride*2)] - val);
    sum += abs((int)hsobel[-(stride-2)] - val);
    sum += abs((int)hsobel[-(stride)]   - val);
    sum += abs((int)hsobel[-(stride+2)] - val);
    sum += abs((int)hsobel[-1]          - val);
    sum += abs((int)hsobel[0]           - val);
    sum += abs((int)hsobel[0]           - val);
    sum += abs((int)hsobel[1]           - val);
    sum += abs((int)hsobel[(stride-2)]  - val);
    sum += abs((int)hsobel[(stride)]    - val);
    sum += abs((int)hsobel[(stride+2)]  - val);
    sum += abs((int)hsobel[(stride*2)]  - val);
    sum += abs((int)vsobel[-stride] - val);
    sum += abs((int)vsobel[-1]      - val);
    sum += abs((int)vsobel[1]       - val);
    sum += abs((int)vsobel[stride]  - val);
    return sum;
}

inline float pixelEstimation(
    int u, int v,
    float plane_a, float plane_b, float plane_c,
    __global const int* disparity_grid, __global const int* grid_nums,
    int grid_width, int grid_height,
    __global const uchar* hsleft, __global const uchar* hsright,
    __global const uchar* vsleft, __global const uchar* vsright,
    int width, int height, int disp_num, int plane_radius, int valid,
    int grid_size, int window_size, int line_offset, int right_image,
    __global const int* d_P)
{
    int d_plane = (int)(plane_a * (float)u + plane_b * (float)v + plane_c);
    int d_plane_min = max(d_plane - plane_radius, 0);
    int d_plane_max = min(d_plane + plane_radius, disp_num - 1);
    int grid_x = (int)floor((float)u / (float)grid_size);
    int grid_y = (int)floor((float)v / (float)grid_size);
    int grid_num = grid_nums[grid_width * grid_y + grid_x];
    int d_curr, u_warp, val;
    int min_val = 2147483647;
    int min_d = -1;
    for (int i = 0; i < grid_num; i++) {
        // Indirect access: disparity_grid[...] → compute u_warp → hsright[u_warp]
        d_curr = disparity_grid[grid_width * grid_height * i + grid_width * grid_y + grid_x];
        if (d_curr < d_plane_min || d_curr > d_plane_max) {
            u_warp = right_image ? u + d_curr : u - d_curr;
            if (u_warp < window_size || u_warp >= width - window_size) continue;
            val = sumAbsoluteErrorAtDescriptor(
                hsleft + line_offset + u, vsleft + line_offset + u, width, width,
                hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
            if (val < min_val) {
                min_val = val;
                min_d = d_curr;
            }
        }
    }
    for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++) {
        u_warp = right_image ? u + d_curr : u - d_curr;
        if (u_warp < window_size || u_warp >= width - window_size) continue;
        val = valid ? d_P[abs(d_curr - d_plane)] : 0;
        val += sumAbsoluteErrorAtDescriptor(
            hsleft + line_offset + u, vsleft + line_offset + u, width, width,
            hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
        if (val < min_val) {
            min_val = val;
            min_d = d_curr;
        }
    }
    if (min_d >= 0) {
        return (float)min_d;
    } else {
        return -1.0f;
    }
}

inline float subpixelEstimation(
    int u, int v,
    float plane_a, float plane_b, float plane_c,
    __global const int* disparity_grid, __global const int* grid_nums,
    int grid_width, int grid_height,
    __global const uchar* hsleft, __global const uchar* hsright,
    __global const uchar* vsleft, __global const uchar* vsright,
    int width, int height, int disp_min, int disp_max, int disp_num, int plane_radius, int valid,
    int grid_size, int window_size, int line_offset, int right_image,
    __global const int* d_P)
{
    int d_plane = (int)(plane_a * (float)u + plane_b * (float)v + plane_c);
    int d_plane_min = max(d_plane - plane_radius, 0);
    int d_plane_max = min(d_plane + plane_radius, disp_num - 1);
    int grid_x = (int)floor((float)u / (float)grid_size);
    int grid_y = (int)floor((float)v / (float)grid_size);
    int grid_num = grid_nums[grid_width * grid_y + grid_x];
    int d_curr, u_warp, val;
    int min_val = 2147483647;
    int min_d = -1;
    float subpixel = 0.0f;
    int min_prev_val = 2147483647;
    int min_next_val = 2147483647;
    int current_val[3];
    for (int i = 0; i < grid_num; i++) {
        d_curr = disparity_grid[grid_width * grid_height * i + grid_width * grid_y + grid_x];
        if (d_curr < d_plane_min || d_curr > d_plane_max) {
            u_warp = right_image ? u + d_curr : u - d_curr;
            if (u_warp < window_size || u_warp >= width - window_size) continue;
            val = sumAbsoluteErrorAtDescriptor(
                hsleft + line_offset + u, vsleft + line_offset + u, width, width,
                hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
            if (val < min_val) {
                min_val = val;
                min_d = d_curr;
                u_warp = right_image ? u + (d_curr - 1) : u - (d_curr - 1);
                if (u_warp < window_size || u_warp >= width - window_size) {
                    min_prev_val = 2147483647;
                } else {
                    min_prev_val = sumAbsoluteErrorAtDescriptor(
                        hsleft + line_offset + u, vsleft + line_offset + u, width, width,
                        hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
                }
                u_warp = right_image ? u + (d_curr + 1) : u - (d_curr + 1);
                if (u_warp < window_size || u_warp >= width - window_size) {
                    min_next_val = 2147483647;
                } else {
                    min_next_val = sumAbsoluteErrorAtDescriptor(
                        hsleft + line_offset + u, vsleft + line_offset + u, width, width,
                        hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
                }
            }
        }
    }
    d_curr = d_plane_min;
    for (int i = -1; i <= 1; i++) {
        u_warp = right_image ? u + (d_curr + i) : u - (d_curr + i);
        if (u_warp < window_size || u_warp >= width - window_size) {
            current_val[i + 1] = 2147483647;
            continue;
        }
        if ((d_curr + i) < d_plane_min || (d_curr + i) > d_plane_max) {
            val = 0;
        } else {
            val = valid ? d_P[abs((d_curr + i) - d_plane)] : 0;
        }
        val += sumAbsoluteErrorAtDescriptor(
            hsleft + line_offset + u, vsleft + line_offset + u, width, width,
            hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
        current_val[i + 1] = val;
    }
    if (current_val[1] < min_val) {
        min_val = current_val[1];
        min_prev_val = current_val[0];
        min_next_val = current_val[2];
        min_d = d_curr;
    }
    for (d_curr = d_plane_min + 1; d_curr <= d_plane_max; d_curr++) {
        current_val[0] = current_val[1];
        current_val[1] = current_val[2];
        u_warp = right_image ? u + (d_curr + 1) : u - (d_curr + 1);
        if (u_warp < window_size || u_warp >= width - window_size) {
            current_val[2] = 2147483647;
        } else {
            if ((d_curr + 1) > d_plane_max) {
                val = 0;
            } else {
                val = valid ? d_P[abs((d_curr + 1) - d_plane)] : 0;
            }
            val += sumAbsoluteErrorAtDescriptor(
                hsleft + line_offset + u, vsleft + line_offset + u, width, width,
                hsright + line_offset + u_warp, vsright + line_offset + u_warp, width, width);
            current_val[2] = val;
        }
        if (current_val[1] < min_val) {
            min_val = current_val[1];
            min_prev_val = current_val[0];
            min_next_val = current_val[2];
            min_d = d_curr;
        }
    }
    if (min_d >= 0) {
        if (min_d > disp_min && min_d < disp_max) {
            if (min_prev_val != 2147483647 && min_next_val != 2147483647) {
                subpixel = computeDisparityWithSubpixel(min_val, min_prev_val, min_next_val);
            }
        }
        return (float)min_d + subpixel;
    } else {
        return -1.0f;
    }
}

inline float findMatch(
    int u, int v,
    float plane_a, float plane_b, float plane_c,
    __global const int* disparity_grid, __global const int* grid_nums,
    int grid_width, int grid_height,
    __global const uchar* hsleft, __global const uchar* hsright,
    __global const uchar* vsleft, __global const uchar* vsright,
    int width, int height, int disp_min, int disp_max, int disp_num, int plane_radius, int valid,
    int match_texture, int grid_size, int estimates_subpixel, int right_image,
    __global const int* d_P)
{
    int window_size = 2;
    if (u < window_size || u >= width - window_size) return -10.0f;
    int line_offset = width * max(min(v, height - (window_size + 1)), window_size);
    int sum = textureAtDescriptor(hsleft + line_offset + u, vsleft + line_offset + u, 128, width);
    if (sum < match_texture) return -10.0f;
    if (estimates_subpixel) {
        return subpixelEstimation(u, v, plane_a, plane_b, plane_c,
            disparity_grid, grid_nums, grid_width, grid_height,
            hsleft, hsright, vsleft, vsright,
            width, height, disp_min, disp_max, disp_num, plane_radius, valid,
            grid_size, window_size, line_offset, right_image, d_P);
    } else {
        return pixelEstimation(u, v, plane_a, plane_b, plane_c,
            disparity_grid, grid_nums, grid_width, grid_height,
            hsleft, hsright, vsleft, vsright,
            width, height, disp_num, plane_radius, valid,
            grid_size, window_size, line_offset, right_image, d_P);
    }
}

__kernel void computeDisparity(
    __global const int* support_us, __global const int* support_vs, __global const int* support_ds, int support_num,
    __global const int* triangle_as, __global const int* triangle_bs, __global const int* triangle_cs,
    __global const float* triangle_t1as, __global const float* triangle_t1bs, __global const float* triangle_t1cs,
    __global const float* triangle_t2as, __global const float* triangle_t2bs, __global const float* triangle_t2cs, int tri_num,
    __global const int* disparity_grid, __global const int* grid_nums,
    int grid_width, int grid_height,
    __global const uchar* hsleft, __global const uchar* hsright,
    __global const uchar* vsleft, __global const uchar* vsright,
    __global const ushort* tri_map, __global float* disp,
    int width, int height,
    int disp_min, int disp_max, int disp_num,
    int match_texture, int grid_size,
    int subsampling, int estimates_subpixel, int plane_radius,
    int right_image,
    __global const int* d_P)
{
    // With subsampling, only pixels with even u and v are computed, into a
    // width/2 x height/2 map (as in the CPU version).
    int d_width = subsampling ? width / 2 : width;
    int d_height = subsampling ? height / 2 : height;

    int total = width * height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u = idx % width;
        int v = idx / width;

        // Indirect access: tri_map[idx] → triangle_t1as[tri_idx]
        ushort tri_idx = tri_map[idx];

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (subsampling && (u % 2 != 0 || v % 2 != 0 || u / 2 >= d_width || v / 2 >= d_height)) {
            continue;
        }
        int d_idx = subsampling ? d_width * (v / 2) + u / 2 : idx;

        if (tri_idx == 0) {
            disp[d_idx] = -1.0f;
            continue;
        }
        tri_idx--;

        float plane_a = right_image ? triangle_t2as[tri_idx] : triangle_t1as[tri_idx];
        float plane_b = right_image ? triangle_t2bs[tri_idx] : triangle_t1bs[tri_idx];
        float plane_c = right_image ? triangle_t2cs[tri_idx] : triangle_t1cs[tri_idx];
        float plane_d = right_image ? triangle_t1as[tri_idx] : triangle_t2as[tri_idx];
        int valid = (fabs(plane_a) < 0.7f && fabs(plane_d) < 0.7f);

        disp[d_idx] = findMatch(u, v, plane_a, plane_b, plane_c,
            disparity_grid, grid_nums, grid_width, grid_height,
            hsleft, hsright, vsleft, vsright,
            width, height, disp_min, disp_max, disp_num, plane_radius, valid,
            match_texture, grid_size, estimates_subpixel, right_image, d_P);
    }
}
