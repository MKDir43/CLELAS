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

// make_disparity_candidate.cl : OpenCL kernel for disparity candidate generation
// Split into two passes (texture check + disparity matching)
//
// The descriptors, patches and checks follow computeSupportMatches /
// computeMatchingDisparity of the CPU version.

// Texture of the descriptor at hsobel/vsobel: sum of |element - val| over 16 elements.
inline int sumAbsoluteErrorAtDescriptor(
    __global const uchar* hsobel, __global const uchar* vsobel, int val, int stride)
{
    int sum = 0;

    // horizontal gradient
    sum += abs((int)hsobel[-(stride * 2)] - val);
    sum += abs((int)hsobel[-(stride - 2)] - val);
    sum += abs((int)hsobel[-stride]       - val);
    sum += abs((int)hsobel[-(stride + 2)] - val);
    sum += abs((int)hsobel[-1]            - val);
    sum += abs((int)hsobel[0]             - val);
    sum += abs((int)hsobel[0]             - val);
    sum += abs((int)hsobel[1]             - val);
    sum += abs((int)hsobel[stride - 2]    - val);
    sum += abs((int)hsobel[stride]        - val);
    sum += abs((int)hsobel[stride + 2]    - val);
    sum += abs((int)hsobel[stride * 2]    - val);

    // vertical gradient
    sum += abs((int)vsobel[-stride] - val);
    sum += abs((int)vsobel[-1]      - val);
    sum += abs((int)vsobel[1]       - val);
    sum += abs((int)vsobel[stride]  - val);

    return sum;
}

// Matching error between two descriptors: sum of |a - b| over 16 elements.
inline int sumAbsoluteErrorAtDescriptorStereo(
    __global const uchar* hs_a, __global const uchar* vs_a,
    __global const uchar* hs_b, __global const uchar* vs_b,
    int stride)
{
    int sum = 0;

    // horizontal gradient
    sum += abs((int)hs_a[-(stride * 2)] - (int)hs_b[-(stride * 2)]);
    sum += abs((int)hs_a[-(stride - 2)] - (int)hs_b[-(stride - 2)]);
    sum += abs((int)hs_a[-stride]       - (int)hs_b[-stride]);
    sum += abs((int)hs_a[-(stride + 2)] - (int)hs_b[-(stride + 2)]);
    sum += abs((int)hs_a[-1]            - (int)hs_b[-1]);
    sum += abs((int)hs_a[0]             - (int)hs_b[0]);
    sum += abs((int)hs_a[0]             - (int)hs_b[0]);
    sum += abs((int)hs_a[1]             - (int)hs_b[1]);
    sum += abs((int)hs_a[stride - 2]    - (int)hs_b[stride - 2]);
    sum += abs((int)hs_a[stride]        - (int)hs_b[stride]);
    sum += abs((int)hs_a[stride + 2]    - (int)hs_b[stride + 2]);
    sum += abs((int)hs_a[stride * 2]    - (int)hs_b[stride * 2]);

    // vertical gradient
    sum += abs((int)vs_a[-stride] - (int)vs_b[-stride]);
    sum += abs((int)vs_a[-1]      - (int)vs_b[-1]);
    sum += abs((int)vs_a[1]       - (int)vs_b[1]);
    sum += abs((int)vs_a[stride]  - (int)vs_b[stride]);

    return sum;
}

// Pass 1: texture check only (initialization is done on the host with clEnqueueFillBuffer)
// texture_mask is pre-initialized to 0 on the host. Set texture_mask = 1 for points with sufficient texture
__kernel void textureCheckPass(
    __global const uchar* hsobel_left, __global const uchar* vsobel_left,
    int width, int height,
    int candi_width, int candi_height, int candi_step,
    int support_texture,
    __global int* texture_mask)
{
    int total = candi_width * candi_height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u_can = idx % candi_width;
        int v_can = idx / candi_width;

        if (u_can < 1 || u_can >= candi_width - 1 || v_can < 1 || v_can >= candi_height - 1) {
            continue;
        }

        int u = u_can * candi_step;
        int v = v_can * candi_step;

        const int u_step = 6;
        const int v_step = 2;
        const int disp_offset = 2;

        if (!(u > disp_offset + u_step && u < width - disp_offset - u_step - 1 &&
              v > disp_offset + v_step && v <= height - disp_offset - v_step - 1)) {
            continue;
        }

        int offset = width * v + u;
        int sum = sumAbsoluteErrorAtDescriptor(
            hsobel_left + offset, vsobel_left + offset, 128, width);

        if (sum >= support_texture) {
            texture_mask[idx] = 1;
        }
    }
}

// Best disparity at (u, v) of image A matched against image B, or -1 when the
// match is not unique enough. Mirrors Elas::computeMatchingDisparity of the CPU
// version: right_image selects the search direction (u - d when A is the left
// image, u + d when A is the right image).
inline short findSupportMatch(
    __global const uchar* hs_a, __global const uchar* vs_a,
    __global const uchar* hs_b, __global const uchar* vs_b,
    int u, int v, int width,
    int disp_min, int disp_max, float support_threshold, int right_image)
{
    const int u_step = 6;
    const int v_step = 2;
    const int disp_offset = 2;

    int disp_min_valid = max(0, disp_min);
    int disp_max_valid = right_image ? min(width - u - disp_offset - u_step - 1, disp_max)
                                     : min(u - disp_offset - u_step - 1, disp_max);

    if (disp_max_valid - disp_min_valid < 10) return -1;

    short min_1_E = 32767;
    short min_1_d = -1;
    short min_2_E = 32767;
    short min_2_d = -1;

    int ul_offset = width * (v - v_step) - u_step;
    int ur_offset = width * (v - v_step) + u_step;
    int dl_offset = width * (v + v_step) - u_step;
    int dr_offset = width * (v + v_step) + u_step;

    for (int d = disp_min_valid; d <= disp_max_valid; d++) {
        int u_warp = right_image ? u + d : u - d;

        int sum = 0;
        sum += sumAbsoluteErrorAtDescriptorStereo(
            hs_a + ul_offset + u, vs_a + ul_offset + u,
            hs_b + ul_offset + u_warp, vs_b + ul_offset + u_warp, width);
        sum += sumAbsoluteErrorAtDescriptorStereo(
            hs_a + ur_offset + u, vs_a + ur_offset + u,
            hs_b + ur_offset + u_warp, vs_b + ur_offset + u_warp, width);
        sum += sumAbsoluteErrorAtDescriptorStereo(
            hs_a + dl_offset + u, vs_a + dl_offset + u,
            hs_b + dl_offset + u_warp, vs_b + dl_offset + u_warp, width);
        sum += sumAbsoluteErrorAtDescriptorStereo(
            hs_a + dr_offset + u, vs_a + dr_offset + u,
            hs_b + dr_offset + u_warp, vs_b + dr_offset + u_warp, width);

        if (sum < min_1_E) {
            min_2_E = min_1_E;
            min_2_d = min_1_d;
            min_1_E = sum;
            min_1_d = d;
        } else if (sum < min_2_E) {
            min_2_E = sum;
            min_2_d = d;
        }
    }

    if (min_1_d >= 0 && min_2_d >= 0 &&
        (float)min_1_E < support_threshold * (float)min_2_E) {
        return min_1_d;
    }
    return -1;
}

// Pass 2: disparity matching for points where the texture mask is valid.
// A candidate is kept only if matching back from the right image gives a
// disparity within lr_threshold (left/right consistency, as in the CPU version).
__kernel void disparityMatchPass(
    __global const uchar* hsobel_left, __global const uchar* vsobel_left,
    __global const uchar* hsobel_right, __global const uchar* vsobel_right,
    int width, int height,
    __global short* candidates, int candi_width, int candi_height, int candi_step,
    int disp_min, int disp_max,
    int support_texture, float support_threshold, int lr_threshold,
    __global const int* texture_mask)
{
    const int u_step = 6;
    const int disp_offset = 2;

    int total = candi_width * candi_height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        // Read texture_mask
        int mask = texture_mask[idx];

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (!mask) continue;

        int u_can = idx % candi_width;
        int v_can = idx / candi_width;
        int u = u_can * candi_step;
        int v = v_can * candi_step;

        // find forwards
        short d = findSupportMatch(hsobel_left, vsobel_left, hsobel_right, vsobel_right,
                                   u, v, width, disp_min, disp_max, support_threshold, 0);
        if (d < 0) continue;

        // find backwards from the matched point in the right image
        int u_right = u - d;
        if (!(u_right > disp_offset + u_step && u_right < width - disp_offset - u_step - 1)) continue;

        int offset = width * v + u_right;
        if (sumAbsoluteErrorAtDescriptor(hsobel_right + offset, vsobel_right + offset, 128, width) < support_texture) continue;

        short d2 = findSupportMatch(hsobel_right, vsobel_right, hsobel_left, vsobel_left,
                                    u_right, v, width, disp_min, disp_max, support_threshold, 1);
        int diff = d - d2;
        if (diff < 0) diff = -diff;
        if (d2 >= 0 && diff <= lr_threshold) {
            candidates[idx] = d;
        }
    }
}
