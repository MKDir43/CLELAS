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

// disparity_planes.cl : OpenCL 1.2 kernel for disparity plane computation
// 1D strided loop + barrier for indirect memory access

// No fused multiply-add, so that the planes match the CPU version bit for bit.
#pragma OPENCL FP_CONTRACT OFF

__kernel void computeDisparityPlanes(
    __global int* sup_us, __global int* sup_vs, __global int* sup_ds,
    __global int* tri_c1s, __global int* tri_c2s, __global int* tri_c3s,
    __global float* tri_t1as, __global float* tri_t1bs, __global float* tri_t1cs,
    __global float* tri_t2as, __global float* tri_t2bs, __global float* tri_t2cs,
    int tri_num)
{
    for (int tri_idx = get_global_id(0); tri_idx < tri_num; tri_idx += get_global_size(0)) {
        // Indirect access: tri_c1s[tri_idx] → sup_us[c1] pattern
        int c1 = tri_c1s[tri_idx];
        int c2 = tri_c2s[tri_idx];
        int c3 = tri_c3s[tri_idx];

        barrier(CLK_GLOBAL_MEM_FENCE);

        int a_u = sup_us[c1]; int a_v = sup_vs[c1]; int a_d = sup_ds[c1];
        int b_u = sup_us[c2]; int b_v = sup_vs[c2]; int b_d = sup_ds[c2];
        int c_u = sup_us[c3]; int c_v = sup_vs[c3]; int c_d = sup_ds[c3];

        int det_inv = a_u*b_v + a_v*c_u + b_u*c_v - b_v*c_u - a_v*b_u - a_u*c_v;

        if (det_inv != 0) {
            tri_t1as[tri_idx] = (1.0f/det_inv)*((b_v-c_v)*a_d+(-a_v+c_v)*b_d+(a_v-b_v)*c_d);
            tri_t1bs[tri_idx] = (1.0f/det_inv)*((-b_u+c_u)*a_d+(a_u-c_u)*b_d+(-a_u+b_u)*c_d);
            tri_t1cs[tri_idx] = (1.0f/det_inv)*((b_u*c_v-b_v*c_u)*a_d+(-a_u*c_v+a_v*c_u)*b_d+(a_u*b_v-a_v*b_u)*c_d);
        } else {
            tri_t1as[tri_idx] = 0.0f;
            tri_t1bs[tri_idx] = 0.0f;
            tri_t1cs[tri_idx] = 0.0f;
        }

        // right triangle
        a_u = sup_us[c1] - sup_ds[c1];
        b_u = sup_us[c2] - sup_ds[c2];
        c_u = sup_us[c3] - sup_ds[c3];
        det_inv = a_u*b_v + a_v*c_u + b_u*c_v - b_v*c_u - a_v*b_u - a_u*c_v;
        if (det_inv != 0) {
            tri_t2as[tri_idx] = (1.0f/det_inv)*((b_v-c_v)*a_d+(-a_v+c_v)*b_d+(a_v-b_v)*c_d);
            tri_t2bs[tri_idx] = (1.0f/det_inv)*((-b_u+c_u)*a_d+(a_u-c_u)*b_d+(-a_u+b_u)*c_d);
            tri_t2cs[tri_idx] = (1.0f/det_inv)*((b_u*c_v-b_v*c_u)*a_d+(-a_u*c_v+a_v*c_u)*b_d+(a_u*b_v-a_v*b_u)*c_d);
        } else {
            tri_t2as[tri_idx] = 0.0f;
            tri_t2bs[tri_idx] = 0.0f;
            tri_t2cs[tri_idx] = 0.0f;
        }
    }
}
