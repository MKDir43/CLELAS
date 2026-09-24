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

// create_triangle_mapping.cl : OpenCL 1.2 kernel for triangle mapping
// 1D strided loop over the rows + barrier for indirect memory access

// No fused multiply-add, so that the triangle edges match the CPU version bit for bit.
#pragma OPENCL FP_CONTRACT OFF

// Span of row v covered by a triangle in the scanline rasterization of
// Elas::createTriangleMapping (CPU version): the rows A_v <= v < C_v of the
// triangle sorted by v, and the columns ceil(min(u_1, u_2)) <= u <= floor(max(u_1, u_2))
// between its edges. Returns 0 when the row is not covered.
inline int triangleRowSpan(int v, float tri_u[3], float tri_v[3], int* u_min, int* u_max)
{
    // sort triangle corners wrt. v (ascending)
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < j; k++) {
            if (tri_v[k] > tri_v[j]) {
                float tri_u_temp = tri_u[j];
                tri_u[j] = tri_u[k];
                tri_u[k] = tri_u_temp;
                float tri_v_temp = tri_v[j];
                tri_v[j] = tri_v[k];
                tri_v[k] = tri_v_temp;
            }
        }
    }
    float A_u = tri_u[0], A_v = tri_v[0];
    float B_u = tri_u[1], B_v = tri_v[1];
    float C_u = tri_u[2], C_v = tri_v[2];

    // straight lines connecting the triangle corners
    float AB_a = 0.0f, AC_a = 0.0f, BC_a = 0.0f;
    if ((int)A_v != (int)B_v) AB_a = (A_u - B_u) / (A_v - B_v);
    if ((int)A_v != (int)C_v) AC_a = (A_u - C_u) / (A_v - C_v);
    if ((int)B_v != (int)C_v) BC_a = (B_u - C_u) / (B_v - C_v);
    float AB_b = A_u - AB_a * A_v;
    float AC_b = A_u - AC_a * A_v;
    float BC_b = B_u - BC_a * B_v;

    float u_1, u_2;
    if (v >= (int)A_v && v < (int)B_v) {         // first part (corner A->B)
        u_1 = AC_a * (float)v + AC_b;
        u_2 = AB_a * (float)v + AB_b;
    } else if (v >= (int)B_v && v < (int)C_v) {  // second part (corner B->C)
        u_1 = AC_a * (float)v + AC_b;
        u_2 = BC_a * (float)v + BC_b;
    } else {
        return 0;
    }
    *u_min = (int)ceil(fmin(u_1, u_2));
    *u_max = (int)floor(fmax(u_1, u_2));
    return 1;
}

// Each work-item draws whole rows: it walks the triangles in order and fills the
// span each one covers, so a pixel covered by several triangles gets the last one,
// as in the CPU version. tri_map has to be cleared to 0 (no triangle) beforehand.
__kernel void createTriangleMapping(
    __global const int* sup_us, __global const int* sup_vs, __global const int* sup_ds, int sup_num,
    __global const int* tri_c1s, __global const int* tri_c2s, __global const int* tri_c3s, int tri_num,
    __global ushort* tri_map, int width, int height, int subsampling, int right_image)
{
    // Every work-item runs the same number of iterations, so that all of them reach the barrier.
    for (int row_base = 0; row_base < height; row_base += get_global_size(0)) {
        int v = row_base + get_global_id(0);
        int active = v < height && !(subsampling && v % 2 != 0);

        for (int tri_idx = 0; tri_idx < tri_num; tri_idx++) {
            // Indirect access: tri_c1s[tri_idx] → sup_us[a]
            int a = tri_c1s[tri_idx];
            int b = tri_c2s[tri_idx];
            int c = tri_c3s[tri_idx];

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (!active || a < 0 || a >= sup_num || b < 0 || b >= sup_num || c < 0 || c >= sup_num) {
                continue;
            }

            float tri_v[3] = {(float)sup_vs[a], (float)sup_vs[b], (float)sup_vs[c]};
            if ((float)v < fmin(fmin(tri_v[0], tri_v[1]), tri_v[2]) ||
                (float)v >= fmax(fmax(tri_v[0], tri_v[1]), tri_v[2])) {
                continue;
            }

            float tri_u[3];
            if (!right_image) {
                tri_u[0] = (float)sup_us[a];
                tri_u[1] = (float)sup_us[b];
                tri_u[2] = (float)sup_us[c];
            } else {
                tri_u[0] = (float)(sup_us[a] - sup_ds[a]);
                tri_u[1] = (float)(sup_us[b] - sup_ds[b]);
                tri_u[2] = (float)(sup_us[c] - sup_ds[c]);
            }

            int u_min, u_max;
            if (!triangleRowSpan(v, tri_u, tri_v, &u_min, &u_max)) {
                continue;
            }
            u_min = max(u_min, 0);
            u_max = min(u_max, width - 1);
            for (int u = u_min; u <= u_max; u++) {
                if (!subsampling || u % 2 == 0) {
                    tri_map[width * v + u] = (ushort)(tri_idx + 1);
                }
            }
        }
    }
}
