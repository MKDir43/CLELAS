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

// correct_supports.cl : OpenCL 1.2 kernels for support point correction
// 1D strided loop

// Invalidates candidates with too few consistent neighbors
// (removeInconsistentSupportPoints of the CPU version, evaluated in parallel).
__kernel void removeInconsistentSupportPoints(
    __global const short* candies,
    __global short* thin_candies,
    int candi_width, int candi_height,
    int incon_window_width, int incon_window_height,
    int incon_threshold, int incon_min_support)
{
    int total = candi_width * candi_height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int u_can = idx % candi_width;
        int v_can = idx / candi_width;

        short disp = candies[idx];

        if (disp < 0) {
            thin_candies[idx] = disp;
            continue;
        }

        int support_count = 0;
        for (int loc_v = -incon_window_height; loc_v <= incon_window_height; loc_v++) {
            for (int loc_u = -incon_window_width; loc_u <= incon_window_width; loc_u++) {
                int nu = u_can + loc_u;
                int nv = v_can + loc_v;
                if (nu >= 0 && nu < candi_width && nv >= 0 && nv < candi_height) {
                    short temp_disp = candies[candi_width * nv + nu];
                    if (temp_disp >= 0 && abs(disp - temp_disp) <= incon_threshold) {
                        support_count++;
                    }
                }
            }
        }

        thin_candies[idx] = support_count < incon_min_support ? -1 : disp;
    }
}

// Invalidates candidates that have a candidate of similar disparity on both sides
// (removeRedundantSupportPoints of the CPU version). A candidate only depends on
// the candidates of its own column (vertical) or row (horizontal), so each
// work-item walks one line in the order of the CPU version, updating in place,
// and gives the same result.
__kernel void removeRedundantSupportPoints(
    __global short* candies,
    int candi_width, int candi_height,
    int redun_max_dist, int redun_threshold, int vertical)
{
    // a line is a column (vertical) or a row (horizontal) of the candidate grid
    int line_num = vertical ? candi_width : candi_height;
    int line_length = vertical ? candi_height : candi_width;
    int stride = vertical ? candi_width : 1;

    for (int line = get_global_id(0); line < line_num; line += get_global_size(0)) {
        __global short* D = candies + (vertical ? line : candi_width * line);

        for (int pos = 0; pos < line_length; pos++) {
            short d = D[stride * pos];
            if (d < 0) {
                continue;
            }

            // redundant if both directions have support
            int redundant = 1;
            for (int dir = -1; dir <= 1 && redundant; dir += 2) {
                int support = 0;
                int pos_2 = pos;
                for (int j = 0; j < redun_max_dist; j++) {
                    pos_2 += dir;
                    if (pos_2 < 0 || pos_2 >= line_length) {
                        break;
                    }
                    short d_2 = D[stride * pos_2];
                    if (d_2 >= 0 && abs(d - d_2) <= redun_threshold) {
                        support = 1;
                        break;
                    }
                }
                redundant = support;
            }

            if (redundant) {
                D[stride * pos] = -1;
            }
        }
    }
}
