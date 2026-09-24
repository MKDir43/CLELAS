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

// create_grid.cl : OpenCL 1.2 kernel for grid creation
// 1D strided loop + barrier for indirect memory access

__kernel void calculateValidGrid(
    __global const int* gpu_us, __global const int* gpu_vs, __global const int* gpu_ds, int sup_num,
    __global uchar* grid_valids, int grid_width, int grid_height, int grid_size, int disp_max, int right_image)
{
    for (int curr_sup = get_global_id(0); curr_sup < sup_num; curr_sup += get_global_size(0)) {
        // Indirect access: compute the index into grid_valids from the value of gpu_us[curr_sup]
        int x_curr = gpu_us[curr_sup];
        int y_curr = gpu_vs[curr_sup];
        int d_curr = gpu_ds[curr_sup];

        barrier(CLK_GLOBAL_MEM_FENCE);

        int d_min = d_curr - 1 > 0 ? d_curr - 1 : 0;
        int d_max = d_curr + 1 < disp_max ? d_curr + 1 : disp_max;
        int grid_x = right_image ? (int)floor((float)(x_curr - d_curr) / (float)grid_size) : (int)floor((float)x_curr / (float)grid_size);
        int grid_y = (int)floor((float)y_curr / (float)grid_size);

        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height) {
            int base_offset = grid_width * grid_y + grid_x;
            for (int d = d_min; d <= d_max; d++) {
                grid_valids[grid_width * grid_height * d + base_offset] = 1;
            }
        }
    }
}

__kernel void computeGrid(
    __global int* grid_data, __global int* grid_nums, __global uchar* grid_valids,
    int grid_width, int grid_height, int disp_max)
{
    int total = grid_width * grid_height;
    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {
        int x = idx % grid_width;
        int y = idx / grid_width;

        if (x > 0 && x < grid_width - 1 && y > 0 && y < grid_height - 1) {
            int curr_ind = 0;
            for (int d = 0; d <= disp_max; d++) {
                int disp_offset = grid_width * grid_height * d;
                int valid =
                    grid_valids[disp_offset + grid_width * (y - 1) + (x - 1)] |
                    grid_valids[disp_offset + grid_width * (y - 1) + (x + 0)] |
                    grid_valids[disp_offset + grid_width * (y - 1) + (x + 1)] |
                    grid_valids[disp_offset + grid_width * (y + 0) + (x - 1)] |
                    grid_valids[disp_offset + grid_width * (y + 0) + (x + 0)] |
                    grid_valids[disp_offset + grid_width * (y + 0) + (x + 1)] |
                    grid_valids[disp_offset + grid_width * (y + 1) + (x - 1)] |
                    grid_valids[disp_offset + grid_width * (y + 1) + (x + 0)] |
                    grid_valids[disp_offset + grid_width * (y + 1) + (x + 1)];
                if (valid > 0) {
                    grid_data[grid_width * grid_height * curr_ind + idx] = d;
                    curr_ind++;
                }
            }
            grid_nums[idx] = curr_ind;
        }
    }
}
