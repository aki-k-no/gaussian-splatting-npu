//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define NOCPP

#include <aie_api/aie.hpp>
#include <aie_api/detail/aie2/transpose.hpp>
#include <lut_based_ops.h>
#include <stdint.h>
#include "../const.hpp"
// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

#include "aie_kernel_utils.h"

using namespace aie;

const int M=4;
const int K=8;
const int N=4;

using bf16 = bfloat16;

template<const int GAUSSIAN_SIZE>
void get_color(bf16 *restrict coeff_data, bf16 *restrict gaussians_data, bf16 *restrict output){

    // load essential variables
    aie::vector<bf16, 8> camera_data = aie::load_v<8>(coeff_data + 16);
    aie::vector<bf16, 8> zeros = aie::zeros<bf16, 8>();

    bf16 zz2s[GAUSSIAN_SIZE];
    bf16 xx_add_yys[GAUSSIAN_SIZE];
    bf16 diff_zs[GAUSSIAN_SIZE];
    
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE, GAUSSIAN_SIZE)
    for(size_t i = 0;i<GAUSSIAN_SIZE;i++){
        
        event0();
        // compute dif vector
        aie::vector<bf16, 8> gaussian_xyz = aie::load_v<8>(gaussians_data);
        
        gaussians_data += 56;
        aie::vector<bf16, 8> diff_vec_unnormed = aie::sub(gaussian_xyz, camera_data);
        diff_vec_unnormed = aie::select(diff_vec_unnormed, zeros, aie::mask<8>::from_uint32(0b11111000));
        
        
        //normalize them
        aie::vector<bf16, 8> diff_powered = aie::mul(diff_vec_unnormed, diff_vec_unnormed);

        aie::vector<bf16, 8> sum = aie::invsqrt(aie::broadcast<bf16, 8>(aie::reduce_add(diff_powered)));
        aie::vector<bf16, 8> diff_vec = aie::zeros<bf16, 8>();
        diff_vec = aie::mul(diff_vec_unnormed, sum);
        bf16 diff_x = diff_vec[0];
        bf16 diff_y = diff_vec[1];
        bf16 diff_z = diff_vec[2];
        
        // replace some elemet for future computation
        diff_vec[3] = diff_y;
        diff_vec[4] = diff_x;
        diff_vec[5] = diff_y;
        diff_vec[6] = diff_z;
        diff_vec[7] = diff_z;
        aie::vector<bf16, 8> diff_vec_trans = aie::transpose(diff_vec, 2, 4);
        // index help: 0:xx 1:xy 2:yz 3:yy 4:xz 7:zz
        aie::vector<bf16, 8> xyz_muled = aie::mul(diff_vec, diff_vec_trans);
        //compute common num
        bf16 xx_add_yy = xyz_muled[0] + xyz_muled[3];
        bf16 xx_sub_yy = xyz_muled[0] - xyz_muled[3];
        bf16 zz2 = xyz_muled[7] * 2;
        bf16 zz4 = xyz_muled[7] * 4;

        //factor for mul
        aie::vector<bf16, 16> calc_vec1 = aie::broadcast<bf16, 16>(bf16(1));
        aie::vector<bf16, 16> calc_vec2 = aie::broadcast<bf16, 16>(bf16(1));
        // calc_vec1[2] = diff_z;
        calc_vec1[4] = xyz_muled[1];
        calc_vec1[5] = xyz_muled[2];
        calc_vec1[6] = zz2 - xx_add_yy;
        calc_vec1[7] = xyz_muled[4];
        calc_vec1[8] = xx_sub_yy;
        calc_vec1[9] = diff_y;
        calc_vec1[10] = diff_z;
        calc_vec1[11] = diff_y;
        //calc_vec1[12] = diff_z;
        calc_vec1[13] = diff_x;
        calc_vec1[14] = diff_z;
        calc_vec1[15] = diff_x;


        bf16 zz4_minus_xx_add_yy = zz4 - xx_add_yy;
        calc_vec2[1] = diff_y;
        calc_vec2[3] = diff_x;
        calc_vec2[9] = (bf16(3) * xyz_muled[0] - xyz_muled[3]);
        calc_vec2[10] = xyz_muled[1];
        calc_vec2[11] = zz4_minus_xx_add_yy;
        //calc_vec2[12] = (zz2 - bf16(3) * xx_add_yy); 
        calc_vec2[13] = zz4_minus_xx_add_yy;
        calc_vec2[14] = xx_sub_yy;
        calc_vec2[15] = xyz_muled[0] - bf16(3) * xyz_muled[3];
        aie::vector<bf16, 16> xyz_factor = aie::mul(calc_vec1, calc_vec2);

        

        //manually store some output, otherwise register pressure too high
        aie::store_v(output, xyz_factor);
        output[2] = diff_z;
        diff_zs[i] = diff_z;
        // bf16 tmp1 = diff_z * (zz2 - bf16(3) * xx_add_yy);
        //output[12] = tmp1;
        event1();
        output += 16;
        
        //compute coeff
        
    }
    output -= 16 * GAUSSIAN_SIZE;
    for(size_t i = 0;i<GAUSSIAN_SIZE / 16;i++){
        int idx = 0;
        
        aie::vector<bf16, 16> zz2_vec = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> xx_add_yy_vec = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> diff_z_vec = aie::zeros<bf16,16>(); 
        for(size_t j = 0;j<16;j++){
            zz2_vec[j] = zz2s[j + idx];
            xx_add_yy_vec[j] = xx_add_yys[j + idx];
            diff_z_vec[j] = output[2 + j * 16];
        }
        idx += 16;
        xx_add_yy_vec = aie::mul(xx_add_yy_vec, bf16(3));
        zz2_vec = aie::sub(zz2_vec, xx_add_yy_vec);
        aie::vector<bf16, 16> final_vec = aie::mul(diff_z_vec, zz2_vec);

        for(size_t j = 0;j<16;j++){
            output[12] = final_vec[j];
            output += 16;
        }

    }


}
extern "C" {

void f32_get_color(bf16 *SH_coeff, bf16 *gaussians_data, bf16 *output) {get_color<TILE_SIZE / CONV3D_TILE_NUM>(SH_coeff, gaussians_data, output);}

} // extern "C"