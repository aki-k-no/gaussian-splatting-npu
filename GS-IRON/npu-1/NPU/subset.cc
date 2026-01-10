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
void get_color(float *restrict coeff_data, float *restrict gaussians_data, float *restrict output){

    aie::vector<float, 16> vec1 = aie::load_v<16>(coeff_data);
    aie::vector<float, 16> gauss_vec = aie::load_v<16>(gaussians_data);

    gauss_vec = aie::mul(vec1, gauss_vec);
    aie::store_v(output, gauss_vec);

}

template <const int GAUSSIAN_SIZE>
void get_conv3D(bf16 *restrict rotations, bf16 *restrict output){

    event0();

    bf16 *scales = rotations + GAUSSIAN_SIZE * 4;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE / 16, GAUSSIAN_SIZE / 16)
    for(size_t i=0;i<GAUSSIAN_SIZE / 16;i++){

        // load rotation quaternions
        aie::vector<bf16, 16> rot_ws = ::aie::load_v<16>(rotations);
        aie::vector<bf16, 16> rot_xs = ::aie::load_v<16>(rotations + 16);
        aie::vector<bf16, 16> rot_ys = ::aie::load_v<16>(rotations + 32);
        aie::vector<bf16, 16> rot_zs = ::aie::load_v<16>(rotations + 48);

        aie::vector<bf16, 16> scale_1 = ::aie::load_v<16>(scales); // ::aie::load_v<16>(scales);
        aie::vector<bf16, 16> scale_2 = ::aie::load_v<16>(scales + 16);
        aie::vector<bf16, 16> scale_3 = ::aie::load_v<16>(scales + 32);
        scales += 48;
        rotations += 64;


        // compute norm
        aie::accum<accfloat, 16> rot_accum = aie::zeros<accfloat, 16>();
        
        rot_accum = aie::mul(rot_ws, rot_ws);
        rot_accum = aie::mac(rot_accum, rot_xs, rot_xs);
        rot_accum = aie::mac(rot_accum, rot_ys, rot_ys);
        rot_accum = aie::mac(rot_accum, rot_zs, rot_zs);
        rot_accum = aie::sqrt(rot_accum.to_vector<float>());
        
        aie::vector<bf16, 16> rot_norm_factor = rot_accum.to_vector<bf16>();
        
        // normalize them with SIMD
        aie::vector<bf16, 16> rot_normed_ws = aie::div(rot_ws, rot_norm_factor);
        aie::vector<bf16, 16> rot_normed_xs = aie::div(rot_xs, rot_norm_factor);
        aie::vector<bf16, 16> rot_normed_ys = aie::div(rot_ys, rot_norm_factor);
        aie::vector<bf16, 16> rot_normed_zs = aie::div(rot_zs, rot_norm_factor);

        


        // compute xy, xz, yz, xx, yy, zz etc...
        
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(16, 16)
        for(size_t j=0;j<16;j++){

            aie::vector<bf16, 8> compute_vec1(rot_normed_xs[j], rot_normed_xs[j], rot_normed_ws[j], rot_normed_ws[j],
                                             rot_normed_zs[j], rot_normed_zs[j], rot_normed_ys[j], rot_normed_ys[j]);
            aie::vector<bf16,8> compute_vec1_trans = aie::transpose(compute_vec1, 4, 2);

            // aie::vector<bf16, 8> compute_vec2(rot_normed_xs[j+1], rot_normed_xs[j+1], rot_normed_ws[j+1], rot_normed_ws[j+1],
            //                                  rot_normed_zs[j+1], rot_normed_zs[j+1], rot_normed_ys[j+1], rot_normed_ys[j+1]);
            // aie::vector<bf16,8> compute_vec2_trans = aie::transpose(compute_vec2, 4, 2);

            //concat them
            aie::vector<bf16, 16> compute_vec_concat = aie::concat(compute_vec1, aie::zeros<bf16,8>());
            aie::vector<bf16, 16> compute_vec_concat_trans = aie::concat(compute_vec1_trans, aie::zeros<bf16,8>());
            compute_vec_concat[8] = bf16(-1) * rot_normed_zs[j];
            compute_vec_concat[9] = rot_normed_xs[j];

            compute_vec_concat_trans[8] = rot_normed_zs[j];
            compute_vec_concat_trans[9] = rot_normed_ys[j];

            //element-wise mul
            // order : xx xw wz wy zx zw yz yy 
            aie::vector<bf16, 16> result_vec = aie::mul(aie::mul(compute_vec_concat, compute_vec_concat_trans).to_vector<bf16>(),bf16(2));


            // matrix for computation
            aie::vector<bf16, 16> compute_R1_1 = aie::zeros<bf16, 16>();
            compute_R1_1[0] = bf16(1);
            compute_R1_1[1] = result_vec[9]; // xy
            compute_R1_1[2] = result_vec[4]; // xz

            compute_R1_1[4] = result_vec[9]; // xy
            compute_R1_1[5] = bf16(1);
            compute_R1_1[6] = result_vec[6]; // yz

            compute_R1_1[8] = result_vec[4]; // xz
            compute_R1_1[9] = result_vec[6]; // yz
            compute_R1_1[10] = bf16(1);
            
            //for sub
            aie::vector<bf16, 16> compute_R1_2 = aie::zeros<bf16, 16>();
            compute_R1_2[0] = result_vec[7]; // -yy
            compute_R1_2[2] = result_vec[3]; // -yw

            compute_R1_2[4] = result_vec[2]; // -zw
            compute_R1_2[5] = result_vec[0]; // -xx

            compute_R1_2[9] = result_vec[1]; // -xw
            compute_R1_2[10] = result_vec[0]; // -xx


            //for leftover computation
            aie::vector<bf16, 16> compute_R1_3 = aie::zeros<bf16, 16>();
            compute_R1_3[0] = result_vec[8]; // -zz
            compute_R1_3[1] = result_vec[2]; // zw

            compute_R1_3[5] = result_vec[8]; // -zz
            compute_R1_3[6] = result_vec[1]; // xw

            compute_R1_3[8] = result_vec[3]; // yw
            compute_R1_3[10] = result_vec[7] * -1; // -1 * yy

            aie::vector<bf16, 16> R1 = aie::add(aie::sub(compute_R1_1, compute_R1_2), compute_R1_3);
            //add scale
            aie::vector<bf16, 16> scale = aie::concat(aie::broadcast<bf16, 8>(scale_1[j]), aie::broadcast<bf16, 8>(scale_3[j]));
            
            scale[4] = scale_2[j];
            scale[5] = scale_2[j];
            scale[6] = scale_2[j];
            R1 = aie::mul(R1, scale);

            aie::vector<bf16, 32> R1_padded = aie::concat(R1, aie::zeros<bf16, 16>());
            aie::vector<bf16, 32> R1_transposed = aie::transpose(R1_padded, 8, 4);
            //somehow, llvm raise an error if we use R1_padded directly
            aie::vector<bf16, 32> R1_padded2 = aie::transpose(R1_transposed, 4, 8);

            MMUL mmul1;
            mmul1.mac(R1_transposed, R1_padded2);
            aie::vector<bf16, 16> output1 = mmul1.to_vector<bf16>();

            // we only need 6 of matrix
            output[0] = output1[0];
            output[1] = output1[1];
            output[2] = output1[2];
            output[3] = output1[5];
            output[4] = output1[6];
            output[5] = output1[10];
        
            output += 8;


        }
        
    }
    event1();
}


extern "C" {

void f32_get_color(float *SH_coeff, float *gaussians_data, float *output) {get_color<TILE_SIZE / CONV3D_TILE_NUM>(SH_coeff, gaussians_data, output);}

} // extern "C"