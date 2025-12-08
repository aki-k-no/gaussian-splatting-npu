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
// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

#include "aie_kernel_utils.h"

using namespace aie;

const int M=4;
const int K=8;
const int N=4;

using bf16 = bfloat16;

using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16>;
// projection from world to view space

// note that size of proj mat is always 4x4, gaussian is 4xN
template <const int GAUSSIAN_SIZE>
void proj_to_view_space(bf16 *restrict proj_mat, bf16 *restrict gaussians, bf16 *restrict output) {    
    // load input data
    aie::vector<bf16, 32> va = ::aie::load_v<32>(proj_mat);
    aie::vector<bf16, 32> va_padded = aie::zeros<bf16, 32>();
    for(size_t i=0;i<4;i++){
        for(size_t j=0;j<4;j++){
            va_padded[i * 8 + j] = va[i * 4 + j];
        }
    }
    
        aie::vector<bf16, 32> y_padded1= aie::load_v<32>(gaussians);
        aie::vector<bf16, 32> y_padded2= aie::load_v<32>(gaussians + 32);
        aie::vector<bf16, 32> y_padded3= aie::load_v<32>(gaussians + 64);
        aie::vector<bf16, 32> y_padded4= aie::load_v<32>(gaussians + 96);

    

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE / 16, GAUSSIAN_SIZE / 16)
    // compute over all elements
    for (size_t i = 0; i < GAUSSIAN_SIZE / 16; i += 1) {
        //load elements

        
        MMUL mmul1;
        MMUL mmul2;
        MMUL mmul3;
        MMUL mmul4;

        mmul1.mac(va_padded,y_padded1);
        mmul2.mac(va_padded,y_padded2);
        mmul3.mac(va_padded,y_padded3);
        mmul4.mac(va_padded,y_padded4);

        gaussians += 128;
        y_padded1= aie::load_v<32>(gaussians);
        y_padded2= aie::load_v<32>(gaussians + 32);
        y_padded3= aie::load_v<32>(gaussians + 64);
        y_padded4= aie::load_v<32>(gaussians + 96);



        

        aie::store_v(output, mmul1.to_vector<bf16>());
        aie::store_v(output + 16, mmul2.to_vector<bf16>());
        aie::store_v(output + 32, mmul3.to_vector<bf16>());
        aie::store_v(output + 48, mmul4.to_vector<bf16>());

        // store data
        output += 64;
        
        
    }

    return;
}

// projection from world to camera
template <const int GAUSSIAN_SIZE>
void get_camera_pos(bf16* restrict camera_mat, bf16 *restrict gaussians, bf16 *restrict output){
    // // load input data
    
    // load input data
    aie::vector<bf16, 16> va = ::aie::load_v<16>(camera_mat);
    aie::vector<bf16, 32> va_padded = aie::zeros<bf16, 32>();
    for(size_t i=0;i<4;i++){
        for(size_t j=0;j<4;j++){
            va_padded[i * 8 + j] = va[i * 4 + j];
        }
    }
    
        aie::vector<bf16, 32> y_padded1= aie::load_v<32>(gaussians);
        aie::vector<bf16, 32> y_padded2= aie::load_v<32>(gaussians + 32);
    

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE / 8, GAUSSIAN_SIZE / 8)
    // compute over all elements
    for (size_t i = 0; i < GAUSSIAN_SIZE / 8; i += 1) {
        event1();
        //load elements
        gaussians += 64;
        MMUL mmul1;
        MMUL mmul2;
        
        mmul1.mul(va_padded,y_padded1);
        mmul2.mul(va_padded,y_padded2);
        
        y_padded1= aie::load_v<32>(gaussians);
        y_padded2= aie::load_v<32>(gaussians + 32);

        aie::vector<bf16, 16> output_nonormed1 = mmul1.to_vector<bf16>();
        aie::vector<bf16, 16> output_nonormed2 = mmul2.to_vector<bf16>();
        // normalize
        aie::vector<bf16, 16> norm_vec = aie::broadcast<bf16, 16>(output_nonormed1[12]);
        event0();
        //hope compiler optimize this
        norm_vec[1] = output_nonormed1[13];
        norm_vec[2] = output_nonormed1[14];
        norm_vec[3] = output_nonormed1[15];
        
        norm_vec[5] = output_nonormed1[13];
        norm_vec[6] = output_nonormed1[14];
        norm_vec[7] = output_nonormed1[15];

        norm_vec[8] = output_nonormed2[12];
        norm_vec[9] = output_nonormed2[13];
        norm_vec[10] = output_nonormed2[14];
        norm_vec[11] = output_nonormed2[15];

        norm_vec[12] = output_nonormed2[12];
        norm_vec[13] = output_nonormed2[13];
        norm_vec[14] = output_nonormed2[14];
        norm_vec[15] = output_nonormed2[15];

        output_nonormed1[8] = output_nonormed2[0];
        output_nonormed1[9] = output_nonormed2[1];
        output_nonormed1[10] = output_nonormed2[2];
        output_nonormed1[11] = output_nonormed2[3];
        output_nonormed1[12] = output_nonormed2[4];
        output_nonormed1[13] = output_nonormed2[5];
        output_nonormed1[14] = output_nonormed2[6];
        output_nonormed1[15] = output_nonormed2[7];

        //aie::store_v(output, aie::add(aie::mul(aie::div(output_nonormed1, norm_vec).to_vector<bf16>(), bf16(0.5)), bf16(0.5)).to_vector<bf16>());
        aie::store_v(output, aie::div(output_nonormed1, norm_vec).to_vector<bf16>());

        // store data
        output += 16;
         
    }
    return;
}

template <const int GAUSSIAN_SIZE>
void get_conv3D(bf16 *restrict rotations, bf16 *restrict output){


    bf16 *scales = rotations + GAUSSIAN_SIZE * 4;
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

        
        event0();


        // compute xy, xz, yz, xx, yy, zz etc...
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
            aie::vector<bf16, 16> scale = aie::zeros<bf16, 16>();
            scale[0] = scale_1[j];
            scale[1] = scale_1[j];
            scale[2] = scale_1[j];
            scale[4] = scale_2[j];
            scale[5] = scale_2[j];
            scale[6] = scale_2[j];
            scale[8] = scale_3[j]; 
            scale[9] = scale_3[j];
            scale[10] = scale_3[j];
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
        
            output += 6;


        }
        
        event1();
    }


}

extern "C" {

void f32_proj_to_view_space(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { proj_to_view_space<128>(proj_in, gaussian_in, out); }

void f32_get_camera_pos(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { get_camera_pos<128>(proj_in, gaussian_in, out); }

void f32_get_conv3D(bf16 *rot_in, bf16 *out) { get_conv3D<64>(rot_in, out); }
} // extern "C"