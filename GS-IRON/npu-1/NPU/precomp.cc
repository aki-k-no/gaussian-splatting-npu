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

        aie::vector<bf16, 16> output_vec1 = mmul1.to_vector<bf16>();
        output_vec1 = aie::transpose(output_vec1, 4, 4);
        aie::vector<bf16, 16> output_vec2 = mmul2.to_vector<bf16>();
        output_vec2 = aie::transpose(output_vec2, 4, 4);
        aie::vector<bf16, 16> output_vec3 = mmul3.to_vector<bf16>();
        output_vec3 = aie::transpose(output_vec3, 4, 4);
        aie::vector<bf16, 16> output_vec4 = mmul4.to_vector<bf16>();
        output_vec4 = aie::transpose(output_vec4, 4, 4);
        aie::store_v(output, output_vec1);
        aie::store_v(output + 16, output_vec2);
        aie::store_v(output + 32, output_vec3);
        aie::store_v(output + 48, output_vec4);

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

        
        event0();


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

template <const int GAUSSIAN_SIZE>
void get_J_R(bf16 *restrict params, bf16 *restrict positions, bf16 *restrict output){

    aie::vector<bf16, 16> param_vec = ::aie::load_v<16>(params);
    bf16 a1 = params[0];
    bf16 a2 = params[1];
    bf16 a3 = params[2];
    bf16 b1 = params[4];
    bf16 b2 = params[5];
    bf16 b3 = params[6];
    bf16 c1 = params[8];
    bf16 c2 = params[9];
    bf16 c3 = params[10];

    
    //vector for computation
    //pack them
    aie::vector<bf16, 8> compute_vec1_half = aie::zeros<bf16, 8>();
    compute_vec1_half[0] = a1;
    compute_vec1_half[1] = a2;
    compute_vec1_half[2] = a3;
    compute_vec1_half[4] = b1;
    compute_vec1_half[5] = b2;
    compute_vec1_half[6] = b3;
    aie::vector<bf16, 16> compute_vec1 = aie::concat(compute_vec1_half, compute_vec1_half);

    aie::vector<bf16, 8> compute_vec2_half = aie::zeros<bf16, 8>();
    compute_vec2_half[0] = c1;
    compute_vec2_half[1] = c2;
    compute_vec2_half[2] = c3;
    compute_vec2_half[4] = c1;
    compute_vec2_half[5] = c2;
    compute_vec2_half[6] = c3;

    aie::vector<bf16, 16> compute_vec2 = aie::concat(compute_vec2_half, compute_vec2_half);




    bf16 fx = params[16];
    bf16 fy = params[17];
    aie::vector<bf16, 16> cam_fx_vec = aie::filter_even(aie::broadcast(fx));
    aie::vector<bf16, 16> cam_fy_vec = aie::filter_even(aie::broadcast(fy));
    fx = bf16(-1) * fx;
    fy = bf16(-1) * fy;

    bf16 tmp[64];
    
    AIE_LOOP_NO_UNROLL
    for(size_t i = 0;i<GAUSSIAN_SIZE / 16;i++){
        aie::vector<bf16,64> before_transpose = ::aie::load_v<64>(positions);
        positions += 64;
        aie::vector<bf16,64> after_transpose = aie::transpose(before_transpose, 16, 4);

        //store it temporarily for splitting
        aie::store_v(tmp, after_transpose);

        aie::vector<bf16,16> xs = aie::load_v<16>(tmp);
        aie::vector<bf16,16> ys = aie::load_v<16>(tmp + 16);
        aie::vector<bf16,16> zs = aie::load_v<16>(tmp + 32);

        aie::vector<bf16,16> fx_z = aie::div(cam_fx_vec, zs);
        aie::vector<bf16,16> fy_z = aie::div(cam_fy_vec, zs);

        aie::vector<bf16,16> z_z = aie::mul(zs, zs).to_vector<bf16>();

        aie::vector<bf16,16> calc1 = aie::div(aie::mul(xs, fx).to_vector<bf16>(), z_z);
        aie::vector<bf16,16> calc2 = aie::div(aie::mul(ys, fy).to_vector<bf16>(), z_z);

        for(size_t j=0; j<16; j+= 2){
            // use element-wise calc
            aie::accum<accfloat, 16> J_R_accum = aie::zeros<accfloat, 16>();

            aie::vector<bf16,16> factor_vec1 = aie::zeros<bf16,16>();
            factor_vec1[0] = fx_z[j];
            factor_vec1[1] = fx_z[j];
            factor_vec1[2] = fx_z[j];
            factor_vec1[4] = fy_z[j];
            factor_vec1[5] = fy_z[j];
            factor_vec1[6] = fy_z[j];
            factor_vec1[8] = fx_z[j + 1];
            factor_vec1[9] = fx_z[j + 1];
            factor_vec1[10] = fx_z[j + 1];
            factor_vec1[12] = fy_z[j + 1];
            factor_vec1[13] = fy_z[j + 1];
            factor_vec1[14] = fy_z[j + 1];

            aie::vector<bf16,16> factor_vec2 = aie::zeros<bf16,16>();
            factor_vec2[0] = calc1[j];
            factor_vec2[1] = calc1[j];
            factor_vec2[2] = calc1[j];
            factor_vec2[4] = calc2[j];
            factor_vec2[5] = calc2[j];
            factor_vec2[6] = calc2[j];
            factor_vec2[8] = calc1[j + 1];
            factor_vec2[9] = calc1[j + 1];
            factor_vec2[10] = calc1[j + 1];
            factor_vec2[12] = calc2[j + 1];
            factor_vec2[13] = calc2[j + 1];
            factor_vec2[14] = calc2[j + 1];

            J_R_accum = aie::mul(compute_vec1, factor_vec1);
            J_R_accum = aie::mac(J_R_accum, compute_vec2, factor_vec2);
            aie::store_v(output, J_R_accum.to_vector<bf16>());

            //store them 
            output += 16;
        }

    }
    return;
}

//cov2D computation
template <const int GAUSSIAN_SIZE>
void get_conv2D(bf16 * mats, bf16 *restrict cov3D, bf16 *restrict output){
    
    return;
}

extern "C" {

void f32_proj_to_view_space(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { proj_to_view_space<TILE_SIZE>(proj_in, gaussian_in, out); }

void f32_get_camera_pos(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { get_camera_pos<TILE_SIZE>(proj_in, gaussian_in, out); }

void f32_get_conv3D(bf16 *rot_in, bf16 *out) { get_conv3D<TILE_SIZE / CONV3D_TILE_NUM>(rot_in, out); }

void f32_get_J_R(bf16 *params_in, bf16 *pos_in, bf16 *out) { get_J_R<TILE_SIZE>(params_in, pos_in, out); }
} // extern "C"