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
using MMUL2 = aie::mmul<4, 8, 4, float, float>;
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
        //load elements
        gaussians += 64;
        MMUL mmul1;
        MMUL mmul2;
        
        mmul1.mul(va_padded,y_padded1);
        mmul2.mul(va_padded,y_padded2);
        
        y_padded1= aie::load_v<32>(gaussians);
        y_padded2= aie::load_v<32>(gaussians + 32);

        aie::vector<float, 16> output_nonormed1 = mmul1.to_vector<float>();
        aie::vector<float, 16> output_nonormed1_trans = aie::transpose(output_nonormed1, 2, 8);
        aie::vector<float, 8> output_nonormed_half1 = aie::filter_even(output_nonormed1_trans);
        output_nonormed_half1 = aie::transpose(output_nonormed_half1, 2, 4);
        aie::vector<float, 16> output_nonormed2 = mmul2.to_vector<float>();
        aie::vector<float, 16> output_nonormed2_trans = aie::transpose(output_nonormed2, 2, 8);
        aie::vector<float, 8> output_nonormed_half2 = aie::filter_even(output_nonormed2_trans);
        output_nonormed_half2 = aie::transpose(output_nonormed_half2, 2, 4);
        aie::vector<float, 16> output_nonormed = aie::concat(output_nonormed_half1, output_nonormed_half2);
        // normalize
        aie::vector<float, 16> norm_vec = aie::broadcast<float, 16>(output_nonormed1[12]);
        //hope compiler optimize this
        norm_vec[2] = output_nonormed1[13];
        norm_vec[3] = output_nonormed1[13];
        norm_vec[4] = output_nonormed1[14];
        norm_vec[5] = output_nonormed1[14];
        norm_vec[6] = output_nonormed1[15];
        norm_vec[7] = output_nonormed1[15];

        norm_vec[8] = output_nonormed2[12];
        norm_vec[9] = output_nonormed2[12];
        norm_vec[10] = output_nonormed2[13];
        norm_vec[11] = output_nonormed2[13];
        norm_vec[12] = output_nonormed2[14];
        norm_vec[13] = output_nonormed2[14];
        norm_vec[14] = output_nonormed2[15];
        norm_vec[15] = output_nonormed2[15];

        // instead of dividing by 2 afterwards, multiply by 2 here
        // or compiler goes crazy
        norm_vec = aie::mul(norm_vec, 2.f);

        aie::vector<float, 16> output_vec = aie::div(output_nonormed, norm_vec).to_vector<float>();
        output_vec = aie::add(output_vec, 0.5f);
        aie::store_v((float *)output, output_vec);

        // store data
        output += 32;
         
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
        
            output += 8;


        }
        
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
template <const int GAUSSIAN_SIZE, const int index>
void get_conv2D(bf16 *restrict JR, bf16 *restrict cov3D, bf16 *restrict output){
    
    JR = JR + index * GAUSSIAN_SIZE / 4 * 8;
    aie::vector<bf16, 16> nan_vec = aie::broadcast<bf16,16>(aie::sqrt(bf16(-1)));
    for(size_t i = 0; i < GAUSSIAN_SIZE / 64; i++){
        
        aie::vector<bf16, 16> vec_cov2D_0_0 = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> vec_cov2D_0_1 = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> vec_cov2D_1_1 = aie::zeros<bf16,16>();
        for(size_t j = 0; j < 16; j++){

            // we want to load as 6 elements, but aie load only support 4/8/16/etc...
            aie::vector<bf16, 8> cov3D_loaded = aie::load_v<8>(cov3D);

        
            // load cov3D and put into vector
            aie::vector<bf16, 16> cov3D_vec = aie::zeros<bf16,16>();

            // load mat
            aie::vector<bf16, 8> mat_loaded = ::aie::load_v<8>(JR);
            JR += 8;
            

            cov3D_vec[0] = cov3D_loaded[0];
            cov3D_vec[1] = cov3D_loaded[1];
            cov3D_vec[2] = cov3D_loaded[2];
            cov3D_vec[4] = cov3D_loaded[1];
            cov3D_vec[5] = cov3D_loaded[3];
            cov3D_vec[6] = cov3D_loaded[4];
            cov3D_vec[8] = cov3D_loaded[2];
            cov3D_vec[9] = cov3D_loaded[4];
            cov3D_vec[10] = cov3D_loaded[5];
            cov3D += 8;

            //copy it
            aie::vector<bf16, 8> mat_loaded2 = mat_loaded;
            //disguise after 3 elements
            mat_loaded[4] = mat_loaded[0];
            mat_loaded[5] = mat_loaded[1];
            mat_loaded[6] = mat_loaded[2];

            mat_loaded2[0] = mat_loaded2[4];
            mat_loaded2[1] = mat_loaded2[5];
            mat_loaded2[2] = mat_loaded2[6];
        
            aie::vector<bf16, 16> mat_vec1 = aie::concat(mat_loaded, mat_loaded);
            aie::vector<bf16, 16> mat_vec2 = aie::concat(mat_loaded2, mat_loaded2);

            aie::vector<bf16, 16> mat_vec1_trans = aie::transpose(mat_vec1, 4, 4);
            aie::vector<bf16, 16> mat_vec2_trans = aie::transpose(mat_vec2, 4, 4);



            // we only need to do accum and reduce add
            aie::vector<bf16, 16> cov2D_accum = aie::zeros<bf16,16>();
            cov2D_accum = aie::mul(aie::mul(cov3D_vec, mat_vec1_trans).to_vector<bf16>(), mat_vec1);
            vec_cov2D_0_0[j] = aie::reduce_add(cov2D_accum);

            cov2D_accum = aie::zeros<bf16,16>();
            cov2D_accum = aie::mul(aie::mul(cov3D_vec, mat_vec2_trans).to_vector<bf16>(), mat_vec1);
            vec_cov2D_0_1[j] = aie::reduce_add(cov2D_accum);

            cov2D_accum = aie::zeros<bf16,16>();
            cov2D_accum = aie::mul(aie::mul(cov3D_vec, mat_vec2_trans).to_vector<bf16>(), mat_vec2);
            vec_cov2D_1_1[j] = aie::reduce_add(cov2D_accum);

        }
        aie::vector<bf16, 16> vec_cov2D_0_1_minus = aie::mul(vec_cov2D_0_1, bf16(-1));

        // compute det
        vec_cov2D_0_0 = aie::add(vec_cov2D_0_0, bf16(0.3));
        vec_cov2D_1_1 = aie::add(vec_cov2D_1_1, bf16(0.3));
        aie::accum<accfloat, 16> det_accum = aie::mul(vec_cov2D_0_0, vec_cov2D_1_1);
        aie::vector<bf16, 16> det_min = aie::mul(det_accum.to_vector<bf16>(), bf16(0.01)).to_vector<bf16>();
        det_accum = aie::mac(det_accum, vec_cov2D_0_1, vec_cov2D_0_1_minus);
        aie::vector<bf16, 16> det_vec = det_accum.to_vector<bf16>();
        // if det is near to zero, or negative, set to nan
        aie::mask<16> det_is_zero = aie::le(det_vec, aie::abs(det_min));
        det_vec = aie::select(det_vec, nan_vec, det_is_zero);
        
        
        aie::vector<bf16, 16> inv_det_vec = aie::div(aie::broadcast<bf16,16>(bf16(1.0)), det_vec);

        // compute inverse
        aie::vector<bf16, 16> inv_cov2D_0_0 = aie::mul(vec_cov2D_1_1, inv_det_vec);
        aie::vector<bf16, 16> inv_cov2D_0_1 = aie::mul(aie::mul(vec_cov2D_0_1, inv_det_vec).to_vector<bf16>(), bf16(-1));
        aie::vector<bf16, 16> inv_cov2D_1_1 = aie::mul(vec_cov2D_0_0, inv_det_vec);

        // calc eigenvalue
        aie::vector<bf16, 16> b = aie::mul(aie::add(vec_cov2D_0_0, vec_cov2D_1_1), bf16(0.5));
        aie::vector<bf16, 16> sqrt_term = aie::sqrt(aie::max(aie::sub(aie::mul(b, b).to_vector<bf16>(), det_vec), bf16(0.1)));
        aie::vector<bf16, 16> radius = aie::add(aie::mul(aie::sqrt(aie::add(b, sqrt_term)), bf16(3)), bf16(0));

        //pack and store
        aie::vector<bf16, 64> packed_untransposed = aie::concat(aie::concat(inv_cov2D_0_0, inv_cov2D_0_1), aie::concat(inv_cov2D_1_1, radius));
        //transpose
        aie::vector<bf16, 64> packed_transposed = aie::transpose(packed_untransposed, 4, 16);
        aie::store_v(output, packed_transposed);
        output += 64;

    }
    return;
}


template<const int GAUSSIAN_SIZE>
void get_color_pre(bf16 *restrict coeff_data, bf16 *restrict gaussians_data, bf16 *restrict output){
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
        //instead, calculate later for vector calculation
        aie::store_v(output, xyz_factor);
        output[2] = diff_z;
        zz2s[i] = zz2;
        xx_add_yys[i] = xx_add_yy;
        diff_zs[i] = diff_z;
        // bf16 tmp1 = diff_z * (zz2 - bf16(3) * xx_add_yy);
        //output[12] = tmp1;
        event1();
        output += 16;
        
        //compute coeff
        
    }
    output -= 16 * GAUSSIAN_SIZE;
    int idx = 0;
    for(size_t i = 0;i<GAUSSIAN_SIZE / 16;i++){
        
        aie::vector<bf16, 16> zz2_vec = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> xx_add_yy_vec = aie::zeros<bf16,16>();
        aie::vector<bf16, 16> diff_z_vec = aie::zeros<bf16,16>(); 
        for(size_t j = 0;j<16;j++){
            zz2_vec[j] = zz2s[j + idx];
            xx_add_yy_vec[j] = xx_add_yys[j + idx];
            diff_z_vec[j] = diff_zs[j + idx];
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

template<const int GAUSSIAN_SIZE>
void get_color_post(bf16 *restrict xyz_factors, bf16 *restrict gaussians_data, bf16 *restrict output){

    // to be implemented if needed
    //compute coeff
    gaussians_data += 8;

    aie::vector<bf16, 16> sh_coeff(bf16(0.28209479177387814f),
                                  bf16(-0.4886025119029199f), bf16(0.4886025119029199f), bf16(-0.4886025119029199f),
                                  bf16(1.0925484305920792f), bf16(-1.0925484305920792f), bf16(0.31539156525252005f), bf16(-1.0925484305920792f), bf16(0.5462742152960396f),
                                  bf16(-0.5900435899266435f), bf16(2.890611442640554f), bf16(-0.4570457994644658f),
                                  bf16(0.3731763325901154f), bf16(-0.4570457994644658f), bf16(1.445305721320277f), bf16(-0.5900435899266435f));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE, GAUSSIAN_SIZE)
    for(size_t i = 0;i<GAUSSIAN_SIZE;i++){

        aie::vector<bf16, 16> xyz_factor = ::aie::load_v<16>(xyz_factors);
        xyz_factors += 16;
        
        //just load and compute three times
        aie::vector<bf16, 16> color1 = aie::concat(aie::load_v<8>(gaussians_data), aie::load_v<8>(gaussians_data + 8));
        color1 = aie::mul(color1, xyz_factor);
        color1 = aie::mul(color1, sh_coeff);
        bf16 result1 = aie::reduce_add(color1);
        output[0] = result1 + bf16(0.5);

        aie::vector<bf16, 16> color2 = aie::concat(aie::load_v<8>(gaussians_data + 16), aie::load_v<8>(gaussians_data + 24));
        color2 = aie::mul(color2, xyz_factor);
        color2 = aie::mul(color2, sh_coeff);
        bf16 result2 = aie::reduce_add(color2);
        output[1] = result2 + bf16(0.5);

        aie::vector<bf16, 16> color3 = aie::concat(aie::load_v<8>(gaussians_data + 32), aie::load_v<8>(gaussians_data + 40));
        color3 = aie::mul(color3, xyz_factor);
        color3 = aie::mul(color3, sh_coeff);
        bf16 result3 = aie::reduce_add(color3);
        output[2] = result3 + bf16(0.5);
        
        gaussians_data += 56;
        output += 4;
    }

}

template<const int GAUSSIAN_SIZE>
void get_special_index(bf16 *camera_xyz, bf16 *opacity, bf16 *output){

    for(size_t i = 0;i<GAUSSIAN_SIZE / 16;i++){
        aie::vector<bf16, 64> camera_vec = ::aie::load_v<64>(camera_xyz);
        camera_xyz += 64;
        aie::vector<bf16, 16> opacity_vec = ::aie::load_v<16>(opacity);
        opacity += 16;

        aie::vector<bf16, 16> camera_zs = aie::filter_odd(aie::filter_even(camera_vec));
        aie::vector<bf16, 32> merged = aie::transpose(aie::concat(opacity_vec, camera_zs), 2, 16);
        aie::store_v(output, merged);
        output += 32;
    }
}


extern "C" {

void f32_proj_to_view_space(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { proj_to_view_space<TILE_SIZE>(proj_in, gaussian_in, out); }

void f32_get_camera_pos(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { get_camera_pos<TILE_SIZE>(proj_in, gaussian_in, out); }

void f32_get_conv3D(bf16 *rot_in, bf16 *out) { get_conv3D<TILE_SIZE / CONV3D_TILE_NUM>(rot_in, out); }

void f32_get_J_R(bf16 *params_in, bf16 *pos_in, bf16 *out) { get_J_R<TILE_SIZE>(params_in, pos_in, out); }

void f32_get_conv2D_0(bf16 *JR_in, bf16 *cov3D_in, bf16 *out) { get_conv2D<TILE_SIZE, 0>(JR_in, cov3D_in, out); }
void f32_get_conv2D_1(bf16 *JR_in, bf16 *cov3D_in, bf16 *out) { get_conv2D<TILE_SIZE, 1>(JR_in, cov3D_in, out); }
void f32_get_conv2D_2(bf16 *JR_in, bf16 *cov3D_in, bf16 *out) { get_conv2D<TILE_SIZE, 2>(JR_in, cov3D_in, out); }
void f32_get_conv2D_3(bf16 *JR_in, bf16 *cov3D_in, bf16 *out) { get_conv2D<TILE_SIZE, 3>(JR_in, cov3D_in, out); }

void f32_get_color_pre(bf16 *SH_coeff, bf16 *gaussians_data, bf16 *output) {get_color_pre<TILE_SIZE / CONV3D_TILE_NUM>(SH_coeff, gaussians_data, output);}

void f32_get_color_post(bf16 *xyz_factors, bf16 *gaussians_data, bf16 *output) {get_color_post<TILE_SIZE / CONV3D_TILE_NUM>(xyz_factors, gaussians_data, output);}

void f32_get_special_index(bf16 *camera_xyz, bf16 *opacity, bf16 *output) {get_special_index<TILE_SIZE>(camera_xyz, opacity, output);}

} // extern "C"