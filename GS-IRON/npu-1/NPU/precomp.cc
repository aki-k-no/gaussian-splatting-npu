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
    
    

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(GAUSSIAN_SIZE / 4, GAUSSIAN_SIZE / 4)
    // compute over all elements
    for (size_t i = 0; i < GAUSSIAN_SIZE / 4; i += 1) {
        //load elements
        aie::vector<bf16, 32> y_padded1= aie::load_v<32>(gaussians);
        // aie::vector<bf16, 32> y_padded2= aie::load_v<32>(gaussians + 32);
        // aie::vector<bf16, 32> y_padded3= aie::load_v<32>(gaussians + 64);
        // aie::vector<bf16, 32> y_padded4= aie::load_v<32>(gaussians + 96);


        
        MMUL mmul1;
        // MMUL mmul2;
        // MMUL mmul3;
        // MMUL mmul4;

        mmul1.mac(va_padded,y_padded1);
        // mmul2.mac(va_padded,y_padded2);
        // mmul3.mac(va_padded,y_padded3);
        // mmul4.mac(va_padded,y_padded4);

        

        aie::store_v(output, mmul1.to_vector<bf16>());
        // aie::store_v(output + 16, mmul2.to_vector<bf16>());
        // aie::store_v(output + 32, mmul3.to_vector<bf16>());
        // aie::store_v(output + 48, mmul4.to_vector<bf16>());

        // store data
        gaussians += 32;
        output += 16;
        
        
    }

    return;
}

// projection from world to camera
template <const int GAUSSIAN_SIZE>
void get_camera_pos(bf16* restrict camera_mat, bf16 *restrict gaussians, bf16 *restrict output){
    // // load input data
    
    // // load input data
    // aie::vector<bf16, 16> va = ::aie::load_v<16>(camera_mat);
    // aie::vector<bf16, 32> va_padded = aie::zeros<bf16, 32>();
    // for(size_t i=0;i<4;i++){
    //     for(size_t j=0;j<4;j++){
    //         va_padded[i * 8 + j] = va[i * 4 + j];
    //     }
    // }
    
    //            event0();
    //     aie::vector<bf16, 32> y_padded1= aie::load_v<32>(gaussians);
    //     aie::vector<bf16, 32> y_padded2= aie::load_v<32>(gaussians + 32);
    //     aie::vector<bf16, 32> y_padded3= aie::load_v<32>(gaussians + 64);
    //     aie::vector<bf16, 32> y_padded4= aie::load_v<32>(gaussians + 96);
    //            event1();
    
    //     MMUL mmul1;
    //     MMUL mmul2;
    //     MMUL mmul3;
    //     MMUL mmul4;

    // AIE_PREPARE_FOR_PIPELINING
    // AIE_LOOP_RANGE(GAUSSIAN_SIZE / 16, GAUSSIAN_SIZE / 16)
    // // compute over all elements
    // for (size_t i = 0; i < GAUSSIAN_SIZE / 16; i += 1) {
    //     //load elements
    //     gaussians += 128;


        

    //     mmul1.mul(va_padded,y_padded1);

    //     mmul2.mul(va_padded,y_padded2);

    //     mmul3.mul(va_padded,y_padded3);

    //     mmul4.mul(va_padded,y_padded4);

        
    //     y_padded1= aie::load_v<32>(gaussians);
    //     y_padded2= aie::load_v<32>(gaussians + 32);
    //     y_padded3= aie::load_v<32>(gaussians + 64);
    //     y_padded4= aie::load_v<32>(gaussians + 96);

    //     aie::store_v(output, mmul1.to_vector<bf16>());
    //     aie::store_v(output + 16, mmul2.to_vector<bf16>());
    //     aie::store_v(output + 32, mmul3.to_vector<bf16>());
    //     aie::store_v(output + 48, mmul4.to_vector<bf16>());

    //     // store data
    //     output += 64;
        
        
    // }


    return;

}

extern "C" {

void f32_proj_to_view_space(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { proj_to_view_space<128>(proj_in, gaussian_in, out); }

void f32_get_camera_pos(bf16 *proj_in, bf16 *gaussian_in, bf16 *out) { get_camera_pos<128>(proj_in, gaussian_in, out); }

} // extern "C"