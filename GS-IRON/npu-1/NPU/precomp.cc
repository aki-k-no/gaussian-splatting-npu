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
#include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <type_traits>

#include "aie_kernel_utils.h"

using namespace aie;

const int M=4;
const int K=8;
const int N=4;

// note that size of proj mat is always 4x4, gaussian is 4xN
template <const int GAUSSIAN_SIZE>
void proj_to_view_space(float *restrict proj_mat, float *restrict gaussians, float *restrict output) {
    // parallel factor

    // load input data
    aie::vector<float, 16> va = ::shuffle(aie::load_v<16>(proj_mat), T32_4x4);

    aie::vector<float, 16> x0 = ::shuffle(::extract_v4float_broadcast_to_v16float(va, 0), T32_4x4);
    aie::vector<float, 16> x1 = ::shuffle(::extract_v4float_broadcast_to_v16float(va, 1), T32_4x4);
    aie::vector<float, 16> x2 = ::shuffle(::extract_v4float_broadcast_to_v16float(va, 2), T32_4x4);
    aie::vector<float, 16> x3 = ::shuffle(::extract_v4float_broadcast_to_v16float(va, 3), T32_4x4);
    
    
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat,16>();
    
    event0();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(64, 64)
    // compute over all elements
    for (size_t i = 0; i < GAUSSIAN_SIZE / 4; i += 1) {
        //load elements
        aie::vector<float, 16> y=aie::load_v<16>(gaussians + i * 16);
        aie::vector<float, 16> y0 = ::extract_v4float_broadcast_to_v16float(y, 0);
        aie::vector<float, 16> y1 = ::extract_v4float_broadcast_to_v16float(y, 1);
        aie::vector<float, 16> y2 = ::extract_v4float_broadcast_to_v16float(y, 2);
        aie::vector<float, 16> y3 = ::extract_v4float_broadcast_to_v16float(y, 3);
        
        acc =  aie::zeros<accfloat,16>();

        acc = ::mul_elem_16(x0, y0);
        acc = ::mac_elem_16(x1, y1, acc);
        acc = ::mac_elem_16(x2, y2, acc);
        
        acc = ::mac_elem_16(x3, y3, acc);
        
        
        

        // store data
        aie::store_v(output + i * 16, acc.to_vector<float>());
        
    }



    
    event1();
    return;
}

extern "C" {

void f32_proj_to_view_space(float *proj_in, float *gaussian_in, float *out) { proj_to_view_space<256>(proj_in, gaussian_in, out); }

} // extern "C"