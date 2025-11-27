//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>


const int M=4;
const int K=8;
const int N=4;
using MMUL_PROJ = aie::mmul<M, K, N, bfloat16, bfloat16>;

// note that size of proj mat is always 4x4, gaussian is 4xN
void proj_to_view_space(bfloat16 *restrict proj_mat, bfloat16 *restrict gaussians, bfloat16 *restrict output, const int GAUSSIAN_SIZE) {
    // parallel factor

    // load input data
    auto pa=aie::begin_vector<M * K>(proj_mat);
    aie::vector<bfloat16,M * K> va=*pa;
    
    event0();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(32, 32)
    // compute over all elements
    for (size_t i = 0; i < GAUSSIAN_SIZE / N; i += N) {
        //load elements
        auto pb=aie::begin_vector<K * N>(proj_mat + sizeof(bfloat16) * i * 4);
        aie::vector<bfloat16,K * N> vb=*pb;
        MMUL_PROJ mmul;
        //matrix multiply
        mmul.mul(va, vb);
        // store data
        aie::store_v(output, mmul.to_vector<bfloat16>());
        aie::store_v(output, vb);
    }
    event1();
    return;
}

extern "C" {

void f32_proj_to_view_space(bfloat16 *proj_in, bfloat16 *gaussian_in, bfloat16 *out) { proj_to_view_space(proj_in, gaussian_in, out, 1024); }

} // extern "C"