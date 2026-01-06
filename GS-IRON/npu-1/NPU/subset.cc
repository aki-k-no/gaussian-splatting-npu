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
extern "C" {

void f32_get_color(float *SH_coeff, float *gaussians_data, float *output) {get_color<TILE_SIZE / CONV3D_TILE_NUM>(SH_coeff, gaussians_data, output);}

} // extern "C"