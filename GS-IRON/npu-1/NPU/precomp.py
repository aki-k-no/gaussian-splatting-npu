# relu/relu.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1, Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_



def precomp(dev):
    xfr_dtype = np.float32

    trace_size = 8192


    @device(dev)
    def device_body():
        
        # Define tensor types
        line_size = 256
        proj_ty = np.ndarray[(4*4,), np.dtype[xfr_dtype]]
        gaussian_ty = np.ndarray[(4*line_size,), np.dtype[xfr_dtype]]

        # AIE Core Function declarations
        proj_func = external_func(
            "f32_proj_to_view_space", inputs=[proj_ty, gaussian_ty, gaussian_ty]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        ComputeTileProj = tile(0, 2)

        # AIE-array data movement with object fifos
        of_proj = object_fifo("proj", ShimTile0, ComputeTileProj, 2, proj_ty)
        of_gaussian = object_fifo("gaussian", ShimTile0, ComputeTileProj, 2, gaussian_ty)
        of_out = object_fifo("out", ComputeTileProj, ShimTile0, 2, gaussian_ty)


        # Compute tile for Projection Mat
        @core(ComputeTileProj, "precomp.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemIn1 = of_proj.acquire(ObjectFifoPort.Consume, 1)
                elemIn2 = of_gaussian.acquire(ObjectFifoPort.Consume, 1)
                proj_func(elemIn1, elemIn2, elemOut)
                of_proj.release(ObjectFifoPort.Consume, 1)
                of_gaussian.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce,1)


        # To/from AIE-array data movement
        @runtime_sequence(proj_ty, gaussian_ty, gaussian_ty)
        def sequence(A, B, C):
            proj_task = shim_dma_single_bd_task(of_proj, A, sizes=[1, 1, 1, 16])
            gaussian_task = shim_dma_single_bd_task(of_gaussian, B, sizes=[1, 1, 1, line_size * 4])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, line_size * 4], issue_token=True
            )

            dma_start_task(proj_task, gaussian_task, out_task)
            dma_await_task(out_task)
            dma_free_task(proj_task)
            dma_free_task(gaussian_task)
    



p = argparse.ArgumentParser()
## Parse command line arguments

## Device name is required to select the AIE device: npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col  # Four columns of NPU1, the maximum available
elif opts.device == "npu2":
    dev = AIEDevice.npu2_1col  # Eight columns of NPU2, the maximum available
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

with mlir_mod_ctx() as ctx:
    precomp(dev)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


