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
import aie.utils.trace as trace_utils



def precomp(dev):
    xfr_dtype = bfloat16

    trace_size = 8192 * 4


    @device(dev)
    def device_body():
        
        # Define tensor types
        line_size = 64
        sub_tiles = 1
        world_to_view_size = 4 * 4
        get_camera_size = 4 * 4
        w2v_ty = np.ndarray[(world_to_view_size,), np.dtype[xfr_dtype]]
        get_camera_ty = np.ndarray[(get_camera_size,), np.dtype[xfr_dtype]]
        gaussian_send_ty = np.ndarray[(8*line_size // sub_tiles,), np.dtype[xfr_dtype]]
        gaussian_back_ty = np.ndarray[(4*line_size // sub_tiles,), np.dtype[xfr_dtype]]
        return_ty = np.ndarray[(4*line_size * 2 // sub_tiles,), np.dtype[xfr_dtype]]

        essentials_ty = np.ndarray[(world_to_view_size + get_camera_size,), np.dtype[xfr_dtype]]

        # AIE Core Function declarations
        w2v_func = external_func(
            "f32_proj_to_view_space", inputs=[w2v_ty, gaussian_send_ty, gaussian_back_ty]
        )

        camera_func = external_func(
            "f32_get_camera_pos", inputs=[get_camera_ty, gaussian_send_ty, gaussian_back_ty]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        MemTile0 = tile(0, 1)
        ComputeTileV2w = tile(0, 2)
        ComputeTileCamera = tile(0, 3)

        # AIE-array data movement with object fifos
        of_essentials = object_fifo("essentials", ShimTile0, MemTile0, 2, essentials_ty)
        of_w2v = object_fifo("w2v", MemTile0, ComputeTileV2w, 2, w2v_ty)
        of_camera = object_fifo("camera", MemTile0, ComputeTileCamera, 2, get_camera_ty)
        object_fifo_link(of_essentials, [of_w2v, of_camera], [], [0, world_to_view_size])

        of_gaussian = object_fifo("gaussian", ShimTile0, [ComputeTileV2w, ComputeTileCamera], 4, gaussian_send_ty)
        of_out1 = object_fifo("out1", ComputeTileV2w, MemTile0, 2, gaussian_back_ty)
        of_out2 = object_fifo("out2", ComputeTileCamera, MemTile0, 2, gaussian_back_ty)
        of_out_unit = object_fifo("out_unit",  MemTile0, ShimTile0, 2, return_ty)
        object_fifo_link([of_out1, of_out2], of_out_unit, [0, 4*line_size // sub_tiles], [])


        # Compute tile for Projection Mat
        @core(ComputeTileV2w, "precomp.o")
        def core_body_v2w():
            for _ in range_(0xFFFFFFFF):
                elemIn1 = of_w2v.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(4):
                    elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_gaussian.acquire(ObjectFifoPort.Consume, 1)
                    w2v_func(elemIn1, elemIn2, elemOut)
                    of_gaussian.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce,1)
                of_w2v.release(ObjectFifoPort.Consume, 1)
                

            
        # Compute tile for Projection Mat
        @core(ComputeTileCamera, "precomp.o")
        def core_body_camera():
            for _ in range_(0xFFFFFFFF):
                elemIn1 = of_camera.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(4):
                    elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_gaussian.acquire(ObjectFifoPort.Consume, 1)
                    camera_func(elemIn1, elemIn2, elemOut)
                    of_gaussian.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce,1)
                of_camera.release(ObjectFifoPort.Consume, 1)
                
        tiles_to_trace = [ComputeTileV2w, ComputeTileCamera, ShimTile0]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile0)


        # To/from AIE-array data movement
        @runtime_sequence(essentials_ty, gaussian_send_ty, return_ty)
        def sequence(A, B, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile0,
                    trace_size=trace_size,
                )
            import_task = shim_dma_single_bd_task(of_essentials, A, sizes=[1, 1, 1, world_to_view_size + get_camera_size])
            gaussian_task = shim_dma_single_bd_task(of_gaussian, B, sizes=[1, 1, 1, line_size * 8])
            out_task = shim_dma_single_bd_task(
                of_out_unit, C, sizes=[1, 1, 1, line_size * 4 * 2], issue_token=True
            )

            dma_start_task(import_task, gaussian_task, out_task)
            dma_await_task(out_task)
            dma_free_task(import_task, gaussian_task)
    



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


