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

    trace_size = 0 # 2048


    @device(dev)
    def device_body():
        
        # Define tensor types
        line_size = 128 * 256
        tile_size = 128
        sub_tiles = line_size // tile_size
        world_to_view_size = 4 * 4 + 2
        get_camera_size = 4 * 4
        conv3D_num = 4


        w2v_ty = np.ndarray[(world_to_view_size,), np.dtype[xfr_dtype]]
        get_camera_ty = np.ndarray[(get_camera_size,), np.dtype[xfr_dtype]]
        
        send1_ty = np.ndarray[(8 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        gaussian_send_ty = np.ndarray[(8*line_size // sub_tiles,), np.dtype[xfr_dtype]]


        
        send2_ty = np.ndarray[(7 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        rot_and_scale_send_ty = np.ndarray[(7 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        gaussian_back1_ty = np.ndarray[(4*line_size // sub_tiles,), np.dtype[xfr_dtype]]
        to_cov2D_ty = np.ndarray[(8 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        gaussian_back2_ty = np.ndarray[(2 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        conv3D_return_ty = np.ndarray[(6 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        
        return1_ty = np.ndarray[(14 * line_size  // sub_tiles,), np.dtype[xfr_dtype]]

        return2_ty = np.ndarray[(6 * line_size  // sub_tiles,), np.dtype[xfr_dtype]]

        essentials_ty = np.ndarray[(world_to_view_size + get_camera_size,), np.dtype[xfr_dtype]]

        # this is required for runtime sequence
        send_ty = np.ndarray[(line_size * 15,), np.dtype[xfr_dtype]]
        return_ty = np.ndarray[(line_size * 18,), np.dtype[xfr_dtype]]

        # AIE Core Function declarations
        w2v_func = external_func(
            "f32_proj_to_view_space", inputs=[w2v_ty, gaussian_send_ty, gaussian_back1_ty]
        )

        camera_func = external_func(
            "f32_get_camera_pos", inputs=[get_camera_ty, gaussian_send_ty, gaussian_back2_ty]
        )

        conv3D_func = external_func(
            "f32_get_conv3D", inputs=[rot_and_scale_send_ty, conv3D_return_ty]
        )

        getJ_R_func = external_func(
            "f32_get_J_R", inputs=[w2v_ty, gaussian_back1_ty,to_cov2D_ty]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        MemTile0 = tile(0, 1)
        ShimTile1 = tile(1, 0)
        MemTile1 = tile(1, 1)
        ComputeTileV2w = tile(0, 2)
        ComputeTileCamera = tile(0, 3)
        ComputeTileJR = tile(0, 4)
        ComputeTileConv3Ds = [tile(1, 2), tile(1,3), tile(1, 4), tile(1,5)]

        # AIE-array data movement with object fifos
        of_essentials = object_fifo("essentials", ShimTile0, MemTile0, 2, essentials_ty)
        of_w2v = object_fifo("w2v", MemTile0, [ComputeTileV2w, ComputeTileJR], 2, w2v_ty)
        of_camera = object_fifo("camera", MemTile0, ComputeTileCamera, 2, get_camera_ty)
        object_fifo_link(of_essentials, [of_w2v, of_camera], [], [0, world_to_view_size])

        of_send1 = object_fifo("send1", ShimTile0, MemTile0, 2, send1_ty)
        of_gaussian = object_fifo("gaussian", MemTile0, [ComputeTileV2w, ComputeTileCamera], 2, gaussian_send_ty)
        
        fifo_send1_link_list =[of_gaussian]
        fifo_send1_offset_list = [0]
        object_fifo_link(of_send1, fifo_send1_link_list, [], fifo_send1_offset_list)


        of_send2 = object_fifo("send2", ShimTile1, MemTile1, 2, send2_ty)
        of_rot_and_scales = [object_fifo("rotscale" + str(i), MemTile1, ComputeTileConv3Ds[i], 2, rot_and_scale_send_ty) for i in range(conv3D_num)]
        
        fifo_send2_link_list = of_rot_and_scales
        fifo_send2_offset_list = [(7*line_size // sub_tiles) // conv3D_num * i for i in range(conv3D_num)]
        object_fifo_link(of_send2, fifo_send2_link_list, [], fifo_send2_offset_list)
        
        of_out1 = object_fifo("out1", ComputeTileV2w, [MemTile0, ComputeTileJR], 2, gaussian_back1_ty)
        of_out2 = object_fifo("out2", ComputeTileCamera, MemTile0, 2, gaussian_back2_ty)
        of_out_cov2D = object_fifo("out_cov2D", ComputeTileJR, MemTile0, 2, to_cov2D_ty)
        of_out1_unit = object_fifo("out1_unit",  MemTile0, ShimTile0, 2, return1_ty)

        fifo_back1_link_list = [of_out1, of_out2, of_out_cov2D]
        fifo_back1_offset_list = [0, 4*line_size // sub_tiles, 6*line_size // sub_tiles]
        object_fifo_link(fifo_back1_link_list, of_out1_unit, fifo_back1_offset_list, [])


        of_out2_unit = object_fifo("out2_unit",  MemTile1, ShimTile1, 2, return2_ty)
        of_out3s = [object_fifo("out3" + str(i), ComputeTileConv3Ds[i], MemTile1, 2, conv3D_return_ty) for i in range(conv3D_num)]
        fifo_back2_link_list = of_out3s
        fifo_back2_offset_list = [(6*line_size // sub_tiles) // conv3D_num * i for i in range(conv3D_num)]
        object_fifo_link(fifo_back2_link_list, of_out2_unit, fifo_back2_offset_list, [])


        # Compute tile for Projection Mat
        @core(ComputeTileV2w, "precomp.a")
        def core_body_v2w():
            for _ in range_(0xFFFFFFFF):
                elemIn1 = of_w2v.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(sub_tiles):
                    elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_gaussian.acquire(ObjectFifoPort.Consume, 1)
                    w2v_func(elemIn1, elemIn2, elemOut)
                    of_gaussian.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce,1)
                of_w2v.release(ObjectFifoPort.Consume, 1)
                

            
        # Compute tile for Projection Mat
        @core(ComputeTileCamera, "precomp.a")
        def core_body_camera():
            for _ in range_(0xFFFFFFFF):
                elemIn1 = of_camera.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(sub_tiles):
                    elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_gaussian.acquire(ObjectFifoPort.Consume, 1)
                    camera_func(elemIn1, elemIn2, elemOut)
                    of_gaussian.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce,1)
                of_camera.release(ObjectFifoPort.Consume, 1)
        
        @core(ComputeTileJR, "precomp.a")
        def core_body_JR():
            for _ in range_(0xFFFFFFFF):
                elemIn1 = of_w2v.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(sub_tiles):
                    elemOut = of_out_cov2D.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_out1.acquire(ObjectFifoPort.Consume, 1)
                    getJ_R_func(elemIn1, elemIn2, elemOut)
                    of_out1.release(ObjectFifoPort.Consume, 1)
                    of_out_cov2D.release(ObjectFifoPort.Produce,1)
                of_w2v.release(ObjectFifoPort.Consume, 1)
                

                
        # Compute conv3D tile
        for i in range(conv3D_num):
            @core(ComputeTileConv3Ds[i], "precomp.a")
            def core_body_conv3D():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(sub_tiles):
                        elemIn1 = of_rot_and_scales[i].acquire(ObjectFifoPort.Consume, 1)
                        elemOut = of_out3s[i].acquire(ObjectFifoPort.Produce, 1)
                        conv3D_func(elemIn1, elemOut)
                        of_out3s[i].release(ObjectFifoPort.Produce,1)
                        of_rot_and_scales[i].release(ObjectFifoPort.Consume, 1)
        
        tiles_to_trace = [ComputeTileConv3Ds[0], ComputeTileCamera, ShimTile0]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile0)


        # To/from AIE-array data movement
        @runtime_sequence(essentials_ty, send_ty, return_ty)
        def sequence(A, B, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile0,
                    trace_size=trace_size,
                )
            import_task = shim_dma_single_bd_task(of_essentials, A, sizes=[1, 1, 1, world_to_view_size + get_camera_size])
            gaussian_task = shim_dma_single_bd_task(of_send1, B, 
                sizes = [1, 1, sub_tiles, 8 * tile_size],
                strides = [0,0,tile_size * 15,1],
                offset = 0
            )
            rot_scale_task = shim_dma_single_bd_task(of_send2, B,
                sizes = [1, 1, sub_tiles, 7 * tile_size],
                strides = [0,0,tile_size * 15,1],
                offset = 8 * tile_size
            )
            out1_task = shim_dma_single_bd_task(
                of_out1_unit, C, issue_token=True,
                sizes = [1, 1, sub_tiles, 14 * tile_size],
                strides = [0,0,tile_size * 20,1],
                offset = 0
            )
            out2_task = shim_dma_single_bd_task(
                of_out2_unit, C, issue_token=True,
                sizes = [1, 1, sub_tiles, 6 * tile_size],
                strides = [0,0,tile_size * 20,1],
                offset = 14 * tile_size
            )

            dma_start_task(import_task, gaussian_task, rot_scale_task, out1_task, out2_task)
            dma_await_task(out1_task, out2_task)
            dma_free_task(import_task, gaussian_task, rot_scale_task)
            if trace_size > 0:
                trace_utils.gen_trace_done_aie2(ShimTile0)
    



p = argparse.ArgumentParser()
## Parse command line arguments

## Device name is required to select the AIE device: npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col  # Four columns of NPU1, the maximum available
elif opts.device == "npu2":
    dev = AIEDevice.npu2_4col  # Eight columns of NPU2, the maximum available
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

with mlir_mod_ctx() as ctx:
    precomp(dev)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

