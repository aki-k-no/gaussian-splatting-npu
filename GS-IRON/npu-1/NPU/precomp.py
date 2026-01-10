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

    trace_size = 2048


    @device(dev)
    def device_body():
        
        # Define tensor types
        line_size = 128 * 256
        tile_size = 128
        sub_tiles = line_size // tile_size
        world_to_view_size = 4 * 4 + 2
        get_camera_size = 4 * 4 + 4
        campos_coeff_size = 8 + 4 * 4
        conv3D_num = 4

        essentials_ty = np.ndarray[(world_to_view_size + get_camera_size + campos_coeff_size,), np.dtype[xfr_dtype]]

        w2v_ty = np.ndarray[(world_to_view_size,), np.dtype[xfr_dtype]]
        get_camera_ty = np.ndarray[(get_camera_size,), np.dtype[xfr_dtype]]
        campos_coeff_ty = np.ndarray[(campos_coeff_size,), np.dtype[xfr_dtype]]
        
        send1_ty = np.ndarray[(8 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        gaussian_send_ty = np.ndarray[(8*line_size // sub_tiles,), np.dtype[xfr_dtype]]
        opacity_send_ty = np.ndarray[(1 * line_size // sub_tiles,), np.dtype[xfr_dtype]]



        
        send2_ty = np.ndarray[(7 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        rot_and_scale_send_ty = np.ndarray[(7 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        gaussian_back1_ty = np.ndarray[(4*line_size // sub_tiles,), np.dtype[xfr_dtype]]
        to_cov2D_ty = np.ndarray[(8 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        gaussian_back2_ty = np.ndarray[(4 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        conv3D_return_ty = np.ndarray[(16 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        conv3D_return_ty_accum = np.ndarray[(16 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        cov2D_return_ty = np.ndarray[(4 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        cov2D_return_accum_ty = np.ndarray[(4 * line_size // sub_tiles,), np.dtype[xfr_dtype]]

        
        send3_ty = np.ndarray[(56 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        color_data_ty = np.ndarray[(56 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        pos_data_ty_accum = np.ndarray[(4 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        pos_data_ty = np.ndarray[(4 * line_size // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        dir_ty = np.ndarray[(9 * line_size // sub_tiles,), np.dtype[xfr_dtype]]
        return1_ty = np.ndarray[(8 * line_size  // sub_tiles,), np.dtype[xfr_dtype]]
        index_back_ty = np.ndarray[(2 * line_size // sub_tiles,), np.dtype[xfr_dtype]]

        return2_ty = np.ndarray[(4 * line_size  // sub_tiles,), np.dtype[xfr_dtype]]

        return3_ty = np.ndarray[(4 * line_size  // sub_tiles,), np.dtype[xfr_dtype]]
        color_return_ty = np.ndarray[(4 * line_size  // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]
        color_inter_ty = np.ndarray[(16 * line_size  // sub_tiles // conv3D_num,), np.dtype[xfr_dtype]]


        # this is required for runtime sequence
        send_ty = np.ndarray[(line_size * 72,), np.dtype[xfr_dtype]]
        return_ty = np.ndarray[(line_size * 14,), np.dtype[xfr_dtype]]

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
            "f32_get_J_R", inputs=[w2v_ty, gaussian_back1_ty, to_cov2D_ty]
        )

        index_func = external_func(
            "f32_get_special_index", inputs=[gaussian_back1_ty, opacity_send_ty, index_back_ty]
        )

        conv2D_funcs = [ external_func(
            "f32_get_conv2D_" + str(i), inputs=[to_cov2D_ty, conv3D_return_ty, cov2D_return_ty]
        ) for i in range(conv3D_num)]

        color_func_pre = external_func(
            "f32_get_color_pre", inputs=[campos_coeff_ty, color_data_ty, color_inter_ty]
        )
        
        color_func_post = external_func(
            "f32_get_color_post", inputs=[color_inter_ty, color_data_ty, color_return_ty]
        )

        dir_func = external_func(
            "f32_get_dir", inputs=[campos_coeff_ty, pos_data_ty, dir_ty]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        MemTile0 = tile(0, 1)
        ShimTile1 = tile(1, 0)
        MemTile1 = tile(1, 1)
        ShimTile2 = tile(2, 0)
        MemTile2 = tile(2, 1)
        ShimTile3 = tile(3, 0)
        MemTile3 = tile(3, 1)
        ShimTile4 = tile(4, 0)
        MemTile4 = tile(4, 1)
        MemTile5 = tile(5, 1)
        ComputeTileV2w = tile(0, 2)
        ComputeTileCamera = tile(0, 3)
        ComputeTileJR = tile(0, 4)
        ComputeTileIndex = tile(0,5)
        ComputeTileConv3Ds = [tile(1, 2), tile(1,3), tile(1, 4), tile(1,5)]
        ComputeTileConv2Ds = [tile(2, 2), tile(2, 3), tile(2, 4), tile(2, 5)]
        ComputeTTileDirs = [tile(3, 2), tile(3, 3), tile(3, 4), tile(3, 5)]
        ComputeTileColorsPre = [tile(4, 2), tile(4, 3), tile(4, 4), tile(4, 5)]
        ComputeTileColorsPost = [tile(5, 2), tile(5, 3), tile(5, 4), tile(5, 5)]

        # AIE-array data movement with object fifos
        of_essentials = object_fifo("essentials", ShimTile0, MemTile0, 2, essentials_ty)
        of_w2v = object_fifo("w2v", MemTile0, [ComputeTileV2w, ComputeTileJR], 2, w2v_ty)
        of_camera = object_fifo("camera", MemTile0, ComputeTileCamera, 2, get_camera_ty)
        of_coeff = object_fifo("coeff", MemTile0, ComputeTileColorsPre, 2, campos_coeff_ty)
        object_fifo_link(of_essentials, [of_w2v, of_camera, of_coeff], [], [0, world_to_view_size, world_to_view_size + get_camera_size])

        of_send1 = object_fifo("send1", ShimTile0, MemTile0, 2, send1_ty)
        of_gaussian = object_fifo("gaussian", MemTile0, [ComputeTileV2w, ComputeTileCamera], 2, gaussian_send_ty)
        of_opacity_send = object_fifo("opacity_send", ShimTile2, MemTile2, 2, opacity_send_ty)
        of_opacity = object_fifo("opacity", MemTile2, ComputeTileIndex, 2, opacity_send_ty)
        object_fifo_link(of_opacity_send, of_opacity, [], [])

        fifo_send1_link_list = [of_gaussian]
        fifo_send1_offset_list = []
        object_fifo_link(of_send1, fifo_send1_link_list, [], fifo_send1_offset_list)


        of_send2 = object_fifo("send2", ShimTile1, MemTile1, 2, send2_ty)
        of_rot_and_scales = [object_fifo("rotscale" + str(i), MemTile1, ComputeTileConv3Ds[i], 2, rot_and_scale_send_ty) for i in range(conv3D_num)]
        
        fifo_send2_link_list = of_rot_and_scales
        fifo_send2_offset_list = [(7*line_size // sub_tiles) // conv3D_num * i for i in range(conv3D_num)]
        object_fifo_link(of_send2, fifo_send2_link_list, [], fifo_send2_offset_list)
        
        of_out1 = object_fifo("out1", ComputeTileV2w, [ComputeTileJR, ComputeTileIndex], 2, gaussian_back1_ty)
        of_out2 = object_fifo("out2", ComputeTileCamera, MemTile0, 2, gaussian_back2_ty)
        of_out_index = object_fifo("out_index", ComputeTileIndex, MemTile2, 2, index_back_ty)
        of_out_index_unit = object_fifo("out_index_unit", MemTile2, ShimTile2, 2, index_back_ty)
        object_fifo_link(of_out_index, of_out_index_unit, [], [])
        of_out2_unit = object_fifo("out2_unit",  MemTile0, ShimTile0, 2, gaussian_back2_ty)
        object_fifo_link(of_out2, of_out2_unit, [], [])

        of_JR_cov2Ds = object_fifo("to_cov2D", ComputeTileJR, ComputeTileConv2Ds, 2, to_cov2D_ty)
    
        of_to_cov2Ds = [object_fifo("cov3Dto2D_" + str(i), ComputeTileConv3Ds[i], ComputeTileConv2Ds[i], 2, conv3D_return_ty) for i in range(conv3D_num)]
        
        of_out3_unit = object_fifo("out3_unit",  MemTile1, ShimTile1, 2, cov2D_return_accum_ty)
        of_cov2D_returns = [object_fifo("cov2D_return_" + str(i), ComputeTileConv2Ds[i], MemTile1, 2, cov2D_return_ty) for i in range(conv3D_num)]
        object_fifo_link(of_cov2D_returns, of_out3_unit, [4 * line_size // sub_tiles // conv3D_num * i for i in range(conv3D_num)], [])

        # of_coeff_dir = object_fifo("coeff_dir_unit", ShimTile3, MemTile3, 2, campos_coeff_ty)
        # of_dir_send = [object_fifo("coeff_dir_" + str(i), MemTile3, ComputeTTileDirs[i], 2, campos_coeff_ty) for i in range(conv3D_num)]
        # object_fifo_link(of_coeff_dir, of_dir_send, [], [4 * line_size // sub_tiles * i for i in range(conv3D_num)])

        # of_dir_returns = [object_fifo("dir_return_" + str(i), ComputeTTileDirs[i], MemTile3, 2, dir_ty) for i in range(conv3D_num)]


        of_color_inters = [object_fifo("color_inter_" + str(i), ComputeTileColorsPre[i], ComputeTileColorsPost[i], 2, color_inter_ty) for i in range(conv3D_num)]

        of_color_return_unit = object_fifo("color_return_unit",  MemTile5, ShimTile4, 2, return3_ty)
        of_color_returns = [object_fifo("color_return_" + str(i), ComputeTileColorsPost[i], MemTile5, 2, color_return_ty) for i in range(conv3D_num)]
        object_fifo_link(of_color_returns, of_color_return_unit, [4 * line_size // sub_tiles // conv3D_num * i for i in range(conv3D_num)], [])
        
        of_color_send_unit = object_fifo("color_send_unit", ShimTile4, MemTile4, 2, send3_ty)
        of_color_sends = [object_fifo("color_send_" + str(i), MemTile4, [ComputeTileColorsPre[i], ComputeTileColorsPost[i]], 2, color_data_ty) for i in range(conv3D_num)]
        object_fifo_link(of_color_send_unit, of_color_sends, [], [56 * line_size // sub_tiles // conv3D_num * i for i in range(conv3D_num)])


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
                    elemOut = of_JR_cov2Ds.acquire(ObjectFifoPort.Produce, 1)
                    elemIn2 = of_out1.acquire(ObjectFifoPort.Consume, 1)
                    getJ_R_func(elemIn1, elemIn2, elemOut)
                    of_out1.release(ObjectFifoPort.Consume, 1)
                    of_JR_cov2Ds.release(ObjectFifoPort.Produce,1)
                of_w2v.release(ObjectFifoPort.Consume, 1)
        
        @core(ComputeTileIndex, "precomp.a")
        def core_body_index():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(sub_tiles):
                    elemIn1 = of_out1.acquire(ObjectFifoPort.Consume, 1)
                    elemIn2 = of_opacity.acquire(ObjectFifoPort.Consume, 1)
                    elemOut = of_out_index.acquire(ObjectFifoPort.Produce, 1)
                    index_func(elemIn1, elemIn2, elemOut)
                    of_out1.release(ObjectFifoPort.Consume, 1)
                    of_opacity.release(ObjectFifoPort.Consume,1)
                    of_out_index.release(ObjectFifoPort.Produce, 1)
                

                
        # Compute conv3D tile
        for i in range(conv3D_num):
            @core(ComputeTileConv3Ds[i], "precomp.a")
            def core_body_conv3D():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(sub_tiles):
                        elemIn1 = of_rot_and_scales[i].acquire(ObjectFifoPort.Consume, 1)
                        elemOut = of_to_cov2Ds[i].acquire(ObjectFifoPort.Produce, 1)
                        conv3D_func(elemIn1, elemOut)
                        of_to_cov2Ds[i].release(ObjectFifoPort.Produce,1)
                        of_rot_and_scales[i].release(ObjectFifoPort.Consume, 1)

        # Compute conv2D tile
        for i in range(conv3D_num):
            @core(ComputeTileConv2Ds[i], "precomp.a")
            def core_body_cov2D():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(sub_tiles):
                        elemIn1 = of_JR_cov2Ds.acquire(ObjectFifoPort.Consume, 1)
                        elemOut = of_cov2D_returns[i].acquire(ObjectFifoPort.Produce, 1)
                        elemIn2 = of_to_cov2Ds[i].acquire(ObjectFifoPort.Consume, 1)
                        conv2D_funcs[i](elemIn1, elemIn2, elemOut)
                        of_to_cov2Ds[i].release(ObjectFifoPort.Consume, 1)
                        of_cov2D_returns[i].release(ObjectFifoPort.Produce,1)
                        of_JR_cov2Ds.release(ObjectFifoPort.Consume, 1)
        
        #color tile 1
        for i in range(conv3D_num):
            @core(ComputeTileColorsPre[i], "precomp.a")
            def core_body_color():
                for _ in range_(0xFFFFFFFF):
                    elemIn1 = of_coeff.acquire(ObjectFifoPort.Consume, 1)
                    for _ in range_(sub_tiles):
                        elemIn2 = of_color_sends[i].acquire(ObjectFifoPort.Consume, 1)
                        elemOut = of_color_inters[i].acquire(ObjectFifoPort.Produce, 1)
                        color_func_pre(elemIn1, elemIn2, elemOut)
                        of_color_inters[i].release(ObjectFifoPort.Produce,1)
                        of_color_sends[i].release(ObjectFifoPort.Consume, 1)
                    of_coeff.release(ObjectFifoPort.Consume, 1)
        
        #color tile 2
        for i in range(conv3D_num):
            @core(ComputeTileColorsPost[i], "precomp.a")
            def core_body_color():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(sub_tiles):
                        elemOut = of_color_returns[i].acquire(ObjectFifoPort.Produce, 1)
                        elemIn1 = of_color_inters[i].acquire(ObjectFifoPort.Consume, 1)
                        elemIn2 = of_color_sends[i].acquire(ObjectFifoPort.Consume, 1)
                        color_func_post(elemIn1, elemIn2, elemOut)
                        of_color_sends[i].release(ObjectFifoPort.Consume, 1)
                        elemIn2 = of_color_inters[i].release(ObjectFifoPort.Consume, 1)
                        of_color_returns[i].release(ObjectFifoPort.Produce, 1)




        tiles_to_trace = [ComputeTileConv3Ds[2],ComputeTileConv2Ds[2], ShimTile3]


        # To/from AIE-array data movement
        @runtime_sequence(essentials_ty, send_ty, return_ty)
        def sequence(A, B, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile3,
                    trace_size=trace_size,
                )
            import_task = shim_dma_single_bd_task(of_essentials, A, sizes=[1, 1, 1, world_to_view_size + get_camera_size + campos_coeff_size])
            gaussian_task = shim_dma_single_bd_task(of_send1, B, 
                sizes = [1, 1, sub_tiles, 8 * tile_size],
                strides = [0,0,tile_size * 72,1],
                offset = 0
            )
            opacity_task = shim_dma_single_bd_task(of_opacity_send, B,
                sizes = [1, 1, sub_tiles, 1 * tile_size],
                strides = [0,0,tile_size * 72,1],
                offset = 8 * tile_size
            )
            rot_scale_task = shim_dma_single_bd_task(of_send2, B,
                sizes = [1, 1, sub_tiles, 7 * tile_size],
                strides = [0,0,tile_size * 72,1],
                offset = 9 * tile_size
            )
            color_task = shim_dma_single_bd_task(of_color_send_unit, B,
                sizes = [1, sub_tiles, 7, 8 * tile_size],
                strides = [0, tile_size * 72, 8 * tile_size,1],
                offset = 16 * tile_size
            )
            # out1_task = shim_dma_single_bd_task(
            #     of_out1_unit, C, issue_token=True,
            #     sizes = [1, 1, 1, 4 * line_size],
            #     strides = [0, 0, 0, 1],
            #     offset = 0
            # )
            out2_task = shim_dma_single_bd_task(
                of_out2_unit, C, issue_token=True,
                sizes = [1, 1, 1, 4 * line_size],
                strides = [0, 0, 0, 1],
                offset = 0 * line_size
            )
            out_index_task = shim_dma_single_bd_task(
                of_out_index_unit, C, issue_token=True,
                sizes = [1, 1, 1, 2 * line_size],
                strides = [0, 0, 0, 1],
                offset = 4 * line_size
            )
            out3_task = shim_dma_single_bd_task(
                of_out3_unit, C, issue_token=True,
                sizes = [1, 1, 1, 4 * line_size],
                strides = [0, 0, 0, 1],
                offset = 6 * line_size
            )
            out4_task = shim_dma_single_bd_task(
                of_color_return_unit, C, issue_token=True,
                sizes = [1, 1, 1, 4 * line_size],
                strides = [0, 0, 0, 1],
                offset = 10 * line_size
            )

            dma_start_task(import_task, gaussian_task, opacity_task, rot_scale_task, color_task, out2_task, out3_task, out4_task, out_index_task)
            dma_await_task(out2_task, out3_task, out4_task, out_index_task)
            dma_free_task(import_task, gaussian_task, opacity_task, rot_scale_task, color_task)
            if trace_size > 0:
                trace_utils.gen_trace_done_aie2(ShimTile3)
    
    
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile3)
    



p = argparse.ArgumentParser()
## Parse command line arguments

## Device name is required to select the AIE device: npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col  # Four columns of NPU1, the maximum available
elif opts.device == "npu2":
    dev = AIEDevice.npu2  # Eight columns of NPU2, the maximum available
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))


with mlir_mod_ctx() as ctx:
    precomp(dev)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

