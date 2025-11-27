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
from aie.iron.device import Tile, NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern


def precomp(dev):
    xfr_dtype = bfloat16

    # Define tensor types
    line_size = 1024
    proj_ty = np.ndarray[(4*4,), np.dtype[xfr_dtype]]
    gaussian_ty = np.ndarray[(4*line_size,), np.dtype[xfr_dtype]]


    # Dataflow with ObjectFifos
    of_proj =  ObjectFifo(proj_ty, name=f"inproj_0_1")
    of_gaussian = ObjectFifo(gaussian_ty, name=f"ingaussian_0_1")
    
    of_out = ObjectFifo(gaussian_ty, name=f"out0_1")
    
    # External, binary kernel definition
    proj_func = Kernel(
        "f32_proj_to_view_space",
        "precomp.cc.o",
        [proj_ty, gaussian_ty, gaussian_ty],
    )

    # Task for the core to perform
    def core_proj_fn(of_proj, of_gaussian, of_out, proj_func):
        elemOut = of_out.acquire(1)
        elemIn1 = of_proj.acquire(1)
        elemIn2 = of_gaussian.acquire(1)
        proj_func(elemIn1, elemIn2, elemOut)
        of_proj.release(1)
        of_gaussian.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    proj_worker = Worker(
            core_proj_fn,
            fn_args = [
                of_proj.cons(),
                of_gaussian.cons(),
                of_out.prod(),
                proj_func,
            ],
        )
    

     

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(proj_ty, gaussian_ty, gaussian_ty) as (a_in, b_in, c_out):
        rt.start(proj_worker)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        rt.fill(of_proj.prod(), a_in, task_group=tg)
        rt.fill(of_gaussian.prod(), b_in, task_group=tg)
        # Drain the output objectFIFOs with data
        rt.drain(of_out.cons(), c_out, wait=True, task_group=tg,)
        rt.finish_task_group(tg)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
## Parse command line arguments

## Device name is required to select the AIE device: npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()  # Four columns of NPU1, the maximum available
elif opts.device == "npu2":
    dev = NPU2()  # Eight columns of NPU2, the maximum available
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

## Call the my_relu function with the parsed arguments
## and print the MLIR as a result
print(precomp(dev))
