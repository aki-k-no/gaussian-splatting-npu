
##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

INCLUDES += -I${srcdir}/../.. -I${srcdir}/../../../mlir-aie/aie_kernels

PROJ_NAME = precomp
KERNEL_SRC = precomp.cc
KERNEL_OBJ = $(KERNEL_SRC:.cc=.cc.o)

PY_SRC = precomp.py

devicename = npu2

.PHONY : all clean build
.SECONDARY:

all: build/final_precomp.xclbin build/insts_precomp.bin

build/precomp.o: precomp.cc
	mkdir -p ${@D}
	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} ${INCLUDES} -c $< -o $@


build/aie_%.mlir: %.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final_%.xclbin: build/aie_%.mlir build/%.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts_$(PROJ_NAME).bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_$(PROJ_NAME).bin $(<:%=../%)
endif

$(PROJ_NAME).exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=$(PROJ_NAME) 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/$(PROJ_NAME).exe $@
else
	cp _build/$(PROJ_NAME) $@ 
endif

run : $(PROJ_NAME).exe build/final_$(PROJ_NAME).xclbin build/insts_precomp.bin 
	${powershell} ./$< -x build/final_$(PROJ_NAME).xclbin -i build/insts_$(PROJ_NAME).bin -k MLIR_AIE

clean:
	rm -f $(PROJ_NAME).exe
	rm -rf build
	rm -rf _build

