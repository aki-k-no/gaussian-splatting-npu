
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../makefile-common

# compiler and flags
CC = g++-13
CXXFLAGS = -std=c++23 -Wall -O2 -Wno-deprecated-enum-enum-conversion -g -fsanitize=address -fsanitize=undefined

#include paths
XRT_INC_DIR = /opt/xilinx/xrt/include
XRT_LIB_DIR = /opt/xilinx/xrt/lib
INCLUDES = -I$(XRT_INC_DIR) -I/usr/include/eigen3 -I/usr/include/opencv4 -I.

MLIR_AIE_DIR := $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

TEST_UTILS_INC_DIR := $(MLIR_AIE_DIR)/runtime_lib/x86_64/test_lib/include
TEST_UTILS_LIB_DIR := $(MLIR_AIE_DIR)/runtime_lib/x86_64/test_lib/lib
TEST_UTILS_RUNTIME_LIB_DIR := $(MLIR_AIE_DIR)/runtime_lib

INCLUDES += -I$(TEST_UTILS_INC_DIR) -I$(TEST_UTILS_RUNTIME_LIB_DIR)

# linker flags
LD_LIBS = `pkg-config --cflags --libs opencv4`

LDFLAGS = -L$(TEST_UTILS_LIB_DIR) -L$(XRT_LIB_DIR)

LD_LIBS += -ltest_utils -lxrt_core -lxrt_coreutil

# source and object files
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

# target executable
TARGET = base

# rules
all: $(TARGET)


$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) $(LDFLAGS) $(OBJS) -o $@ $(LD_LIBS)

#$(TARGET): $(OBJS)
#	rm -rf _build
#	mkdir -p _build
#	cd _build && cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=$(TARGET) 
#	cd _build && cmake --build . --config Release
#ifeq "${powershell}" "powershell.exe"
#	cp _build/$(TARGET).exe $@
#else
#	cp _build/$(TARGET) $@ 
#endif

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

cp :
	cp -r NPU/build/ ./

run: $(TARGET)
	export LSAN_OPTIONS=verbosity=1:log_threads=1
	${powershell} ./$< -x build/final_precomp.xclbin -i build/insts_precomp.bin -k MLIR_AIE

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean cp