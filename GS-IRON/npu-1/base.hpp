#ifndef BASE_HPP
#define BASE_HPP

#define __USE_NPU


#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


using DATATYPE_IN1 = float;
using DATATYPE_IN2 = float;
using DATATYPE_OUT = float;
extern int verbosity;

extern std::vector<uint32_t> instr_v;
extern xrt::device device;
extern xrt::kernel kernel;
extern xrt::bo bo_instr;
extern xrt::bo bo_inA;
extern xrt::bo bo_inB;
extern xrt::bo bo_outC;
extern void *bufInstr;
extern DATATYPE_IN1 *bufInA;
extern DATATYPE_IN2 *bufInB;
extern DATATYPE_OUT *bufOut;
#endif