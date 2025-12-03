
#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <cstdint>

using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_IN2 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;


// Functional correctness verifyer
int verify(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
    int errors = 0;
    verbosity = 2;

    for (int iter = 0; iter <= SIZE / 8; iter++){
        for (int i = 0; i < 8; i++) {
            for(int j=0;j<4;j++){
                int32_t ref = bufIn1[j * 8] * bufIn2[iter * 32 + i] + bufIn1[j * 8 + 1] * bufIn2[iter * 32 + i + 8]
                              + bufIn1[j * 8 + 2] * bufIn2[iter * 32 + i + 16] + bufIn1[j * 8 + 3] * bufIn2[iter * 32 + i + 24];
                int32_t test = bufOut[i + j * 8];
                if (test != ref) {
                    if (verbosity >= 1)
                        std::cout << "Error in output " << i + j * 8<< ":" << test << " != " << ref << std::endl;
                        errors++;
                } else {
                    if (verbosity >= 1)
                        std::cout << "Correct output " << test << " == " << ref << std::endl;
                }
            }
        }
    }
    return errors;
}

int main(int argc, const char *argv[]) {

    const int IN1_SIZE = 4;
    const int IN2_SIZE = 256;
    const int OUT_SIZE = IN2_SIZE;

    // Program arguments parsing
    cxxopts::Options options("section-3");
    test_utils::add_default_options(options);

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();

    // Load instruction sequence
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    

    // Start the XRT context and load the kernel
    xrt::device device;
    xrt::kernel kernel;

    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inA = xrt::bo(device, IN1_SIZE * 4 * 2 * sizeof(DATATYPE_IN1),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inB = xrt::bo(device, IN2_SIZE * 4 * sizeof(DATATYPE_IN2),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_outC = xrt::bo(device, OUT_SIZE * 4 * sizeof(DATATYPE_OUT),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize buffer bo_inA
    DATATYPE_IN1 *bufInA = bo_inA.map<DATATYPE_IN1 *>();
    for (int i = 0; i < IN1_SIZE ; i++){
        for (int j=0; j < 4; j++){
            bufInA[8 * i + j] = i;
            bufInA[8 * i + 4 + j] = 0;
        }

    }
    // Initialize buffer bo_inFactor
    DATATYPE_IN2 *bufInB = bo_inB.map<DATATYPE_IN2 *>();
    for (int i = 0; i < IN2_SIZE * 4; i++)
        bufInB[i] = i + 1;

    // Zero out buffer bo_outC
    DATATYPE_OUT *bufOut = bo_outC.map<DATATYPE_OUT *>();
    for (int i = 0; i < OUT_SIZE * 4; i++)
        bufOut[i] = 1;

    
    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
    run.wait();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    verify(bufInA, bufInB, bufOut, IN2_SIZE, verbosity);
}

