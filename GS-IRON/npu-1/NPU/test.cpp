
#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstdint>
#include <random>

using DATATYPE_IN1 = std::float32_t;
using DATATYPE_IN2 = std::float32_t;
using DATATYPE_OUT = std::float32_t;

// helper function to generate random bf16
void generate_random_bfloat16(std::bfloat16_t* buf, size_t n, float min_val, float max_val) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        float r = distr(eng);
        memcpy(buf + i, reinterpret_cast<char *>(&r) + sizeof(float) - sizeof(std::bfloat16_t), sizeof(std::bfloat16_t));
        // std::cout << "generated" << buf[i] << " " << sizeof(std::bfloat16_t) << "\n";
    }
}


// Functional correctness verifyer
int verify(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
    int errors = 0;
    verbosity = 2;

    for (int iter = 0; iter < SIZE / 4; iter++){
        for(int j=0;j<4;j++){
            for (int i = 0; i < 4; i++) {
                DATATYPE_OUT ref = bufIn1[j * 4] * bufIn2[iter * 16 + i] + bufIn1[j * 4 + 1] * bufIn2[iter * 16 + i + 4]
                              + bufIn1[j * 4 + 2] * bufIn2[iter * 16 + i + 8] + bufIn1[j * 4 + 3] * bufIn2[iter * 16 + i + 12];
                DATATYPE_OUT test = bufOut[iter * 16 + i + j * 4];
                if (test < ref - 0.1 || test > ref + 0.1) {
                    if (verbosity >= 1){
                        
                        std::cout << "Error in output " << iter * 16 + i + j * 4 << " : " << test << " != " << ref << std::endl;
                    }
                    errors++;
                    
                } else {
                    if (verbosity >= 1)
                        std::cout << "Correct in output " << iter * 16 + i + j * 4 << " : " << test << " == " << ref << std::endl;
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
    const int TRACE_SIZE = 8192 * 4;

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
    auto bo_inA = xrt::bo(device, IN1_SIZE * 4 * sizeof(DATATYPE_IN1),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inB = xrt::bo(device, IN2_SIZE * 4 * sizeof(DATATYPE_IN2),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_outC = xrt::bo(device, OUT_SIZE * 4 * sizeof(DATATYPE_OUT),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_trace = xrt::bo(device, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(7));

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize buffer bo_inA
    DATATYPE_IN1 *bufInA = bo_inA.map<DATATYPE_IN1 *>();
    for (int i = 0; i < IN1_SIZE ; i++){
        for (int j=0; j < 4; j++){
            bufInA[4 * i + j] = 4 * i + j + 0.5;
        }
    }
    
//    generate_random_bfloat16(bufInA, IN1_SIZE * 4, 0, 10);

    // Initialize buffer bo_inFactor
    DATATYPE_IN2 *bufInB = bo_inB.map<DATATYPE_IN2 *>();
    for (int i = 0; i < IN2_SIZE * 4; i++)
        bufInB[i] = i + 1 + 0.2;

//    generate_random_bfloat16(bufInB, IN2_SIZE * 8, 0, 10);

    // Zero out buffer bo_outC
    DATATYPE_OUT *bufOut = bo_outC.map<DATATYPE_OUT *>();
    for (int i = 0; i < OUT_SIZE * 4; i++)
        bufOut[i] = 14;

    
    char *bufTrace = bo_trace.map<char *>();
    memset(bufTrace, 0, TRACE_SIZE);

    
    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC, 0, bo_trace);
    run.wait();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    test_utils::write_out_trace((char *)bufTrace, TRACE_SIZE, "trace.txt");

    int errors = verify(bufInA, bufInB, bufOut, IN2_SIZE, verbosity);
    if(errors == 0){
        std::cout << "PASS!\n";
    }else{
        std::cout << "FAIL with " << errors << "errors\n";
    }
}

