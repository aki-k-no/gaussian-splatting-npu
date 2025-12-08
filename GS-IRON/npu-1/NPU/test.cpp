
#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../const.hpp"

#include <cstdint>
#include <random>

#define __ENABLE_TRACE

using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_IN2 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;

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

void check_each(int verbosity, DATATYPE_OUT test, DATATYPE_OUT ref, int index, float tol) {
    if (test < ref - tol || test > ref + tol) {
        if (verbosity >= 1){
            std::cout << "Error in output " << index << " : " << test << " != " << ref << std::endl;
        }
    } else {
        if (verbosity >= 2)
            std::cout << "Correct in output " << index << " : " << test << " == " << ref << std::endl;
    }
}

void compute_3x3_mat(DATATYPE_OUT *tests, DATATYPE_OUT *output){
    //comput 3x3 3x3 matrix multiplication
    for(size_t i=0;i<3;i++){
        for(size_t j=0;j<3;j++){
            DATATYPE_OUT acc = 0;
            for(size_t k=0;k<3;k++){
                acc += tests[i * 3 + k] * tests[j * 3 + k];
            }
            output[i * 3 + j] = (DATATYPE_OUT) acc;
        }
    }


}


// Functional correctness verifyer
int verify(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int TILE_SIZE, int TILE_COUNT, int verbosity) {
    int errors = 0;
    verbosity = 1;

    //first
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        for (int iter = 0; iter < TILE_SIZE / 4; iter++){
            for(int j=0;j<4;j++){
                for (int i = 0; i < 4; i++) {
                    int offset = tile * TILE_SIZE * 15;
                    DATATYPE_OUT ref = bufIn1[4 * j] * bufIn2[offset + iter * 32 + i] + bufIn1[4 * j + 1] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[4 * j + 2] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[4 * j + 3] * bufIn2[offset + iter * 32 + i + 12];
                    DATATYPE_OUT test = bufOut[tile * TILE_SIZE * 12 + iter * 16 + i + j * 4];
                    if (test < ref - 0.25 || test > ref + 0.25) {
                        if (verbosity >= 1){

                            std::cout << "Error in output " << tile * TILE_SIZE * 12 + iter * 16 + i + j * 4 << " : " << test << " != " << ref << std::endl;
                        }
                        errors++;

                    } else {
                        if (verbosity >= 2)
                            std::cout << "Correct in output " << tile * TILE_SIZE * 12 + iter * 16 + i + j * 4 << " : " << test << " == " << ref << std::endl;
                    }
                    
                }
        
            }
        }
    }
    
    //second
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        for (int iter = 0; iter < TILE_SIZE / 4; iter++){
            for(int j=0;j<2;j++){
                for (int i = 0; i < 4; i++) {
                    int offset = tile * TILE_SIZE * 15;
                    DATATYPE_OUT ref = (bufIn1[4 * j + 16] * bufIn2[offset + iter * 32 + i] + bufIn1[4 * j + 1 + 16] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[4 * j + 2 + 16] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[4 * j + 3 + 16] * bufIn2[offset + iter * 32 + i + 12])
                                  / (bufIn1[12 + 16] * bufIn2[offset + iter * 32 + i] + bufIn1[13 + 16] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[14 + 16] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[15 + 16] * bufIn2[offset + iter * 32 + i + 12]);
                                  ref = (ref + 1) * 0.5;
                    DATATYPE_OUT test = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 4 + iter * 8 + i + j * 4];
                    if(test < ref - 0.05 || test > ref + 0.05) {
                        if (verbosity >= 1){
                        
                            std::cout << "Error in output (w) " << tile * TILE_SIZE * 12 + iter * 8 + i + j * 4  << " : " << test << " != " << ref << std::endl;
                        }
                        errors++;
                    
                    } else {
                        if (verbosity >= 2){
                            std::cout << "Correct in output (w) " << tile * TILE_SIZE * 12 + iter * 8 + i + j * 4  << " : " << test << " == " << ref << std::endl;
                        }
                    }
                }
            }
        }
    }

    //third
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        for (int iter = 0; iter < TILE_SIZE / 16; iter++){
                for (int i = 0; i < 16; i++) {
                    
                    int offset = tile * TILE_SIZE * 15 + TILE_SIZE * 8;
                    DATATYPE_IN2 w = bufIn2[offset + iter * 64 + i];
                    DATATYPE_IN2 x = bufIn2[offset + iter * 64 + i + 16];
                    DATATYPE_IN2 y = bufIn2[offset + iter * 64 + i + 32];
                    DATATYPE_IN2 z = bufIn2[offset + iter * 64 + i + 48];

                    DATATYPE_IN2 scale1 = bufIn2[offset + TILE_SIZE * 4 + iter * 48 + i];
                    DATATYPE_IN2 scale2 = bufIn2[offset + TILE_SIZE * 4 + iter * 48 + i + 16];
                    DATATYPE_IN2 scale3 = bufIn2[offset + TILE_SIZE * 4 + iter * 48 + i + 32];
                    DATATYPE_IN2 norm = std::sqrt(w * w + x * x + y * y + z * z);
                    w /= norm;
                    x /= norm;
                    y /= norm;
                    z /= norm;

                    
                    DATATYPE_OUT test1 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6];
                    DATATYPE_OUT test2 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 1];
                    DATATYPE_OUT test3 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 2];
                    DATATYPE_OUT test4 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 3];
                    DATATYPE_OUT test5 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 4];
                    DATATYPE_OUT test6 = bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 5];

                    


                    DATATYPE_OUT elem1 = 1 - 2 * y * y - 2 * z * z;
                    DATATYPE_OUT elem2 = 2 * x * y - 2 * w * z;
                    DATATYPE_OUT elem3 = 2 * x * z + 2 * w * y;

                    DATATYPE_OUT elem5 = 2 * x * y + 2 * w * z;
                    DATATYPE_OUT elem6 = 1 - 2 * x * x - 2 * z * z;
                    DATATYPE_OUT elem7 = 2 * y * z - 2 * w * x;

                    DATATYPE_OUT elem9 = 2 * x * z - 2 * w * y;
                    DATATYPE_OUT elem10 = 2 * y * z + 2 * w * x;
                    DATATYPE_OUT elem11 = 1 - 2 * x * x - 2 * y * y;

                    DATATYPE_OUT calc_mat[9] = {
                        scale1 * elem1, scale2 * elem2, scale3 * elem3,
                        scale1 * elem5, scale2 * elem6, scale3 * elem7,
                        scale1 * elem9, scale2 * elem10, scale3 * elem11
                    };
                    DATATYPE_OUT ref_mat[9];

                    compute_3x3_mat(calc_mat, ref_mat);
                    
                    check_each(verbosity, test1, ref_mat[0], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 , 0.5);
                    check_each(verbosity, test2, ref_mat[1], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 1, 0.5);
                    check_each(verbosity, test3, ref_mat[2], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 2, 0.5);
                    check_each(verbosity, test4, ref_mat[4], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 3, 0.5);
                    check_each(verbosity, test5, ref_mat[5], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 4, 0.5);
                    check_each(verbosity, test6, ref_mat[8], tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 6 + 5, 0.5);
                    
                    // //additional debug log
                    // std::cout << scale1 * elem1 << " " << scale2 * elem2 << " " << scale3 * elem3 << "\n";
                    // std::cout << scale1 * elem5 << " " << scale2 * elem6 << " " << scale3 * elem7 << "\n";
                    // std::cout << scale1 * elem9 << " " << scale2 * elem10 << " " << scale3 * elem11 << "\n";
                    // std::cout << scale1 << " " << scale2 << " " << scale3 << "\n";
                    // std::cout << "----\n";

                    // for(int m = 0;m<32;m++){
                    //     std::cout << bufOut[tile * TILE_SIZE * 12 + TILE_SIZE * 6 + iter * 96 + i * 8 + m] << " ";
                    // }
                    // std::cout << "\n";
                }
        }
    }
    return errors;
}

void fill_bufB(DATATYPE_IN2 *bufInB){
    // fill randomly first
    generate_random_bfloat16(bufInB, CHUNK_SIZE * 15, 0, 3);

    // fill some of them with zero (the gaussian padding)
    for (int i=0; i < TILE_COUNT; i++){
        int offset = TILE_SIZE * i * 15;
        for(int j=0;j<TILE_SIZE / 4; j++){
            for(int k=0;k<16;k++){
                bufInB[offset + j * 32 + k + 16] = 0;
            }
        }
    }
}

int main(int argc, const char *argv[]) {


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
    auto bo_inA = xrt::bo(device, IN1_SIZE * sizeof(DATATYPE_IN1),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inB = xrt::bo(device, CHUNK_SIZE * 15 * sizeof(DATATYPE_IN2),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_outC = xrt::bo(device, CHUNK_SIZE * 12 * sizeof(DATATYPE_OUT),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    #ifdef __ENABLE_TRACE
    auto bo_trace = xrt::bo(device, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));
    std::cout << "Trace enabled\n";
    #endif
    

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize buffer bo_inA
    DATATYPE_IN1 *bufInA = bo_inA.map<DATATYPE_IN1 *>();
    for (int i = 0; i < IN1_SIZE ; i++){
        bufInA[i] = i + 1;
    }
    generate_random_bfloat16(bufInA, IN1_SIZE , 0, 3);
    

    // Initialize buffer bo_inFactor
    DATATYPE_IN2 *bufInB = bo_inB.map<DATATYPE_IN2 *>();
    fill_bufB(bufInB);

    

    // Zero out buffer bo_outC
    DATATYPE_OUT *bufOut = bo_outC.map<DATATYPE_OUT *>();
    for (int i = 0; i < CHUNK_SIZE * 12; i++)
        bufOut[i] = 14;

    #ifdef __ENABLE_TRACE
    char *bufTrace = bo_trace.map<char *>();
    memset(bufTrace, 0, TRACE_SIZE);
    #endif

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    #ifdef __ENABLE_TRACE
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    #endif

    for(int i = 0; i< 1; i++){
    

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";
    unsigned int opcode = 3;
    auto run =
    #ifndef __ENABLE_TRACE
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
        #else
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC, 0, bo_trace);
    #endif
    run.wait();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    #ifdef __ENABLE_TRACE
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    test_utils::write_out_trace((char *)bufTrace, TRACE_SIZE, "trace.txt");
    #endif

    int errors = verify(bufInA, bufInB, bufOut, TILE_SIZE, TILE_COUNT, verbosity);
    if(errors == 0){
         std::cout << "PASS!\n";
    }else{
        std::cout << "FAIL with " << errors << "errors\n";
    }
    std::cout << "Iteration " << i << " done.\n";
    
}
}

