
#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../const.hpp"

#include <cstdint>
#include <string>
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

void check_each(std::string msg, int verbosity, DATATYPE_OUT test, DATATYPE_OUT ref, int index, float tol, int& error) {
    if (test < ref - tol || test > ref + tol) {
        if (verbosity >= 1){
            std::cout << "Error in output (" << msg << ") " << index << " : " << test << " != " << ref << std::endl;
        
            error++;
        }
    } else {
        if (verbosity >= 2)
            std::cout << "Correct in output ( " << msg << ") " << index << " : " << test << " == " << ref << std::endl;
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

void compute_2x3_3x3_mat(DATATYPE_OUT *mat2_3, DATATYPE_OUT *mat3_3, DATATYPE_OUT *output){
    //comput 2x3 3x3 matrix multiplication
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<3;j++){
            DATATYPE_OUT acc = 0;
            for(size_t k=0;k<3;k++){
                acc += mat2_3[i * 3 + k] * mat3_3[j + k * 3];
            }
            output[i * 3 + j] = (DATATYPE_OUT) acc;
        }
    }
}

void compute_2x3_3x3_3x2_mat(DATATYPE_OUT *mat2_3, DATATYPE_OUT *mat3_3, DATATYPE_OUT *output){
    //comput 2x3 3x3 matrix multiplication
    DATATYPE_OUT temp[6];
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<3;j++){
            DATATYPE_OUT acc = 0;
            for(size_t k=0;k<3;k++){
                acc += mat2_3[i * 3 + k] * mat3_3[j + k * 3];
            }
            temp[i * 3 + j] = (DATATYPE_OUT) acc;
        }
    }
    // then multiply 3x2
    for(size_t i=0;i<2;i++){
        for(size_t j=0;j<2;j++){
            DATATYPE_OUT acc = 0;
            for(size_t k=0;k<3;k++){
                acc += temp[i * 3 + k] * mat2_3[j * 3 + k];
            }
            output[i * 2 + j] = (DATATYPE_OUT) acc;
        }
    }
}

inline float combine_bfloat16(std::bfloat16_t high, std::bfloat16_t low) {
    uint16_t lo16 = 0, hi16 = 0;
    memcpy(&lo16, &low, sizeof(uint16_t));
    memcpy(&hi16, &high, sizeof(uint16_t));

    uint32_t u32 = static_cast<uint32_t>(lo16) | (static_cast<uint32_t>(hi16) << 16);
    float combined;
    memcpy(&combined, &u32, sizeof(float));
    return combined;

}


// Functional correctness verifyer
int verify(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int TILE_SIZE, int TILE_COUNT, int verbosity) {
    int errors = 0;
    verbosity = 1;

    //first
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        for (int iter = 0; iter < TILE_SIZE / 4; iter++){
            int offset = tile * TILE_SIZE * 72;
            for (int i = 0; i < 4; i++) {
                
                

                // for(int j=0;j<4;j++){

                //     DATATYPE_OUT ref = bufIn1[4 * j] * bufIn2[offset + iter * 32 + i] + bufIn1[4 * j + 1] * bufIn2[offset + iter * 32 + i + 4]
                //                   + bufIn1[4 * j + 2] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[4 * j + 3] * bufIn2[offset + iter * 32 + i + 12];
                //     DATATYPE_OUT test = bufOut[tile * TILE_SIZE * 4 + iter * 16 + i * 4 + j];
                //     if (test < ref - 0.25 || test > ref + 0.25) {
                //         if (verbosity >= 1){

                //             std::cout << "Error in output " << tile * TILE_SIZE * 14 + iter * 16 + i + j * 4 << " : " << test << " != " << ref << std::endl;
                //         }
                //         errors++;

                //     } else {
                //         if (verbosity >= 2)
                //             std::cout << "Correct in output " << tile * TILE_SIZE * 14 + iter * 16 + i + j * 4 << " : " << test << " == " << ref << std::endl;
                //     }
                    
                // }

                

                DATATYPE_OUT ref_x = bufIn1[0] * bufIn2[offset + iter * 32 + i] + bufIn1[1] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[2] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[3] * bufIn2[offset + iter * 32 + i + 12];

                DATATYPE_OUT ref_y = bufIn1[4] * bufIn2[offset + iter * 32 + i] + bufIn1[5] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[6] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[7] * bufIn2[offset + iter * 32 + i + 12];
                
                DATATYPE_OUT ref_z = bufIn1[8] * bufIn2[offset + iter * 32 + i] + bufIn1[9] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[10] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[11] * bufIn2[offset + iter * 32 + i + 12];
                DATATYPE_OUT ref_opa = bufIn2[offset + TILE_SIZE * 8 + iter * 4 + i];

                float ref_index = combine_bfloat16(ref_z, ref_opa); // add opacity for sorting faster

                DATATYPE_OUT fx = bufIn1[16];
                DATATYPE_OUT fy = bufIn1[17];
                DATATYPE_OUT ref_R1[6];
                ref_R1[0] = fx / ref_z;
                ref_R1[1] = 0;
                ref_R1[2] = -1 * fx * ref_x / (ref_z * ref_z);
                ref_R1[3] = 0;
                ref_R1[4] = fy / ref_z;
                ref_R1[5] = -1 * fy * ref_y / (ref_z * ref_z);

                DATATYPE_OUT ref_R2[9];
                ref_R2[0] = bufIn1[0];
                ref_R2[1] = bufIn1[1];
                ref_R2[2] = bufIn1[2];
                ref_R2[3] = bufIn1[4];
                ref_R2[4] = bufIn1[5];
                ref_R2[5] = bufIn1[6];
                ref_R2[6] = bufIn1[8];
                ref_R2[7] = bufIn1[9];
                ref_R2[8] = bufIn1[10];
                // this test_R is J_R matrix
                DATATYPE_OUT test_R[6];
                compute_2x3_3x3_mat(ref_R1, ref_R2, test_R);

                // next, we make cov3D mat 
                int true_iter = iter * 4 + i;
                int sub_tile_id = true_iter / (TILE_SIZE / CONV3D_TILE_NUM);
                int sub_tile_res = true_iter % (TILE_SIZE / CONV3D_TILE_NUM);
                
                int offset2 = tile * TILE_SIZE * 72 + TILE_SIZE * 9 + sub_tile_id * (TILE_SIZE / CONV3D_TILE_NUM) * 7;
                DATATYPE_IN2 w = bufIn2[offset2 + (sub_tile_res / 16) * 64 + (sub_tile_res % 16)];
                DATATYPE_IN2 x = bufIn2[offset2 + (sub_tile_res / 16) * 64 + (sub_tile_res % 16) + 16];
                DATATYPE_IN2 y = bufIn2[offset2 + (sub_tile_res / 16) * 64 + (sub_tile_res % 16) + 32];
                DATATYPE_IN2 z = bufIn2[offset2 + (sub_tile_res / 16) * 64 + (sub_tile_res % 16) + 48];

                DATATYPE_IN2 scale1 = bufIn2[offset2 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + (sub_tile_res / 16) * 48 + (sub_tile_res % 16)];
                DATATYPE_IN2 scale2 = bufIn2[offset2 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + (sub_tile_res / 16) * 48 + (sub_tile_res % 16) + 16];
                DATATYPE_IN2 scale3 = bufIn2[offset2 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + (sub_tile_res / 16) * 48 + (sub_tile_res % 16) + 32];
                DATATYPE_IN2 norm = std::sqrt(w * w + x * x + y * y + z * z);
                w /= norm;
                x /= norm;
                y /= norm;
                z /= norm;                    


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

                DATATYPE_OUT cov3D_mat[9];

                
                compute_3x3_mat(calc_mat, cov3D_mat);
                


                DATATYPE_OUT ref_mat[4];
                compute_2x3_3x3_3x2_mat(test_R, cov3D_mat, ref_mat);
                //compute inv
                ref_mat[0] = ref_mat[0] + 0.3;
                ref_mat[3] = ref_mat[3] + 0.3;
                DATATYPE_OUT det = ref_mat[0] * ref_mat[3] - ref_mat[1] * ref_mat[2];
                DATATYPE_OUT inv_det = 1.0 / det;
                DATATYPE_OUT inv_cov2D_0_0 = ref_mat[3] * inv_det;
                DATATYPE_OUT inv_cov2D_0_1 = -1 * ref_mat[1] * inv_det;
                DATATYPE_OUT inv_cov2D_1_1 = ref_mat[0] * inv_det;
                DATATYPE_OUT b = (ref_mat[0] + ref_mat[3]) * 0.5;
                DATATYPE_OUT sqrt_term = std::sqrt(std::max((DATATYPE_OUT)0.1, b * b - det));
                DATATYPE_OUT radius = std::sqrt(b + sqrt_term) * 3 + 1;

                
                float test_index = combine_bfloat16(bufOut[CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2 + 1],
                                                            bufOut[CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2]);
                // check
                if(test_index < ref_index - 0.25 || test_index > ref_index + 0.25) {
                    if (verbosity >= 1){
                        std::cout << ref_index << " " << ref_z << " " << ref_opa << "\n";
                        std::cout << bufOut[CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2 + 1] << " " << bufOut[CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2] << "\n";
                        std::cout << "Error in output (index) " << CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2 << " : " << test_index << " != " << ref_index << std::endl;
                    }
                    errors++;
                } else {
                    if (verbosity >= 2)
                        std::cout << "Correct in output (index) " << CHUNK_SIZE * 4 + (tile * TILE_SIZE + iter * 4 + i) * 2 << " : " << test_index << " == " << ref_index << std::endl;
                }

                //check
                // for(int k = 0;k<16;k++){
                    
                //     std::cout << "output data " << bufOut[CHUNK_SIZE * 6 + (tile * TILE_SIZE + iter * 4 + i) * 4 + k] << " ";
                // }
                check_each("cov2D", verbosity, bufOut[CHUNK_SIZE * 6 + (tile * TILE_SIZE + iter * 4 + i) * 4], inv_cov2D_0_0, tile * TILE_SIZE * 14 + TILE_SIZE * 6 + (iter * 4 + i) * 4, 0.25, errors);
                check_each("cov2D", verbosity, bufOut[CHUNK_SIZE * 6 + (tile * TILE_SIZE + iter * 4 + i) * 4 + 1], inv_cov2D_0_1, tile * TILE_SIZE * 14 + TILE_SIZE * 6 + (iter * 4 + i) * 4 + 1, 0.25, errors);
                check_each("cov2D", verbosity, bufOut[CHUNK_SIZE * 6 + (tile * TILE_SIZE + iter * 4 + i) * 4 + 2], inv_cov2D_1_1, tile * TILE_SIZE * 14 + TILE_SIZE * 6 + (iter * 4 + i) * 4 + 2, 0.25, errors);
                check_each("cov2D", verbosity, bufOut[CHUNK_SIZE * 6 + (tile * TILE_SIZE + iter * 4 + i) * 4 + 3], radius, tile * TILE_SIZE * 14 + TILE_SIZE * 6 + (iter * 4 + i) * 4 + 3, 5, errors);
                
            
        
            }
        }
    }

    //second
    for (int tile = 0; tile < TILE_COUNT; tile++) {
        for (int iter = 0; iter < TILE_SIZE / 4; iter++){
            for (int i = 0; i < 4; i++) {
                for(int j=0;j<2;j++){
                    
                    int offset = tile * TILE_SIZE * 72;
                    float ref = (bufIn1[4 * j + 18] * bufIn2[offset + iter * 32 + i] + bufIn1[4 * j + 1 + 18] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[4 * j + 2 + 18] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[4 * j + 3 + 18] * bufIn2[offset + iter * 32 + i + 12])
                                  / (bufIn1[12 + 18] * bufIn2[offset + iter * 32 + i] + bufIn1[13 + 18] * bufIn2[offset + iter * 32 + i + 4]
                                  + bufIn1[14 + 18] * bufIn2[offset + iter * 32 + i + 8] + bufIn1[15 + 18] * bufIn2[offset + iter * 32 + i + 12]);
                    
                                  ref = (ref + 1) * 0.5 * 800.f - 0.5f;

                    
                    // Combine two bf16 outputs into a single 32-bit float (little endian)
                    size_t idx = CHUNK_SIZE * 0 + tile * TILE_SIZE * 4 + (iter * 8 + i * 2 + j) * 2;

                    float combined = combine_bfloat16(bufOut[idx + 1], bufOut[idx]);

                    float ref_f = static_cast<float>(ref);

                    if (std::isnan(combined) || combined < ref_f - 0.05f || combined > ref_f + 0.05f) {
                        if (verbosity >= 1) {
                            std::cout << "Error in output (w) " << CHUNK_SIZE * 4 + tile * TILE_SIZE * 4 + (iter * 8 + i * 2 + j) * 2
                                << " : " << combined << " != " << ref_f << std::endl;
                        }
                        errors++;
                    } else {
                        if (verbosity >= 2) {
                            std::cout << "Correct in output (w) " << CHUNK_SIZE * 4 + tile * TILE_SIZE * 4 + (iter * 8 + i * 2 + j) * 2
                                << " : " << combined << " == " << ref_f << std::endl;
                        }
                    }
                    
                    
                }
            }
        }
    }
    

    {
        DATATYPE_IN1 sh[16];
        for(int i=0;i<16;i++){
            sh[i] = bufIn1[38 + i];
        }
        DATATYPE_IN1 cam_x;
        DATATYPE_IN1 cam_y;
        DATATYPE_IN1 cam_z;
        cam_x = bufIn1[54];
        cam_y = bufIn1[55];
        cam_z = bufIn1[56];


        for (int tile = 0; tile < TILE_COUNT; tile++) {
            for (int i = 0; i < TILE_SIZE; i++){
                int offset = tile * TILE_SIZE * 72 + TILE_SIZE * 16 + i * 56;

                int out_offset = CHUNK_SIZE * 10 + (tile * TILE_SIZE + i) * 4;

                //normalize
                DATATYPE_IN1 x = bufIn2[offset] - cam_x;
                DATATYPE_IN1 y = bufIn2[offset + 1] - cam_y;
                DATATYPE_IN1 z = bufIn2[offset + 2] - cam_z;
                DATATYPE_IN1 len = std::sqrt(x * x + y * y + z * z);
                x /= len;
                y /= len;
                z /= len;
                DATATYPE_IN1 xx = x * x;
                DATATYPE_IN1 yy = y * y;
                DATATYPE_IN1 zz = z * z;
                DATATYPE_IN1 xy = x * y;
                DATATYPE_IN1 xz = x * z;
                DATATYPE_IN1 yz = y * z;

                for(int j=0;j<3;j++){
                    DATATYPE_IN2 fs[16];
                    for(int k=0;k<16;k++){
                        fs[k] = bufIn2[offset + 8 + j * 16 + k];
                    }

                    fs[0] = sh[0] * fs[0];
                    fs[1] = sh[1] * -1 * y * fs[1];
                    fs[2] = sh[2] * z * fs[2];
                    fs[3] = sh[3] * -1 * x * fs[3];
                    fs[4] = sh[4] * xy * fs[4];
                    fs[5] = sh[5] * yz * fs[5];
                    fs[6] = sh[6] * (2 * zz - xx - yy) * fs[6];
                    fs[7] = sh[7] * xz * fs[7];
                    fs[8] = sh[8] * (xx - yy) * fs[8];
                    fs[9] = sh[9] * y * (3 * xx - yy) * fs[9];
                    fs[10] = sh[10] * xy * z * fs[10];
                    fs[11] = sh[11] * y * (4 * zz - xx - yy) * fs[11];
                    fs[12] = sh[12] * z * (2 * zz - 3 * xx - 3 * yy) * fs[12];
                    fs[13] = sh[13] * x * (4 * zz - xx - yy) * fs[13];
                    fs[14] = sh[14] * z * (xx - yy) * fs[14];
                    fs[15] = sh[15] * x * (xx - 3.0f * yy) * fs[15];
                    DATATYPE_OUT sum = 0.5;
                    for(int k=0;k<16;k++){
                        sum += fs[k];
                    }
                    check_each("color", verbosity, sum, bufOut[out_offset + j], i, 0.25, errors);
                    
                    
                    
                    
                }

            }
        }
    }
    return errors;
}

void fill_bufB(DATATYPE_IN2 *bufInB){
    // fill randomly first
    generate_random_bfloat16(bufInB, CHUNK_SIZE * 72, 0, 3);

    // fill some of them with zero (the gaussian padding)
    for (int i=0; i < TILE_COUNT; i++){
        int offset = TILE_SIZE * i * 72;
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
    auto bo_inB = xrt::bo(device, CHUNK_SIZE * 72 * sizeof(DATATYPE_IN2),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_outC = xrt::bo(device, CHUNK_SIZE * 14 * sizeof(DATATYPE_OUT),
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
    
    generate_random_bfloat16(bufInA, IN1_SIZE, 0, 3);
    bufInA[38] = 0.28209479177387814f;

    bufInA[39] = 0.4886025119029199f;
    bufInA[40] = 0.4886025119029199f;
    bufInA[41] = 0.4886025119029199f;

    bufInA[42] = 1.0925484305920792f;
    bufInA[43] = -1.0925484305920792f;
    bufInA[44] = 0.31539156525252005f;
    bufInA[45] = -1.0925484305920792f;
    bufInA[46] = 0.5462742152960396f;
    bufInA[47] = -0.5900435899266435f;
    bufInA[48] = 2.890611442640554f;
    bufInA[49] = -0.4570457994644658f;
    bufInA[50] = 0.3731763325901154f;
    bufInA[51] = -0.4570457994644658f;
    bufInA[52] = 1.445305721320277f;
    bufInA[53] = -0.5900435899266435f;
    

    // Initialize buffer bo_inFactor
    DATATYPE_IN2 *bufInB = bo_inB.map<DATATYPE_IN2 *>();
    fill_bufB(bufInB);
    // bufInB[0] = -0.706435;
    // bufInB[4] = -0.888309;
    // bufInB[8] = -0.871879;
    // bufInB[12] = 1;

    // bufInB[TILE_SIZE * 8] = 0.892785;
    // bufInB[TILE_SIZE * 8 + 16] = 0.222361;
    // bufInB[TILE_SIZE * 8 + 32] = -0.168303;
    // bufInB[TILE_SIZE * 8 + 48] = -0.118424;

    // bufInB[TILE_SIZE * 8 + (TILE_SIZE / CONV3D_TILE_NUM) * 4] = 0.000952771;
    // bufInB[TILE_SIZE * 8 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + 16] = 0.0301535;
    // bufInB[TILE_SIZE * 8 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + 32] = 0.120647;

    





    

    // Zero out buffer bo_outC
    DATATYPE_OUT *bufOut = bo_outC.map<DATATYPE_OUT *>();
    for (int i = 0; i < CHUNK_SIZE * 14; i++)
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

    std::cout << "calculation done, start to verify the result...\n";
    
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

