
#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include <cstdint>

using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_IN2 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;


// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < 16; i++)
    bufIn1[i] = i;
}

// Initialize Input buffer 2
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
  for (int i = 0; i < 4 * SIZE; i++)
  bufIn2[i] = 2;

}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 1, 4 * SIZE); // you need to set this to non-zero value or it may be optimized away
}

// Functional correctness verifyer
int verify(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
    int errors = 0;
    verbosity = 2;

    for (int i = 0; i < SIZE; i++) {
        for(int j=0;j<4;j++){
            int32_t ref = bufIn1[j] * bufIn2[4 * i] + bufIn1[4 + j] * bufIn2[4 * i + 1]
                          + bufIn1[8 + j] * bufIn2[4 * i + 2] + bufIn1[12 + j] * bufIn2[4 * i + 3];
            int32_t test = bufOut[4 * i + j];
            if (test != ref) {
                if (verbosity >= 1)
                    std::cout << "Error in output " << test << " != " << ref << std::endl;
                errors++;
            } else {
                if (verbosity >= 1)
                    std::cout << "Correct output " << test << " == " << ref << std::endl;
            }
        }
    }
    return errors;
}

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = 4;
  constexpr int IN2_VOLUME = 1024;
  constexpr int OUT_VOLUME = 1024;

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_IN2, DATATYPE_OUT,
                              initialize_bufIn1, initialize_bufIn2,
                              initialize_bufOut, verify>(
      IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs, true);
  return res;
}