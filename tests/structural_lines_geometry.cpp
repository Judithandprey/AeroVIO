#include "viobench_abi.h"
#include "structural_detector_cases.h"
#include "structural_gate_cases.h"
#include <cstdio>

extern "C" VIOBENCH_EXPORT(run_tests) int run_tests() {
    const int failed=detector_cases()+gate_cases();
    std::printf("structural line failures: %d\n",failed);
    std::fflush(stdout);
    return failed;
}
