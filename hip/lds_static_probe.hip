// Static __shared__ sized at compile time, so the code object records what a
// kernel of this shape actually asks the toolchain for.
//
// This exists because the dynamic-LDS probe disagreed with the counters at
// 10752 bytes: the runtime API divided 64 KiB by the requested byte count and
// never applied the 1024-byte rounding the counter model needs. The measured
// PEPS kernel allocates statically, so this asks whether the rounding is
// visible in the compiled object at all, before any GPU is involved.

#include <hip/hip_runtime.h>

#include <cstdio>

#ifndef PROBE_LDS_BYTES
#define PROBE_LDS_BYTES 8704
#endif

__global__ void static_lds(float *sink) {
    __shared__ char lds[PROBE_LDS_BYTES];
    lds[threadIdx.x] = static_cast<char>(threadIdx.x);
    __syncthreads();
    if (sink != nullptr && threadIdx.x == 0) {
        sink[0] = static_cast<float>(lds[0]);
    }
}

int main() {
    std::printf("{\"requested_lds_bytes\": %d}\n", PROBE_LDS_BYTES);
    return 0;
}
