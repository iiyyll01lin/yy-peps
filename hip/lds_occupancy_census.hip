// Count how many workgroups are actually resident at once, by asking them.
//
// The runtime occupancy API was rejected as an oracle: it reports an integer
// per compute unit, and on RDNA the LDS pool belongs to a workgroup processor
// spanning two CUs, so an odd workgroup count cannot survive the division. At
// 13312 bytes the counters measured 8.97 waves per CU and the API said 8.
//
// This does not divide anything. Every block registers on entry, holds its LDS
// for a fixed wall-clock period, and deregisters on exit; the peak registration
// is the number that were resident together. It is a device-wide integer, so an
// odd workgroup per workgroup processor is representable.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void census_dynamic(int *live, int *peak, char *sink,
                               unsigned long long spin) {
    extern __shared__ char lds[];
    if (threadIdx.x == 0) {
        atomicMax(peak, atomicAdd(live, 1) + 1);
    }
    lds[threadIdx.x] = static_cast<char>(threadIdx.x);
    __syncthreads();

    unsigned long long start = wall_clock64();
    while (wall_clock64() - start < spin) {
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        sink[0] = lds[0];
        atomicSub(live, 1);
    }
}

int main(int argc, char **argv) {
    int device = (argc > 1) ? std::atoi(argv[1]) : 0;
    int block = (argc > 2) ? std::atoi(argv[2]) : 64;
    int spin_ms = (argc > 3) ? std::atoi(argv[3]) : 20;

    if (hipSetDevice(device) != hipSuccess) {
        std::fprintf(stderr, "cannot select device %d\n", device);
        return 1;
    }
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, device);

    int wall_khz = 0;
    hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, device);
    if (wall_khz <= 0) {
        std::fprintf(stderr, "no wall clock rate\n");
        return 1;
    }
    const unsigned long long spin =
        static_cast<unsigned long long>(wall_khz) * spin_ms;

    int *live = nullptr;
    int *peak = nullptr;
    char *sink = nullptr;
    hipMalloc(&live, sizeof(int));
    hipMalloc(&peak, sizeof(int));
    hipMalloc(&sink, 1);

    const std::vector<int> footprints = {
        8704,  9216,  9728,  10240, 10752, 11264,
        11776, 12288, 13312, 16384, 32768,
    };

    std::printf("{\n");
    std::printf("  \"gcn_arch\": \"%s\",\n", prop.gcnArchName);
    std::printf("  \"multi_processor_count\": %d,\n", prop.multiProcessorCount);
    std::printf("  \"warp_size\": %d,\n", prop.warpSize);
    std::printf("  \"block_threads\": %d,\n", block);
    std::printf("  \"spin_ms\": %d,\n", spin_ms);
    std::printf("  \"wall_clock_khz\": %d,\n", wall_khz);
    std::printf("  \"rows\": [\n");

    for (size_t i = 0; i < footprints.size(); ++i) {
        const int bytes = footprints[i];
        // Enough blocks that residency saturates several times over.
        const int grid = 64 * prop.multiProcessorCount;

        int zero = 0;
        hipMemcpy(live, &zero, sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(peak, &zero, sizeof(int), hipMemcpyHostToDevice);

        hipLaunchKernelGGL(census_dynamic, dim3(grid), dim3(block), bytes, 0,
                           live, peak, sink, spin);
        hipError_t launched = hipGetLastError();
        hipError_t ran = hipDeviceSynchronize();

        int observed = -1;
        hipMemcpy(&observed, peak, sizeof(int), hipMemcpyDeviceToHost);

        std::printf("    {\"footprint\": %d, \"grid\": %d, ", bytes, grid);
        if (launched != hipSuccess || ran != hipSuccess) {
            std::printf("\"peak_resident_blocks\": null, \"error\": \"%s\"}",
                        hipGetErrorString(launched != hipSuccess ? launched : ran));
        } else {
            const double per_mp =
                static_cast<double>(observed) / prop.multiProcessorCount;
            std::printf(
                "\"peak_resident_blocks\": %d, \"per_multiprocessor\": %.4f}",
                observed, per_mp);
        }
        std::printf("%s\n", (i + 1 < footprints.size()) ? "," : "");
        std::fflush(stdout);
    }

    std::printf("  ]\n}\n");
    hipFree(live);
    hipFree(peak);
    hipFree(sink);
    return 0;
}
