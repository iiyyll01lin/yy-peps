// What the occupancy model claims is about LDS bytes, a rounding granule and a
// pool division. None of that involves WMMA, so this probe carries no matrix
// instructions and compiles for CDNA as well as RDNA.
//
// The sweep uses dynamic shared memory so one binary covers every footprint.
// The committed measurements came from static __shared__, so the first thing
// stage one has to show is that the two are accounted the same way.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void lds_only(float *sink) {
    extern __shared__ char lds[];
    // Keep the allocation observable so it cannot be optimised away.
    if (sink != nullptr && threadIdx.x == 0) {
        sink[0] = static_cast<float>(lds[0]);
    }
}

int main(int argc, char **argv) {
    int device = (argc > 1) ? std::atoi(argv[1]) : 0;
    int block = (argc > 2) ? std::atoi(argv[2]) : 64;

    if (hipSetDevice(device) != hipSuccess) {
        std::fprintf(stderr, "cannot select device %d\n", device);
        return 1;
    }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        std::fprintf(stderr, "cannot read device properties\n");
        return 1;
    }

    // The seven footprints the counters were read at, plus the granule
    // boundaries either side of them and two larger points.
    const std::vector<int> footprints = {
        8704,  9216,  9728,  10240, 10752, 11264, 11776,
        12288, 13312, 16384, 32768, 65536,
    };

    std::printf("{\n");
    std::printf("  \"gcn_arch\": \"%s\",\n", prop.gcnArchName);
    std::printf("  \"multi_processor_count\": %d,\n", prop.multiProcessorCount);
    std::printf("  \"warp_size\": %d,\n", prop.warpSize);
    std::printf("  \"max_shared_memory_per_block\": %zu,\n",
                static_cast<size_t>(prop.sharedMemPerBlock));
    std::printf("  \"max_threads_per_multiprocessor\": %d,\n",
                prop.maxThreadsPerMultiProcessor);
    std::printf("  \"block_threads\": %d,\n", block);
    std::printf("  \"rows\": [\n");

    for (size_t i = 0; i < footprints.size(); ++i) {
        int blocks = 0;
        hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, reinterpret_cast<const void *>(lds_only), block,
            static_cast<size_t>(footprints[i]));

        std::printf("    {\"footprint\": %d, ", footprints[i]);
        if (status != hipSuccess) {
            std::printf("\"blocks_per_multiprocessor\": null, \"error\": \"%s\"}",
                        hipGetErrorString(status));
        } else {
            int waves_per_block = (block + prop.warpSize - 1) / prop.warpSize;
            std::printf(
                "\"blocks_per_multiprocessor\": %d, \"waves_per_block\": %d, "
                "\"waves_per_multiprocessor\": %d}",
                blocks, waves_per_block, blocks * waves_per_block);
        }
        std::printf("%s\n", (i + 1 < footprints.size()) ? "," : "");
    }

    std::printf("  ]\n}\n");
    return 0;
}
