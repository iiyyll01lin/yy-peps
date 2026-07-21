// wmma_mlp.hip — WMMA-accelerated MLP matmul for the PEPS decoder (fp16 + int8).
//
// 繁體中文:用 rocWMMA(Wave Matrix Multiply-Accumulate)加速 PEPS 解碼器的
// MLP 矩陣乘。一層 MLP 即 Y = X * W + b 後接非線性;此 kernel 用 16x16x16 的
// WMMA fragment 做 X(MxK) * W(KxN) 的 tile 累加。支援兩種精度:
//   * fp16 輸入 / fp32 累加(標準混合精度)
//   * int8 (iu8) 輸入 / int32 累加(RTXNTC 走的量化快路徑)
// 針對 RDNA3.5(gfx1151)與 RDNA4(gfx1201)皆可編譯與執行。
//
// Per-ISA selection is REAL, not cosmetic (see the PEPS_WMMA_GEN block below):
//   * rocWMMA emits the correct low-level WMMA intrinsic for the *device* arch
//     (gfx11 vs gfx12 have different WMMA encodings).
//   * The `__gfx1201__` / `__gfx1151__` predefined device macros additionally
//     select the int8 code path: hardware iu8 WMMA on archs known to support it,
//     otherwise a correct scalar int8 fallback (so it never silently mis-runs).
//   * The compiled WMMA generation is reported at runtime (read back from device).
//
// Two run modes / 兩種執行模式:
//   1. bench   : ./wmma_mlp [M K N iters]                  (fp16, random fill)
//                ./wmma_mlp bench <fp16|int8> [M K N iters]
//   2. fixture : ./wmma_mlp fixture <fp16|int8> <in.bin> <out.bin>
//
// Build / 編譯:
//   Box B (RDNA4):  hipcc --offload-arch=gfx1201 wmma_mlp.hip -o wmma_mlp
//   Box A (RDNA3.5):hipcc --offload-arch=gfx1151 wmma_mlp.hip -o wmma_mlp

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

using rocwmma::float16_t;
using rocwmma::float32_t;

#define HIP_CHECK(cmd)                                                              \
    do {                                                                           \
        hipError_t _e = (cmd);                                                     \
        if (_e != hipSuccess) {                                                    \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),      \
                    __FILE__, __LINE__);                                           \
            return 3;                                                              \
        }                                                                          \
    } while (0)

// WMMA tile dimensions (16x16x16 is supported on both RDNA3.5 and RDNA4).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// --- REAL per-ISA selection via the predefined *device* macros ----------------
// These macros are only defined during the GPU compilation pass, so the choice
// below genuinely depends on --offload-arch. gfx120x = RDNA4 (WMMA gen12);
// gfx110x/gfx115x = RDNA3/RDNA3.5 (WMMA gen11). Both generations expose iu8 WMMA;
// any other/unknown arch falls back to a correct scalar int8 loop.
#if defined(__gfx1200__) || defined(__gfx1201__)
#  define PEPS_WMMA_GEN 12
#  define PEPS_HAS_IU8_WMMA 1
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) ||       \
      defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) ||       \
      defined(__gfx1153__)
#  define PEPS_WMMA_GEN 11
#  define PEPS_HAS_IU8_WMMA 1
#else
#  define PEPS_WMMA_GEN 0
#  define PEPS_HAS_IU8_WMMA 0
#endif

// Report the compiled WMMA generation back to the host (verifiable at runtime).
__global__ void report_wmma_gen(int* gen) {
    if (threadIdx.x == 0) *gen = PEPS_WMMA_GEN;
}

// ---------------------------------------------------------------- fp16 kernel
// One block computes one 16x16 output tile: C[16x16] = A[16xK] * B[Kx16].
// A = activations (row-major M x K), B = weights (row-major K x N).
__global__ void wmma_mlp_tile(const float16_t* A, const float16_t* B,
                              float32_t* C, int M, int N, int K) {
    using namespace rocwmma;

    int tileRow = blockIdx.y * WMMA_M;
    int tileCol = blockIdx.x * WMMA_N;
    if (tileRow >= M || tileCol >= N) return;

    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, float16_t, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, float16_t, row_major>();
    auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float32_t>();
    fill_fragment(fragC, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        load_matrix_sync(fragA, A + tileRow * K + k, K);
        load_matrix_sync(fragB, B + k * N + tileCol, N);
        mma_sync(fragC, fragA, fragB, fragC);
    }
    store_matrix_sync(C + tileRow * N + tileCol, fragC, N, mem_row_major);
}

// ---------------------------------------------------------------- int8 kernel
// C[16x16] (int32) = A[16xK] (int8) * B[Kx16] (int8), int32 accumulate.
// On archs with confirmed iu8 WMMA this uses the hardware path; otherwise it
// falls back to a correct scalar int32 tile loop (documented, never faked).
__global__ void wmma_i8_tile(const int8_t* A, const int8_t* B,
                             int32_t* C, int M, int N, int K) {
    using namespace rocwmma;

    int tileRow = blockIdx.y * WMMA_M;
    int tileCol = blockIdx.x * WMMA_N;
    if (tileRow >= M || tileCol >= N) return;

#if PEPS_HAS_IU8_WMMA
    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major>();
    auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t>();
    fill_fragment(fragC, 0);

    for (int k = 0; k < K; k += WMMA_K) {
        load_matrix_sync(fragA, A + tileRow * K + k, K);
        load_matrix_sync(fragB, B + k * N + tileCol, N);
        mma_sync(fragC, fragA, fragB, fragC);
    }
    store_matrix_sync(C + tileRow * N + tileCol, fragC, N, mem_row_major);
#else
    // Scalar int8 fallback for archs without confirmed iu8 WMMA.
    if (threadIdx.x == 0) {
        for (int r = 0; r < WMMA_M && tileRow + r < M; r++) {
            for (int c = 0; c < WMMA_N && tileCol + c < N; c++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += int(A[(tileRow + r) * K + k]) * int(B[k * N + (tileCol + c)]);
                C[(tileRow + r) * N + (tileCol + c)] = acc;
            }
        }
    }
#endif
}

// ---------------------------------------------------------------- host helpers
static int query_wmma_gen() {
    int* dGen; int hGen = -1;
    if (hipMalloc(&dGen, sizeof(int)) != hipSuccess) return -1;
    report_wmma_gen<<<1, 1>>>(dGen);
    hipDeviceSynchronize();
    hipMemcpy(&hGen, dGen, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(dGen);
    return hGen;
}

static void print_arch_banner() {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, 0) == hipSuccess) {
        int gen = query_wmma_gen();
        const char* path = gen ? "hardware iu8 WMMA" : "scalar int8 fallback";
        printf("arch=%s  compiled WMMA gen=%d  warpSize=%d  int8 path=%s\n",
               p.gcnArchName, gen, p.warpSize, path);
    }
}

static void roundup3(int& M, int& K, int& N) {
    auto ru = [](int v, int m) { return (v + m - 1) / m * m; };
    M = ru(M, WMMA_M); K = ru(K, WMMA_K); N = ru(N, WMMA_N);
}

// ---------------------------------------------------------------- fp16 compute
static int launch_fp16(const std::vector<float16_t>& hA, const std::vector<float16_t>& hB,
                       int M, int N, int K, std::vector<float32_t>& hC) {
    float16_t *dA, *dB; float32_t* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float32_t)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float16_t), hipMemcpyHostToDevice));

    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(rocwmma::Constants::AMDGCN_WAVE_SIZE);
    wmma_mlp_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hC.resize((size_t)M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float32_t), hipMemcpyDeviceToHost));
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}

// ---------------------------------------------------------------- int8 compute
static int launch_i8(const std::vector<int8_t>& hA, const std::vector<int8_t>& hB,
                     int M, int N, int K, std::vector<int32_t>& hC) {
    int8_t *dA, *dB; int32_t* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dB, hB.size()));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));

    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(rocwmma::Constants::AMDGCN_WAVE_SIZE);
    wmma_i8_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hC.resize((size_t)M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}

// ---------------------------------------------------------------- fixture I/O
// Binary layout (little-endian), matching tests/test_hip_parity.py:
//   input : int32 magic, dtype(0=fp16,1=int8), M, K, N ; then A[M*K], B[K*N]
//   output: int32 magic, dtype, M, N                    ; then C[M*N]
// fp16 payload is stored as float32 (already rounded to fp16 by the writer);
// int8 payload is raw int8; int8 output is int32.
static const int32_t WMMA_MAGIC = 0x574D4D31;  // 'WMM1'

static int run_fixture_fp16(const char* inpath, const char* outpath) {
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", inpath); return 2; }
    int32_t hdr[5];
    if (fread(hdr, sizeof(int32_t), 5, f) != 5 || hdr[0] != WMMA_MAGIC || hdr[1] != 0) {
        fprintf(stderr, "bad fp16 fixture header in %s\n", inpath); fclose(f); return 2;
    }
    int M = hdr[2], K = hdr[3], N = hdr[4];
    std::vector<float> fA((size_t)M * K), fB((size_t)K * N);
    bool ok = fread(fA.data(), sizeof(float), fA.size(), f) == fA.size()
              && fread(fB.data(), sizeof(float), fB.size(), f) == fB.size();
    fclose(f);
    if (!ok) { fprintf(stderr, "short read %s\n", inpath); return 2; }

    std::vector<float16_t> hA(fA.size()), hB(fB.size());
    for (size_t i = 0; i < fA.size(); i++) hA[i] = float16_t(fA[i]);
    for (size_t i = 0; i < fB.size(); i++) hB[i] = float16_t(fB[i]);

    std::vector<float32_t> hC;
    int rc = launch_fp16(hA, hB, M, N, K, hC);
    if (rc) return rc;

    FILE* g = fopen(outpath, "wb");
    if (!g) { fprintf(stderr, "cannot open %s\n", outpath); return 2; }
    int32_t ohdr[4] = {WMMA_MAGIC, 0, M, N};
    fwrite(ohdr, sizeof(int32_t), 4, g);
    fwrite(hC.data(), sizeof(float32_t), hC.size(), g);
    fclose(g);
    printf("wmma fp16 fixture: wrote %dx%d to %s\n", M, N, outpath);
    return 0;
}

static int run_fixture_i8(const char* inpath, const char* outpath) {
    FILE* f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", inpath); return 2; }
    int32_t hdr[5];
    if (fread(hdr, sizeof(int32_t), 5, f) != 5 || hdr[0] != WMMA_MAGIC || hdr[1] != 1) {
        fprintf(stderr, "bad int8 fixture header in %s\n", inpath); fclose(f); return 2;
    }
    int M = hdr[2], K = hdr[3], N = hdr[4];
    std::vector<int8_t> hA((size_t)M * K), hB((size_t)K * N);
    bool ok = fread(hA.data(), 1, hA.size(), f) == hA.size()
              && fread(hB.data(), 1, hB.size(), f) == hB.size();
    fclose(f);
    if (!ok) { fprintf(stderr, "short read %s\n", inpath); return 2; }

    std::vector<int32_t> hC;
    int rc = launch_i8(hA, hB, M, N, K, hC);
    if (rc) return rc;

    FILE* g = fopen(outpath, "wb");
    if (!g) { fprintf(stderr, "cannot open %s\n", outpath); return 2; }
    int32_t ohdr[4] = {WMMA_MAGIC, 1, M, N};
    fwrite(ohdr, sizeof(int32_t), 4, g);
    fwrite(hC.data(), sizeof(int32_t), hC.size(), g);
    fclose(g);
    printf("wmma int8 fixture: wrote %dx%d (int32) to %s\n", M, N, outpath);
    return 0;
}

// ---------------------------------------------------------------- benchmarks
static int bench_fp16(int M, int K, int N, int iters) {
    roundup3(M, K, N);
    std::vector<float16_t> hA((size_t)M * K), hB((size_t)K * N);
    srand(1234);
    for (auto& x : hA) x = float16_t(((rand() % 200) - 100) / 100.0f);
    for (auto& x : hB) x = float16_t(((rand() % 200) - 100) / 100.0f);

    std::vector<float32_t> hC;
    if (launch_fp16(hA, hB, M, N, K, hC)) return 3;   // warmup + correctness
    double ref = 0.0;
    for (int k = 0; k < K; k++) ref += float(hA[k]) * float(hB[k * N]);
    printf("correctness C[0,0]=%.3f  ref=%.3f  (M=%d K=%d N=%d)\n",
           float(hC[0]), ref, M, K, N);

    float16_t *dA, *dB; float32_t* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float32_t)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float16_t), hipMemcpyHostToDevice));
    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(rocwmma::Constants::AMDGCN_WAVE_SIZE);

    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        wmma_mlp_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("WMMA MLP layer [fp16]: %.4f ms/iter  (%d iters)\n", ms, iters);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}

static int bench_i8(int M, int K, int N, int iters) {
    roundup3(M, K, N);
    std::vector<int8_t> hA((size_t)M * K), hB((size_t)K * N);
    srand(1234);
    for (auto& x : hA) x = int8_t((rand() % 255) - 127);
    for (auto& x : hB) x = int8_t((rand() % 255) - 127);

    std::vector<int32_t> hC;
    if (launch_i8(hA, hB, M, N, K, hC)) return 3;   // warmup + correctness
    long ref = 0;
    for (int k = 0; k < K; k++) ref += int(hA[k]) * int(hB[k * N]);
    printf("correctness C[0,0]=%d  ref=%ld  (M=%d K=%d N=%d)\n",
           hC[0], ref, M, K, N);

    int8_t *dA, *dB; int32_t* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dB, hB.size()));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(rocwmma::Constants::AMDGCN_WAVE_SIZE);

    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        wmma_i8_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("WMMA MLP layer [int8]: %.4f ms/iter  (%d iters)\n", ms, iters);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}

int main(int argc, char** argv) {
    std::string a1 = argc > 1 ? argv[1] : "";

    if (a1 == "fixture") {
        if (argc < 5) {
            fprintf(stderr, "usage: %s fixture <fp16|int8> <in.bin> <out.bin>\n", argv[0]);
            return 2;
        }
        if (std::strcmp(argv[2], "int8") == 0) return run_fixture_i8(argv[3], argv[4]);
        return run_fixture_fp16(argv[3], argv[4]);
    }

    // bench mode (default). Backward compatible with `wmma_mlp M K N iters`.
    std::string dtype = "fp16";
    int off = 1;
    if (a1 == "bench") {
        off = 2;
        if (argc > 2 && (std::strcmp(argv[2], "fp16") == 0 || std::strcmp(argv[2], "int8") == 0)) {
            dtype = argv[2];
            off = 3;
        }
    }
    int M = argc > off     ? atoi(argv[off])     : 4096;
    int K = argc > off + 1 ? atoi(argv[off + 1]) : 64;
    int N = argc > off + 2 ? atoi(argv[off + 2]) : 64;
    int iters = argc > off + 3 ? atoi(argv[off + 3]) : 200;

    print_arch_banner();
    if (dtype == "int8") return bench_i8(M, K, N, iters);
    return bench_fp16(M, K, N, iters);
}
