// wmma_mlp.hip — WMMA-accelerated MLP matmul for the PEPS decoder.
//
// 繁體中文:用 rocWMMA(Wave Matrix Multiply-Accumulate)加速 PEPS 解碼器的
// MLP 矩陣乘。一層 MLP 即 Y = X * W + b 後接非線性;此 kernel 用 16x16x16 的
// WMMA fragment 做 X(MxK) * W(KxN) 的 tile 累加。針對 RDNA3.5(gfx1151)與
// RDNA4(gfx1201)皆可編譯;fragment 形狀由 rocWMMA 依 arch 選擇。
//
// Build / 編譯:
//   Box B (RDNA4):  hipcc --offload-arch=gfx1201 wmma_mlp.hip -o wmma_mlp
//   Box A (RDNA3.5):hipcc --offload-arch=gfx1151 wmma_mlp.hip -o wmma_mlp
//
// This is a teaching kernel: correctness-first, single WMMA tile per block, FP16
// inputs with FP32 accumulate (the standard mixed-precision path).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

using rocwmma::float16_t;
using rocwmma::float32_t;

// WMMA tile dimensions (16x16x16 is supported on both RDNA3.5 and RDNA4).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

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

static void fill_rand(std::vector<float16_t>& v) {
    for (auto& x : v) x = float16_t(((rand() % 200) - 100) / 100.0f);
}

int main(int argc, char** argv) {
    // MLP layer size: M rows (points), K in-features, N out-features.
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int K = argc > 2 ? atoi(argv[2]) : 64;
    int N = argc > 3 ? atoi(argv[3]) : 64;
    int iters = argc > 4 ? atoi(argv[4]) : 200;

    // round up to WMMA tiles
    auto roundup = [](int v, int m) { return (v + m - 1) / m * m; };
    M = roundup(M, WMMA_M); K = roundup(K, WMMA_K); N = roundup(N, WMMA_N);

    std::vector<float16_t> hA(M * K), hB(K * N);
    std::vector<float32_t> hC(M * N, 0.0f);
    srand(1234); fill_rand(hA); fill_rand(hB);

    float16_t *dA, *dB; float32_t* dC;
    hipMalloc(&dA, hA.size() * sizeof(float16_t));
    hipMalloc(&dB, hB.size() * sizeof(float16_t));
    hipMalloc(&dC, hC.size() * sizeof(float32_t));
    hipMemcpy(dA, hA.data(), hA.size() * sizeof(float16_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), hB.size() * sizeof(float16_t), hipMemcpyHostToDevice);

    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(rocwmma::Constants::AMDGCN_WAVE_SIZE);  // one wave per tile

    // warmup + correctness (compare tile [0,0] to CPU)
    wmma_mlp_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();
    hipMemcpy(hC.data(), dC, hC.size() * sizeof(float32_t), hipMemcpyDeviceToHost);
    double ref = 0.0;
    for (int k = 0; k < K; k++) ref += float(hA[k]) * float(hB[k * N]);
    printf("correctness C[0,0]=%.3f  ref=%.3f  (M=%d K=%d N=%d)\n",
           float(hC[0]), ref, M, K, N);

    // timed loop
    hipDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        wmma_mlp_tile<<<grid, block>>>(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("WMMA MLP layer: %.4f ms/iter  (%d iters)\n", ms, iters);

    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
