// fused_peps_kernel.hip — encoder-sample + MLP first layer fused into one kernel.
//
// 繁體中文:把 PEPS 推論路徑的「grid 雙線性取樣」與「MLP 第一層」融合成單一 kernel,
// 示範 kernel fusion 對即時材質解碼的意義:少一次全域記憶體來回。每個 thread 處理
// 一個查詢點:對 2D grid 做雙線性取樣得 latent,直接乘上第一層權重並加 bias、ReLU。
// 這是教學版(未用 WMMA;WMMA 版見 wmma_mlp.hip),重點在展示 fusion 概念與 HIP 基礎。
//
// Build:
//   hipcc --offload-arch=gfx1201 fused_peps_kernel.hip -o fused_peps   # Box B RDNA4
//   hipcc --offload-arch=gfx1151 fused_peps_kernel.hip -o fused_peps   # Box A RDNA3.5

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>

// Bilinear sample of a (Cx H x W) feature grid at normalized (u,v) in [0,1].
__device__ inline void bilinear_sample(const float* grid, int C, int H, int W,
                                        float u, float v, float* out) {
    float fx = u * (W - 1), fy = v * (H - 1);
    int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float wx = fx - x0, wy = fy - y0;
    for (int c = 0; c < C; c++) {
        const float* g = grid + c * H * W;
        float v00 = g[y0 * W + x0], v01 = g[y0 * W + x1];
        float v10 = g[y1 * W + x0], v11 = g[y1 * W + x1];
        float top = v00 * (1 - wx) + v01 * wx;
        float bot = v10 * (1 - wx) + v11 * wx;
        out[c] = top * (1 - wy) + bot * wy;
    }
}

// One thread per query point: sample latent (C) then first MLP layer (C -> Hn) + ReLU.
__global__ void fused_sample_mlp(const float* grid, int C, int H, int W,
                                 const float* coords, int nPoints,
                                 const float* W1, const float* b1, int Hn,
                                 float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nPoints) return;
    float u = coords[i * 2 + 0], v = coords[i * 2 + 1];

    float latent[32];                       // assumes C <= 32 (teaching cap)
    bilinear_sample(grid, C, H, W, u, v, latent);

    for (int j = 0; j < Hn; j++) {
        float acc = b1[j];
        for (int c = 0; c < C; c++) acc += latent[c] * W1[c * Hn + j];
        out[i * Hn + j] = acc > 0.f ? acc : 0.f;   // ReLU
    }
}

int main(int argc, char** argv) {
    int C = 8, H = 128, W = 128, Hn = 64;
    int nPoints = argc > 1 ? atoi(argv[1]) : 1 << 18;   // 262144
    int iters = argc > 2 ? atoi(argv[2]) : 200;

    std::vector<float> grid(C * H * W), coords(nPoints * 2), W1(C * Hn), b1(Hn);
    srand(7);
    for (auto& x : grid) x = (rand() % 200 - 100) / 100.0f;
    for (auto& x : coords) x = (rand() % 1000) / 1000.0f;
    for (auto& x : W1) x = (rand() % 200 - 100) / 100.0f;
    for (auto& x : b1) x = 0.0f;

    float *dGrid, *dCoords, *dW1, *db1, *dOut;
    hipMalloc(&dGrid, grid.size() * 4); hipMalloc(&dCoords, coords.size() * 4);
    hipMalloc(&dW1, W1.size() * 4); hipMalloc(&db1, b1.size() * 4);
    hipMalloc(&dOut, (size_t)nPoints * Hn * 4);
    hipMemcpy(dGrid, grid.data(), grid.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dCoords, coords.data(), coords.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dW1, W1.data(), W1.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(db1, b1.data(), b1.size() * 4, hipMemcpyHostToDevice);

    int block = 256, gridDim = (nPoints + block - 1) / block;
    fused_sample_mlp<<<gridDim, block>>>(dGrid, C, H, W, dCoords, nPoints, dW1, db1, Hn, dOut);
    hipDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fused_sample_mlp<<<gridDim, block>>>(dGrid, C, H, W, dCoords, nPoints, dW1, db1, Hn, dOut);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("fused sample+MLP: %.4f ms/iter  (%d points, C=%d Hn=%d)\n", ms, nPoints, C, Hn);

    hipFree(dGrid); hipFree(dCoords); hipFree(dW1); hipFree(db1); hipFree(dOut);
    return 0;
}
