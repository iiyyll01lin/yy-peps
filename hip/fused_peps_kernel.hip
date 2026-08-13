// fused_peps_kernel.hip — paper-scale fused PEPS inference for AMD GPUs.
//
// The fp32 path is the correctness oracle: one thread executes the complete
// coordinate-to-RGB path:
//   projection -> every shared-grid sample -> concat or paper Pink aggregation
//   -> three 64-wide hidden layers -> output layer.
//
// The fp16 path keeps that entire pipeline in one kernel. One wave builds a
// 16-query feature tile (including selective Pink channel loads), then executes
// all four Linear layers with rocWMMA and applies biases/activations in LDS.
// No encoded feature or hidden activation is written to global memory.
//
// Modes:
//   ./fused_peps fixture [fp32|fp16] in.bin out.bin
//   ./fused_peps benchmark <all|method> [side=1024] [warmup=20] [iters=100]
//   ./fused_peps micro [points=262144] [iters=200]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using rocwmma::float16_t;
using rocwmma::float32_t;

#ifndef PEPS_GIT_SHA
#define PEPS_GIT_SHA "unknown"
#endif
#ifndef PEPS_TARGET_ISA
#define PEPS_TARGET_ISA "unknown"
#endif

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t _e = (cmd);                                                      \
        if (_e != hipSuccess) {                                                     \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),       \
                    __FILE__, __LINE__);                                            \
            return 3;                                                               \
        }                                                                           \
    } while (0)

static constexpr int MODE_BASELINE = 0;
static constexpr int MODE_CONCAT = 1;
static constexpr int MODE_PINK = 2;
static constexpr int ACT_RELU = 0;
static constexpr int ACT_GELU = 1;
static constexpr int ACT_LEAKY_RELU = 2;
static constexpr int MAX_CHANNELS = 32;
static constexpr int MAX_FREQUENCIES = 8;
// LDS tile caps.  These size the four __shared__ tiles in
// integrated_peps_wmma, so they set the per-workgroup LDS footprint and
// therefore the occupancy limit -- independently of the dimensions a run
// actually uses.  They are guarded so a build can narrow them to the
// configuration under test; the defaults below reproduce the original
// 32 KB footprint exactly.  check_config() rejects any run whose
// aggregated input exceeds the cap, so a too-narrow build fails closed
// with a diagnostic instead of reading past the tile.
#ifndef PEPS_MAX_INPUT_DIM
#define PEPS_MAX_INPUT_DIM 512
#endif
#ifndef PEPS_MAX_HIDDEN_DIM
#define PEPS_MAX_HIDDEN_DIM 128
#endif
static constexpr int MAX_INPUT_DIM = PEPS_MAX_INPUT_DIM;
static constexpr int MAX_HIDDEN_DIM = PEPS_MAX_HIDDEN_DIM;
static constexpr int MAX_OUTPUT_DIM = 16;
static constexpr float PEPS_PI = 3.14159265358979323846f;
static constexpr int WMMA_TILE = 16;

#if defined(__gfx1200__) || defined(__gfx1201__)
static constexpr int PEPS_WMMA_GENERATION = 12;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
      defined(__gfx1153__)
static constexpr int PEPS_WMMA_GENERATION = 11;
#else
static constexpr int PEPS_WMMA_GENERATION = 0;
#endif

static const int32_t WORKLOAD_MAGIC = 0x50505332;  // 'PPS2'
static const int32_t WORKLOAD_SCHEMA = 2;

struct WorkloadConfig {
    int mode;
    int channels;
    int grid_h;
    int grid_w;
    int points;
    int frequencies;
    int hidden;
    int output;
    int activation;
};

__host__ __device__ inline int pink_width(int channels, int frequency_index) {
    int width = channels / (1 << frequency_index);
    return width > 0 ? width : 1;
}

__host__ __device__ inline int aggregate_dim(int mode, int channels, int frequencies) {
    if (mode == MODE_BASELINE) return channels;
    if (mode == MODE_CONCAT) return (2 * frequencies + 1) * channels;
    int result = channels;
    for (int i = 1; i <= frequencies; ++i) result += 2 * pink_width(channels, i);
    return result;
}

static const char* mode_name(int mode) {
    if (mode == MODE_BASELINE) return "baseline";
    if (mode == MODE_CONCAT) return "peps";
    if (mode == MODE_PINK) return "pink";
    return "unknown";
}

static int parse_mode(const char* name) {
    if (std::strcmp(name, "baseline") == 0) return MODE_BASELINE;
    if (std::strcmp(name, "peps") == 0 || std::strcmp(name, "concat") == 0)
        return MODE_CONCAT;
    if (std::strcmp(name, "pink") == 0) return MODE_PINK;
    return -1;
}

static int validate_config(const WorkloadConfig& cfg) {
    if (cfg.mode < MODE_BASELINE || cfg.mode > MODE_PINK ||
        cfg.channels < 1 || cfg.channels > MAX_CHANNELS ||
        cfg.grid_h < 2 || cfg.grid_w < 2 || cfg.points < 1 ||
        cfg.frequencies < 0 || cfg.frequencies > MAX_FREQUENCIES ||
        cfg.hidden < 1 || cfg.hidden > MAX_HIDDEN_DIM ||
        cfg.output < 1 || cfg.output > MAX_OUTPUT_DIM ||
        cfg.activation < ACT_RELU || cfg.activation > ACT_LEAKY_RELU) {
        fprintf(stderr, "invalid integrated workload dimensions\n");
        return 2;
    }
    int input = aggregate_dim(cfg.mode, cfg.channels, cfg.frequencies);
    if (input > MAX_INPUT_DIM) {
        fprintf(stderr, "aggregated input %d exceeds cap %d\n", input, MAX_INPUT_DIM);
        return 2;
    }
    return 0;
}

__device__ inline float sample_channel(
    const float* grid, int channels, int height, int width,
    float u, float v, int channel
) {
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float fx = u * float(width - 1);
    float fy = v * float(height - 1);
    int x0 = int(floorf(fx));
    int y0 = int(floorf(fy));
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= width) x0 = width - 1;
    if (y0 >= height) y0 = height - 1;
    int x1 = x0 + 1 < width ? x0 + 1 : width - 1;
    int y1 = y0 + 1 < height ? y0 + 1 : height - 1;
    float wx = fx - float(x0);
    float wy = fy - float(y0);
    const float* values = grid + size_t(channel) * height * width;
    float top = values[y0 * width + x0] * (1.0f - wx)
              + values[y0 * width + x1] * wx;
    float bottom = values[y1 * width + x0] * (1.0f - wx)
                 + values[y1 * width + x1] * wx;
    return top * (1.0f - wy) + bottom * wy;
}

__device__ inline void projected_point(
    float u, float v, int frequency_index, bool cosine,
    float& projected_u, float& projected_v
) {
    float phi = PEPS_PI * float(1 << frequency_index);
    if (cosine) {
        projected_u = 0.5f * (1.0f + cosf(phi * u));
        projected_v = 0.5f * (1.0f + cosf(phi * v));
    } else {
        projected_u = 0.5f * (1.0f + sinf(phi * u));
        projected_v = 0.5f * (1.0f + sinf(phi * v));
    }
}

__device__ inline int positive_mod(int value, int modulus) {
    int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

__device__ inline int fill_aggregated(
    const float* grid, const WorkloadConfig& cfg, float u, float v, float* input
) {
    int cursor = 0;
    for (int channel = 0; channel < cfg.channels; ++channel)
        input[cursor++] = sample_channel(
            grid, cfg.channels, cfg.grid_h, cfg.grid_w, u, v, channel
        );
    if (cfg.mode == MODE_BASELINE) return cursor;

    if (cfg.mode == MODE_CONCAT) {
        for (int trig = 0; trig < 2; ++trig) {
            for (int frequency = 1; frequency <= cfg.frequencies; ++frequency) {
                float pu, pv;
                projected_point(u, v, frequency, trig == 1, pu, pv);
                for (int channel = 0; channel < cfg.channels; ++channel)
                    input[cursor++] = sample_channel(
                        grid, cfg.channels, cfg.grid_h, cfg.grid_w,
                        pu, pv, channel
                    );
            }
        }
        return cursor;
    }

    // Paper Algorithm 1:
    // a_i=floor(C/2^i), G_i=sum_j<=i a_j,
    // sin uses [-G_i:-G_{i-1}], cos uses [G_{i-1}:G_i], circularly.
    int cumulative = 0;
    for (int frequency = 1; frequency <= cfg.frequencies; ++frequency) {
        int width = pink_width(cfg.channels, frequency);
        int next = cumulative + width;
        float pu, pv;
        projected_point(u, v, frequency, false, pu, pv);
        for (int index = -next; index < -cumulative; ++index) {
            int channel = positive_mod(index, cfg.channels);
            input[cursor++] = sample_channel(
                grid, cfg.channels, cfg.grid_h, cfg.grid_w, pu, pv, channel
            );
        }
        cumulative = next;
    }
    cumulative = 0;
    for (int frequency = 1; frequency <= cfg.frequencies; ++frequency) {
        int width = pink_width(cfg.channels, frequency);
        int next = cumulative + width;
        float pu, pv;
        projected_point(u, v, frequency, true, pu, pv);
        for (int index = cumulative; index < next; ++index) {
            int channel = positive_mod(index, cfg.channels);
            input[cursor++] = sample_channel(
                grid, cfg.channels, cfg.grid_h, cfg.grid_w, pu, pv, channel
            );
        }
        cumulative = next;
    }
    return cursor;
}

__device__ inline float activate(float value, int activation) {
    if (activation == ACT_GELU)
        return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    if (activation == ACT_LEAKY_RELU)
        return value >= 0.0f ? value : 0.01f * value;
    return value > 0.0f ? value : 0.0f;
}

__global__ void integrated_peps_workload(
    const float* grid, const float* coords, WorkloadConfig cfg,
    const float* weight1, const float* bias1,
    const float* weight2, const float* bias2,
    const float* weight3, const float* bias3,
    const float* weight4, const float* bias4,
    float* output
) {
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= cfg.points) return;

    float input[MAX_INPUT_DIM];
    float hidden_a[MAX_HIDDEN_DIM];
    float hidden_b[MAX_HIDDEN_DIM];
    float u = coords[point * 2];
    float v = coords[point * 2 + 1];
    int input_dim = fill_aggregated(grid, cfg, u, v, input);

    for (int out = 0; out < cfg.hidden; ++out) {
        float acc = bias1[out];
        for (int in = 0; in < input_dim; ++in)
            acc += input[in] * weight1[in * cfg.hidden + out];
        hidden_a[out] = activate(acc, cfg.activation);
    }
    for (int out = 0; out < cfg.hidden; ++out) {
        float acc = bias2[out];
        for (int in = 0; in < cfg.hidden; ++in)
            acc += hidden_a[in] * weight2[in * cfg.hidden + out];
        hidden_b[out] = activate(acc, cfg.activation);
    }
    for (int out = 0; out < cfg.hidden; ++out) {
        float acc = bias3[out];
        for (int in = 0; in < cfg.hidden; ++in)
            acc += hidden_b[in] * weight3[in * cfg.hidden + out];
        hidden_a[out] = activate(acc, cfg.activation);
    }
    for (int out = 0; out < cfg.output; ++out) {
        float acc = bias4[out];
        for (int in = 0; in < cfg.hidden; ++in)
            acc += hidden_a[in] * weight4[in * cfg.output + out];
        output[point * cfg.output + out] = acc;
    }
}

__device__ inline float sample_half_channel(
    const float16_t* grid, int channels, int height, int width,
    float u, float v, int channel
) {
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float fx = u * float(width - 1);
    float fy = v * float(height - 1);
    int x0 = int(floorf(fx));
    int y0 = int(floorf(fy));
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= width) x0 = width - 1;
    if (y0 >= height) y0 = height - 1;
    int x1 = x0 + 1 < width ? x0 + 1 : width - 1;
    int y1 = y0 + 1 < height ? y0 + 1 : height - 1;
    const float16_t* values = grid + size_t(channel) * height * width;
    float wx = fx - float(x0);
    float wy = fy - float(y0);
    float v00 = float(values[y0 * width + x0]);
    float v01 = float(values[y0 * width + x1]);
    float v10 = float(values[y1 * width + x0]);
    float v11 = float(values[y1 * width + x1]);
    float top = v00 * (1.0f - wx) + v01 * wx;
    float bottom = v10 * (1.0f - wx) + v11 * wx;
    return top * (1.0f - wy) + bottom * wy;
}

// Resolve one aggregate column without materializing unselected Pink channels.
// This is the GPU equivalent of PinkAggregator.channel_indices_for_point().
__device__ inline float selected_feature(
    const float16_t* grid, const WorkloadConfig& cfg,
    float u, float v, int aggregate_column
) {
    if (cfg.mode == MODE_BASELINE) {
        return sample_half_channel(
            grid, cfg.channels, cfg.grid_h, cfg.grid_w,
            u, v, aggregate_column
        );
    }

    if (cfg.mode == MODE_CONCAT) {
        int projected = aggregate_column / cfg.channels;
        int channel = aggregate_column - projected * cfg.channels;
        if (projected == 0) {
            return sample_half_channel(
                grid, cfg.channels, cfg.grid_h, cfg.grid_w,
                u, v, channel
            );
        }
        bool cosine = projected > cfg.frequencies;
        int frequency = cosine
            ? projected - cfg.frequencies
            : projected;
        float pu, pv;
        projected_point(u, v, frequency, cosine, pu, pv);
        return sample_half_channel(
            grid, cfg.channels, cfg.grid_h, cfg.grid_w,
            pu, pv, channel
        );
    }

    if (aggregate_column < cfg.channels) {
        return sample_half_channel(
            grid, cfg.channels, cfg.grid_h, cfg.grid_w,
            u, v, aggregate_column
        );
    }
    int column = aggregate_column - cfg.channels;
    int cumulative = 0;
    for (int trig = 0; trig < 2; ++trig) {
        for (int frequency = 1; frequency <= cfg.frequencies; ++frequency) {
            int width = pink_width(cfg.channels, frequency);
            if (column < width) {
                int channel = trig == 0
                    ? positive_mod(-(cumulative + width) + column, cfg.channels)
                    : positive_mod(cumulative + column, cfg.channels);
                float pu, pv;
                projected_point(
                    u, v, frequency, trig == 1, pu, pv
                );
                return sample_half_channel(
                    grid, cfg.channels, cfg.grid_h, cfg.grid_w,
                    pu, pv, channel
                );
            }
            column -= width;
            cumulative += width;
        }
        cumulative = 0;
    }
    return 0.0f;
}

__device__ inline void wmma_hidden_layer(
    const float16_t* input, int input_dim,
    const float16_t* weight, int output_dim,
    const float* bias, int activation,
    float16_t* output, float* accumulator_buffer
) {
    using namespace rocwmma;
    for (int column = 0; column < output_dim; column += WMMA_TILE) {
        auto frag_a = fragment<
            matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE,
            float16_t, row_major
        >();
        auto frag_b = fragment<
            matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE,
            float16_t, row_major
        >();
        auto frag_c = fragment<
            accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float32_t
        >();
        fill_fragment(frag_c, 0.0f);
        for (int k = 0; k < input_dim; k += WMMA_TILE) {
            load_matrix_sync(frag_a, input + k, input_dim);
            load_matrix_sync(
                frag_b, weight + k * output_dim + column, output_dim
            );
            mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        store_matrix_sync(
            accumulator_buffer + column, frag_c, output_dim, mem_row_major
        );
    }
    __syncthreads();
    for (
        int index = int(threadIdx.x);
        index < WMMA_TILE * output_dim;
        index += int(blockDim.x)
    ) {
        int column = index % output_dim;
        output[index] = float16_t(
            activate(accumulator_buffer[index] + bias[column], activation)
        );
    }
    __syncthreads();
}

__device__ inline void wmma_output_layer(
    const float16_t* input, int input_dim,
    const float16_t* weight, int padded_output,
    const float* bias, int output_dim,
    float* output, int output_stride, int first_point,
    int points, float* accumulator_buffer
) {
    using namespace rocwmma;
    for (int column = 0; column < padded_output; column += WMMA_TILE) {
        auto frag_a = fragment<
            matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE,
            float16_t, row_major
        >();
        auto frag_b = fragment<
            matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE,
            float16_t, row_major
        >();
        auto frag_c = fragment<
            accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float32_t
        >();
        fill_fragment(frag_c, 0.0f);
        for (int k = 0; k < input_dim; k += WMMA_TILE) {
            load_matrix_sync(frag_a, input + k, input_dim);
            load_matrix_sync(
                frag_b, weight + k * padded_output + column, padded_output
            );
            mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        store_matrix_sync(
            accumulator_buffer + column, frag_c, padded_output, mem_row_major
        );
    }
    __syncthreads();
    for (
        int index = int(threadIdx.x);
        index < WMMA_TILE * output_dim;
        index += int(blockDim.x)
    ) {
        int row = index / output_dim;
        int column = index - row * output_dim;
        if (first_point + row < points) {
            output[(first_point + row) * output_stride + column] =
                accumulator_buffer[row * padded_output + column] + bias[column];
        }
    }
}

// One block is exactly one wave and computes sixteen complete RGB queries.
// Feature and hidden matrices stay in LDS; the only global output is RGB.
__global__ void integrated_peps_wmma(
    const float16_t* grid, const float* coords, WorkloadConfig cfg,
    const float16_t* weight1, const float* bias1,
    const float16_t* weight2, const float* bias2,
    const float16_t* weight3, const float* bias3,
    const float16_t* weight4, const float* bias4,
    int padded_input, int padded_hidden, int padded_output,
    float* output
) {
    __shared__ float16_t feature_tile[WMMA_TILE * MAX_INPUT_DIM];
    __shared__ float16_t hidden_a[WMMA_TILE * MAX_HIDDEN_DIM];
    __shared__ float16_t hidden_b[WMMA_TILE * MAX_HIDDEN_DIM];
    __shared__ float accumulator[WMMA_TILE * MAX_HIDDEN_DIM];

    int first_point = int(blockIdx.x) * WMMA_TILE;
    int input_dim = aggregate_dim(
        cfg.mode, cfg.channels, cfg.frequencies
    );
    for (
        int index = int(threadIdx.x);
        index < WMMA_TILE * padded_input;
        index += int(blockDim.x)
    ) {
        int row = index / padded_input;
        int column = index - row * padded_input;
        int point = first_point + row;
        float value = 0.0f;
        if (point < cfg.points && column < input_dim) {
            float u = coords[point * 2];
            float v = coords[point * 2 + 1];
            value = selected_feature(grid, cfg, u, v, column);
        }
        feature_tile[index] = float16_t(value);
    }
    __syncthreads();

    wmma_hidden_layer(
        feature_tile, padded_input, weight1, padded_hidden,
        bias1, cfg.activation, hidden_a, accumulator
    );
    wmma_hidden_layer(
        hidden_a, padded_hidden, weight2, padded_hidden,
        bias2, cfg.activation, hidden_b, accumulator
    );
    wmma_hidden_layer(
        hidden_b, padded_hidden, weight3, padded_hidden,
        bias3, cfg.activation, hidden_a, accumulator
    );
    wmma_output_layer(
        hidden_a, padded_hidden, weight4, padded_output,
        bias4, cfg.output, output, cfg.output,
        first_point, cfg.points, accumulator
    );
}

struct HostWorkload {
    std::vector<float> grid, coords;
    std::vector<float> weight1, bias1, weight2, bias2;
    std::vector<float> weight3, bias3, weight4, bias4;
};

struct DeviceWorkload {
    float *grid = nullptr, *coords = nullptr;
    float *weight1 = nullptr, *bias1 = nullptr, *weight2 = nullptr, *bias2 = nullptr;
    float *weight3 = nullptr, *bias3 = nullptr, *weight4 = nullptr, *bias4 = nullptr;
    float* output = nullptr;
};

struct HostWmmaWorkload {
    std::vector<float16_t> grid;
    std::vector<float> coords;
    std::vector<float16_t> weight1, weight2, weight3, weight4;
    std::vector<float> bias1, bias2, bias3, bias4;
    int padded_input = 0;
    int padded_hidden = 0;
    int padded_output = 0;
};

struct DeviceWmmaWorkload {
    float16_t *grid = nullptr;
    float *coords = nullptr;
    float16_t *weight1 = nullptr, *weight2 = nullptr;
    float16_t *weight3 = nullptr, *weight4 = nullptr;
    float *bias1 = nullptr, *bias2 = nullptr;
    float *bias3 = nullptr, *bias4 = nullptr;
    float* output = nullptr;
};

static void release_device_allocation(void* pointer) {
    if (pointer) {
        hipError_t ignored = hipFree(pointer);
        (void)ignored;
    }
}

static void release_event(hipEvent_t event) {
    if (event) {
        hipError_t ignored = hipEventDestroy(event);
        (void)ignored;
    }
}

static int round_up_wmma(int value) {
    return (value + WMMA_TILE - 1) / WMMA_TILE * WMMA_TILE;
}

static std::vector<float16_t> pad_half_matrix(
    const std::vector<float>& source,
    int rows, int columns, int padded_rows, int padded_columns
) {
    std::vector<float16_t> result(
        size_t(padded_rows) * padded_columns, float16_t(0.0f)
    );
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            result[size_t(row) * padded_columns + column] =
                float16_t(source[size_t(row) * columns + column]);
        }
    }
    return result;
}

static std::vector<float> pad_bias(
    const std::vector<float>& source, int padded_size
) {
    std::vector<float> result(padded_size, 0.0f);
    std::copy(source.begin(), source.end(), result.begin());
    return result;
}

static HostWmmaWorkload make_wmma_host(
    const HostWorkload& host, const WorkloadConfig& cfg
) {
    HostWmmaWorkload result;
    int input_dim = aggregate_dim(cfg.mode, cfg.channels, cfg.frequencies);
    result.padded_input = round_up_wmma(input_dim);
    result.padded_hidden = round_up_wmma(cfg.hidden);
    result.padded_output = round_up_wmma(cfg.output);
    result.grid.reserve(host.grid.size());
    for (float value : host.grid) result.grid.push_back(float16_t(value));
    result.coords = host.coords;
    result.weight1 = pad_half_matrix(
        host.weight1, input_dim, cfg.hidden,
        result.padded_input, result.padded_hidden
    );
    result.weight2 = pad_half_matrix(
        host.weight2, cfg.hidden, cfg.hidden,
        result.padded_hidden, result.padded_hidden
    );
    result.weight3 = pad_half_matrix(
        host.weight3, cfg.hidden, cfg.hidden,
        result.padded_hidden, result.padded_hidden
    );
    result.weight4 = pad_half_matrix(
        host.weight4, cfg.hidden, cfg.output,
        result.padded_hidden, result.padded_output
    );
    result.bias1 = pad_bias(host.bias1, result.padded_hidden);
    result.bias2 = pad_bias(host.bias2, result.padded_hidden);
    result.bias3 = pad_bias(host.bias3, result.padded_hidden);
    result.bias4 = pad_bias(host.bias4, result.padded_output);
    return result;
}

static void free_device(DeviceWorkload& device) {
    release_device_allocation(device.grid);
    release_device_allocation(device.coords);
    release_device_allocation(device.weight1);
    release_device_allocation(device.bias1);
    release_device_allocation(device.weight2);
    release_device_allocation(device.bias2);
    release_device_allocation(device.weight3);
    release_device_allocation(device.bias3);
    release_device_allocation(device.weight4);
    release_device_allocation(device.bias4);
    release_device_allocation(device.output);
}

static void free_wmma_device(DeviceWmmaWorkload& device) {
    release_device_allocation(device.grid);
    release_device_allocation(device.coords);
    release_device_allocation(device.weight1);
    release_device_allocation(device.bias1);
    release_device_allocation(device.weight2);
    release_device_allocation(device.bias2);
    release_device_allocation(device.weight3);
    release_device_allocation(device.bias3);
    release_device_allocation(device.weight4);
    release_device_allocation(device.bias4);
    release_device_allocation(device.output);
}

static int upload_wmma(
    const HostWmmaWorkload& host, const WorkloadConfig& cfg,
    DeviceWmmaWorkload& device
) {
    HIP_CHECK(hipMalloc(&device.grid, host.grid.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&device.coords, host.coords.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device.weight1, host.weight1.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&device.bias1, host.bias1.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device.weight2, host.weight2.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&device.bias2, host.bias2.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device.weight3, host.weight3.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&device.bias3, host.bias3.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&device.weight4, host.weight4.size() * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&device.bias4, host.bias4.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &device.output, size_t(cfg.points) * cfg.output * sizeof(float)
    ));

#define COPY_WMMA(field, type) HIP_CHECK(hipMemcpy(                              \
    device.field, host.field.data(), host.field.size() * sizeof(type),           \
    hipMemcpyHostToDevice))
    COPY_WMMA(grid, float16_t); COPY_WMMA(coords, float);
    COPY_WMMA(weight1, float16_t); COPY_WMMA(bias1, float);
    COPY_WMMA(weight2, float16_t); COPY_WMMA(bias2, float);
    COPY_WMMA(weight3, float16_t); COPY_WMMA(bias3, float);
    COPY_WMMA(weight4, float16_t); COPY_WMMA(bias4, float);
#undef COPY_WMMA
    return 0;
}

static int launch_wmma(
    const DeviceWmmaWorkload& device, const HostWmmaWorkload& host,
    const WorkloadConfig& cfg
) {
    int blocks = (cfg.points + WMMA_TILE - 1) / WMMA_TILE;
    int wave_size = rocwmma::Constants::AMDGCN_WAVE_SIZE;
    integrated_peps_wmma<<<blocks, wave_size>>>(
        device.grid, device.coords, cfg,
        device.weight1, device.bias1,
        device.weight2, device.bias2,
        device.weight3, device.bias3,
        device.weight4, device.bias4,
        host.padded_input, host.padded_hidden, host.padded_output,
        device.output
    );
    HIP_CHECK(hipGetLastError());
    return 0;
}

static int upload(
    const HostWorkload& host, const WorkloadConfig& cfg, DeviceWorkload& device
) {
    const size_t f = sizeof(float);
    HIP_CHECK(hipMalloc(&device.grid, host.grid.size() * f));
    HIP_CHECK(hipMalloc(&device.coords, host.coords.size() * f));
    HIP_CHECK(hipMalloc(&device.weight1, host.weight1.size() * f));
    HIP_CHECK(hipMalloc(&device.bias1, host.bias1.size() * f));
    HIP_CHECK(hipMalloc(&device.weight2, host.weight2.size() * f));
    HIP_CHECK(hipMalloc(&device.bias2, host.bias2.size() * f));
    HIP_CHECK(hipMalloc(&device.weight3, host.weight3.size() * f));
    HIP_CHECK(hipMalloc(&device.bias3, host.bias3.size() * f));
    HIP_CHECK(hipMalloc(&device.weight4, host.weight4.size() * f));
    HIP_CHECK(hipMalloc(&device.bias4, host.bias4.size() * f));
    HIP_CHECK(hipMalloc(&device.output, size_t(cfg.points) * cfg.output * f));

#define COPY_FIELD(field) HIP_CHECK(hipMemcpy(                                      \
    device.field, host.field.data(), host.field.size() * f, hipMemcpyHostToDevice))
    COPY_FIELD(grid); COPY_FIELD(coords);
    COPY_FIELD(weight1); COPY_FIELD(bias1);
    COPY_FIELD(weight2); COPY_FIELD(bias2);
    COPY_FIELD(weight3); COPY_FIELD(bias3);
    COPY_FIELD(weight4); COPY_FIELD(bias4);
#undef COPY_FIELD
    return 0;
}

static int launch_integrated(const DeviceWorkload& device, const WorkloadConfig& cfg) {
    int block = 256;
    int blocks = (cfg.points + block - 1) / block;
    integrated_peps_workload<<<blocks, block>>>(
        device.grid, device.coords, cfg,
        device.weight1, device.bias1, device.weight2, device.bias2,
        device.weight3, device.bias3, device.weight4, device.bias4,
        device.output
    );
    HIP_CHECK(hipGetLastError());
    return 0;
}

static int run_once(
    const HostWorkload& host, const WorkloadConfig& cfg, std::vector<float>& output
) {
    if (validate_config(cfg)) return 2;
    DeviceWorkload device;
    int rc = upload(host, cfg, device);
    if (rc) { free_device(device); return rc; }
    rc = launch_integrated(device, cfg);
    if (!rc) {
        HIP_CHECK(hipDeviceSynchronize());
        output.resize(size_t(cfg.points) * cfg.output);
        HIP_CHECK(hipMemcpy(
            output.data(), device.output, output.size() * sizeof(float),
            hipMemcpyDeviceToHost
        ));
    }
    free_device(device);
    return rc;
}

static int run_once_wmma(
    const HostWorkload& host, const WorkloadConfig& cfg,
    std::vector<float>& output
) {
    if (validate_config(cfg)) return 2;
    HostWmmaWorkload half_host = make_wmma_host(host, cfg);
    DeviceWmmaWorkload device;
    int rc = upload_wmma(half_host, cfg, device);
    if (rc) {
        free_wmma_device(device);
        return rc;
    }
    rc = launch_wmma(device, half_host, cfg);
    if (!rc) {
        HIP_CHECK(hipDeviceSynchronize());
        output.resize(size_t(cfg.points) * cfg.output);
        HIP_CHECK(hipMemcpy(
            output.data(), device.output, output.size() * sizeof(float),
            hipMemcpyDeviceToHost
        ));
    }
    free_wmma_device(device);
    return rc;
}

static bool read_floats(FILE* file, std::vector<float>& values, size_t count) {
    values.resize(count);
    return fread(values.data(), sizeof(float), count, file) == count;
}

// Fixture input header (11 int32):
// magic, schema, mode, C, H, W, N, L, hidden, output, activation.
// Payload: grid, coords, W1,b1,W2,b2,W3,b3,W4,b4 as float32. Matrices are
// input-major ([in,out]), matching NumPy/PyTorch transposed Linear weights.
// Output header: magic, schema, mode, N, output; followed by float32 results.
static int run_fixture(
    const char* input_path, const char* output_path, bool use_wmma
) {
    FILE* file = fopen(input_path, "rb");
    if (!file) { fprintf(stderr, "cannot open %s\n", input_path); return 2; }
    int32_t header[11];
    if (fread(header, sizeof(int32_t), 11, file) != 11 ||
        header[0] != WORKLOAD_MAGIC || header[1] != WORKLOAD_SCHEMA) {
        fprintf(stderr, "bad integrated fixture header\n");
        fclose(file);
        return 2;
    }
    WorkloadConfig cfg{
        header[2], header[3], header[4], header[5], header[6],
        header[7], header[8], header[9], header[10]
    };
    if (validate_config(cfg)) { fclose(file); return 2; }
    int input_dim = aggregate_dim(cfg.mode, cfg.channels, cfg.frequencies);
    HostWorkload host;
    bool ok =
        read_floats(file, host.grid, size_t(cfg.channels) * cfg.grid_h * cfg.grid_w) &&
        read_floats(file, host.coords, size_t(cfg.points) * 2) &&
        read_floats(file, host.weight1, size_t(input_dim) * cfg.hidden) &&
        read_floats(file, host.bias1, cfg.hidden) &&
        read_floats(file, host.weight2, size_t(cfg.hidden) * cfg.hidden) &&
        read_floats(file, host.bias2, cfg.hidden) &&
        read_floats(file, host.weight3, size_t(cfg.hidden) * cfg.hidden) &&
        read_floats(file, host.bias3, cfg.hidden) &&
        read_floats(file, host.weight4, size_t(cfg.hidden) * cfg.output) &&
        read_floats(file, host.bias4, cfg.output);
    fclose(file);
    if (!ok) { fprintf(stderr, "short integrated fixture payload\n"); return 2; }

    std::vector<float> output;
    int rc = use_wmma
        ? run_once_wmma(host, cfg, output)
        : run_once(host, cfg, output);
    if (rc) return rc;
    FILE* result = fopen(output_path, "wb");
    if (!result) { fprintf(stderr, "cannot open %s\n", output_path); return 2; }
    int32_t output_header[5] = {
        WORKLOAD_MAGIC, WORKLOAD_SCHEMA, cfg.mode, cfg.points, cfg.output
    };
    bool wrote =
        fwrite(output_header, sizeof(int32_t), 5, result) == 5 &&
        fwrite(output.data(), sizeof(float), output.size(), result) == output.size();
    fclose(result);
    if (!wrote) { fprintf(stderr, "short write to %s\n", output_path); return 2; }
    printf(
        "integrated fixture mode=%s implementation=%s points=%d output=%d\n",
        mode_name(cfg.mode), use_wmma ? "fused_rocwmma_fp16" : "scalar_fp32",
        cfg.points, cfg.output
    );
    return 0;
}

static float random_value(float scale) {
    return float((rand() % 2001) - 1000) * scale;
}

static HostWorkload make_paper_workload(const WorkloadConfig& cfg, int side) {
    int input_dim = aggregate_dim(cfg.mode, cfg.channels, cfg.frequencies);
    HostWorkload host;
    host.grid.resize(size_t(cfg.channels) * cfg.grid_h * cfg.grid_w);
    host.coords.resize(size_t(cfg.points) * 2);
    host.weight1.resize(size_t(input_dim) * cfg.hidden);
    host.bias1.resize(cfg.hidden);
    host.weight2.resize(size_t(cfg.hidden) * cfg.hidden);
    host.bias2.resize(cfg.hidden);
    host.weight3.resize(size_t(cfg.hidden) * cfg.hidden);
    host.bias3.resize(cfg.hidden);
    host.weight4.resize(size_t(cfg.hidden) * cfg.output);
    host.bias4.resize(cfg.output);
    srand(20260427 + cfg.mode);
    for (float& value : host.grid) value = random_value(0.001f);
    for (float& value : host.weight1) value = random_value(0.0001f);
    for (float& value : host.weight2) value = random_value(0.0001f);
    for (float& value : host.weight3) value = random_value(0.0001f);
    for (float& value : host.weight4) value = random_value(0.0001f);
    for (float& value : host.bias1) value = random_value(0.00001f);
    for (float& value : host.bias2) value = random_value(0.00001f);
    for (float& value : host.bias3) value = random_value(0.00001f);
    for (float& value : host.bias4) value = random_value(0.00001f);
    float denominator = side > 1 ? float(side - 1) : 1.0f;
    for (int point = 0; point < cfg.points; ++point) {
        host.coords[point * 2] = float(point % side) / denominator;
        host.coords[point * 2 + 1] = float(point / side) / denominator;
    }
    return host;
}

static int run_paper_workload(int mode, int side, int iters) {
    if (side < 1 || iters < 1) {
        fprintf(stderr, "side and iters must be positive\n");
        return 2;
    }
    WorkloadConfig cfg{
        mode, 16, 1024, 1024, side * side, 3, 64, 3, ACT_GELU
    };
    if (validate_config(cfg)) return 2;
    HostWorkload host = make_paper_workload(cfg, side);
    DeviceWorkload device;
    int rc = upload(host, cfg, device);
    if (rc) { free_device(device); return rc; }
    for (int warmup = 0; warmup < 5; ++warmup) {
        rc = launch_integrated(device, cfg);
        if (rc) { free_device(device); return rc; }
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int iteration = 0; iteration < iters; ++iteration) {
        rc = launch_integrated(device, cfg);
        if (rc) { free_device(device); return rc; }
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    std::vector<float> checksum(cfg.output);
    HIP_CHECK(hipMemcpy(
        checksum.data(), device.output, cfg.output * sizeof(float),
        hipMemcpyDeviceToHost
    ));
    release_event(start);
    release_event(stop);
    free_device(device);
    float sum = 0.0f;
    for (float value : checksum) sum += value;
    printf(
        "integrated paper-workload mode=%s implementation=scalar_fp32 "
        "side=%d grid=1024 C=16 L=3 hidden=64x3 out=3 activation=gelu "
        "input=%d: %.4f ms/iter (%d iters, checksum=%.6g)\n",
        mode_name(mode), side, aggregate_dim(mode, 16, 3),
        elapsed_ms / iters, iters, sum
    );
    return 0;
}

struct BenchmarkMethod {
    const char* name;
    int mode;
    int frequencies;
    float paper_ms;
};

static const BenchmarkMethod BENCHMARK_METHODS[] = {
    {"bi-grid", MODE_BASELINE, 0, 4.32f},
    {"grid-peps-3f", MODE_CONCAT, 3, 5.47f},
    {"grid-pink-peps-3f", MODE_PINK, 3, 4.86f},
    {"grid-pink-peps-4f", MODE_PINK, 4, 4.99f},
};

static const BenchmarkMethod* find_benchmark_method(const char* name) {
    for (const auto& method : BENCHMARK_METHODS) {
        if (std::strcmp(name, method.name) == 0) return &method;
    }
    return nullptr;
}

static std::string json_escape(const char* input) {
    std::string result;
    for (const unsigned char* p =
             reinterpret_cast<const unsigned char*>(input); *p; ++p) {
        switch (*p) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (*p < 0x20) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", int(*p));
                    result += escaped;
                } else {
                    result += char(*p);
                }
        }
    }
    return result;
}

static float percentile_sorted(const std::vector<float>& sorted, float quantile) {
    size_t index = size_t(ceilf(quantile * float(sorted.size()))) - 1;
    if (index >= sorted.size()) index = sorted.size() - 1;
    return sorted[index];
}

static int run_wmma_benchmark(
    const BenchmarkMethod& method, int side, int warmup, int iters,
    int channels = 16
) {
    if (side < 2 || side > 32768 || warmup < 0 || iters < 1) {
        fprintf(stderr, "invalid benchmark side/warmup/iters\n");
        return 2;
    }
    WorkloadConfig cfg{
        method.mode, channels, 1024, 1024, side * side,
        method.frequencies, 64, 3, ACT_GELU
    };
    if (validate_config(cfg)) return 2;
    HostWorkload full_host = make_paper_workload(cfg, side);
    HostWmmaWorkload half_host = make_wmma_host(full_host, cfg);
    DeviceWmmaWorkload device;
    int rc = upload_wmma(half_host, cfg, device);
    if (rc) {
        free_wmma_device(device);
        return rc;
    }
    for (int iteration = 0; iteration < warmup; ++iteration) {
        rc = launch_wmma(device, half_host, cfg);
        if (rc) {
            free_wmma_device(device);
            return rc;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(iters);
    for (int iteration = 0; iteration < iters; ++iteration) {
        HIP_CHECK(hipEventRecord(start));
        rc = launch_wmma(device, half_host, cfg);
        if (rc) {
            release_event(start);
            release_event(stop);
            free_wmma_device(device);
            return rc;
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(milliseconds);
    }
    std::vector<float> checksum(cfg.output);
    HIP_CHECK(hipMemcpy(
        checksum.data(), device.output, checksum.size() * sizeof(float),
        hipMemcpyDeviceToHost
    ));
    release_event(start);
    release_event(stop);
    free_wmma_device(device);

    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
    float mean = sum / float(samples.size());
    float squared = 0.0f;
    for (float value : samples) {
        float delta = value - mean;
        squared += delta * delta;
    }
    float stddev = sqrtf(squared / float(samples.size()));
    float median = sorted.size() % 2
        ? sorted[sorted.size() / 2]
        : 0.5f * (
            sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]
        );
    float p95 = percentile_sorted(sorted, 0.95f);
    float output_checksum =
        std::accumulate(checksum.begin(), checksum.end(), 0.0f);

    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    int runtime_version = 0;
    int driver_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));
    printf(
        "{"
        "\"schema_version\":3,"
        "\"benchmark_kind\":\"integrated_paper_workload\","
        "\"method\":\"%s\","
        "\"mode\":\"%s\","
        "\"implementation\":\"fused_rocwmma_fp16\","
        "\"precision\":\"fp16_fp32_accumulate\","
        "\"timing\":\"hip_events\","
        "\"isa\":\"%s\","
        "\"compiled_target\":\"%s\","
        "\"git_sha\":\"%s\","
        "\"gpu_name\":\"%s\","
        "\"runtime_version\":%d,"
        "\"driver_version\":%d,"
        "\"wave_size\":%d,"
        "\"compute_units\":%d,"
        // The name is kept for schema compatibility, but the value is a WGP
        // count on RDNA and reading it as compute units halves the part.
        "\"compute_units_semantics\":\"hipDeviceProp_t multiProcessorCount; "
        "on RDNA this counts WGPs and each WGP is two compute units\","
        "\"output_width\":%d,"
        "\"output_height\":%d,"
        "\"grid_width\":1024,"
        "\"grid_height\":1024,"
        "\"feature_dim\":%d,"
        "\"num_frequencies\":%d,"
        "\"selected_feature_dim\":%d,"
        "\"hidden_dim\":%d,"
        "\"hidden_layers\":3,"
        "\"out_dim\":3,"
        "\"activation\":\"gelu\","
        "\"warmup\":%d,"
        "\"iters\":%d,"
        "\"median_ms\":%.6f,"
        "\"p95_ms\":%.6f,"
        "\"mean_ms\":%.6f,"
        "\"stddev_ms\":%.6f,"
        "\"min_ms\":%.6f,"
        "\"max_ms\":%.6f,"
        "\"paper_reference_ms\":%.6f,"
        "\"paper_reference_source\":\"arXiv:2604.24167v1\","
        "\"comparable_to_paper\":false,"
        "\"checksum\":%.9g"
        "}\n",
        method.name, mode_name(method.mode),
        json_escape(properties.gcnArchName).c_str(), PEPS_TARGET_ISA,
        PEPS_GIT_SHA, json_escape(properties.name).c_str(),
        runtime_version, driver_version, properties.warpSize,
        properties.multiProcessorCount, side, side, cfg.channels,
        cfg.frequencies,
        aggregate_dim(cfg.mode, cfg.channels, cfg.frequencies), cfg.hidden,
        warmup, iters, median, p95, mean, stddev,
        sorted.front(), sorted.back(), method.paper_ms, output_checksum
    );
    return 0;
}

// Supplementary diagnostic retained from the original course kernel:
// one grid sample plus only the first ReLU layer.
__global__ void micro_sample_first_layer(
    const float* grid, int channels, int height, int width,
    const float* coords, int points,
    const float* weight, const float* bias, int hidden, float* output
) {
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= points) return;
    float u = coords[point * 2], v = coords[point * 2 + 1];
    for (int out = 0; out < hidden; ++out) {
        float acc = bias[out];
        for (int channel = 0; channel < channels; ++channel)
            acc += sample_channel(
                grid, channels, height, width, u, v, channel
            ) * weight[channel * hidden + out];
        output[point * hidden + out] = acc > 0.0f ? acc : 0.0f;
    }
}

static int run_micro(int points, int iters) {
    if (points < 1 || iters < 1) return 2;
    int channels = 8, height = 128, width = 128, hidden = 64;
    std::vector<float> grid(size_t(channels) * height * width);
    std::vector<float> coords(size_t(points) * 2);
    std::vector<float> weight(size_t(channels) * hidden), bias(hidden, 0.0f);
    srand(7);
    for (float& value : grid) value = random_value(0.001f);
    for (float& value : coords) value = float(rand() % 1001) / 1000.0f;
    for (float& value : weight) value = random_value(0.001f);
    float *d_grid, *d_coords, *d_weight, *d_bias, *d_output;
    HIP_CHECK(hipMalloc(&d_grid, grid.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_coords, coords.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, bias.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, size_t(points) * hidden * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_grid, grid.data(), grid.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_coords, coords.data(), coords.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice));
    int block = 256, blocks = (points + block - 1) / block;
    for (int warmup = 0; warmup < 5; ++warmup)
        micro_sample_first_layer<<<blocks, block>>>(
            d_grid, channels, height, width, d_coords, points,
            d_weight, d_bias, hidden, d_output
        );
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int iteration = 0; iteration < iters; ++iteration)
        micro_sample_first_layer<<<blocks, block>>>(
            d_grid, channels, height, width, d_coords, points,
            d_weight, d_bias, hidden, d_output
        );
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipGetLastError());
    printf("micro sample+first-layer: %.4f ms/iter (%d points, C=%d hidden=%d)\n",
           elapsed_ms / iters, points, channels, hidden);
    release_event(start);
    release_event(stop);
    release_device_allocation(d_grid);
    release_device_allocation(d_coords);
    release_device_allocation(d_weight);
    release_device_allocation(d_bias);
    release_device_allocation(d_output);
    return 0;
}

static void usage(const char* program) {
    fprintf(stderr,
        "usage:\n"
        "  %s fixture [fp32|fp16] <in.bin> <out.bin>\n"
        "  %s benchmark <all|bi-grid|grid-peps-3f|"
        "grid-pink-peps-3f|grid-pink-peps-4f> [side] [warmup] [iters]\n"
        "  %s geometry <baseline|peps|pink> <channels> <frequencies>"
        " [side] [warmup] [iters]\n"
        "  %s workload <baseline|peps|pink|all> [side] [iters]\n"
        "  %s micro [points] [iters]\n",
        program, program, program, program, program
    );
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    std::string command = argv[1];
    if (command == "fixture") {
        if (argc == 4) return run_fixture(argv[2], argv[3], false);
        if (argc != 5) { usage(argv[0]); return 2; }
        bool use_wmma = std::strcmp(argv[2], "fp16") == 0;
        if (!use_wmma && std::strcmp(argv[2], "fp32") != 0) {
            usage(argv[0]);
            return 2;
        }
        return run_fixture(argv[3], argv[4], use_wmma);
    }
    if (command == "benchmark") {
        if (argc < 3) { usage(argv[0]); return 2; }
        int side = argc > 3 ? atoi(argv[3]) : 1024;
        int warmup = argc > 4 ? atoi(argv[4]) : 20;
        int iters = argc > 5 ? atoi(argv[5]) : 100;
        if (std::strcmp(argv[2], "all") == 0) {
            for (const auto& method : BENCHMARK_METHODS) {
                int rc = run_wmma_benchmark(method, side, warmup, iters);
                if (rc) return rc;
            }
            return 0;
        }
        const BenchmarkMethod* method = find_benchmark_method(argv[2]);
        if (!method) { usage(argv[0]); return 2; }
        return run_wmma_benchmark(*method, side, warmup, iters);
    }
    if (command == "geometry") {
        if (argc < 5) { usage(argv[0]); return 2; }
        int mode = parse_mode(argv[2]);
        if (mode < 0) { usage(argv[0]); return 2; }
        int channels = atoi(argv[3]);
        int frequencies = atoi(argv[4]);
        int side = argc > 5 ? atoi(argv[5]) : 1024;
        int warmup = argc > 6 ? atoi(argv[6]) : 20;
        int iters = argc > 7 ? atoi(argv[7]) : 100;
        char label[64];
        std::snprintf(
            label, sizeof(label), "geometry-%s-c%d-%df",
            mode_name(mode), channels, frequencies
        );
        // An arbitrary geometry has no counterpart in the paper's table.
        BenchmarkMethod method{label, mode, frequencies, 0.0f};
        return run_wmma_benchmark(method, side, warmup, iters, channels);
    }
    if (command == "workload") {
        if (argc < 3) { usage(argv[0]); return 2; }
        int side = argc > 3 ? atoi(argv[3]) : 1024;
        int iters = argc > 4 ? atoi(argv[4]) : 20;
        if (std::strcmp(argv[2], "all") == 0) {
            for (int mode = MODE_BASELINE; mode <= MODE_PINK; ++mode) {
                int rc = run_paper_workload(mode, side, iters);
                if (rc) return rc;
            }
            return 0;
        }
        int mode = parse_mode(argv[2]);
        if (mode < 0) { usage(argv[0]); return 2; }
        return run_paper_workload(mode, side, iters);
    }
    if (command == "micro") {
        int points = argc > 2 ? atoi(argv[2]) : 1 << 18;
        int iters = argc > 3 ? atoi(argv[3]) : 200;
        return run_micro(points, iters);
    }
    // Backward-compatible numeric invocation remains the supplementary microbench.
    char* end = nullptr;
    long points = std::strtol(argv[1], &end, 10);
    if (end && *end == '\0') {
        int iters = argc > 2 ? atoi(argv[2]) : 200;
        return run_micro(int(points), iters);
    }
    usage(argv[0]);
    return 2;
}
