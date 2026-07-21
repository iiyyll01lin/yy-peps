# Part IV — Quantization (W10 extension) / 量化延伸

The PEPS paper does not evaluate quantization. W10 therefore poses a new
question—whether projected shared-grid sampling changes PTQ sensitivity—but
treats the explanation as a **hypothesis**, not a result.

PEPS 論文未評估量化。W10 因此提出新問題:projected shared-grid sampling 是否改變
PTQ 敏感度;目前的機制解釋僅是**待驗證假說**,不是研究結論。

> **Artifact status:** `results/w10_rate_distortion.csv` is currently
> `legacy-unverified` in `results/manifest.json`. The values below are historical
> teaching output, not accepted reproduction or extension evidence.

## What is counted / 位元率計算

`peps.quant` defines an explicit packed-model container. A reported rate includes:

- every named model parameter, including fp32 biases or unquantized parameters;
- byte-packed quantized payloads and final-byte padding;
- every scale for per-tensor or per-channel quantization;
- tensor names, ranks, shapes, bit widths, granularity/axis records, payload
  lengths, and the model header.

The primary storage value is `total_encoded_bits`. W10 also reports:

- `bits_per_parameter = total_encoded_bits / total_parameters`;
- `bpp = total_encoded_bits / number_of_pixels`;
- `bpt = total_encoded_bits / number_of_texels`.

`bpt` means **bits per texel** here; bits per token is exposed separately. These
are model rates, not complete codec rates: image headers, entropy coding,
decoder executable code, and runtime activation memory remain outside the model
container and must be reported separately when relevant.

`peps.quant` 以明確封裝格式計數:所有參數、byte-packed payload、scale、tensor 名稱與
shape、量化軸及 model header 都納入。主要數值是 `total_encoded_bits`;並依像素或
texel 數計算 bpp/bpt。這仍是**模型**位元率,不是含影像 header、entropy coding、
decoder 程式與 activation memory 的完整 codec 位元率。

## Required ablations / 必要消融

The generated W10 notebook runs four plans:

1. fp32 reference;
2. symmetric int8, per-tensor latents and weights;
3. symmetric int8, per-channel latents and weights;
4. mixed precision: 6-bit latents and per-channel 8-bit weights.

Biases stay fp32 in the weight-only PTQ plans. Parameter-name glob overrides in
`QuantizationConfig` support further mixed-precision studies without silently
dropping unmatched parameters.

## Evidence threshold / 證據門檻

The baseline is accurately called `grid`, not RTXNTC or the paper's `NTC_N`.
Before training, W10 chooses a PEPS grid resolution with a matched total
parameter count. After encoding, each pair must also be within 2.5% in
`total_encoded_bits`. The protocol uses three fixed seeds and stores every raw
row; only repeated, matched-size paired effects can support a comparative claim.

The legacy schema-v2 CSV currently contains grid **140,163 params** and PEPS
**140,235 params** (0.051% apart). Its recorded three-seed means are:

| plan | bpp grid / PEPS | grid PSNR | PEPS PSNR | paired gap |
|---|---:|---:|---:|---:|
| fp32 | 45.661 / 45.688 | 39.380 | 42.022 | +2.642 dB |
| int8 per-tensor | 11.491 / 11.500 | 36.506 | 41.100 | **+4.594 dB** |
| int8 per-channel | 11.556 / 11.564 | 38.702 | 41.623 | +2.921 dB |
| latent-6 / weight-8 per-channel | 8.887 / 9.019 | 33.678 | 37.825 | +4.147 dB |

Within that legacy artifact, all 12 encoded pairs are inside the 2.5% size
tolerance and every paired gap is positive. Those rows remain unverified until
the complete release-policy manifest and raw evidence are accepted; they do not
currently support a repository claim.

Baseline 應稱為 `grid`,不是 RTXNTC 或論文的 `NTC_N`。legacy schema-v2 CSV 記錄
grid **140,163 參數**與 PEPS **140,235 參數**(差 0.051%)。三 seed 平均:
int8 per-tensor 為 **36.506 / 41.100 dB**(差 **+4.594 dB**),int8 per-channel
為 **38.702 / 41.623 dB**(差 **+2.921 dB**),latent-6 / weight-8 為
**33.678 / 37.825 dB**(差 **+4.147 dB**)。12 組 encoded pair 都通過 2.5%
大小容忍且差距為正;但完整 manifest 與原始證據通過前仍是
`legacy-unverified`,目前不能支撐 repo 結論。

## Interpretation boundary / 解讀邊界

It is plausible that multiple projected samples alter how quantization error
propagates, but “errors average out” has not been isolated experimentally.
The matched-size design is specified, but its checked-in artifact is not yet
accepted as verified. Establishing the mechanism additionally requires a
targeted control that varies sample count/aggregation while holding the decoder,
training budget, and encoded size fixed, plus broader data.

多點 sampling 可能改變量化誤差傳播,但「誤差會平均掉」尚未被實驗隔離。repeated
matched-size 設計已明確,但 checked-in artifact 尚未驗證;要建立因果機制,仍需固定
decoder、training budget 與 encoded size、只改 sample count/aggregation 的控制實驗,
並擴大資料集。
