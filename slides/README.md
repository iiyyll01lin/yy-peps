# Slides — bilingual Marp decks / 投影片(雙語 Marp)

Six core [Marp](https://marp.app/) decks, plus any numbered capstone decks. Each
slide pairs an
**English title/points** with **繁體中文要點**, mirroring the textbook (`../docs`)
and artifact status in (`../results/manifest.json`).

六份 [Marp](https://marp.app/) 投影片,對應課程各視圖。每張投影片:**英文標題/要點**
搭配**繁體中文要點**,與教材(`../docs`)與重現數字(`../results`)一致。

| Deck | Part | Source |
|---|---|---|
| `00_overview.md` | Course overview / 課程總覽 | all |
| `01_foundations.md` | I — Foundations (W01–03) | `docs/01` |
| `02_peps_core.md` | II — PEPS core (W04–06) | `docs/02`, `table1_image.csv` |
| `03_applications.md` | III — Applications (W07–09) | `docs/03`, `table1/2/3_*.csv` |
| `04_quantization.md` | IV — Quantization (W10) | `docs/04`, `w10_rate_distortion.csv` |
| `05_amd_hardware.md` | V — AMD hardware (W11–12) | `docs/05`, `hip_latency.csv` |

## Build / 建置

The decks render with `@marp-team/marp-cli` **4.5.0**, pinned in the Makefile.
You need **Node.js >=18 + npx**; first use requires network access.

投影片以 `@marp-team/marp-cli` 輸出。需 **Node.js + npx**;`npx` 首次執行時抓取
marp-cli(首次需連網),無需全域安裝。

```bash
cd slides

make                     # build every deck -> PDF
make 03_applications.pdf # build a single deck
make pptx                # every deck -> PPTX
make html                # every deck -> HTML
make validate            # every deck -> build/*.html (CPU CI target)
make version             # print marp-cli version (sanity check)
make clean               # remove generated PDF/PPTX/HTML
```

Without `make`, call marp directly (same command the `Makefile` wraps):

不使用 `make` 時,可直接呼叫 marp(與 `Makefile` 內相同):

```bash
npx --yes @marp-team/marp-cli@4.5.0 --html \
  slides/01_foundations.md -o slides/01_foundations.pdf
```

## Style convention / 風格慣例

- Front-matter: `marp: true`, `theme: default`, `paginate: true`, a `title:`.
- Slides separated by `---`; **English title + 繁中要點** on each.
- `<!-- fit -->` for a big closing slide, `<br>` to separate the EN/繁中 blocks.
- Keep each deck focused (~8–14 slides): motivation, key concepts, the target
  protocol or explicitly status-labelled historical output, and an
  **honest-limitations** slide.

All currently tracked result CSVs are `legacy-unverified`. A rendered deck must
not turn them into a verified or paper-reproduction claim.

投影片以 `---` 分隔;每張**英文標題 + 繁中要點**;每份聚焦約 8–14 張,含一張
**誠實限制**投影片。
