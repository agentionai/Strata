# Gyro (agentionai rotor quants) on Strata

Validated on Linux with an RTX 5090 (32 GB, CUDA 13.0, sm_120; 2026-10-02 and 2026-10-03), an RTX 3090 (24 GB,
sm_86; 2026-10-03) and two RTX 4090s (24 GB each, sm_89; 2026-10-04, single and dual GPU). A short end-to-end run on AMD (Strix Halo, gfx1151, ROCm) passed on 2026-10-03.

The targets are the published files of `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`, used unchanged:

| File | Experts | GPU memory for the experts | Status |
|---|---|---:|---|
| `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (58.5 GB) | TQK6 gate/up, TQK7 down, block-128 Hadamard | 24.0 GiB | validated end to end (RTX 5090, RTX 3090) |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (92.0 GB) | TQ2_T, block-128 Hadamard | 29.9 GiB | validated end to end (RTX 5090, RTX 3090); the experts do not all fit a 32 GB card, `auto` streams the rest |

Both files carry their own experts in the trellis types TQ2_T/TQK6/TQK7 and record the activation rotation in
`prism.hadamard.*` metadata. Strata reads both natively. About half of each file is the
n-gram (PLE) table, which stays on disk and is read row by row; it is not loaded into RAM or VRAM.

The draft head is Strata's own MTP runtime, prepared from the original model's MTP weights as for the other
Qwen3.8-Flash-Next models. It is not the GGUF draft file from the Gyro repository.

## Quick setup

Gyro support lives on the `rc2` branch of the agentionai Strata fork. The Apple Silicon port combines its trellis
formats (TQ2_T / TQK6 / TQK7), Hadamard rotation and CPU kernels with upstream Strata v0.1.44.1's Metal engine.

```sh
git clone -b rc2 https://github.com/agentionai/Strata
cd Strata
```

`tools/gyro_setup.sh` builds, downloads, packs and writes a server config on Linux or an Apple Silicon Mac:

```sh
tools/gyro_setup.sh --model S --backend metal                     # Apple Silicon, macOS 15+
tools/gyro_setup.sh --model S --backend metal --variant huihui    # Huihui abliterated Gyro
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 120       # RTX 50 series
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 89        # RTX 40 series
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 86        # RTX 30 series / RTX A6000
tools/gyro_setup.sh --model S --backend hip  --hip-arch gfx1201    # Radeon RX 9070 / AI PRO R9700 (experimental)
```

It prints the command that starts the server. The steps, by hand:

## Apple Silicon

Use `--backend metal` on macOS 15 or newer, an M1 or newer, with the Command Line Tools installed and Python 3.10
or newer. It builds `build-metal/strata` with Apple's metal-cpp headers and the agentionai llama.cpp fork (a new
checkout is pinned to `dc255f2a4b6e4211b555b0c05260b2523281f314`; `--llama DIR` keeps an existing checkout).
`--variant huihui` selects `agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF` and uses a separate
`gyro-s-huihui` / `gyro-m-huihui` pack and config, so the two variants can coexist.
The ordinary Mac installer still offers its existing model families; use the Gyro setup command above for these files.

The setup writes a page-aligned `<model>-metal.gguf` containing the dense weights and experts, with each tensor
checked by SHA-256 against the download. It preserves the `prism.hadamard.*` metadata. The n-gram table stays in
the original download, passed separately as `--ple-gguf`. Keep both files. Packing uses the Metal file so every
expert offset matches the file the engine maps or streams.

Both the decode kernels and the two prompt GEMM implementations decode the trellis formats from the canonical
FP16 Hyb codebook. The Hadamard kernels rotate only routed inputs, before gate/up and before down; shared experts
and routing read their original inputs. Decode rotates FP32. Prompt operands are FP16; the butterflies run in
FP32 and round on the final store. Identity signs, explicit signs, per-layer roles and disk-streamed experts use
the same paths. A mismatched or missing Hadamard pack is refused. The fused HC reader also handles Gyro's mixed
Q6_K down/inject and Q8_0 up matrices: it dequantizes the former and prepares Q8_0 copies at load time.
`STRATA_HC_Q8=0` prepares BF16 copies for the BF16 reader instead. The Mac's arena and memory limits are described
in [MAC.md](MAC.md).

Synthetic validation on an M5 Max (128 GB, macOS 26.6, Apple clang 21): both Gyro-S and Gyro-M format pairs,
windows of 1/4/8 tokens, four-chunk expert arenas, prompt v2/v3, signed Hadamard prompt operands, and the ARM CPU
path against the fork's GGML graph. The reference trellis dequantizers match GGML bit for bit. Ordinary Q2_0
decode output matched upstream byte for byte (same synthetic four-token input). These checks do not establish
speed or full-model token parity against llama.cpp. M1-M4 and Gyro-M full-model runs have not been measured.

Full-model smoke checks on that Mac (2026-10-13) used the requested Huihui Gyro-S file
(`Qwen3.8-Flash-Next-Gyro-S-abliterated-TQ1_0.gguf`, SHA-256
`783e1af72859c48146162840f1a2173922c03e4befaeea4add7578ad25c303a3`). The repack's 1,223 tensors matched the source
bytes. The engine answered "The capital of France is Paris." and recalled "SILICON" from a 234-token prompt.
The local OpenAI-compatible chat API also answered correctly. These were smoke checks at 4K context without an
MTP draft layer, not a speed or quality benchmark. The original Gyro-S file also passed these generation checks,
and a forced 4 GB expert arena matched its resident short-prompt tokens (48 generated tokens, prompt chunks 128).

The same Huihui Gyro-S also passed a resident MTP/context sweep on that Mac, with 131,072-token capacity,
INT8 KV, greedy decoding, `--spec 2 --spec-min-p 0.5 --suffix-draft 0`. Every run generated 256 prose tokens;
these are single fresh-process measurements, not medians. MTP used Strata's Q2_0 draft head from the pinned
original checkpoint (all 31 source tensors verified).

| Input tokens | MTP decode tokens/s | Prompt tokens/s |
|---:|---:|---:|
| 4,096 | 75.48 | 1,201.67 |
| 32,768 | 70.10 | 1,127.06 |
| 65,536 | 65.17 | 1,095.35 |
| 130,816 | 61.85 | 1,024.08 |

The top run filled the context with 130,816 input tokens plus 256 generated tokens. All 256 outputs matched
the no-MTP reference at short, 4K and full context; the full-context no-MTP reference measured 38.77 tokens/s
while draft preparation was still finishing, so its absolute timing is approximate. A short API request also
passed with MTP and 128K capacity configured. These checks do not establish long-context recall or quality;
see the [report and raw timings](../bench/results/2026-10-13-community-m5-max-huihui-gyro/README.md).

To run the synthetic checks (after setup has obtained both header dependencies):

```sh
cmake -S metal -B build-metal -DMETAL_CPP_DIR="$PWD/third_party/metal-cpp" \
  -DSTRATA_GGML_DIR="$PWD/../agention-llama.cpp" -DSTRATA_METAL_GYRO_TESTS=ON
cmake --build build-metal -j
ctest --test-dir build-metal --output-on-failure
```

The remaining manual build instructions are for NVIDIA/AMD. On a Mac use the Metal setup command above.

## 1. Build

From the `rc2` checkout above. Strata's GGML dependency must be the agentionai llama.cpp fork, which has the trellis types:

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120: issues #220, #224)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental: kernel parity and an end-to-end run validated on gfx1151; in a non-root container add
# --group-add $(getent group render | cut -d: -f3), otherwise HIP finds no device)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 2. Download and pack

```sh
.venv/bin/pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --local-dir models
STRATA_GGUF_PY=../agention-llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --out packs/gyro-s --compat-bf16
build/native_expert_parity --load-check packs/gyro-s models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf
```

The pack is 1.5 GB (dense weights, tokenizer, the Hadamard and expert tables); the experts are read from the
GGUF. Keep the GGUF on a fast local SSD: the n-gram table is read from it for every token.
`--load-check` must end with `0 failures`.

## 3. Draft head

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

## 4. Serve

Save as `strata-gyro-s.json` at the repository root:

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/gyro-s",
    "--native", "models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "131072", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/gyro-s/tokenizer",
  "model_name": "gyro-s-strata",
  "log": "strata-gyro-s.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`; API clients use `http://127.0.0.1:8080/v1` (OpenAI) or
`/v1/messages` (Anthropic). `--expert-cache auto` sizes the expert cache to the free VRAM: on a 32 GB card at
128k context it holds 23,464 of 24,576 experts (22.9 GiB). On smaller cards it holds fewer and streams the rest
from RAM; nothing else changes.

## Measured

All runs: greedy or the stated sampling, same prompt tokens as llama.cpp, MTP draft `--spec 4 --spec-min-p 0.5`,
int8 KV. "llama.cpp" is the agentionai fork with the CUDA trellis fixes (in its `main` since 2026-10-04).

### RTX 5090 (32 GB), Gyro-S, all experts on the GPU (2026-10-03)

Decode tokens/s, single prompts of 34-177 tokens, 384 new tokens, `--max-context 16384` (all 24,576 experts cached):

| | prose | JSON | code |
|---|---:|---:|---:|
| Strata | 139.9 | 221.3 | 209.0 |

Server, `tools/gyro_setup.sh` config (128k context, 23,463 experts cached), thinking off, temperature 0.7,
top-k 20, top-p 0.95, min-p 0.05, 512 new tokens, 3 repetitions:

| | prose | JSON | code | copy |
|---|---:|---:|---:|---:|
| Strata | 131.7 | 200.1 | 195.2 | 201.8 |
| llama.cpp, MTP draft | 116.8 | 213.0 | 179.0 | 220.7 |
| llama.cpp, no draft | 121.8 | 123.2 | 122.5 | 121.2 |

Long real-text prompts, prefill tokens/s: 4k tokens 3,456 (llama.cpp 2,520), 16k tokens 4,883 (llama.cpp 2,471).

### RTX 5090, Gyro-M (2026-10-03)

The experts (29.9 GiB) do not all fit next to the KV cache; `auto` caches 18,007 of 24,576. Same server protocol:
prose 113.8, JSON 170.8, code 153.0, copy 175.7 tokens/s; prefill 4k 3,369, 16k 4,854 tokens/s. llama.cpp needs
`-ncmoe 8` for this file on a 32 GB card and decodes at 70.8 tokens/s without a draft.

### RTX 3090 (24 GB), 2026-10-03

Same server protocol, 128k context, `auto` cache:

| | experts cached | prose | JSON | code | copy | prefill 4k | prefill 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-S | 15,756 | 31.8 | 45.2 | 42.2 | 45.1 | 1,402 | 1,863 |
| Gyro-M | 11,811 | 27.8 | 39.4 | 34.4 | 37.7 | 1,363 | 1,910 |
| llama.cpp, Gyro-S, `-ncmoe 10`, 32k context, no draft | (10 layers on the CPU) | 16.0 | 16.2 | 16.2 | 16.5 | 468 (`-ncmoe 11`) | 479 |

On a 24 GB card, cache misses run on the CPU, so the CPU matters: this box had a Zen 2 EPYC (AVX2, no AVX-512).
The llama.cpp row is the build before the CUDA trellis fixes; those added about 3 % at 24 GB, where the CPU layers
dominate.

The Strata rows predate Strata's own CPU kernels for the trellis types (`src/kernels/cpu/tq_*.cpp`, AVX2 and
AVX-512, picked at run time). Before them a cache miss ran ggml-cpu's `vec_dot`, one token at a time; the new kernels
take about 45 % of its time per expert on AVX2 and decode a block once for all the tokens of a verify window
(`build/tq_cpu_parity --bench` measures them on your CPU). `STRATA_NO_TQ_KERNELS=1` goes back to ggml-cpu;
`STRATA_TQ_ISA=avx2` or `scalar` caps the instruction set (for comparisons).

### Smaller caches (RTX 5090, 2026-10-02, before the int8 decode kernels)

| Expert cache | prose | JSON | code |
|---|---:|---:|---:|
| 24,576 slots (all experts) | 116.4 | 194.3 | 173.5 |
| 22,000 slots | 111.4 | 175.7 | 166.1 |
| 9,000 slots (about a 16 GB card's room) | 81.4 | 74.0 | 96.8 |
| 3,000 slots (about an 8 GB card's room) | 48.0 | 52.4 | 57.9 |

The smaller caches were emulated on the 5090 (fast GPU, PCIe 5.0 x16 at ~51 GB/s). A real 8 or 16 GB card is a
slower GPU, often on a slower link: read those rows as "it stays usable", not as a prediction for a specific card.
The decoded output was read and is fluent and correct; token-level parity against llama.cpp has not been
checked. A real chat at 128k context worked through the web interface.

### Two GPUs (2026-10-04, 2× RTX 4090, no NVLink/P2P)

- **Layer split** (Strata splits the layers across both cards automatically) works with Gyro, including the
  idle-card prompt helper (`STRATA_PREFILL_HELP=1`). Repeated runs give identical output.
- **Decode with a second card** (`--peer-device 1` or `--expert-cache-device1 auto`) works and matches single-GPU output.
- **Prompt processing on the peer card** needs GPU-to-GPU access (NVLink or data-centre GPUs); without it Strata keeps
  the prompt on the first GPU and logs `no P2P`.

Single RTX 4090 smoke run (Gyro-S, MTP draft, greedy): code 107.5, prose 84.3, JSON 76.3 tokens/s.

## Numerics

- **Decode** (Strata's own expert kernels) reads the trellis weights through an int8 copy of the codebook with one
  global scale (`tq_lut_i8`, relative RMS error 0.85 % of the codebook's standard deviation, far below the
  quantization error itself), against q8_1 activations.
- **Prompt processing** decodes them otherwise: the FP16 path through the fp16 codebook (`tq_lut_f16`, what GGML's
  own dequantizer reads); the MMQ path through the llama.cpp fork's kernels, which read the int8 codebook as well
  on the fork's current `main`.
- So decode and prompt results agree within tolerance, not bit for bit. `native_expert_parity` checks one expert
  on the decode kernels against GGML's float dequantizer and a float SwiGLU: relative L1 error below 3e-2, the
  bound every expert type gets there. Its "bitwise equal" lines compare Strata's decode-once and per-entry decode
  kernels with each other, not with a float reference.

## Known limits

- **Not measured yet:** token parity against llama.cpp.
- **Benchmark mode only:** `strata --tokens-file ... --max-new N` decodes N tokens past the end of the answer;
  compare speeds only up to the answer's end.
- **AMD:** the trellis kernels pass parity on gfx1151 (ROCm 7.2.1) and a short end-to-end run there generated
  correctly (~28-31 tokens/s); no speed tuning or RDNA4 run yet.
- **Gyro-M:** 29.9 GiB of experts do not all fit a 32 GB card next to the KV cache; `auto` streams the rest.
- **Not validated end to end yet** (Strata runs them and logs a warning): an AMD GPU other than gfx1151, and
  prompt processing on a peer GPU over P2P.
