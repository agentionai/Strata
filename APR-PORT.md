# Porting Agention's APR expert types to Strata

Goal: run Agention's published "Gyro" GGUFs of Qwen3.8-Flash-Next (APR, the rotor-quant encoder) on Strata's
hybrid engine: hot experts in a VRAM cache, all experts in RAM computed by the CPU pool.

This file maps every place the port touches, records what each stage changed, and lists what is still open.
It was checked against Strata `d9ab843` (engine 0.1.35) and the agention llama.cpp fork `073bc9e68`.

## 1. What the Gyro files contain

| | Gyro-S (`...-Gyro-S-TQ1_0.gguf`, 58.5 GB) | Gyro-M (`...-Gyro-M-TQ2_0.gguf`, 92.0 GB) |
| --- | --- | --- |
| routed experts gate/up | TQK6 (145), 1.625 bpw | TQ2_T (144), 2.125 bpw |
| routed experts down | TQK7 (146), 1.875 bpw | TQ2_T (144) |
| experts total | 25.8 GB (24.0 GiB) | |
| attention / GDN / shared-expert projections | Q5_K (shexp down Q8_0) | Q8_0 |
| hyper-connection, PLE key/value | Q6_K / Q8_0 | Q8_0 |
| `output_hc_*` | IQ4_NL | Q8_0 |
| router `ffn_gate_inp` | BF16 | BF16 |
| `token_embd`, `output` | Q6_K | Q8_0 |
| PLE n-gram table `per_layer_token_embd` | IQ4_NL (28.8 GB) | **Q8_0** |
| `prism.hadamard` | v1, block 128, explicit signs (widths 640, 2560); all 48 x {gate, up, down}_exps | same |

The trellis formats (ggml-common.h of the fork): 128-value blocks, `{fp16 d; qs}`; qs is 32 trellis steps of K bits
(K = 8 TQ2_T, 6 TQK6, 7 TQK7, tail-biting, bit-packed for TQK). Step t's 16-bit state s gives four weights:
`x = (s & 0x7fff) * 0x9e3779b1`, codebook points `x >> 21` and `(x >> 10) & 2047` of the 2048-entry fp16-pair table
`tq2t_lut_f16`, times d. ggml-cpu's `vec_dot_type` for all three is Q8_0.

`prism.hadamard`: the encoder folded W' = W diag(s) H^T per 128-column block (H the normalized Sylvester-Walsh-
Hadamard matrix). The runtime must feed every folded weight `H (s * x)` per 128-block of its input: the gate/up
input x (2560 = 20 blocks) and the down input h (640 = 5 blocks), on every path. In the fork this is
`build_lora_mm_id` + `llama_mul_mat_hadamard`.

## 2. Strata's pipeline (as it applies here)

1. `tools/iq_pack.py --gguf <model.gguf> --out <pack> [--compat-bf16]` writes a "native pack": `index.txt`,
   `dense.bin` (small float tensors), `native_experts.txt` (per layer: gu_type d_type blob offsets in the GGUF),
   `tokenizer/`, `conversions.json`. Experts stay in the GGUF (optionally copied to `experts.bin`).
2. The engine runs with `--pack <pack> --native <model.gguf>`: dense quantized projections, head and embedding are
   read from the GGUF (`NativeDense`, `NativeHead`, `NativeEmbed`), the PLE table from the GGUF on disk (`PleTable`),
   experts from the GGUF into pinned RAM and a VRAM cache.
3. The CPU computes uncached experts with ggml-cpu's own type traits (`src/kernels/cpu/native_expert.cpp`). ggml
   is built from `-DSTRATA_GGML_DIR=<llama.cpp checkout>`, so pointing it at the agention fork gives the CPU path
   the trellis types and their AVX-512/AVX2 kernels (c7864adca) without porting them.

## 3. Non-expert tensors: accepted as-is?

| Tensor (Gyro-S / Gyro-M) | Verdict |
| --- | --- |
| Q5_K / Q8_0 attention, GDN, shared-expert projections | yes: `native_mmvq` has Q5_K, Q6_K, Q8_0 (`native_mmvq_supported`, native_mmvq.cu:1441) |
| `output.weight` Q6_K / Q8_0 | yes (`NativeHead`, any `native_mmvq` type) |
| hc_*, `output_hc_*`, ple_key, ple_value (Q6_K, Q8_0, IQ4_NL) | converted to BF16 by `iq_pack.py --compat-bf16` (the engine reads them as BF16; max abs err 0.0144, recorded in conversions.json). Gyro-M's Q8_0 ple_key could stay native. |
| router BF16, norms F32 | yes |
| `token_embd` Q6_K (Gyro-S) | **no GPU dequantizer** (`embed_type_supported`: is_iq types + BF16). Workaround added: `tools/embd_bf16_pack.py --gguf <model> --out embd.gguf`, then `--embd-gguf embd.gguf` (1.27 GB BF16, max rel err 2^-8). A `dq_q6_k` in `dq_dispatch` would remove the step. |
| PLE table IQ4_NL (Gyro-S) | yes (`PleTable`) |
| PLE table Q8_0 (Gyro-M) | **no** (`PleTable`: IQ4_NL, Q5_0 or FP8). Use `--ple-gguf <Gyro-S.gguf>` (the PLE table is not Hadamard-folded; same tensor, IQ4_NL), or add a Q8_0 reader to `ngram.cpp`. |
| MTP draft layer | its own S2-form experts (`mtp.cpp:182`), not the model's: unaffected. |

## 4. Where a new expert type must be added

CPU (done by building against the fork; nothing per-type in Strata):
- `native_fmt` / `native_gu_rows` / `native_down_rows` (native_expert.cpp) are driven by ggml-cpu traits.
  `iq512_supported` / `kq` fast paths do not claim types 144-146, so they take ggml's vec_dot (the fork's SIMD).

Type tables (done in stage 2):
- `include/strata/artifact/gguf_reader.hpp`: `ggml_type_name`, `block_geometry` (143-146).
- `tools/gguf_reader.py`: `GGML_TYPES`, `BLOCK_GEOMETRY` (used by every packer).

GPU (`src/kernels/cuda/iq_kernels.cu`, stage 4):
- `Fmt<T>` traits (:448-474: `qk`, `ipb`, `step`, `dot(v, block_q8_1*, kbx, iqs)`): a 128-block reads 4 q8_1 blocks.
- `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS` (:478-480), optional `Split<T>` (:523).
- `dq_dispatch` (:1231, 256 values per call = two 128-blocks), `is_iq` (:1268), `gu_qk` / `d_qk` (:1273),
  `iq_row_bytes` (:1335), `native_expert_supported` (:1417, checked at generate.cpp:1690).
- Device codebook: the 8 KiB `tq2t_lut_f16` (in `__constant__` or global memory, as the fork's CUDA backend does).
- Prompt path MMQ: `src/prefill/moe_mmq.cu:92 supported()` - leave the types out (the layer then takes the FP16
  fallback, which needs only `dq_dispatch`); or compile the fork's own `mmq-instance-tq*.cu` (f62fdead0).

## 5. How activations reach the experts; where the rotation goes

None of Strata's native-expert paths fuse SwiGLU or the quantization of h, so the rotation is a separate step
before each quantizer:

| Path | x (gate/up input) | h (down input) |
| --- | --- | --- |
| CPU pool (decode/verify) | `native_quant_act` (expert_source.cpp:1753 quantizes the fp32 doorbell copy) | `native_quant_h` (pool.cpp:680) |
| GPU, local cache (verify.cpp) | before `quantize_q8_1_rows(xm, ...)` :717, into a separate buffer (xm is shared with the shared expert, and the doorbell copy at :691 must stay unrotated for the CPU) | in `native_expert_grouped` between `swiglu_entries_kernel` and `quantize_q8_1_kernel` (iq_kernels.cu:1462-1463) |
| remote GPUs (remote_experts.cpp) | before `quantize_q8_1_rows` :297 | inside `native_expert_grouped` (same as above) |
| prompt, FP16 fallback (prefill.cpp) | rotate `m.Xs` after `gather_rows16` :1648 (private copy; `mixed_h` is shared) | rotate `Hh` after `swiglu_interleaved` :1756 |
| prompt, MMQ (prefill.cpp) | rotated copy of `m.mixed` before `mmq::quantize` :1628 | between `mmq::swiglu` :1734 and `mmq::quantize` :1736 |

Design choice: the rotation belongs to the consumer's quantizer, not to the published activation, so the CPU pool,
the local GPU and remote GPUs each rotate their own copy and the shared expert / attention never see a rotated x.

## 6. Packing and the runtime contract (stage 2)

- `iq_pack.py` validates `prism.hadamard.*` (version 1, normalized-sylvester-walsh-hadamard, input-last-dimension,
  identity/explicit signs, weight_names only `blk.N.ffn_{gate,up,down}_exps`, gate and up together, no
  `inverse_weight_names`, sign vectors for the widths used) and writes `<pack>/hadamard.txt`
  (`block`, `signs <width> ...`, `layer l g u d`). Anything else is refused before the pack changes.
- With Hadamard experts `native_experts.txt` becomes **v5**: an older engine refuses the pack instead of running the
  folded experts unrotated. `expert_layout_load` reads hadamard.txt (`src/kernels/cpu/hadamard.cpp`) and sets
  `NativeFmt::{had_block, had_gu, had_d, gu_signs, d_signs}` per layer; a v5 pack without hadamard.txt, or the
  reverse, is an error.
- The engine (`generate.cpp`) refuses a Hadamard pack until the GPU paths of section 5 rotate (stage 4).
- Why a sidecar and not the GGUF: `strata::GgufFile` keeps only the first 64 items of a metadata array, so it cannot
  read `sign_values` (3200 items).

## 7. Status

| Stage | State | Evidence |
| --- | --- | --- |
| 1 map | this file | |
| 2 loader + packing | done | Gyro-S and Gyro-M pack (`--compat-bf16`; 8 s each); 26/26 `tools/test_iq_pack.py` incl. 3 new Hadamard tests |
| 3 CPU experts + rotation | done | `native_hadamard_test`: Strata's CPU path is bit-identical (rel. diff 0) to the fork's ggml graph (`ggml_mul(signs)`, `llama_mul_mat_hadamard`, `ggml_mul_mat`) on real Gyro-S (TQK6/TQK7) and Gyro-M (TQ2_T) experts, also through `ExpertPool::run_split_multi_native`; skipping the rotation is 1.2-1.5 off. Synthetic Q8_0: the folded expert reproduces the unrotated float expert to 1.2%. |
| 4 GPU (CUDA/HIP) | see below | |

## 8. Running it

Build (CPU tests only, no GPU):

    cmake -S . -B build-cpu -DSTRATA_GGML_DIR=<agention llama.cpp> -DSTRATA_BUILD_TESTS=ON
    cmake --build build-cpu -j8 --target native_hadamard_test
    build-cpu/native_hadamard_test                                   # synthetic, ~1 min (ggml's placeholder TQ encoder)
    build-cpu/native_hadamard_test --pack <pack> --gguf <Gyro.gguf> 0 0 47 511

Pack:

    STRATA_GGUF_PY=<agention llama.cpp>/gguf-py python tools/iq_pack.py --gguf <Gyro-S.gguf> --out packs/gyro-s --compat-bf16
    STRATA_GGUF_PY=... python tools/embd_bf16_pack.py --gguf <Gyro-S.gguf> --out packs/gyro-s/token-embd-bf16.gguf
