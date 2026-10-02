# Draft PR: Agention APR (trellis) expert types and Hadamard-folded experts

**Not opened.** This is a local description draft for `feat/apr-types`.

## What this adds

- **Expert types.** Strata can now run Agention's APR "Gyro" GGUFs of Qwen3.8-Flash-Next. Their routed experts are
  trellis-coded: TQ2_T (ggml type 144, 2.125 bpw), TQK6 (145, 1.625 bpw) and TQK7 (146, 1.875 bpw). Each block is
  128 values: an fp16 scale and 32 steps of K bits. Each step holds a 15-bit state that decodes through a 2048-point
  fp16-pair codebook.
- **Hadamard-folded experts.** The Gyro experts are stored in a block-128 Hadamard-rotated basis, declared by the
  `prism.hadamard.*` metadata. The runtime must feed each folded expert `H (s * x)`, per block of 128, for both the
  gate/up input and the down input, on every path.

### CPU path

The CPU path uses ggml-cpu's own type traits. Build with `-DSTRATA_GGML_DIR=<agentionai/llama.cpp>`; that tree
carries the types and their AVX-512 / AVX2 kernels.

The rotation is applied in `native_quant_act` and `native_quant_h`. These are the only two places the CPU quantizes
an expert activation.

### GPU path

- **Decode.** `Fmt<144..146>` is a lane decode against q8_1, 4 q8_1 blocks per 128-value block. `dq_tq` adds the
  types to the FP16 prompt path and to the embedding dequantizer.
- **Rotation.** `hadamard_rows` / `hadamard_rows_f16` is a shared-memory FWHT that runs the butterflies in the same
  order as the CPU, so it gives the same floats. It is applied:
  - before `quantize_q8_1_rows` in the verify window;
  - on the helper GPUs (`remote_experts.cpp`);
  - in the FP16 prompt fallback;
  - inside `native_expert_grouped` for h.
- **Separate buffers.** Each consumer rotates its own copy of x. The shared expert and the CPU's doorbell copy keep
  the unrotated x.
- **MMQ.** A folded layer never takes the MMQ prompt path.

### Packing

- `tools/iq_pack.py` validates the metadata and refuses anything the engine cannot apply. For a valid file it
  writes `hadamard.txt` and bumps `native_experts.txt` to **v5**, so an older engine refuses the pack instead of
  running the folded experts unrotated.
- `tools/embd_bf16_pack.py --gguf` dequantizes a GGUF's own `token_embd` to BF16.
- `tools/gguf_replace_tensor.py` swaps a tensor of a different type into a GGUF, leaving everything else
  byte-identical.

## Proposal: a metadata hook for activation transforms, instead of a sidecar

This branch carries the transform in a pack sidecar, `hadamard.txt`. That is a stopgap. The long-term design we
would propose is to read the transform from the GGUF itself, through a small, engine-generic hook:
"activation transform before the matmul of weight W".

- **Contract.** For each folded weight W, the input activation a becomes `T(a)` before quantization, on every path.
  T is described by:
  - `block_size`;
  - `transform` (`normalized-sylvester-walsh-hadamard`);
  - `axis` (`input-last-dimension`);
  - `sign_mode`: `identity`, or `explicit` with `sign_widths` / `sign_values`;
  - the list of weights it applies to.

  These are the `prism.hadamard.*` keys llama.cpp forks already use. Strata would only need the routed-expert case.
- **Engine side.** `NativeFmt` already carries the per-layer transform. It would be filled from the GGUF at load
  instead of from `hadamard.txt`. The v5 bump stays as the refusal mechanism for older engines.
- **Blocker in today's reader.** `strata::GgufFile` keeps only the first 64 items of any metadata array
  (`if (i < 64) v.items.push_back(...)`). It cannot read `sign_values`, which has 3200 items for Gyro (widths 640
  and 2560). The hook needs either:
  - an accessor that reads a full array lazily from the mapping (offset + count, decoded on demand), or
  - an exemption list for arrays the engine consumes.

  The sidecar side-steps this today.
- **Refusal semantics.** An engine that sees a transform it cannot apply must refuse the file. A missing transform
  produces fluent garbage, not an error. This branch also refuses transforms on dense weights, inverse
  (lookup-side) tables, gate without up, and Q2_0 experts.

## Validation

- **CPU.** On real Gyro-S experts (TQK6/TQK7, layers 0, 5, 11, 23, 47) and Gyro-M experts (TQ2_T), the output of
  `native_hadamard_test` is bit-identical to the fork's ggml graph (`ggml_mul(signs)`, `llama_mul_mat_hadamard`,
  `ggml_mul_mat`). This also holds through `ExpertPool::run_split_multi_native`.
- **GPU.** `native_expert_parity --synthetic` and `--synthetic-hadamard` for `tqk6/tqk7` and `tq2_t/tq2_t` pass on
  gfx1151 (ROCm 7.2.1):
  - the GPU and CPU outputs differ from the float reference by the same amount;
  - GPU vs CPU differs by 5e-5 to 1e-3;
  - the dequantizers and the rotation match the CPU bit for bit.

  The CUDA build (12.8, sm_120) compiles and is not run here.
- **Not yet.** No end-to-end generation has been compared against llama.cpp on the same prompt.
