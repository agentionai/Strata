# Gyro-S / R9700: rc2 settings-aligned follow-up

Measured 2026-10-11 on one R9700 (32 GiB, gfx1201), Ryzen 7 9800X3D,
186 GiB RAM, system ROCm 7.2.1 and hipBLASLt 1.2.2. This uses the verified
October 10 Gyro-S update, not different weights between arms.

The [author's rc2 guide](https://github.com/agentionai/Strata/blob/rc2/docs/GYRO.md)
sets 128K configured context, int8 KV, `--spec 4 --spec-min-p 0.5`, thinking
off, temperature 0.7, top-k 20, top-p 0.95, min-p 0.05, 512 new tokens,
three repeats, prose/JSON/code/copy and 4K/16K real-text prefill. These settings
are used here, with fixed seed 42 and **disclosed substitute prompts**.
The exact original author prompts are not published in that guide. These are
settings-aligned measurements, **not exact model-card replication**; the author's
October 3 runs also used earlier weights/engine code.

## Reproducibility and controls

Separate rc2 checkout at upstream `9ff02c9fa9c86a88088370925b3c9131cc5d6c26`,
minimal gather port saved locally as `e5489dd2281e3da03c2cd5109fc9d1f9e2716297`.
The [complete port patch](../bench/results/2026-10-11-gyro-r9700/rc2-routed-gather.patch)
includes its header and tests. It retains rc2's transfer-counted staging
lookahead, flushing pending gather readers before overwriting their ring slot.
The switch remains off by default, HIP-only and single-device.

One fresh Release HIP binary for all four arms:
SHA-256 `d8fdb392ca1d6399a7184932753ab2c638de265392c96a8c26d2048ea2649e6d`.
GGML fork remains `dc255f2a4b6e4211b555b0c05260b2523281f314`.
The original installation and its working 16K profile were preserved.

OFF/ON/OFF/ON, fresh server per arm. OFF disables both WMMA attention and routed
gather; ON enables both. Ring 96, stream threshold 1024, 32 I/O threads, native
MMQ, APR and automatic cache stay fixed. **No GEMM table in either arm**.
All four cache receipts show 24,393/24,576 expert pairs (23.84 GiB), not full
residency. No diagnostic timing or builds overlapped throughput runs.

Eleven completed-output length screens run before each arm, warming the batched
paths. An additional 512-token warmup is retained separately. Every measured
request reads its entire prompt with zero prefix reuse and reaches the 512-token
cap; no generation is forced beyond EOS. The 72 measured responses are capped
throughput probes, not complete-answer quality grades.

Saved [requests](../bench/results/2026-10-11-gyro-r9700/rc2-protocol-requests.json)
contain three each of prose, JSON, code and copy (63–69 prompt tokens), plus
three each of exact 4,096- and 16,384-token real-text prompts drawn from pinned
rc2 `docs/DETAILS.md`. The
[receipt](../bench/results/2026-10-11-gyro-r9700/rc2-protocol.json) includes
corpus/request/model/build hashes, per-request timing, output hashes,
draft acceptance and cache counts. Full raw responses/logs remain in the local
attempt archive. Seed 42 does not imply identical output across attention paths.

## Results

Each cell is the median of three requests, tokens/s:

| Prefill | OFF A | ON A | OFF B | ON B |
|---|---:|---:|---:|---:|
| 4K real text | 1,615.1 | 2,134.0 | 1,614.9 | 2,132.6 |
| 16K real text | 1,671.7 | 2,415.4 | 1,667.4 | 2,411.7 |

Prefill gain: **+32.13% / +32.06% at 4K; +44.49% / +44.64% at 16K**.
Total request wall time falls about 5.4% / 17.2% respectively with these
512-token replies. This is a combined-switch comparison; it does not separately
attribute a gain to routed gather. Long chunks use the existing streamed walk;
the earlier isolated attention/gather studies remain separate evidence.

| Short-prompt decode | OFF A | ON A | OFF B | ON B |
|---|---:|---:|---:|---:|
| Prose | 67.8 | 68.9 | 67.7 | 68.8 |
| JSON | 81.9 | 84.0 | 81.4 | 83.8 |
| Code | 90.3 | 87.9 | 89.9 | 87.7 |
| Copy | 118.6 | 119.1 | 118.1 | 118.8 |

No decode speedup claimed. Reply text differs across paths except the copy
cases, and acceptance differs too: approximately 0.50→0.535 for prose,
0.682→0.701 JSON and 0.737→0.700 code. Copy acceptance stays about 0.998.
The engine's `--spec 4` denotes a verify window (up to three MTP drafts), not
four drafts; default suffix lookup can extend a window to six tokens. Keep
the setting and observed acceptance/window behavior distinct in future sweeps.

## Validation and boundaries

- Focused rc2 CTest 4/4: WMMA attention, native byte-gather parity, CPU staging
  lifetime model, CPU trellis parity. Not the full upstream suite.
- CPU model: 400 schedules across both ordinal and transfer-counted staging;
  removing the lifetime guard detects 163 corruptions.
- Native model/pack load check: zero failures, Hadamard block 128, 48 trellis
  layers, 302 supported native dense tensors and direct-I/O PLE.
- 44/44 completed arithmetic screens pass, actual prompt lengths 64–16000,
  zero reuse, nine generated tokens, normal stop. Narrow task, not broad quality.
- 72/72 measured protocol samples: expected prompt counts, zero reuse, 512 outputs.
- Working experimental 16K service restored and verified loaded/healthy.

128K is configured capacity, **not full-window inference validation**: the
longest actual prompt tested here is 16K. Other architectures, Windows, CUDA,
multi-GPU, low-VRAM stress, mmap/transient sources, images and broad completed-code
quality remain unvalidated. The published 3090 row is a 24 GB, partly resident
setup on a different CPU; it is not a GPU-only bandwidth comparison.

Next: obtain exact author prompts or retain this disclosure, then an isolated
greedy `--spec 2/3/4` verify-window sweep with matched replies and cache receipts.
Do not adopt a new default from these single-card screens.
