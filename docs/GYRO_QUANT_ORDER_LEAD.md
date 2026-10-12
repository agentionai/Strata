# Gyro / R9700: activation-quant ordering and MMQ tile audit

Measured 2026-10-11 after the rc2 protocol screen. Claude's ranked leads
correctly marked short-prompt gather as already done. Relevant starting
worktrees were clean; the ordering change was absent from the bundled
dependency `dc255f2a`.

The hypothesis comes from [ROCmFPX #44](https://github.com/ROCmFPX/ROCmFPX/pull/44),
merged October 9: on gfx1151, consecutive workgroups walking a token's chunks
rather than different tokens improved large-ubatch prefill. Those outside-engine
results are references, not results on this machine. No outside engine was run.

## Controls and validation

Private local dependency copy, separate Strata checkout based on the tested rc2
gather port `e5489dd`. One fresh gfx1201 binary contains both ordering kernels;
`GGML_HIP_QUANT_CHUNK_FASTEST=0/1` is the only throughput-arm change.
OFF retains the previous non-scatter mapping, ON permutes workgroup assignment;
output layout/arithmetic are unchanged. Scatter and CUDA retain the old mapping.
The experimental switch defaults OFF. The dependency code was not committed,
pushed or submitted, and its private patch is not published in this PR.

R9700 32 GiB, Ryzen 7 9800X3D, ROCm 7.2.1, hipBLASLt 1.2.2. Native October 10
Gyro-S and existing pack/MTP. Configured context 16K, int8 KV, greedy, thinking
off, seed 42, 128-output cap. WMMA and routed gather stay ON; calibrated table,
ring 96, stream threshold 1024, 32 I/O threads stay fixed. All 24,576 expert
pairs initially cached; prompt buffers still borrow cache space.

OFF/ON/OFF/ON, fresh server each time, same 12 saved requests per arm: three
each of code/prose/JSON at 512 tokens and code at 8192/8192/8191. Warmup and cold
observations retained. Diagnostic timing and tile logging run separately; no
builds overlap throughput. These are optimization probes, not model-card rows.

- 104 full-byte ordering cases pass: all three activation layouts, TQK6/TQK7,
  widths 640/2560/6144/10240, row indirection, channels/samples, scatter, zero
  rows, padding guards and alternating finite inputs.
- Focused CTest 4/4 per mode: byte ordering, MMQ products against CPU-dequantized
  FP32 reference, native gather, 400 staging schedules. Not the full backend suite.
- Initial legacy product fixture timed out after 120.12 s while doing CPU-heavy
  offline trellis encoding. Three other tests passed; no throughput ran; service
  restored. Retry uses deterministic valid encoded trellis blocks with finite
  FP16 scales (every byte string is a valid trellis path), retaining the FP32
  product reference. It does not skip product parity or alter the runtime.
- All 48 measured requests have expected prompt counts, zero reuse and 128 outputs.
- 24/24 paired greedy messages identical; 24/24 completed smoke checks pass.

## Result: do not select the ordering change on this setup

Median prefill tokens/s, three samples per cell:

| Prompt | OFF A | ON A | OFF B | ON B |
|---|---:|---:|---:|---:|
| Code, 512 | 1000.9 | 984.5 | 993.5 | 994.8 |
| Prose, 512 | 1014.0 | 1005.2 | 1001.8 | 1002.9 |
| JSON, 512 | 1002.5 | 998.3 | 994.7 | 997.2 |
| Code, ~8K | 2430.2 | 2409.8 | 2418.1 | 2408.6 |

Long prompt: **-0.84% / -0.39%**, no measured improvement. Short controls are
mixed/noise. Isolated kernel screens also show no clear win; at 8192 rows and
K=2560 the reordered quantizer takes about 0.1837 ms versus 0.1744 ms (+5.37%
time). These repeated-buffer kernel timings are not end-to-end speed claims.

Separate diagnostic compute-stream intervals put activation quantization at
about 2.2% for a 505-token batched call, 3.3% for 8184/8185-token calls with
ordering OFF, and 3.6% ON. Expert gate/up plus down occupy about 52% / 35% at
those lengths. Intervals include waits and host-emission gaps, not isolated
kernel attribution; they should not be pooled with throughput observations.

## Tile audit: no smallest-tile or no-tile fallback issue found

The diagnostic logger reproduces the pinned dependency's selector from the
same runtime product arguments/configuration; it does not override any tile.
25,348 product calls logged: 12,626 TQK6 gate/up (N=1280,K=2560), 12,626 TQK7
down (N=2560,K=640), and 96 Q8_0 draft-layer products. Full shape tuples remain
local; the public receipt groups them into 50 entries with logical prompt rows,
activation/expert-row ranges, sizes, flags and call counts.

TQK6/TQK7 both select J=16/32/48/64/80/96/112/128; **no J=8** observed.
I=64 with 128 threads for J<=64, I=128 with 256 threads for J>=80. Largest
configuration uses 61,952 shared-memory bytes. Every logged expert product has
unaligned-row fallback=0 and stream_k=0; no #420 no-tile warning appears.
This is a dispatch audit, not proof that the current tile is fastest. A tile
change still needs per-shape numerical checks and controlled model A/B.

## Revised next leads

rc2 already contains an AMD quad GDN recurrence behind `STRATA_GDN_HEAD=1`,
default OFF on gfx1201 (ON by arch default on gfx1151). Test its existing HIP
parity harness and one-variable A/B before porting the CUDA key-head kernel.
GDN recurrence is about 3.6% / 10% of these diagnostic short/long intervals.
No GDN variant was enabled or benchmarked in this attempt.

The model's native output head is Q6_K (type 14); the static draft vocabulary
has 106,299 tokens. rc2 already offers `--mtp-q4 head` and a draft-vocabulary
file override. Dynamic/per-sequence selection remains a separate proposal.
Draft-head time/acceptance should be measured before choosing any of them;
neither Q4 nor a smaller vocabulary was tested here.

Public [results receipt](../bench/results/2026-10-11-gyro-r9700/quant-order-lead.json)
and [requests](../bench/results/2026-10-11-gyro-r9700/quant-order-requests.json)
include hashes, per-request timing/output hashes, kernel timings, phase records,
tile counts and the failed-validation/retry history. No dependency patch is
submitted through this results PR. Original source/dependency remain clean;
working 16K Gyro restored healthy. Maya was not started or messaged; the R9700
remains reserved by that Gyro service, not released to the Maya queue.
