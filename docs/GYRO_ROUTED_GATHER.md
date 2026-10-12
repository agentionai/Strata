# Gyro-S: opt-in routed-only gather batching on HIP

2026-10-11, R9700 gfx1201, ROCm 7.2.1. Follow-up to the
[RDNA4 roadmap](GYRO_RDNA4_ROADMAP.md): a scheduling change, not a new trellis
decoder, fused expert kernel, quantization format or setup default.

## Change and lifetime rule

`STRATA_PREFILL_ROUTED_GATHER=1` lets HIP's routed-only native MMQ path use the
existing grouped byte-copy kernel for resident and streamed experts. It stays
**off by default**, single-device only, and respects
`STRATA_PREFILL_GROUP_GATHER=0`. FP16, non-native and multi-device paths retain
their existing behavior. The long streamed walk already groups gathers.

The short walk has eight staging slots but an MMQ product group can hold 16
experts. Merely removing the old `stream_all` gate can overwrite a blob before
its deferred gather reads it. Before staging into a slot retained by an open
gather, this change flushes the partial gather, waits for its copies and records
its reader-release event. The existing copy stream waits on that new event
before reuse. Resident experts neither consume nor advance the staging ring.
The final partial group is gathered before its products.

Weight bytes, expert ordering, MMQ products, activation quantization, rotations
and sampler are unchanged. Grouped copies retain the unaligned fallback. No
extra device buffers are allocated.

With `STRATA_PREFILL_TIMING=1`, per-call receipts also count actual MMQ/FP16/fused
layer passes, grouped/single native gathers, standalone FP16 expert dequants,
copied bytes and routed-grouped passes. The summarizer retains these optional
receipts. The existing `dequant` interval is not renamed: in this Gyro MMQ path
it includes native packing/gather, not standalone FP16 weight dequantization.

## Controlled result

Fresh CMake HIP/gfx1201 build on `rc1` base `434e136` plus this patch. Binary SHA-256:
`5b6c5dbd15fe62257d99c38bc502c35664965ce185d686680d9333a98e2e53ea`.
Same executable for OFF/ON/OFF/ON, fresh server each time; no compilation or
diagnostic timing overlapped throughput runs.

Existing Gyro-S revision `ae0277dbabcbb12944e6887b356bcc1ef1adbafd`, 16K context,
int8 KV, auto prefill (8192), MTP spec 4/min-p 0.5; all 24,576 pairs initially
resident, prefill borrows 3390 slots. WMMA attention is **on in both arms**.
Stream threshold 1024, ring 96, MMQ/group-gather on, I/O threads 32, calibrated
table unchanged. Only `STRATA_PREFILL_ROUTED_GATHER=0/1` changes between arms.

Same twelve requests each arm: three code/prose/JSON pp512 cases and three ~4K
code cases. Greedy, thinking off, seed 42, 128 output tokens, zero prefix reuse.
Small API warmup and cold first batched observation are retained. Each cell is
the median of three trials, not a confidence interval.

| Median prefill tok/s | OFF A | ON A | OFF B | ON B |
|---|---:|---:|---:|---:|
| Code, 512 | 872.4 | 979.3 | 864.8 | 974.4 |
| Prose, 512 | 869.8 | 989.0 | 868.1 | 981.5 |
| JSON, 512 | 868.6 | 978.8 | 863.8 | 975.3 |
| Code, ~4K | 2012.3 | 2010.8 | 2009.5 | 2007.2 |

Short prefill gains repeat at **12–14%**. Median total short-request wall time
improves about **2–4%**; decoding dominates a 128-token reply. Long prompts are
unchanged within run variation: their existing grouped path is unchanged. No
decode speedup is claimed. All 24 paired replies match; this is greedy text
identity on these cases, not universal numerical or answer-quality parity.

A separate diagnostic overlay confirms 48 MMQ passes, zero FP16/fused passes
and zero standalone FP16 expert dequants. Typical short requests use ~950
grouped launches instead of ~14,500 single gathers, with equal byte volume.
CPU compilation overlapped that diagnostic pass: its elapsed rates are **not**
throughput evidence.

## Verification and reproduction

- Fresh HIP engine and both new CMake targets built.
- `hip_prefill_gather_parity`: eight byte-exact Gyro-sized cases, partial groups,
  nonzero first index, padded strides/guards and unaligned/oversized declines.
- `prefill_gather_schedule`: 200 CPU schedules using the engine's lifetime
  helper; removing the guard is detected as corruption in 76 cases.
- 13 parser/CLI tests and five portable-probe contract tests pass.
- Six completed-output arithmetic/sort/recall checks per OFF/ON path pass.
- Identical completed arithmetic requests at 64, 128, 256, 511, 512, 513, 768,
  1024, 2048, 8192 and 16000 actual prompt tokens pass on both paths, without
  reuse or truncation. This screens tails, boundaries and multi-chunk behavior,
  not broad quality.

```sh
cmake --build build-gfx1201 --target strata hip_prefill_gather_parity prefill_gather_schedule -j2
HIP_VISIBLE_DEVICES=1 ctest --test-dir build-gfx1201 --output-on-failure \
  -R '^(hip_prefill_gather_parity|prefill_gather_schedule)$'
python3 tools/hip/test_summarize_prefill.py
python3 tools/hip/test_gyro_probe.py
```

Use your own device number. Replay `gyro_probe.py` with identical settings/table
and toggle only the new switch; do not compare instrumented throughput.
Observations, hashes, dispatch and length/smoke receipts:
`bench/results/2026-10-11-gyro-r9700/routed-gather.json`.

## Rejected or deferred alternatives

Streaming threshold 1024→256 gave only ~3–4% short prefill gain in two clean
rounds; replies matched. An initial diagnostic-only decode slowdown did not
reproduce without timing. Keep 1024. Resident-only batching applied to just
0–2 layers because cache loans are spread across layers; it was not selected.
These attempts remain in the local ledger.

## Model-card work and limits

Later [rc2 integration and protocol screen](GYRO_RC2_PROTOCOL.md) validates a
minimal port against the newer staging schedule and runs 128K-configured,
512-output settings with disclosed substitute prompts. The limits below
describe the original rc1 study; neither study establishes full 128K-window
inference or broad quality equivalence.

This is a 16K-context optimization comparison on the October 10 Gyro-S update,
**not a model-card
row**. The author's [current guide](https://github.com/agentionai/Strata/blob/rc2/docs/GYRO.md)
points to `rc2` and describes the 128K-context, temperature-0.7, 512-new-token,
three-repeat prose/JSON/code/copy protocol plus 4K/16K real-text prefill.
Existing weights are checksum-verified as current; earlier older-weight wording
was incorrect. Card-compatible work still needs that separate protocol; obtain
the original prompt text or disclose substitute prompts.

The published 3090 has **24 GB**, caches 15,756 pairs in that run, and uses a
different CPU/older CPU kernels. It is not a same-capacity, fully-resident
bandwidth comparison with this R9700. At 128K even the author's 32 GB setup
caches fewer than all expert pairs.

Not validated: `rc2` integration, 128K context, low-VRAM cache
stress, mmap/transient expert sources, Windows, other architectures, CUDA,
multiple devices, images or broad completed-code quality. The HIP-only switch
stays opt-in; no universal default should follow from this single-card screen.
Trellis-incompatible fused expert kernels are not used.
