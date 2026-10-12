# Gyro-S on RDNA4: measured starting point and next work

This is an R9700 experiment on `rc1` at `434e136`, not a new default or a claim
that GLM/Maya kernels can be used unchanged. Model files and engine code were
not changed. The first useful move is an existing opt-in QSA attention path;
the next work should follow the measured expert preparation/product costs.

Follow-up: [guarded routed-only gather batching](GYRO_ROUTED_GATHER.md) changes
engine scheduling behind an opt-in HIP switch. Fresh same-binary tests show
12–14% short-prompt prefill gains; the initial experiments above remain an
unchanged-engine comparison. No setup default or model file has been changed.

Follow-up: [rc2 settings-aligned protocol screen](GYRO_RC2_PROTOCOL.md), with
128K configured context and 512-token replies, repeats +32% 4K / +44.5% 16K
prefill across OFF/ON/OFF/ON. Exact author prompts are unavailable; substitute
prompts are published. rc2 staging integration passes focused validation; no
decode speedup or full 128K-window quality claim follows.

Claude's next lead is screened in the
[activation-quant/tile audit](GYRO_QUANT_ORDER_LEAD.md): reordered quantization
passes correctness checks but gives no R9700 gain (-0.84%/-0.39% at ~8K), so
is not selected. Tile logging finds no smallest-tile/no-tile fallback issue.
rc2 already has a HIP GDN alternative; test that existing option before a port.

## Starting point, 2026-10-11

- Radeon AI PRO R9700, 32 GiB, gfx1201; Ryzen 7 9800X3D, 186 GiB system RAM.
- Linux 7.0.0-28-generic, system ROCm 7.2.1; hipBLASLt 1.2.2 (`100202`, header
  build `dabb6df2b9`). No runtime/compiler upgrade.
- Strata `434e136f73a7da2bfd11cde3bb0aa0acaf1328c6`, Release HIP/gfx1201,
  `STRATA_PREFILL_MMQ=ON`; agentionai llama.cpp checkout `dc255f2a4b6e4211b555b0c05260b2523281f314`.
  Executable SHA-256: `856e84ef41206ca7637ac246864019eb557e22b0d322b20407a33258072a0598`.
- Existing Gyro-S download revision `ae0277dbabcbb12944e6887b356bcc1ef1adbafd`.
  This already contains the October 10 Gyro-S update advertised on the current
  [model card](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF).
  Full-file SHA-256: `c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87`.
  The earlier older-weight label was incorrect; see the
  [checksum and pack provenance receipt](../bench/results/2026-10-11-gyro-r9700/model-update-verification.json).
  Keep the model, generated pack and tokenizer together for comparisons.
- 16,384 context, int8 KV, auto prefill (8,192-token chunk), MTP `--spec 4
  --spec-min-p 0.5`, the existing Strata MTP runtime. All 24,576 expert pairs
  initially resident; prefill borrows 3,390 cache slots (3.31 GiB). This is not
  proof that every prompt operation is a residency hit or that longer contexts fit.
- `HIP_VISIBLE_DEVICES=1` on this host; `STRATA_APR=1`, `STRATA_PREFILL_MMQ=1`,
  `STRATA_PREFILL_RING=96`, `STRATA_IO_THREADS=32`. Device numbering is local;
  check your own device list. Expert worker pool: 7 workers plus host thread.

The earlier three-arm table test measured code pp512 at 840.5/833.0/841.0 tok/s
(no table/shipped table/freshly calibrated table), and code ~4K at
1543.1/1550.4/1548.5. Three trials per type: no meaningful prefill gain was
established. Different replies prevent attributing decode-rate changes to the
table. The calibrated table covers 16 observed dense geometries at T=512/4096;
32 winning rows passed the tuner's numerical/padding gates, and the extra
four-case padded/beta smoke used actual Lt launches without fallbacks.

## First move: test existing matrix-core prompt attention

Two alternating OFF/ON rounds, fresh server per arm, **one unchanged binary**.
Each arm replayed the same twelve saved API requests: three each of code,
prose and JSON at 512 tokens, plus three code prompts of 4096/4096/4095 tokens.
Greedy, thinking off, seed 42, max output 128; every measured request reused
zero prefix tokens and generated 128 tokens. The code/prose/JSON outputs are
deliberately capped, not graded complete answers.

The table, context, ring and all other settings stayed fixed. Both arms used
the newly calibrated table; `STRATA_HIP_WMMA=0/1` is the intended difference.
Neither `STRATA_PREFILL_TIMING` nor verbose Lt logging was enabled for these
throughput measurements. The 25-token warmup does not warm the batched path:
the first pp512 code request is cold. Medians below include three samples per
arm; cold timing is retained in the machine-readable evidence.

| Median prefill tok/s | OFF A | ON A | OFF B | ON B |
|---|---:|---:|---:|---:|
| Code, 512 tokens | 836.7 | 864.1 | 839.8 | 866.2 |
| Prose, 512 tokens | 843.5 | 866.3 | 841.3 | 864.7 |
| JSON, 512 tokens | 840.8 | 863.5 | 837.6 | 864.1 |
| Code, ~4K tokens | 1546.9 | 2008.2 | 1544.3 | 2010.2 |

The ~4K prefill gain repeats at about **30%**. Short-prompt median gains are
about 3%; no claim is made that this closes the Vulkan pp512 gap or improves
decode. Replies can change with different floating-point accumulation, and
the existing attention switch remains opt-in. This is validation of a switch
already in Strata, not authorship of a new attention kernel.

`hip_prompt_attn_wmma` passed three int8-KV geometries against FP64 with zero
failures: contexts 32,768/1,500/2,100, query counts 2,048/1,500/256. Its isolated
kernel gains were about 7x (6.94–7.23x on the saved final repeat); those are **not** whole-model speedups. Broad
answer-quality equivalence, 128K inference, images and other cards are not established.

An additional OFF/ON run through the portable probe reproduced the same trend
(~4K medians 1544.8/2008.0 tok/s). Six completed-output JSON checks per path
(arithmetic, sorted distinct numbers, access-code recall, at two reference
lengths) passed 6/6 in both arms, without truncation. The smoke arms have
different identifying prefixes, so this is a basic functionality check, not a
controlled quality-equivalence test. The probe itself has five CPU-only replay
contract tests; the phase summarizer has eleven CPU-only parser/CLI tests.

## What the diagnostic profile says

Separate instrumented OFF/ON passes, not the throughput runs above. The
existing timer charges elapsed compute-stream intervals, including GPU waits
for host work/copies, to phases. Integer-millisecond rounding applies.

| Interval | OFF | ON | Selection |
|---|---:|---:|---|
| `dequant` | 31.5% | 32.5% | Eight warm 505-token batched calls |
| `gemm gate/up` | 28.6% | 29.7% | Same short calls |
| `gemm down` | 16.4% | 16.9% | Same short calls |
| `qsa attn` | 26.2% | 4.9% | Two 4089-token batched calls |
| `qsa attn`, mean elapsed | 705 ms | 103.5 ms | Same long calls |

The prompt batch processes 505/4089/4088 tokens here; the API counts the full
512/4096/4095 prompt. The first batched call is excluded from the short
profile, not inferred to be warmed by the small API warmup.

Important: `dequant` is the existing phase **label**. In the MMQ branch its
interval includes native expert packing/gather; in the non-MMQ branch it covers
FP16 dequantization. Gate/up and down phases also include adjacent activation,
Hadamard or quantization work. These percentages do not establish a specific
dequant kernel bottleneck, and dense native projection dequantization is
inside its parent attention/shared-expert phases. Audit actual dispatch and
subdivide the intervals before choosing a kernel change.

## Roadmap and acceptance gates

1. **Completed: controlled baseline, table economics, profile and attention A/B.**
   Preserve individual observations and exact prompts; report cold/warm and
   prefix reuse separately. Do not ship the new table as an automatic default:
   it produced no meaningful gain and its solution ids are library-build-specific.
2. **Next: break down expert preparation/product intervals.** Add diagnostic
   receipts for actual per-layer MMQ/FP16/fused dispatch, packing/gather, trellis
   decode, rotations and activation quantization. Prioritize the short-prompt
   path, where the preparation label plus expert products dominate. Prove
   avoidable copies/repeated work before implementing zero-copy/direct-indexed
   access or fused trellis-aware preparation. Require format/rotation/padding
   parity and cost-inclusive same-binary OFF/ON/OFF/ON end-to-end results.
3. **Then: quality and length screen for the attention switch.** Complete-output
   code/math/JSON/recall checks, identical generation caps and graded tasks;
   test 8K/16K before longer context. Track MTP acceptance and changed outputs.
   A throughput win alone is not a reason to enable it in setup by default.
4. **Shared runtime audit.** Compare cache-lending/resident-mode transitions,
   pinned-memory publication and router nonfinite/bounds handling with the
   corresponding Maya work. Port only demonstrably missing shared fixes,
   with targeted race/replay tests; do not copy GLM-only state transitions.
5. **Only if the profile justifies it:** native dense weight reuse/producer
   fusion, speculation-window tuning, then long-context cache admission.
   Charge extra resident storage against expert capacity. Measure MTP with
   matched tasks; output-dependent decode rates are not isolated kernel timings.

Do not directly port Maya's GLM attention or IQ-format fused expert kernels:
Gyro's TQK6/TQK7 and block-128 Hadamard rotations need their own implementation
and tests. Strata explicitly excludes folded trellis layers from its current
fused-native prompt branch. Maya's no-MTP final-layer skip is not applicable to
this MTP setup. RAM-tier/prefetch work is lower priority while Gyro-S initially
fits; it becomes relevant if a larger model/context causes actual misses.

Keep ROCm 7.2.1 for this comparison. A runtime change is a separate experiment,
not bundled with a table or kernel change. Keep the 7900 XT idle for this
single-card baseline; dual-card work is a separate lane.

## Reproduce and inspect

Build/pack/load-check as in [GYRO.md](GYRO.md), but use the recorded model
revision when reproducing these numbers. Use a dedicated otherwise idle server
on loopback, no concurrent API clients. Configure the above engine arguments;
select your own R9700 HIP number and keep the same runtime/table in both arms.
The calibration file in `bench/results/2026-10-11-gyro-r9700/hipblaslt.txt` is
an experiment receipt, **not** a portable default. Recalibrate for another build
and inspect actual Lt launches/fallbacks before adopting it.

```sh
.venv/bin/python tools/hip/gyro_probe.py prepare --pack packs/gyro-s --output requests.json
# With the GYRO.md server config adjusted to max-context=16384, backend=hip,
# port=18080 and the corresponding engine log:
HIP_VISIBLE_DEVICES=1 STRATA_APR=1 STRATA_PREFILL_MMQ=1 \
  STRATA_PREFILL_RING=96 STRATA_IO_THREADS=32 STRATA_HIP_WMMA=0 \
  STRATA_HIPBLASLT_TUNING="$PWD/bench/results/2026-10-11-gyro-r9700/hipblaslt.txt" \
  .venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json \
  --host 127.0.0.1 --port 18080
# Restart the dedicated server with STRATA_HIP_WMMA=0, then =1, then =0, then =1.
# Do not set STRATA_PREFILL_TIMING for throughput; keep all other settings fixed.
.venv/bin/python tools/hip/gyro_probe.py run --requests requests.json \
  --engine-log strata-gyro-s.log --label off-a --output off-a.json
```

Preparation matched the original saved requests byte for byte (SHA-256
`2bfaebeba890527045eb1291af692fed71dbdca7b157201811b28353d3c571cd`).
Choose distinct result paths; the probe refuses to overwrite prior outputs.
It checks actual prompt counts, zero prefix reuse and the 128-token output cap.
The probe uses a dedicated loopback server without an API key; authenticated
or remote deployment is outside this experiment.

For a separate diagnostic restart with `STRATA_PREFILL_TIMING=1`:

```sh
python3 tools/hip/summarize_prefill.py strata-gyro-s.log --skip 1 --tokens 505 --output short-phases.json
python3 tools/hip/summarize_prefill.py strata-gyro-s.log --tokens 4089 --output long-phases.json
python3 tools/hip/test_summarize_prefill.py
python3 tools/hip/test_gyro_probe.py
```

`--skip` counts parsed prefill calls before filtering, not API requests. The
summarizer preserves per-call records and computes percentages from summed
elapsed milliseconds, not averages of rounded percentages. Host fields and
`host staging` are retained as reported; some existing engine stats accumulate
across calls, so they must not be summed as independent per-request costs.
Machine-readable receipts live in `bench/results/2026-10-11-gyro-r9700/`.
