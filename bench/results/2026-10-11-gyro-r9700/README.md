# Gyro-S R9700 receipts, 2026-10-11

See [the report and roadmap](../../../docs/GYRO_RDNA4_ROADMAP.md) for methods,
hardware, model/build provenance, reproduction commands and limits.

- `measurements.json`: 78 throughput observations (six arms, including one
  warmup per arm) and 24 separate instrumented prefill calls. The headline table
  uses only the first four alternating OFF/ON arms; the two `portable-*` arms
  validate the portable probe. Cold first batched requests are retained. The
  `host_staging_ms` and PLE fields include existing cumulative engine statistics.
- `hipblaslt.txt`: exact table used by every attention A/B arm. It is a receipt,
  not a default installation table. Its solution ids require this architecture,
  version and compatible library build; verify actual launches and recalibrate
  on another build. No table speedup is claimed.
- `attention-parity.txt`: fresh existing numerical-test output against FP64.
  Timing is an isolated kernel measurement, not an end-to-end benchmark.
- `smoke.json`: six completed-output checks per attention path, all passing.
  They ask for `17*23` as a JSON object, sorted distinct `[9,1,9,-3,2]`, and
  `GYRO-7391` access-code recall. Reference material contains 60 or 550 lines
  of `Reference row i: unit i%11, reading i*7, status nominal.`, with the code
  inserted midway. Each arm has a different identifying prefix. These checks
  are not controlled quality equivalence or a full benchmark suite.

The deterministic prompt generator is `tools/hip/gyro_probe.py prepare`.
It reproduced the original saved request file byte for byte. Responses in the
performance probe are deliberately capped at 128 tokens; hashes show reply
identity, not correctness. All benchmark input text was generated for this
experiment; no private user documents or model files are included.
