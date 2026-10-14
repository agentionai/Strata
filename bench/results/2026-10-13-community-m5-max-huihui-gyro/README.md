# Huihui Gyro-S: MTP and 128K context on Apple Silicon

Measured on an Apple M5 Max with 128 GiB unified memory, macOS 26.6. Model: agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF, Gyro-S TQ1_0; source SHA256 783e1af72859c48146162840f1a2173922c03e4befaeea4add7578ad25c303a3.

Every run allocated a 131,072-token context, used INT8 KV and the resident Metal model, greedy decoding, `--prefill auto`, `--spec-min-p 0.5`, and `--suffix-draft 0`. MTP used the pinned original model draft head, quantized to Q2_0 by Strata. All 31 source tensors passed the pinned SHA256 checks.

Each speed run generated 256 tokens of prose with no EOS encountered. Long inputs contain numbered synthetic archive records followed by a story-writing request. Each row is one fresh engine process; there is no reused prompt cache. These are single-run measurements, not medians, and do not establish long-context recall or model quality.

| Input tokens | MTP window | Decode tokens/sec | Prefill tokens/sec | Prompt seconds |
|---:|---:|---:|---:|---:|
| 45 | 2 | 71.69 | 162.16 | 0.27 |
| 4,096 | 2 | 75.48 | 1201.67 | 3.41 |
| 32,768 | 2 | 70.10 | 1127.06 | 29.07 |
| 65,536 | 2 | 65.17 | 1095.35 | 59.83 |
| 130,816 | 2 | 61.85 | 1024.08 | 127.74 |

The engine counts all but the final input token under prefill; the final one enters the decode path. At the top end, 130,816 input tokens plus 256 generated tokens fill the 131,072-token context.

| Input tokens | Without MTP | With MTP (window 2) | Output parity |
|---:|---:|---:|---|
| 45 | 48.33 | 71.69 | all 256 tokens identical |
| 4,096 | 46.29 | 75.48 | all 256 tokens identical |
| 130,816 | 38.77 | 61.85 | all 256 tokens identical |

Short-prompt MTP windows 2 / 3 / 4 measured 71.69 / 72.22 / 68.72 tokens/sec. The small difference between 2 and 3 is below what a single run can settle; window 2 was used for the full context sweep. Every short-prompt window matched the no-MTP output.

The first 128K no-MTP reference overlapped the final draft-weight download and preparation. Treat its absolute timing as approximate. All MTP measurements ran after preparation completed, one inference process at a time.

Build commit: `67ba504da769b36acb22a3618d63532a60379049`. Engine SHA256: `a13cd444c3383f55e09e9436e5537ff725ed6b003f6839280e3be414735214c5`. MTP GGUF SHA256: `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`.

The companion [JSON](results.json) contains raw output tokens, acceptance counts, prompt/decode timings and generated text; [CSV](results.csv) contains the timing rows.

The local OpenAI-compatible API passed with MTP and 131,072-token capacity configured: a 45-token story prompt produced 256 tokens at 67.3 tokens/sec, with 82 of 121 draft tokens accepted. This API check used a short input; the full-length input sweep above used the engine directly. The test server and engine were stopped after validation.
