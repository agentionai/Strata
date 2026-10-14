#!/usr/bin/env bash
# Set up agentionai Gyro (Qwen3.8-Flash-Next rotor quants) on Strata: build, download, pack, draft head, config.
# Linux (CUDA/HIP) or Apple Silicon (Metal). See docs/GYRO.md for what each step does.
#   tools/gyro_setup.sh --model S|M --backend cuda|hip|metal [--cuda-arch 120] [--hip-arch gfx1201]
#                       [--variant original|huihui] [--models DIR] [--llama DIR] [--ctx 131072] [--port 8080]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=S VARIANT=original BACKEND=cuda CUDA_ARCH=120 HIP_ARCH=gfx1201 MODELS=models LLAMA=../agention-llama.cpp CTX=131072 PORT=8080
while [ $# -gt 0 ]; do case "$1" in
  --variant) VARIANT=$2; shift 2;; --model) MODEL=$2; shift 2;; --backend) BACKEND=$2; shift 2;; --cuda-arch) CUDA_ARCH=$2; shift 2;;
  --hip-arch) HIP_ARCH=$2; shift 2;; --models) MODELS=$2; shift 2;; --llama) LLAMA=$2; shift 2;;
  --ctx) CTX=$2; shift 2;; --port) PORT=$2; shift 2;; -h|--help) sed -n 2,5p "$0"; exit 0;;
  *) echo "unknown option $1"; exit 2;; esac; done
case "$BACKEND" in cuda|hip|metal) ;; *) echo "--backend must be cuda, hip or metal"; exit 2;; esac
if [ "$BACKEND" = metal ] && { [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; }; then
  echo "--backend metal requires an Apple Silicon Mac"; exit 2
fi
case "$MODEL" in S) F=Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf; P=gyro-s;; M) F=Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf; P=gyro-m;;
  *) echo "--model must be S or M"; exit 2;; esac
REPO=agentionai/Qwen3.8-Flash-Next-Gyro-GGUF
case "$VARIANT" in
  original) ;;
  huihui)
    REPO=agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF
    case "$MODEL" in
      S) F=Qwen3.8-Flash-Next-Gyro-S-abliterated-TQ1_0.gguf;;
      M) F=Qwen3.8-Flash-Next-Gyro-M-abliterated-TQ2_0.gguf;;
    esac
    P=$P-huihui;;
  *) echo "--variant must be original or huihui"; exit 2;;
esac
step() { printf '\n== %s\n' "$*"; }
for t in git python3; do command -v $t >/dev/null || { echo "missing: $t"; exit 1; }; done

step "python environment (also provides cmake >= 3.24 and ninja)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt huggingface_hub hf_transfer
export PATH="$PWD/.venv/bin:$PATH"

step "agentionai llama.cpp fork (GGML with the trellis types) -> $LLAMA"
if [ ! -d "$LLAMA/ggml" ]; then
  git clone --depth 1 https://github.com/agentionai/llama.cpp "$LLAMA"
  # Pin a new checkout to the GGML revision validated with the Metal port. Existing --llama trees are kept.
  git -C "$LLAMA" fetch --depth 1 origin dc255f2a4b6e4211b555b0c05260b2523281f314
  git -C "$LLAMA" checkout --detach dc255f2a4b6e4211b555b0c05260b2523281f314
fi
LLAMA=$(cd "$LLAMA" && pwd)
export STRATA_GGUF_PY="$LLAMA/gguf-py"   # iq_pack.py and mtp_pack.py read GGUF through it

EXE=build/strata SPEC=4
step "build Strata ($BACKEND)"
if [ "$BACKEND" = cuda ]; then
  NV=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' || true)
  if [ "${CUDA_ARCH}" -ge 120 ] && [ "${NV:-0}" -lt 13 ]; then
    echo "warning: RTX 50 (sm_${CUDA_ARCH}) with CUDA ${NV:-?}: use CUDA 13 (12.8 crashes the prompt path, issues #220/#224)"; fi
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DSTRATA_GGML_DIR="$LLAMA"
elif [ "$BACKEND" = hip ]; then
  echo "note: AMD is experimental for Gyro (kernel parity and an end-to-end run validated on gfx1151 only)"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" -DSTRATA_GGML_DIR="$LLAMA" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}"
else
  METAL_CPP=$(.venv/bin/python -c 'from metal.setup_mac import metal_cpp; print(metal_cpp())' | tail -1)
  cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMETAL_CPP_DIR="$METAL_CPP" -DSTRATA_GGML_DIR="$LLAMA"
  EXE=build-metal/strata SPEC=2
fi
cmake --build "${EXE%/*}" -j

step "download $F -> $MODELS (keep it on a fast local SSD: the n-gram table is read per token)"
mkdir -p "$MODELS"
if [ -s "$MODELS/$F" ]; then echo "already there: $MODELS/$F (delete it to download again)"
else HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download "$REPO" "$F" --local-dir "$MODELS"; fi
G=$(cd "$MODELS" && pwd)/$F

PLE=$G
if [ "$BACKEND" = metal ]; then
  step "Metal model (without the disk-only n-gram table, aligned for Apple Silicon)"
  MG="$MODELS/${F%.gguf}-metal.gguf"
  [ -s "$MG" ] || .venv/bin/python metal/tools/metal_repack.py --shard1 "$G" --out "$MG" --verify
  G=$(cd "$MODELS" && pwd)/${F%.gguf}-metal.gguf
fi
step "pack -> packs/$P"
.venv/bin/python tools/iq_pack.py --gguf "$G" --out "packs/$P" --compat-bf16
if [ "$BACKEND" != metal ]; then
  LC=$(build/native_expert_parity --load-check "packs/$P" "$G" 2>&1) || true; echo "$LC"
  echo "$LC" | grep -q " 0 failures" || { echo "load check failed: see above"; exit 1; }
fi

step "draft head (Strata's MTP runtime)"
if [ ! -f mtp/rt/draft_vocab.bin ]; then
  .venv/bin/python tools/mtp_fetch.py fetch --out mtp
  .venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
  .venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
  cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
fi

step "config strata-$P.json"
.venv/bin/python - "$EXE" "$G" "$PLE" "$P" "$CTX" "$PORT" "$BACKEND" "$SPEC" <<'PYCONFIG'
import json, pathlib, sys
exe, native, ple, pack, ctx, port, backend, spec = sys.argv[1:]
args = ["--pack", "packs/" + pack, "--native", native, "--ple-gguf", ple]
if backend == "metal":
    args += ["--native-dense-gguf", native]
args += ["--expert-profile", "data/expert-profile.bin", "--expert-cache", "64" if backend == "metal" else "auto",
         "--prefill", "auto", "--spec", spec, "--spec-min-p", "0.5", "--mtp", "mtp/rt",
         "--max-context", ctx, "--kv", "int8"]
config = dict(exe=exe, args=args, cwd=".", tokenizer="packs/" + pack + "/tokenizer", model_name=pack + "-strata",
              log="strata-" + pack + ".log", host="127.0.0.1", port=int(port))
pathlib.Path("strata-" + pack + ".json").write_text(json.dumps(config, indent=2) + "\n")
PYCONFIG

step "done. Start the server with:"
echo "  .venv/bin/python -m serve.server --engine strata --config strata-$P.json --port $PORT"
echo "  web: http://127.0.0.1:$PORT   API: http://127.0.0.1:$PORT/v1"
