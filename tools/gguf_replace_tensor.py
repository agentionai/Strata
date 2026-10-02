"""tools/gguf_replace_tensor.py - a GGUF with some tensors taken from other GGUFs, everything else byte-identical.

    python tools/gguf_replace_tensor.py --src model.gguf --out model-strata.gguf \
        --replace token_embd.weight=embd-bf16.gguf [--replace per_layer_token_embd.weight=other.gguf ...]

Unlike a byte splice, a replacement may change the tensor's type and byte size (same shape required): the header's
metadata is copied verbatim, the tensor table is rewritten with the new types and offsets, and the tensor data is
streamed in the source's order.  Used to make Agention's Gyro GGUFs readable by Strata's GPU path (APR-PORT.md):
Gyro-S's Q6_K token_embd -> BF16 (tools/embd_bf16_pack.py --gguf), Gyro-M's Q8_0 PLE table -> Gyro-S's IQ4_NL one.
Check the result with `gguf-pack diff --data`.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
CHUNK = 64 << 20


class Header:
    """Byte positions of a GGUF v3 header: the metadata KV bytes and the tensor table."""

    def __init__(self, path: pathlib.Path):
        with open(path, "rb") as f:
            self.raw = f.read(64 << 20)          # headers here are a few MB (token lists)
        b = self.raw
        magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", b, 0)
        if magic != 0x46554747 or ver != 3:
            raise SystemExit(f"{path}: not a GGUF v3 file")
        p = 24
        self.alignment = 32

        def string(p):
            n = struct.unpack_from("<Q", b, p)[0]
            return b[p + 8:p + 8 + n].decode("utf-8"), p + 8 + n

        def value(p, t):
            if t == 8:
                return string(p)
            if t == 9:
                et, n = struct.unpack_from("<IQ", b, p)
                p += 12
                if et == 8:
                    for _ in range(n):
                        _, p = string(p)
                    return None, p
                return None, p + SIZES[et] * n
            v = struct.unpack_from({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                                    10: "<Q", 11: "<q", 12: "<d"}[t], b, p)[0]
            return v, p + SIZES[t]

        for _ in range(n_kv):
            key, p = string(p)
            t = struct.unpack_from("<I", b, p)[0]
            v, p = value(p + 4, t)
            if key == "general.alignment":
                self.alignment = int(v)
        self.kv_end = p
        self.tensors = []                        # (name, dims, type, offset)
        for _ in range(n_t):
            name, p = string(p)
            nd = struct.unpack_from("<I", b, p)[0]
            dims = list(struct.unpack_from("<%dQ" % nd, b, p + 4))
            p += 4 + 8 * nd
            typ, off = struct.unpack_from("<IQ", b, p)
            p += 12
            self.tensors.append((name, dims, typ, off))
        self.info_end = p
        self.data_start = (p + self.alignment - 1) // self.alignment * self.alignment


def nbytes(dims, typ) -> int:
    name = G.GGML_TYPES.get(typ)
    be, bb = G.BLOCK_GEOMETRY[name]
    n = 1
    for d in dims:
        n *= d
    assert n % be == 0
    return n // be * bb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace", action="append", required=True, help="NAME=FILE.gguf (same tensor name and shape)")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    h = Header(src)
    repl = {}
    for r in a.replace:
        name, path = r.split("=", 1)
        sh = Header(pathlib.Path(path))
        t = next((t for t in sh.tensors if t[0] == name), None)
        mine = next((t for t in h.tensors if t[0] == name), None)
        if t is None or mine is None:
            sys.exit(f"{name}: not in both files")
        if t[1] != mine[1]:
            sys.exit(f"{name}: shape {t[1]} in {path} differs from {mine[1]}")
        repl[name] = (pathlib.Path(path), sh.data_start + t[3], t[2], nbytes(t[1], t[2]))
        print(f"{name}: {G.GGML_TYPES[mine[2]]} {nbytes(mine[1], mine[2])} B -> {G.GGML_TYPES[t[2]]} "
              f"{repl[name][3]} B from {path}")
    # the new tensor table: same order, new types, offsets packed in the source's data order
    order = sorted(range(len(h.tensors)), key=lambda i: h.tensors[i][3])
    new_off, at = {}, 0
    plan = []
    for i in order:
        name, dims, typ, off = h.tensors[i]
        if name in repl:
            p, o, t2, n = repl[name]
            plan.append((name, p, o, n))
            typ_new = t2
        else:
            n = nbytes(dims, typ)
            plan.append((name, src, h.data_start + off, n))
            typ_new = typ
        new_off[name] = (at, typ_new)
        at += (n + h.alignment - 1) // h.alignment * h.alignment
    head = bytearray(h.raw[:h.kv_end])
    for name, dims, typ, off in h.tensors:
        o, t = new_off[name]
        nb = name.encode("utf-8")
        head += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        head += struct.pack("<IQ", t, o)
    head += b"\0" * ((-len(head)) % h.alignment)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as w:
        w.write(head)
        files = {}
        for k, (name, path, o, n) in enumerate(plan):
            f = files.get(path) or files.setdefault(path, open(path, "rb"))
            f.seek(o)
            left = n
            while left:
                buf = f.read(min(CHUNK, left))
                if not buf:
                    sys.exit(f"short read: {name} in {path}")
                w.write(buf)
                left -= len(buf)
            w.write(b"\0" * ((-n) % h.alignment))
            if k % 100 == 0:
                print(f"  {k}/{len(plan)} tensors", flush=True)
        for f in files.values():
            f.close()
    expect = len(head) + at
    if tmp.stat().st_size != expect:
        sys.exit(f"size check failed: {tmp.stat().st_size} != {expect}")
    os.replace(tmp, out)
    print(f"wrote {out} ({expect} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
