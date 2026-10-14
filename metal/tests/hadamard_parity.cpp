#include "mtl_rt.h"
#include "ref_quant.h"
#include "strata/kernels/cpu/hadamard.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"hadamard.metal"}, "#define NO_GGML_COMMON 1\n");
    bool ok = true;
    for (int width : {640, 2560}) for (int block : {32, 128}) for (bool half : {false, true})
    for (bool identity : {false, true}) for (bool inplace : {false, true}) {
        constexpr int rows = 8, cap = 11;
        auto x = cx.alloc((half ? 2 : 4) * cap * width), y = cx.alloc(x.bytes);
        auto signs = cx.alloc(4 * width), meta = cx.alloc(16);
        ref::Rng rng(73);
        for (int i = 0; i < width; ++i) signs.as<float>()[i] = identity || (rng.next() & 1) ? 1.f : -1.f;
        std::vector<float> want(cap * width);
        for (int i = 0; i < cap * width; ++i) {
            float v = rng.sym();
            if (half) { x.as<uint16_t>()[i] = ref::f32_to_f16(v); v = ref::f16_to_f32(x.as<uint16_t>()[i]); }
            else x.as<float>()[i] = v;
            want[i] = v;
        }
        std::memcpy(y.ptr, x.ptr, x.bytes);
        for (int r = 0; r < rows; ++r)
            strata::kernels::cpu::hadamard_rotate(want.data() + r * width, width, block, signs.as<float>());
        meta.as<uint32_t>()[1] = rows;
        struct { uint32_t width, rows, block, indirect; } a{(uint32_t)width, cap, (uint32_t)block, 1};
        auto p = cx.pipeline(lib, half ? "hadamard_f16" : "hadamard_f32");
        mtl::Dispatch d; d.p = &p;
        d.args = {mtl::Arg::B(x), mtl::Arg::B(inplace ? x : y), mtl::Arg::B(signs), mtl::Arg::V(a), mtl::Arg::B(meta)};
        d.grid[0] = width / block; d.grid[1] = cap; d.tg[0] = 128;
        cx.run({d});
        const auto& out = inplace ? x : y;
        double maxerr = 0;
        for (int i = 0; i < cap * width; ++i) {
            const float expected = half ? ref::f16_to_f32(ref::f32_to_f16(want[i])) : want[i];
            const float got = half ? ref::f16_to_f32(out.as<uint16_t>()[i]) : out.as<float>()[i];
            maxerr = std::max(maxerr, double(std::fabs(got - expected)));
        }
        const bool pass = maxerr == 0;
        ok &= pass;
        std::printf("Hadamard width %d block %d %s %s %s indirect rows: max error %.2g %s\n", width, block,
                    half ? "FP16" : "FP32", identity ? "identity" : "explicit signs", inplace ? "in place" : "separate", maxerr, pass ? "PASS" : "FAIL");
        cx.free(x); cx.free(y); cx.free(signs); cx.free(meta);
    }
    // Independent reference: the fork's own to_float vs Metal's CPU reference, all bit-packed boundary steps.
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, true});
    for (int type : {ref::TQ2_T, ref::TQK6, ref::TQK7}) {
        if (type >= GGML_TYPE_COUNT) { std::fprintf(stderr, "trellis parity requires agentionai/llama.cpp\n"); return 2; }
        constexpr int n = 128 * 256;
        std::vector<uint8_t> w(ref::row_bytes(type, n));
        std::vector<float> own(n), ggml(n);
        ref::random_blocks(type, w.data(), 1, n, 123);
        ref::dequant(type, w.data(), own.data(), n);
        ggml_get_type_traits((ggml_type)type)->to_float(w.data(), ggml.data(), n);
        const bool pass = std::memcmp(own.data(), ggml.data(), n * sizeof(float)) == 0;
        ok &= pass;
        std::printf("%s reference vs ggml to_float: %s\n", ref::type_name(type), pass ? "PASS" : "FAIL");
    }
    ggml_free(ctx);
    return ok ? 0 : 1;
}
