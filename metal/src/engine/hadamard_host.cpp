// Shared host setup can request CUDA-style signs; the Metal Window owns its own sign buffers.
// CUDA/HIP rotation entry points remain fail-fast because Metal dispatches through its Window.
#include "strata/kernels/iq_kernels.hpp"
namespace strata::metal_engine { [[noreturn]] void missing(const char*); }
namespace strata::kernels {
const float* hadamard_device_signs(const float*, int) { return nullptr; }
void hadamard_rows(const float*, float*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("hadamard_rows (use the Metal window)");
}
void hadamard_rows_f16(uint16_t*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("hadamard_rows_f16 (use the Metal prompt)");
}
}
