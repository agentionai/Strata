// Byte-exact native gather screens, including Gyro's gate/up/down byte sizes.
// No quantization/dequantization is performed; padding and untouched slots must stay intact.
#include "strata/prefill/moe_mmq.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace {
void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
struct Buffer {
    uint8_t* ptr = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&ptr, bytes)); }
    ~Buffer() { if (ptr) (void) hipFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

void run(int first, int n, bool padded) {
    constexpr size_t half = 640 * (2560 / 128) * 26; // TQK6, gate and up
    constexpr size_t down = 2560 * (640 / 128) * 30; // TQK7
    const size_t up_off = half + (padded ? 32 : 0);
    const size_t down_off = up_off + half + (padded ? 64 : 0);
    const size_t blob_stride = down_off + down + 32;
    const size_t gu_stride = 2 * half + (padded ? 16 : 0);
    const size_t d_stride = down + (padded ? 32 : 0);
    constexpr int slots = mmq::kGatherGroupMax;
    std::vector<uint8_t> input(slots * blob_stride);
    for (size_t i = 0; i < input.size(); ++i) input[i] = (uint8_t) ((i * 17 + i / 13) % 251);
    Buffer src(input.size()), gu(slots * gu_stride + 64), dn(slots * d_stride + 64);
    check(hipMemcpy(src.ptr, input.data(), input.size(), hipMemcpyHostToDevice));
    check(hipMemset(gu.ptr, 0xa5, slots * gu_stride + 64));
    check(hipMemset(dn.ptr, 0xa5, slots * d_stride + 64));
    mmq::GatherGroup group;
    group.first = first; group.n = n;
    for (int q = 0; q < slots; ++q) group.blob[q] = src.ptr + q * blob_stride;
    if (!mmq::gather_native_group(group, up_off, half, down_off, down,
                                 gu.ptr, gu_stride, dn.ptr, d_stride, nullptr))
        throw std::runtime_error("aligned gather declined");
    std::vector<uint8_t> expected_gu(slots * gu_stride + 64, 0xa5), expected_dn(slots * d_stride + 64, 0xa5);
    for (int q = first; q < n; ++q) {
        const auto start = input.begin() + q * blob_stride;
        std::copy_n(start, half, expected_gu.begin() + q * gu_stride);
        std::copy_n(start + up_off, half, expected_gu.begin() + q * gu_stride + half);
        std::copy_n(start + down_off, down, expected_dn.begin() + q * d_stride);
    }
    std::vector<uint8_t> actual_gu(expected_gu.size()), actual_dn(expected_dn.size());
    check(hipMemcpy(actual_gu.data(), gu.ptr, actual_gu.size(), hipMemcpyDeviceToHost));
    check(hipMemcpy(actual_dn.data(), dn.ptr, actual_dn.size(), hipMemcpyDeviceToHost));
    if (actual_gu != expected_gu || actual_dn != expected_dn)
        throw std::runtime_error("byte mismatch or padding write");
    // A declined unaligned request must enqueue nothing and leave every byte unchanged.
    group.blob[first] += 1;
    if (mmq::gather_native_group(group, up_off, half, down_off, down,
                                gu.ptr, gu_stride, dn.ptr, d_stride, nullptr))
        throw std::runtime_error("unaligned gather accepted");
    check(hipMemcpy(actual_gu.data(), gu.ptr, actual_gu.size(), hipMemcpyDeviceToHost));
    check(hipMemcpy(actual_dn.data(), dn.ptr, actual_dn.size(), hipMemcpyDeviceToHost));
    if (actual_gu != expected_gu || actual_dn != expected_dn)
        throw std::runtime_error("declined gather changed bytes");
    group.n = slots + 1;
    if (mmq::gather_native_group(group, up_off, half, down_off, down,
                                gu.ptr, gu_stride, dn.ptr, d_stride, nullptr))
        throw std::runtime_error("oversized group accepted");
    std::printf("PASS gather first=%d n=%d padded=%d, byte-exact, guards intact, declines intact\n", first, n, padded);
}
} // namespace

int main() {
    try {
        for (bool padded : {false, true}) {
            run(0, 1, padded); run(0, 7, padded); run(0, 16, padded); run(5, 16, padded);
        }
        std::puts("FAILURES: 0");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL gather: %s\n", error.what());
        return 1;
    }
}
