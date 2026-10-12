// CPU-only model of the routed lookahead and partial-group lifetime rule.
// The helper tested here is the same helper the engine calls before stage_one.
#include "strata/prefill/gather_schedule.hpp"
#include <array>
#include <cstdio>
#include <vector>

namespace {
bool simulate(const std::vector<bool>& resident, bool guard) {
    constexpr int stage = 8, group = 16;
    std::array<int, stage> owners{};
    std::array<int, group> pending_slots{}, experts{}, source_slots{};
    std::vector<int> stage_of(resident.size(), -1), output(resident.size(), -1);
    int next = 0, nslots = 0, first = 0, n = 0;
    size_t staged = 0;
    auto flush = [&]() {
        for (int i = first; i < n; ++i)
            output[experts[i]] = source_slots[i] < 0 ? experts[i] : owners[source_slots[i]];
        first = n; nslots = 0;
    };
    for (size_t j = 0; j < resident.size(); ++j) {
        while (staged < resident.size() && staged <= j + stage - 1) {
            if (guard && strata::prefill::gather_before_reuse(pending_slots.data(), nslots, next, resident[staged]))
                flush();
            if (!resident[staged]) {
                owners[next] = (int) staged;
                stage_of[staged] = next;
                next = (next + 1) % stage;
            }
            ++staged;
        }
        const int q = (int) j % group;
        experts[q] = (int) j;
        source_slots[q] = stage_of[j];
        n = q + 1;
        if (stage_of[j] >= 0) {
            if (nslots == group) return false;
            pending_slots[nslots++] = stage_of[j];
        }
        if (q + 1 == group || j + 1 == resident.size()) {
            flush(); first = n = 0;
        }
    }
    for (size_t i = 0; i < output.size(); ++i) if (output[i] != (int) i) return false;
    return true;
}
} // namespace

int main() {
    int cases = 0, caught = 0;
    for (int length : {1, 7, 8, 9, 15, 16, 17, 31, 127, 511}) {
        for (int pattern = 0; pattern < 20; ++pattern) {
            std::vector<bool> resident(length);
            for (int i = 0; i < length; ++i)
                resident[i] = pattern == 0 || (pattern != 1 && ((i * 37 + i / 7 + pattern * 11) % 23) < pattern);
            if (!simulate(resident, true)) { std::puts("FAIL guarded schedule"); return 1; }
            if (!simulate(resident, false)) ++caught;
            ++cases;
        }
    }
    if (caught == 0) { std::puts("FAIL model did not detect missing lifetime guard"); return 1; }
    std::printf("PASS %d schedules; detected %d corruptions when the reuse guard is removed\n", cases, caught);
    return 0;
}
