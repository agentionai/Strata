#pragma once

namespace strata::prefill {
// Before a routed-only copy overwrites `next_slot`, every pending gather reader
// of that slot must be queued and its release event recorded. Resident experts
// do not use or advance the staging ring. `n` entries are live in `slots`.
inline bool gather_before_reuse(const int* slots, int n, int next_slot, bool resident) {
    if (resident) return false;
    for (int i = 0; i < n; ++i) if (slots[i] == next_slot) return true;
    return false;
}
} // namespace strata::prefill
