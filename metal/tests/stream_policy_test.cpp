// metal/tests/stream_policy_test.cpp - the adaptive expert tier's arithmetic (src/engine/stream_policy.h): the decay, the
// per-layer counts, and the bounded heat choice of an eviction victim (docs/MAC_ADAPTIVE_TIER.md).  No GPU, no model:
// plain arrays.  Exit 0 = every check passed.
#include "../src/engine/stream_policy.h"

#include <cstdio>
#include <vector>

using namespace strata::metal_engine::policy;

static int failures = 0;
static void check(bool ok, const char* what, double got) {
    std::printf("%s %-72s %.3f\n", ok ? "ok  " : "FAIL", what, got);
    failures += !ok;
}

int main() {
    // ---- decay: every count scaled once a window
    std::vector<float> heat = {1.0f, 2.0f, 4.0f};
    decay_all(heat.data(), heat.size(), 0.5f);
    check(heat[0] == 0.5f && heat[1] == 1.0f && heat[2] == 2.0f, "decay 0.5 halves every count", heat[2]);

    // ---- one layer's counts: stride 4, layer 1, ids {0,3}; an out-of-range id is ignored
    std::vector<float> grid(8, 0.0f);
    const int32_t ids[] = {0, 3, 9, -1};
    bump_layer(grid.data(), 4, 1, ids, 4);
    check(grid[4] == 1.0f && grid[7] == 1.0f && grid[0] == 0.0f && grid[5] == 0.0f,
          "bump_layer: layer 1's ids 0 and 3, nothing else", grid[4] + grid[7]);
    bump_layer(grid.data(), 4, -1, ids, 2);
    check(grid[0] == 0.0f, "bump_layer: a negative layer does nothing", grid[0]);

    // ---- the victim: five slots, the LRU tail at 4 (walk 4 -> 3 -> 2 -> 1 -> 0; prev[k] = the next slot towards the head)
    const int prev[5] = {-1, 0, 1, 2, 3};
    const int16_t layer[5] = {0, 0, -1, 0, 0};
    const int16_t expert[5] = {1, 2, 0, 3, 4};
    const float h[2 * 4] = {0.0f, 5.0f, 1.0f, 3.0f, 9.0f, 0.0f, 0.0f, 0.0f};
    auto heat_of = [&](int k) { return layer[k] < 0 ? -1e30f : h[layer[k] * 4 + expert[k]]; };
    auto all = [](int) { return true; };

    check(coldest(4, 5, [&](int k) { return prev[k]; }, all, heat_of) == 2, "an empty slot is taken before any expert",
          coldest(4, 5, [&](int k) { return prev[k]; }, all, heat_of));
    // with slot 2 (empty) unusable, the coldest expert wins: slot 1 has 1.0, the tail slot 4 has 9.0
    auto not2 = [](int k) { return k != 2; };
    check(coldest(4, 5, [&](int k) { return prev[k]; }, not2, heat_of) == 1, "coldest wins, not the tail (1.0 against 9.0)",
          coldest(4, 5, [&](int k) { return prev[k]; }, not2, heat_of));
    // scan 1 looks at the tail only: age decides, as it does today
    check(coldest(4, 1, [&](int k) { return prev[k]; }, not2, heat_of) == 4, "scan 1 is the plain LRU tail",
          coldest(4, 1, [&](int k) { return prev[k]; }, not2, heat_of));
    // ties go to the slot nearer the tail: equal counts, the tail first
    auto equal = [&](int) { return 0.0f; };
    check(coldest(4, 5, [&](int k) { return prev[k]; }, not2, equal) == 4, "a tie keeps the older slot (nearer the tail)",
          coldest(4, 5, [&](int k) { return prev[k]; }, not2, equal));
    // every slot busy: nothing to take
    auto none = [](int) { return false; };
    check(coldest(4, 5, [&](int k) { return prev[k]; }, none, heat_of) == -1, "no usable slot returns -1",
          coldest(4, 5, [&](int k) { return prev[k]; }, none, heat_of));

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
