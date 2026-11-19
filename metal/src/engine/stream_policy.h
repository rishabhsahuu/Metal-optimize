// metal/src/engine/stream_policy.h - the adaptive expert tier's arithmetic, as plain functions so it can be tested
// without a Mac (metal/tests/stream_policy_test.cpp), the way arena_budget.h is.
//
// The full design is docs/MAC_ADAPTIVE_TIER.md. In short: the streaming arena used to choose its eviction victim by age
// alone (the LRU tail), which evicts a hot expert that a topic change left untouched for a few windows. These functions
// add decayed routing counts and let the choice prefer the coldest slot within a bounded scan, never a pinned or busy
// one. Nothing here changes what an expert's bytes are: on the Mac a slot holds the GGUF bytes unchanged, so placement
// changes speed only, never a token.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::metal_engine::policy {

/// Scale every count by `d`. Called once a window, at the window's first layer.
inline void decay_all(float* heat, std::size_t n, float d) {
    for (std::size_t i = 0; i < n; ++i) heat[i] *= d;
}

/// Add one to each routed expert of one layer. `ids` must not repeat; the caller passes the window's distinct ids.
inline void bump_layer(float* heat, int stride, int layer, const int32_t* ids, int n) {
    if (layer < 0) return;
    float* row = heat + (std::size_t) layer * (std::size_t) stride;
    for (int i = 0; i < n; ++i)
        if (ids[i] >= 0 && ids[i] < stride) row[ids[i]] += 1.0f;
}

/// The coldest usable slot among at most `scan` slots walked from the LRU tail, or -1 when none is usable.
///
/// `scan` bounds the work on the host turn (a full walk is O(slots) per miss). An expert the run has never routed counts
/// 0, colder than any routed one; an empty slot ranks coldest of all, since nothing is lost by taking it. Ties keep the
/// slot nearer the tail, so the age order still decides between equals.
///
/// `prev(k)` is slot k's link towards the head, `usable(k)` false for a busy, pinned, retired or held slot, and `heat(k)`
/// the count of the expert it holds (-1e30 for an empty slot).
template <class Prev, class Usable, class Heat>
inline int coldest(int tail, int scan, Prev prev, Usable usable, Heat heat) {
    int best = -1;
    float best_h = 0.0f;
    int visited = 0;
    for (int k = tail; k >= 0 && visited < scan; k = prev(k)) {
        ++visited;
        if (!usable(k)) continue;
        const float h = heat(k);
        if (best < 0 || h < best_h) { best = k; best_h = h; }
    }
    return best;
}

}  // namespace strata::metal_engine::policy
