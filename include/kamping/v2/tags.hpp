#pragma once

namespace kamping::v2 {

struct resize_t {};
inline constexpr resize_t resize{};

/// Tag for user-provided displacements that are known to be monotonically non-decreasing
/// (e.g. computed via exclusive_scan). Enables the O(1) tight-bound formula
/// displs.back() + counts.back() in resize_v_view instead of O(p) max(counts[i]+displs[i]).
struct monotonic_t {};
inline constexpr monotonic_t monotonic{};

} // namespace kamping::v2
