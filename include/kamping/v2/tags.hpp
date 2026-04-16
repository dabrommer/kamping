#pragma once

#include <mpi/sentinels.hpp>

namespace kamping::v2 {

/// Tag used to request in-place resize of a counts or data buffer.
struct resize_t {};
inline constexpr resize_t resize{};

/// Tag for user-provided displacements that are known to be monotonically
/// non-decreasing (e.g. computed via exclusive_scan). Enables the O(1)
/// tight-bound formula displs.back() + counts.back() in resize_v_view
/// instead of the O(p) max(counts[i]+displs[i]) formula.
struct monotonic_t {};
inline constexpr monotonic_t monotonic{};

// Re-export core sentinels into kamping::v2 for backwards compatibility.
using mpi::experimental::bottom;
using mpi::experimental::bottom_t;
using mpi::experimental::inplace;
using mpi::experimental::inplace_t;
using mpi::experimental::null_buf;
using mpi::experimental::null_buf_t;

} // namespace kamping::v2
