#pragma once

#include <mpi.h>

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
using mpi::experimental::null_buf_v;
using mpi::experimental::null_buf_v_t;

/// Deferred variadic null recv buffer for non-root ranks in gather-family collectives.
///
/// Satisfies both `recv_buffer_v` (accepted by core gatherv/scatterv) and
/// `deferred_recv_buf_v` (triggers the count-gathering phase in infer()).
/// All deferred-protocol methods (`set_comm_size`, `commit_counts`) are no-ops;
/// `mpi_counts()` returns a null-data span that MPI ignores on non-root.
///
/// Use `auto_null_recv_v()` (the factory function) rather than storing an instance,
/// so the object is always a non-const rvalue — required for `deferred_recv_buf_v`
/// which needs mutable access to `mpi_counts()`:
///
///   v2::gatherv(local_data, v2::auto_null_recv_v(), root, comm);   // non-root
///   v2::gatherv(local_data, v2::auto_recv_v<int>(), root, comm);   // root
struct auto_null_recv_v_t {
    static void*                mpi_ptr()    noexcept { return nullptr; }
    static MPI_Datatype         mpi_type()   noexcept { return MPI_DATATYPE_NULL; }
    std::span<int>       mpi_counts()        noexcept { return {}; }
    std::span<int const> mpi_counts() const  noexcept { return {}; }
    static std::span<int const> mpi_displs() noexcept { return {}; }
    static void set_comm_size(int)           noexcept {}
    static void commit_counts()              noexcept {}
};

/// Factory function returning a fresh `auto_null_recv_v_t` rvalue.
/// Always call as `auto_null_recv_v()` — never store the result in a `const` variable,
/// as `deferred_recv_buf_v` requires mutable access to `mpi_counts()`.
inline auto_null_recv_v_t auto_null_recv_v() noexcept { return {}; }

} // namespace kamping::v2
