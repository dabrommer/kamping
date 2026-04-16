#pragma once

#include <cstddef>

#include <mpi.h>

namespace kamping::v2 {

struct resize_t {};
inline constexpr resize_t resize{};

/// Tag for user-provided displacements that are known to be monotonically non-decreasing
/// (e.g. computed via exclusive_scan). Enables the O(1) tight-bound formula
/// displs.back() + counts.back() in resize_v_view instead of O(p) max(counts[i]+displs[i]).
struct monotonic_t {};
inline constexpr monotonic_t monotonic{};

/// Sentinel buffer that passes MPI_BOTTOM as the data pointer.
/// Used with derived datatypes that embed absolute addresses.
/// Not a complete data_buffer on its own: compose with views::with_type and views::with_size.
///
///   v2::send(v2::bottom | views::with_type(my_abs_type) | views::with_size(1), dest, comm);
struct bottom_t {
    static void* mpi_data() {
        return MPI_BOTTOM;
    }
};
inline constexpr bottom_t bottom{};

/// Sentinel buffer that passes MPI_IN_PLACE as the data pointer.
/// Used for in-place collectives (e.g. allgather, gather on root) where each rank's
/// contribution is already in the correct position in the receive buffer.
/// The receive buffer must be pre-sized; deferred resize is not supported for in-place operations.
struct inplace_t {
    static void*          mpi_data() { return MPI_IN_PLACE; }
    static std::ptrdiff_t mpi_count() { return 0; }
    static MPI_Datatype   mpi_type() { return MPI_DATATYPE_NULL; }
};
inline constexpr inplace_t inplace{};

/// Sentinel buffer that passes a null pointer as the data pointer.
/// Used for optional root-only buffers in gather/scatter: non-root ranks pass null_buf
/// for the receive/send buffer that MPI ignores on their side.
///
///   v2::gather(local_data, v2::null_buf, root, comm);  // non-root
struct null_buf_t {
    static void*          mpi_data() { return nullptr; }
    static std::ptrdiff_t mpi_count() { return 0; }
    static MPI_Datatype   mpi_type() { return MPI_DATATYPE_NULL; }
};
inline constexpr null_buf_t null_buf{};

} // namespace kamping::v2
