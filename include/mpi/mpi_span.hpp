#pragma once

#include <span>

#include <mpi.h>

/// @file
/// Concrete minimal buffer types for callers who want to talk directly to the
/// core MPI wrappers without going through the view pipeline.
///
/// `mpi_span`   — satisfies `send_buffer` and `recv_buffer`
/// `mpi_span_v` — satisfies `send_buffer_v` and `recv_buffer_v`

namespace mpi::experimental {

/// @brief Non-owning view over a contiguous buffer for use as an MPI send or
/// receive buffer. Satisfies `send_buffer` (data() is void*, convertible to
/// void const*) and `recv_buffer` (data() is void*).
struct mpi_span {
    void*          data; ///< Pointer to the first element.
    std::ptrdiff_t size; ///< Number of elements.
    MPI_Datatype   type; ///< MPI datatype of each element.

    void*          mpi_data()  noexcept       { return data; }
    std::ptrdiff_t mpi_count() const noexcept { return size; }
    MPI_Datatype   mpi_type()  const noexcept { return type; }
};

/// @brief Non-owning view over a variadic (per-rank) MPI buffer. Satisfies
/// `send_buffer_v` and `recv_buffer_v`.
///
/// `counts` and `displs` must point to arrays of length `comm_size`. The
/// caller is responsible for keeping those arrays alive.
struct mpi_span_v {
    void*        data;      ///< Pointer to the first element.
    MPI_Datatype type;      ///< MPI datatype of each element.
    int const*   counts;    ///< Per-rank element counts (length: comm_size).
    int const*   displs;    ///< Per-rank displacements  (length: comm_size).
    int          comm_size; ///< Number of ranks.

    void*        mpi_data() noexcept       { return data; }
    MPI_Datatype mpi_type() const noexcept { return type; }

    std::span<int const> mpi_counts() const noexcept {
        return {counts, static_cast<std::size_t>(comm_size)};
    }

    std::span<int const> mpi_displs() const noexcept {
        return {displs, static_cast<std::size_t>(comm_size)};
    }
};

} // namespace mpi::experimental
