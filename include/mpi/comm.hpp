#pragma once

#include <mpi.h>

namespace mpi::experimental {

// ── CRTP mixin ──────────────────────────────────────────────────────────────
// Provides `.rank()`, `.size()`, and `.native()` for any communicator wrapper.
// Derived must implement `mpi_handle() const -> MPI_Comm`.

template <typename Derived>
class comm_accessors {
    MPI_Comm comm() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @return The rank of the calling process in this communicator.
    [[nodiscard]] int rank() const {
        int r;
        MPI_Comm_rank(comm(), &r);
        return r;
    }

    /// @return The number of processes in this communicator.
    [[nodiscard]] int size() const {
        int s;
        MPI_Comm_size(comm(), &s);
        return s;
    }

    /// @return The underlying `MPI_Comm` (escape hatch).
    [[nodiscard]] MPI_Comm native() const noexcept { return comm(); }
};

// ── comm_view ────────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Comm`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Comm>` so it can be passed
/// directly to any `mpi::experimental::` operation. Does not free the
/// communicator on destruction — use the owning `kamping::v2::comm` for that.
class comm_view : public comm_accessors<comm_view> {
public:
    /// @brief Construct from a raw `MPI_Comm`. The communicator must outlive this view.
    explicit comm_view(MPI_Comm comm) noexcept : _comm(comm) {}

    /// @return The underlying `MPI_Comm` (for `handle()` dispatch).
    [[nodiscard]] MPI_Comm mpi_handle() const noexcept { return _comm; }

private:
    MPI_Comm _comm;
};

} // namespace mpi::experimental
