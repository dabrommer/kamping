#pragma once

#include <span>
#include <vector>

#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/tags.hpp"

namespace kamping::ranges {

/// Standalone counts buffer for variadic collectives.
///
/// Wraps a user-provided (or internally owned) counts container. The infer()
/// machinery pre-allocates via set_comm_size() (when resize=true) and writes
/// directly into counts().data() via MPI — no copy needed.
///
/// Displacements are intentionally not provided here; compose with with_displs()
/// or with_auto_displs() to attach them.
///
/// Template parameters:
///   Counts — the wrapped counts range (after all() wrapping)
///   resize — if true, set_comm_size() resizes via resize_for_receive(); if false,
///            the buffer must already have the right size before infer() runs.
///
/// Typical infer() usage (resize=true):
///   counts_buf.set_comm_size(comm_size);
///   MPI_Allgather(&send_count, 1, MPI_INT, counts_buf.counts().data(), 1, MPI_INT, comm);
///   counts_buf.commit_counts();
///   // mpi_sizev() is now valid; pass counts_buf.counts() to with_counts()
///   // or pipe with with_displs / with_auto_displs to attach displacements
template <count_range Counts, bool resize = false>
    requires(!resize || has_resize<Counts> || has_mpi_resize_for_receive<Counts>)
class auto_counts_view {
    Counts counts_;

public:
    /// Construct from a user-provided counts buffer (no resize; buffer must be pre-sized).
    template <typename C>
    explicit auto_counts_view(C&& counts) : counts_(kamping::ranges::all(std::forward<C>(counts))) {}

    /// Construct from a user-provided counts buffer with resize enabled (tag dispatch).
    template <typename C>
    auto_counts_view(kamping::v2::resize_t, C&& counts)
        : counts_(kamping::ranges::all(std::forward<C>(counts))) {}

    /// Direct access to the wrapped counts container for ownership transfer and MPI writes.
    Counts const& counts() const& { return counts_; }
    Counts&       counts() &      { return counts_; }
    Counts&&      counts() &&     { return std::move(counts_); }

    /// Pre-allocates or resizes the counts buffer for comm_size processes.
    /// Only resizes when resize=true; otherwise this is a no-op (buffer must already
    /// have the correct size).
    void set_comm_size(int n) {
        if constexpr (resize) {
            kamping::ranges::resize_for_receive(counts_, static_cast<std::ptrdiff_t>(n));
        }
    }

    /// Signal that MPI has finished writing into counts(). Currently a no-op,
    /// but present as an explicit protocol step for clarity and future extensibility.
    void commit_counts() {}

    std::span<int const> mpi_sizev() const {
        return {counts_};
    }
};

template <typename C>
auto_counts_view(C&&) -> auto_counts_view<kamping::ranges::all_t<C>>;

template <typename C>
auto_counts_view(kamping::v2::resize_t, C&&) -> auto_counts_view<kamping::ranges::all_t<C>, true>;

template <typename Counts, bool resize>
inline constexpr bool enable_borrowed_buffer<auto_counts_view<Counts, resize>> = enable_borrowed_buffer<Counts>;

} // namespace kamping::ranges

namespace kamping::views {

/// Factory for auto_counts_view.
///
/// Usage:
///   auto_counts()         — owned std::vector<int>, resized by infer() via set_comm_size()
///   auto_counts(buf)      — user-provided buffer, no resize (must be pre-sized)
///   auto_counts(resize, buf) — user-provided buffer, resized by infer() via set_comm_size()
///
/// The returned view is passed to a collective as the recv_counts argument.
/// Pipe with with_displs() or with_auto_displs() to attach displacements.
inline constexpr struct auto_counts_fn {
    /// 0-arg: construct an owned Container (default std::vector<int>) that will be
    /// auto-resized by infer() via set_comm_size().
    template <typename Container = std::vector<int>>
    constexpr auto operator()() const {
        return kamping::ranges::auto_counts_view(kamping::v2::resize, Container{});
    }

    /// 1-arg: user-provided buffer, no resize (buffer must already have correct size).
    template <typename C>
        requires std::ranges::range<std::remove_cvref_t<C>>
    constexpr auto operator()(C&& counts) const {
        return kamping::ranges::auto_counts_view(std::forward<C>(counts));
    }

    /// 2-arg: resize tag + user-provided buffer; infer() will resize via set_comm_size().
    template <typename C>
        requires std::ranges::range<std::remove_cvref_t<C>>
    constexpr auto operator()(kamping::v2::resize_t, C&& counts) const {
        return kamping::ranges::auto_counts_view(kamping::v2::resize, std::forward<C>(counts));
    }
} auto_counts{};

} // namespace kamping::views
