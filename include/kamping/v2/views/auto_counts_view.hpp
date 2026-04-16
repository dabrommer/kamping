#pragma once

#include <span>
#include <vector>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/views/view_interface.hpp"
#include "kamping/v2/tags.hpp"

namespace kamping::ranges {

/// Deferred counts buffer for variadic collectives, wrapping an underlying data buffer.
///
/// Combines a data buffer (Base) with a counts container (Counts). The infer()
/// machinery pre-allocates via set_comm_size() (when resize=true) and writes
/// directly into counts().data() via MPI — no copy needed.
///
/// Displacements are intentionally not provided here; compose with with_displs()
/// or auto_displs() to attach them, e.g.:
///   vec | auto_counts() | auto_displs()
///
/// Template parameters:
///   Base   — the wrapped data buffer (forwarded through base())
///   Counts — the wrapped counts range (after all() wrapping)
///   resize — if true, set_comm_size() resizes via resize_for_receive(); if false,
///            the buffer must already have the right size before infer() runs.
template <typename Base, count_range Counts, bool resize = false>
    requires(!resize || has_resize<Counts> || has_mpi_resize_for_receive<Counts>)
class auto_counts_view : public kamping::ranges::view_interface<auto_counts_view<Base, Counts, resize>> {
    Base   base_;
    Counts counts_;

public:
    constexpr Base const& base() const& noexcept { return base_; }
    constexpr Base&       base() &      noexcept { return base_; }

    /// Construct from a data buffer and counts buffer (no resize; counts must be pre-sized).
    template <typename R, typename C>
    auto_counts_view(R&& base, C&& counts)
        : base_(kamping::ranges::all(std::forward<R>(base))),
          counts_(kamping::ranges::all(std::forward<C>(counts))) {}

    /// Construct from a data buffer and counts buffer with resize enabled (tag dispatch).
    template <typename R, typename C>
    auto_counts_view(kamping::v2::resize_t, R&& base, C&& counts)
        : base_(kamping::ranges::all(std::forward<R>(base))),
          counts_(kamping::ranges::all(std::forward<C>(counts))) {}

    /// Direct access to the wrapped counts container for ownership transfer and MPI writes.
    constexpr Counts const& counts() const& { return counts_; }
    constexpr Counts&       counts() &      { return counts_; }
    constexpr Counts&&      counts() &&     { return std::move(counts_); }

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

template <typename R, typename C>
auto_counts_view(R&&, C&&) -> auto_counts_view<kamping::ranges::all_t<R>, kamping::ranges::all_t<C>>;

template <typename R, typename C>
auto_counts_view(kamping::v2::resize_t, R&&, C&&)
    -> auto_counts_view<kamping::ranges::all_t<R>, kamping::ranges::all_t<C>, true>;

template <typename Base, typename Counts, bool resize>
inline constexpr bool enable_borrowed_buffer<auto_counts_view<Base, Counts, resize>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Counts>;

} // namespace kamping::ranges

namespace kamping::views {

/// 0-arg: owned Container (default std::vector<int>) auto-resized by infer() via set_comm_size().
template <typename Container = std::vector<int>>
constexpr auto auto_counts() {
    return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& counts) {
        return kamping::ranges::auto_counts_view(
            kamping::v2::resize,
            std::forward<decltype(r)>(r),
            std::forward<decltype(counts)>(counts)
        );
    })>{}(Container{});
}

/// 1-arg: user-provided counts buffer, no resize (buffer must already have correct size).
template <typename C>
    requires std::ranges::range<std::remove_cvref_t<C>>
constexpr auto auto_counts(C&& counts) {
    return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& counts) {
        return kamping::ranges::auto_counts_view(
            std::forward<decltype(r)>(r),
            std::forward<decltype(counts)>(counts)
        );
    })>{}(std::forward<C>(counts));
}

/// 2-arg: resize tag + user-provided counts buffer; infer() will resize via set_comm_size().
template <typename C>
    requires std::ranges::range<std::remove_cvref_t<C>>
constexpr auto auto_counts(kamping::v2::resize_t, C&& counts) {
    return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& counts) {
        return kamping::ranges::auto_counts_view(
            kamping::v2::resize,
            std::forward<decltype(r)>(r),
            std::forward<decltype(counts)>(counts)
        );
    })>{}(std::forward<C>(counts));
}

} // namespace kamping::views
