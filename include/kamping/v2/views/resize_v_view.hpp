#pragma once

#include <algorithm>
#include <ranges>

#include "kamping/v2/ranges/adaptor.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/ranges/view_interface.hpp"

namespace kamping::ranges {

/// Variadic-receive counterpart of resize_buf_view.
///
/// Wraps a base buffer that already exposes mpi_sizev() (per-process counts) and
/// mpi_displs() (per-process displacements). On mpi_data() the underlying data
/// buffer is resized to fit all incoming elements.
///
/// The required size is max(counts[i] + displs[i]) over all i — the same formula
/// as kamping::internal::compute_required_recv_buf_size_in_vectorized_communication.
/// Correct for both monotonically-increasing (auto-computed) and user-provided
/// non-monotonic displacements.
///
/// Typical composition:
///   recv_buf | with_counts(auto_counts()) | with_auto_displs(resize, displs) | resize_v
///   recv_buf | with_counts(auto_counts()) | with_displs(user_displs)         | resize_v
template <typename Base>
    requires has_mpi_sizev<Base> && has_mpi_displs<Base>
class resize_v_view : public view_interface<resize_v_view<Base>> {
    Base base_;

public:
    template <typename R>
    explicit resize_v_view(R&& base) : base_(kamping::ranges::all(std::forward<R>(base))) {}

    constexpr Base&       base() &      noexcept { return base_; }
    constexpr Base const& base() const& noexcept { return base_; }

    // mpi_sizev, mpi_displs, mpi_type, mpi_size are all forwarded through view_interface.

    auto mpi_data() {
        auto const& counts     = kamping::ranges::sizev(base_);
        auto const& displs     = kamping::ranges::displs(base_);
        auto const* counts_ptr = std::ranges::data(counts);
        auto const* displs_ptr = std::ranges::data(displs);
        auto const  n          = std::ranges::size(counts);
        std::ptrdiff_t total   = 0;
        // Fast path: monotonically increasing displs (e.g. exclusive_scan or user-declared).
        // Tight O(1) bound: last_displ + last_count.
        // General path: non-monotonic displs require max(displs[i] + counts[i]) over all i.
        if constexpr (has_monotonic_displs<Base>) {
            if (n > 0 && base_.displs_monotonic()) {
                total = static_cast<std::ptrdiff_t>(displs_ptr[n - 1]) + counts_ptr[n - 1];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                total = std::max(total, static_cast<std::ptrdiff_t>(displs_ptr[i] + counts_ptr[i]));
            }
        }
        kamping::ranges::resize_for_receive(base_, total);
        return kamping::ranges::data(base_);
    }
};

template <typename R>
resize_v_view(R&&) -> resize_v_view<kamping::ranges::all_t<R>>;

template <typename Base>
inline constexpr bool enable_borrowed_buffer<resize_v_view<Base>> = enable_borrowed_buffer<Base>;

} // namespace kamping::ranges

namespace kamping::views {

/// Wraps a base buffer (which must already provide mpi_sizev() and mpi_displs()
/// via e.g. with_counts | with_auto_displs or with_counts | with_displs) so the
/// underlying data buffer is resized to the correct total size on mpi_data().
/// Use as: buf | with_counts(...) | with_auto_displs(...) | resize_v
inline constexpr struct resize_v_fn : kamping::ranges::adaptor_closure<resize_v_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::ranges::resize_v_view(kamping::ranges::all(std::forward<R>(r)));
    }
} resize_v{};

} // namespace kamping::views
