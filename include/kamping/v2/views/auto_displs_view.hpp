#pragma once

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "kamping/v2/ranges/adaptor.hpp"
#include "kamping/v2/ranges/all.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/ranges/view_interface.hpp"
#include "kamping/v2/tags.hpp"

namespace kamping {
namespace ranges {

template <typename Base, count_range Displs, bool resize = false>
    requires has_mpi_sizev<Base> && std::ranges::output_range<Displs, int>
             && (!resize || has_resize<Displs> || has_mpi_resize_for_receive<Displs>)
class auto_displs_view : public kamping::ranges::view_interface<auto_displs_view<Base, Displs>> {
    Base           base_;
    mutable Displs displs_;
    mutable bool   needs_to_compute_displs_ = true;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R, typename C>
    auto_displs_view(R&& base, C&& displs)
        : base_(kamping::ranges::all(std::forward<R>(base))),
          displs_(kamping::ranges::all(std::forward<C>(displs))) {}

    template <typename R, typename C>
    auto_displs_view(kamping::v2::resize_t, R&& base, C&& displs)
        : base_(kamping::ranges::all(std::forward<R>(base))),
          displs_(kamping::ranges::all(std::forward<C>(displs))) {}

    constexpr Displs const& displs() const& { return displs_; }
    constexpr Displs&       displs() &      { return displs_; }
    constexpr Displs&&      displs() &&     { return std::move(displs_); }

    constexpr std::pair<Base, Displs> extract() && {
        return {std::move(base_), std::move(displs_)};
    }

    /// Displacements are always computed via exclusive_scan — monotonically non-decreasing.
    constexpr bool displs_monotonic() const { return true; }

    /// Invalidates the cached displacements so they will be recomputed on the
    /// next mpi_displs() call. Called automatically when commit_counts() propagates
    /// through from an inner auto_counts_view.
    void commit_counts() {
        needs_to_compute_displs_ = true;
    }

    std::span<int const> mpi_displs() const {
        if (needs_to_compute_displs_) {
            auto&& counts = kamping::ranges::sizev(base());
            if constexpr (resize) {
                if (std::ranges::size(displs_) < std::ranges::size(counts)) {
                    kamping::ranges::resize_for_receive(
                        displs_, static_cast<std::ptrdiff_t>(std::ranges::size(counts))
                    );
                }
            }
            KAMPING_ASSERT(std::ranges::size(displs_) >= std::ranges::size(counts));
            std::exclusive_scan(std::ranges::begin(counts), std::ranges::end(counts), std::ranges::begin(displs_), 0);
            needs_to_compute_displs_ = false;
        }
        return {displs_};
    }
};

template <typename R, typename C>
auto_displs_view(R&&, C&&) -> auto_displs_view<kamping::ranges::all_t<R>, kamping::ranges::all_t<C>>;

template <typename R, typename C>
auto_displs_view(kamping::v2::resize_t, R&&, C&&)
    -> auto_displs_view<kamping::ranges::all_t<R>, kamping::ranges::all_t<C>, true>;

template <typename Base, typename Displs>
inline constexpr bool enable_borrowed_buffer<auto_displs_view<Base, Displs>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Displs>;

} // namespace ranges

namespace views {

inline constexpr struct auto_displs_fn {
    // 0-arg: owned default container (std::vector<int>), auto-resized on mpi_displs()
    template <typename Container = std::vector<int>>
    constexpr auto operator()() const {
        return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& displs) {
            return kamping::ranges::auto_displs_view(
                kamping::v2::resize,
                std::forward<decltype(r)>(r),
                std::forward<decltype(displs)>(displs)
            );
        })>{}(Container{});
    }

    // (container) partial or (r, container) full — no resize.
    // count-based (adaptor<1>): single arg is always the displs container, not a piped range.
    template <typename... Args>
        requires(sizeof...(Args) >= 1
                 && !std::same_as<
                     std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Args...>>>,
                     kamping::v2::resize_t>)
    constexpr auto operator()(Args&&... args) const {
        return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& displs) {
            return kamping::ranges::auto_displs_view(
                std::forward<decltype(r)>(r),
                std::forward<decltype(displs)>(displs)
            );
        })>{}(std::forward<Args>(args)...);
    }

    // (resize, container) partial or (resize, r, container) full — with resize.
    // Strip the resize tag, then delegate to adaptor<1> for count-based disambiguation.
    template <typename... Args>
        requires(sizeof...(Args) >= 1)
    constexpr auto operator()(kamping::v2::resize_t, Args&&... args) const {
        return kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& displs) {
            return kamping::ranges::auto_displs_view(
                kamping::v2::resize,
                std::forward<decltype(r)>(r),
                std::forward<decltype(displs)>(displs)
            );
        })>{}(std::forward<Args>(args)...);
    }
} auto_displs{};

} // namespace views
} // namespace kamping
