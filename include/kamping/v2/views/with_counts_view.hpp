#pragma once

#include <span>
#include <utility>

#include "kamping/v2/views/adaptor.hpp"
#include "kamping/v2/views/all.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/views/view_interface.hpp"

namespace kamping {
namespace ranges {

template <typename Base, count_range Counts>
class with_counts_view : public kamping::ranges::view_interface<with_counts_view<Base, Counts>> {
    Base   base_;
    Counts counts_;

public:
    constexpr Base const& base() const& noexcept {
        return base_;
    }
    constexpr Base& base() & noexcept {
        return base_;
    }

    template <typename R, typename C>
    with_counts_view(R&& base, C&& counts)
        : base_(kamping::ranges::all(std::forward<R>(base))),
          counts_(kamping::ranges::all(std::forward<C>(counts))) {}

    constexpr Counts const& counts() const& { return counts_; }
    constexpr Counts&       counts() &      { return counts_; }
    constexpr Counts&&      counts() &&     { return std::move(counts_); }

    constexpr std::pair<Base, Counts> extract() && {
        return {std::move(base_), std::move(counts_)};
    }

    std::span<int const> mpi_sizev() const {
        return {counts_};
    }
};

template <typename R, typename C>
with_counts_view(R&&, C&&) -> with_counts_view<kamping::ranges::all_t<R>, kamping::ranges::all_t<C>>;

template <typename Base, typename Counts>
inline constexpr bool enable_borrowed_buffer<with_counts_view<Base, Counts>> =
    enable_borrowed_buffer<Base> && enable_borrowed_buffer<Counts>;

} // namespace ranges

namespace views {

inline constexpr kamping::ranges::adaptor<1, decltype([](auto&& r, auto&& counts) {
                                              return kamping::ranges::with_counts_view(
                                                  std::forward<decltype(r)>(r),
                                                  std::forward<decltype(counts)>(counts)
                                              );
                                          })>
    with_counts{};

} // namespace views
} // namespace kamping
