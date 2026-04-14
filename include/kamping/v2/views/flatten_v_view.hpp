#pragma once

#include <algorithm>
#include <cstddef>
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

/// Flattens a range-of-ranges into a contiguous MPI buffer with per-rank counts
/// and displacements.
///
/// The source range-of-ranges is lazily flattened (copied) into the flat buffer
/// on first access to any MPI protocol method. Counts are derived from inner
/// range sizes, displacements via exclusive scan.
///
/// mpi_type() is forwarded from the flat buffer via view_interface — if the
/// element type is a builtin MPI type (or has buffer_traits), it works out of
/// the box. Otherwise, compose with with_type().
///
/// Template parameters:
///   Source        — the range-of-ranges (after all() wrapping)
///   FlatBuf       — the flat data buffer (after all() wrapping)
///   Counts        — the counts container (after all() wrapping)
///   Displs        — the displacements container (after all() wrapping)
///   resize_buf    — if true, flat buffer is resized as needed
///   resize_counts — if true, counts buffer is resized as needed
///   resize_displs — if true, displs buffer is resized as needed
///
/// Typical usage:
///   std::vector<std::vector<int>> per_rank = ...;
///   kamping::v2::alltoallv(per_rank | flatten_v(), rbuf);
template <typename Source, typename FlatBuf, count_range Counts, count_range Displs,
          bool resize_buf = false, bool resize_counts = false, bool resize_displs = false>
    requires std::ranges::forward_range<Source> && std::ranges::sized_range<Source>
             && std::ranges::input_range<std::ranges::range_value_t<Source>>
             && std::ranges::sized_range<std::ranges::range_value_t<Source>>
class flatten_v_view
    : public view_interface<flatten_v_view<Source, FlatBuf, Counts, Displs, resize_buf, resize_counts, resize_displs>> {
    Source               source_;
    mutable FlatBuf      flat_buf_;
    mutable Counts       counts_;
    mutable Displs       displs_;
    mutable std::ptrdiff_t   total_size_ = 0;
    mutable bool             needs_flatten_ = true;

    void ensure_flattened() const {
        if (!needs_flatten_) return;

        auto const num_ranks = std::ranges::size(source_);

        // Resize counts if needed
        if constexpr (resize_counts) {
            kamping::ranges::resize_for_receive(counts_, static_cast<std::ptrdiff_t>(num_ranks));
        }

        // Compute counts
        auto* counts_ptr = std::ranges::data(counts_);
        total_size_ = 0;
        std::size_t idx = 0;
        for (auto&& inner : source_) {
            auto const s = static_cast<int>(std::ranges::size(inner));
            counts_ptr[idx++] = s;
            total_size_ += s;
        }

        // Resize displacements if needed
        if constexpr (resize_displs) {
            kamping::ranges::resize_for_receive(displs_, static_cast<std::ptrdiff_t>(num_ranks));
        }

        // Compute displacements
        auto* displs_ptr = std::ranges::data(displs_);
        std::exclusive_scan(counts_ptr, counts_ptr + num_ranks, displs_ptr, 0);

        // Resize flat buffer if needed
        if constexpr (resize_buf) {
            kamping::ranges::resize_for_receive(flat_buf_, total_size_);
        }

        // Copy data
        using elem_t = std::ranges::range_value_t<std::ranges::range_value_t<Source>>;
        elem_t* dest = std::ranges::data(flat_buf_);
        for (auto&& inner : source_) {
            dest = std::copy(std::ranges::begin(inner), std::ranges::end(inner), dest);
        }

        needs_flatten_ = false;
    }

public:
    constexpr FlatBuf const& base() const& noexcept { return flat_buf_; }
    constexpr FlatBuf&       base() &      noexcept { return flat_buf_; }

    template <typename S, typename F, typename C, typename D>
    flatten_v_view(S&& source, F&& flat_buf, C&& counts, D&& displs)
        : source_(kamping::ranges::all(std::forward<S>(source))),
          flat_buf_(kamping::ranges::all(std::forward<F>(flat_buf))),
          counts_(kamping::ranges::all(std::forward<C>(counts))),
          displs_(kamping::ranges::all(std::forward<D>(displs))) {}

    auto mpi_data() {
        ensure_flattened();
        return kamping::ranges::data(flat_buf_);
    }

    auto mpi_data() const {
        ensure_flattened();
        return kamping::ranges::data(std::as_const(flat_buf_));
    }

    std::ptrdiff_t mpi_size() const {
        ensure_flattened();
        return total_size_;
    }

    std::span<int const> mpi_sizev() const {
        ensure_flattened();
        return {std::ranges::data(counts_), std::ranges::size(counts_)};
    }

    std::span<int const> mpi_displs() const {
        ensure_flattened();
        return {std::ranges::data(displs_), std::ranges::size(displs_)};
    }

    /// Displacements are always computed via exclusive_scan — monotonically non-decreasing.
    constexpr bool displs_monotonic() const { return true; }

    constexpr Counts const& counts() const& { return counts_; }
    constexpr Counts&       counts() &      { return counts_; }
    constexpr Counts&&      counts() &&     { return std::move(counts_); }

    constexpr Displs const& displs() const& { return displs_; }
    constexpr Displs&       displs() &      { return displs_; }
    constexpr Displs&&      displs() &&     { return std::move(displs_); }
};

template <typename S, typename F, typename C, typename D>
flatten_v_view(S&&, F&&, C&&, D&&)
    -> flatten_v_view<kamping::ranges::all_t<S>, kamping::ranges::all_t<F>,
                      kamping::ranges::all_t<C>, kamping::ranges::all_t<D>>;

template <typename Source, typename FlatBuf, typename Counts, typename Displs, bool rb, bool rc, bool rd>
inline constexpr bool enable_borrowed_buffer<flatten_v_view<Source, FlatBuf, Counts, Displs, rb, rc, rd>> =
    enable_borrowed_buffer<FlatBuf> && enable_borrowed_buffer<Counts> && enable_borrowed_buffer<Displs>;

} // namespace ranges

namespace views {

inline constexpr struct flatten_v_fn {
    /// 0-arg: allocate flat buffer, counts, and displs internally — all auto-resized.
    constexpr auto operator()() const {
        return kamping::ranges::adaptor<0, decltype([](auto&& source) {
            using Source  = std::remove_cvref_t<decltype(source)>;
            using inner_t = std::ranges::range_value_t<Source>;
            using elem_t  = std::ranges::range_value_t<inner_t>;
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::owning_view<std::vector<elem_t>>;
            using C = kamping::ranges::owning_view<std::vector<int>>;
            using D = kamping::ranges::owning_view<std::vector<int>>;
            return kamping::ranges::flatten_v_view<S, F, C, D, true, true, true>(
                std::forward<decltype(source)>(source),
                std::vector<elem_t>{},
                std::vector<int>{},
                std::vector<int>{}
            );
        })>{}();
    }

    /// 1-arg (non-resize): user flat buffer, internal counts and displs (auto-resized).
    template <typename Fb>
        requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
    constexpr auto operator()(Fb&& flat_buf) const {
        return kamping::ranges::adaptor<1, decltype([](auto&& source, auto&& fb) {
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::all_t<decltype(fb)>;
            using C = kamping::ranges::owning_view<std::vector<int>>;
            using D = kamping::ranges::owning_view<std::vector<int>>;
            return kamping::ranges::flatten_v_view<S, F, C, D, false, true, true>(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::vector<int>{},
                std::vector<int>{}
            );
        })>{}(std::forward<Fb>(flat_buf));
    }

    /// 2-arg (non-resize): user flat buffer + user counts, internal displs (auto-resized).
    template <typename Fb, typename Ct>
        requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
    constexpr auto operator()(Fb&& flat_buf, Ct&& counts) const {
        return kamping::ranges::adaptor<2, decltype([](auto&& source, auto&& fb, auto&& c) {
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::all_t<decltype(fb)>;
            using C = kamping::ranges::all_t<decltype(c)>;
            using D = kamping::ranges::owning_view<std::vector<int>>;
            return kamping::ranges::flatten_v_view<S, F, C, D, false, false, true>(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::forward<decltype(c)>(c),
                std::vector<int>{}
            );
        })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts));
    }

    /// 3-arg (non-resize): user flat buffer + user counts + user displs — no resize for any.
    template <typename Fb, typename Ct, typename Dt>
        requires(!std::same_as<std::remove_cvref_t<Fb>, kamping::v2::resize_t>)
    constexpr auto operator()(Fb&& flat_buf, Ct&& counts, Dt&& displs) const {
        return kamping::ranges::adaptor<3, decltype([](auto&& source, auto&& fb, auto&& c, auto&& d) {
            return kamping::ranges::flatten_v_view(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::forward<decltype(c)>(c),
                std::forward<decltype(d)>(d)
            );
        })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts), std::forward<Dt>(displs));
    }

    /// 1-arg (resize): user flat buffer with resize, internal counts and displs (auto-resized).
    template <typename Fb>
    constexpr auto operator()(kamping::v2::resize_t, Fb&& flat_buf) const {
        return kamping::ranges::adaptor<1, decltype([](auto&& source, auto&& fb) {
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::all_t<decltype(fb)>;
            using C = kamping::ranges::owning_view<std::vector<int>>;
            using D = kamping::ranges::owning_view<std::vector<int>>;
            return kamping::ranges::flatten_v_view<S, F, C, D, true, true, true>(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::vector<int>{},
                std::vector<int>{}
            );
        })>{}(std::forward<Fb>(flat_buf));
    }

    /// 2-arg (resize): user flat buffer + user counts — both resized, internal displs (auto-resized).
    template <typename Fb, typename Ct>
    constexpr auto operator()(kamping::v2::resize_t, Fb&& flat_buf, Ct&& counts) const {
        return kamping::ranges::adaptor<2, decltype([](auto&& source, auto&& fb, auto&& c) {
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::all_t<decltype(fb)>;
            using C = kamping::ranges::all_t<decltype(c)>;
            using D = kamping::ranges::owning_view<std::vector<int>>;
            return kamping::ranges::flatten_v_view<S, F, C, D, true, true, true>(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::forward<decltype(c)>(c),
                std::vector<int>{}
            );
        })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts));
    }

    /// 3-arg (resize): user flat buffer + user counts + user displs — all resized.
    template <typename Fb, typename Ct, typename Dt>
    constexpr auto operator()(kamping::v2::resize_t, Fb&& flat_buf, Ct&& counts, Dt&& displs) const {
        return kamping::ranges::adaptor<3, decltype([](auto&& source, auto&& fb, auto&& c, auto&& d) {
            using S = kamping::ranges::all_t<decltype(source)>;
            using F = kamping::ranges::all_t<decltype(fb)>;
            using C = kamping::ranges::all_t<decltype(c)>;
            using D = kamping::ranges::all_t<decltype(d)>;
            return kamping::ranges::flatten_v_view<S, F, C, D, true, true, true>(
                std::forward<decltype(source)>(source),
                std::forward<decltype(fb)>(fb),
                std::forward<decltype(c)>(c),
                std::forward<decltype(d)>(d)
            );
        })>{}(std::forward<Fb>(flat_buf), std::forward<Ct>(counts), std::forward<Dt>(displs));
    }
} flatten_v{};

} // namespace views
} // namespace kamping
