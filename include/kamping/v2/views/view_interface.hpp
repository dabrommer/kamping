#pragma once

#include <ranges>

#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"

namespace kamping::ranges {
namespace detail {
template <typename D>
concept has_base_range = requires(D& d) {
    { d.base() } -> std::ranges::range;
};

template <typename D>
concept has_const_base_range = requires(D const& d) {
    { d.base() } -> std::ranges::range;
};
} // namespace detail

template <typename Derived>
struct view_interface : public view_interface_base, public std::ranges::view_interface<Derived> {
    constexpr Derived& derived() noexcept {
        return static_cast<Derived&>(*this);
    }

    constexpr Derived const& derived() const noexcept {
        return static_cast<Derived const&>(*this);
    }

    constexpr auto begin()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end()
        requires detail::has_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    constexpr auto begin() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::begin(derived().base());
    }

    constexpr auto end() const
        requires detail::has_const_base_range<Derived>
    {
        return std::ranges::end(derived().base());
    }

    template <typename _Derived = Derived>
    auto mpi_type() const
        requires mpi::experimental::has_mpi_type<decltype(derived().base())>
    {
        return mpi::experimental::type(derived().base());
    }

    constexpr auto mpi_count() const
        requires mpi::experimental::has_mpi_count<decltype(derived().base())>
    {
        return mpi::experimental::count(derived().base());
    }

    constexpr auto mpi_data()
        requires mpi::experimental::has_mpi_data<decltype(derived().base())>
    {
        return mpi::experimental::data(derived().base());
    }

    constexpr auto mpi_data() const
        requires mpi::experimental::has_mpi_data<decltype(derived().base())>
    {
        return mpi::experimental::data(derived().base());
    }

    constexpr auto mpi_sizev() const
        requires mpi::experimental::has_mpi_sizev<decltype(derived().base())>
    {
        return mpi::experimental::sizev(derived().base());
    }

    constexpr auto mpi_displs() const
        requires mpi::experimental::has_mpi_displs<decltype(derived().base())>
    {
        return mpi::experimental::displs(derived().base());
    }

    decltype(auto) counts() const&
        requires kamping::ranges::has_counts_accessor<decltype(derived().base())>
    {
        return derived().base().counts();
    }
    decltype(auto) counts() &
        requires kamping::ranges::has_counts_accessor<decltype(derived().base())>
    {
        return derived().base().counts();
    }
    decltype(auto) counts() &&
        requires kamping::ranges::has_counts_accessor<decltype(derived().base())>
    {
        return std::move(derived().base()).counts();
    }

    decltype(auto) displs() const&
        requires kamping::ranges::has_displs_accessor<decltype(derived().base())>
    {
        return derived().base().displs();
    }
    decltype(auto) displs() &
        requires kamping::ranges::has_displs_accessor<decltype(derived().base())>
    {
        return derived().base().displs();
    }
    decltype(auto) displs() &&
        requires kamping::ranges::has_displs_accessor<decltype(derived().base())>
    {
        return std::move(derived().base()).displs();
    }

    void mpi_resize_for_receive(std::ptrdiff_t n)
        requires(
            kamping::ranges::has_mpi_resize_for_receive<decltype(derived().base())>
            || kamping::ranges::has_resize<decltype(derived().base())>
        )
    {
        kamping::ranges::resize_for_receive(derived().base(), n);
    }

    void commit_counts()
        requires kamping::ranges::has_commit_counts<decltype(derived().base())>
    {
        derived().base().commit_counts();
    }

    void set_comm_size(int n)
        requires kamping::ranges::has_set_comm_size<decltype(derived().base())>
    {
        derived().base().set_comm_size(n);
    }

    constexpr bool displs_monotonic() const
        requires kamping::ranges::has_monotonic_displs<decltype(derived().base())>
    {
        return derived().base().displs_monotonic();
    }
};
} // namespace kamping::ranges

/* template <typename Derived> */
/* inline constexpr bool std::ranges::enable_borrowed_range<kamping::ranges::view_interface<Derived>> = */
/*     std::ranges::borrowed_range<decltype(std::declval<Derived&>().base())>; */
