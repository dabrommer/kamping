#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "kamping/builtin_types.hpp"
#include "kamping/v2/ranges/adaptor_closure.hpp"
#include "kamping/v2/ranges/concepts.hpp"

namespace kamping::ranges {

/// Wraps a Kokkos::View and packs it into a contiguous Kokkos::View.
///
/// T is the wrapped Kokkos::View type: a (possibly const) lvalue reference for non-owning
/// wrappers, or a value type for owning wrappers.
///
/// Send path: mpi_size()/mpi_data() lazily create a contiguous view and deep_copy() the
///            wrapped view into it.
/// Recv path: set_recv_count(n) only works on resizable Kokkos::views with rank = 1 and size = 0.
///            in this case the wrapped view will be resized
///
template <typename T>
class kokkos_view {
    static constexpr bool is_owning = !std::is_lvalue_reference_v<T>;
    using view_type                 = std::remove_reference_t<T>;

    static_assert(Kokkos::is_view<view_type>::value, "kokkos_view requires a Kokkos::View type");

    using scalar_type     = std::remove_const_t<typename view_type::value_type>;
    using execution_space = view_type::execution_space;
    using memory_space    = view_type::memory_space;
    using stored_t        = std::conditional_t<is_owning, view_type, view_type*>;

    // Ensure deep_copy can run in the wrapped view's exec space
    static_assert(
        Kokkos::SpaceAccessibility<execution_space, memory_space>::accessible,
        "kokkos_view requires the execution space to access the wrapped view memory space"
    );
    // Ensure the given view is in HostSpace (for now)
    static_assert(
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible,
        "kokkos_view currently host-accessible memory space"
    );

    using packed_view_t = Kokkos::View<typename view_type::non_const_data_type, Kokkos::LayoutRight, memory_space>;

    mutable stored_t      base_;
    mutable packed_view_t packed_storage_;

    mutable bool packed_       = false;
    mutable bool needs_unpack_ = false;

    view_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return base_;
        else
            return *base_;
    }

    static packed_view_t make_packed(view_type const& v) {
        execution_space   exec;
        std::string const label = std::string(v.label()) + "-kamping-kokkos-view";

        return [&exec, &v, &label]<std::size_t... Is>(std::index_sequence<Is...>) {
            return packed_view_t(
                Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
                static_cast<typename packed_view_t::size_type>(v.extent(Is))...
            );
        }(std::make_index_sequence<view_type::rank>{});
    }

    void pack() const {
        packed_storage_ = make_packed(base_ref());
        Kokkos::deep_copy(packed_storage_, base_ref());
        packed_       = true;
        needs_unpack_ = true;
    }

    void unpack() const {
        Kokkos::deep_copy(base_ref(), packed_storage_);
        needs_unpack_ = false;
    }

public:
    explicit kokkos_view(view_type& view)
        requires(!is_owning)
        : base_(&view) {}

    explicit kokkos_view(view_type&& view)
        requires(is_owning)
        : base_(std::move(view)) {}

    view_type& operator*() {
        if (needs_unpack_)
            unpack();
        return base_ref();
    }

    view_type const& operator*() const {
        if (needs_unpack_)
            unpack();
        return base_ref();
    }

    view_type* operator->() {
        return std::addressof(**this);
    }
    view_type const* operator->() const {
        return std::addressof(**this);
    }

    void set_recv_count(std::ptrdiff_t n)
        requires(
            view_type::rank == 1 && requires(view_type& v, typename view_type::size_type m) { Kokkos::resize(v, m); }
        )
    {
        auto const current_size = static_cast<std::ptrdiff_t>(base_ref().size());
        if (n == current_size) {
            return;
        }

        KAMPING_ASSERT(current_size == 0, "Wrapped kokkos_view size must be zero for resizing");
        Kokkos::resize(base_ref(), static_cast<view_type::size_type>(n));

        packed_       = false;
        needs_unpack_ = false;
    }

    std::ptrdiff_t mpi_size() const {
        return static_cast<std::ptrdiff_t>(base_ref().size());
    }

    MPI_Datatype mpi_type()
        requires has_mpi_type<std::span<scalar_type>>
    {
        return kamping::ranges::type(std::span{packed_storage_.data(), packed_storage_.size()});
    }

    void* mpi_data() const {
        if (!needs_unpack_ && !packed_)
            pack();
        return packed_storage_.data();
    }
};

template <typename T>
kokkos_view(T&) -> kokkos_view<T&>;

template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
kokkos_view(T&&) -> kokkos_view<T>;

} // namespace kamping::ranges

namespace kamping::views {
inline constexpr struct kokkos_fn : kamping::ranges::adaptor_closure<kokkos_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::ranges::kokkos_view(std::forward<R>(r));
    }
} kokkos{};

/// Returns an owning rank-1 kokkos_view for receive with a custom label.
template <typename T>
auto unpack(std::string const& label) {
    using view_t = Kokkos::View<T*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    return kamping::ranges::kokkos_view<view_t>(view_t(label, 0));
}

/// Returns an owning rank-1 kokkos_view for receive
/// Template parameter is the element type, e.g. unpack<int>()
template <typename T>
auto unpack() {
    return unpack<T>("kamping-kokkos-unpack");
}

} // namespace kamping::views
