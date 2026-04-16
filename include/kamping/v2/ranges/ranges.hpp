#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include <mpi.h>

#include "kamping/builtin_types.hpp"

namespace kamping::ranges {

// Tag base for all kamping views. Used to guard the std::ranges::size fallback
// against ADL circularity (std::ranges::size → ADL mpi::experimental::size → std::ranges::sized_range).
struct view_interface_base {};

/// A type that is a "view" in the kamping sense: either a std::ranges::view
/// (lightweight, copyable, e.g. std::span) or a kamping view (derived from
/// view_interface_base — may be owning and non-copyable like owning_view).
/// Mirrors the role of the std::ranges::view concept but extended for kamping.
template <typename T>
concept view = std::ranges::view<T> || std::derived_from<T, view_interface_base>;

/// Type implements the custom MPI resize protocol (preferred over plain resize()).
template <typename T>
concept has_mpi_resize_for_receive = requires(T& t, std::ptrdiff_t n) { t.mpi_resize_for_receive(n); };

/// Type is a standard resizable container (e.g. std::vector).
template <typename T>
concept has_resize = requires(T& t, std::size_t n) { t.resize(n); };


/// Resize t to hold n MPI elements before a receive. Dispatches to:
///   1. t.mpi_resize_for_receive(n) — custom protocol (e.g. resize_and_overwrite, NUMA alloc)
///   2. t.resize(n)                 — standard containers
template <has_mpi_resize_for_receive T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.mpi_resize_for_receive(n);
}

template <typename T>
    requires(!has_mpi_resize_for_receive<T>) && has_resize<T>
void resize_for_receive(T& t, std::ptrdiff_t n) {
    t.resize(static_cast<std::size_t>(n));
}

} // namespace kamping::ranges
