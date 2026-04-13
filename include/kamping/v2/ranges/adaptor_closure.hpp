#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

namespace kamping::ranges {

/// Tag base class — any type inheriting (transitively) from this is recognized as a pipe-able closure.
struct adaptor_closure_base {};

/// Checks whether T is a pipe-able closure (i.e., something produced by partial application of an adaptor).
template <typename T>
concept is_adaptor_closure = std::derived_from<std::remove_cvref_t<T>, adaptor_closure_base>;

/// Detects external range adaptor closures (e.g. std::views::take(2)) by duck typing:
/// they are not ranges themselves but are callable with a range argument.
/// Used to route closure | closure to composition rather than closure(value).
template <typename T>
concept is_external_closure =
    !std::ranges::range<std::remove_cvref_t<T>> &&
    !is_adaptor_closure<T> &&
    requires(std::remove_cvref_t<T> const& t, std::ranges::empty_view<int> r) { t(r); };

template <typename First, typename Second>
struct composed_closure;

/// CRTP base that makes Derived pipe-able. Provides two operator| overloads found via ADL:
///   - val | closure        → closure(val)            when val is NOT itself a closure
///   - closure1 | closure2  → composed_closure{...}   when both sides are closures
/// The !is_adaptor_closure constraint on the left side is intentionally the *only* constraint,
/// so any value type (range, MPI buffer, scalar, ...) can appear on the left side of |.
template <typename Derived>
struct adaptor_closure : adaptor_closure_base {
    // val | closure  →  closure(val)
    template <typename T>
        requires(!is_adaptor_closure<T>) && (!is_external_closure<T>)
    friend constexpr auto operator|(T&& val, Derived const& self) {
        return self(std::forward<T>(val));
    }

    // val | rvalue_closure  →  std::move(closure)(val)
    // Moves stored arguments out of temporary closures so that views take ownership
    // rather than holding a dangling ref_view into the closure's storage.
    template <typename T>
        requires(!is_adaptor_closure<T>) && (!is_external_closure<T>)
    friend constexpr auto operator|(T&& val, Derived&& self) {
        return std::move(self)(std::forward<T>(val));
    }

    // closure | closure  →  composed_closure that applies them left-to-right
    template <typename Other>
        requires is_adaptor_closure<Other> || is_external_closure<Other>
    friend constexpr auto operator|(Other other, Derived self) {
        return composed_closure<Other, Derived>(std::move(other), std::move(self));
    }
};

/// The result of closure1 | closure2. Itself a closure, so chains of arbitrary length work:
///   (c1 | c2) | c3  →  composed_closure<composed_closure<C1,C2>, C3>
template <typename First, typename Second>
struct composed_closure : adaptor_closure<composed_closure<First, Second>> {
    [[no_unique_address]] First  first_;
    [[no_unique_address]] Second second_;

    constexpr composed_closure(First first, Second second)
        : first_(std::move(first)),
          second_(std::move(second)) {}

    template <typename T>
    constexpr auto operator()(T&& val) const {
        return second_(first_(std::forward<T>(val)));
    }
};

} // namespace kamping::ranges
