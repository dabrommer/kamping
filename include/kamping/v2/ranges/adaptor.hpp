#pragma once

#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>

#include "kamping/v2/ranges/adaptor_closure.hpp"
#include "kamping/v2/ranges/all.hpp"

namespace kamping::ranges {

/// A closure with pre-bound arguments. Created by calling an adaptor with only the extra arguments
/// (partial application). When invoked (directly or via |), prepends the incoming value and calls fn_.
///
/// Example: with_type(MPI_INT) returns a bound_adaptor holding fn_ and MPI_INT in bound_.
///          vec | that_closure  →  fn_(vec, MPI_INT)  →  with_type_view(vec, MPI_INT)
template <typename Fn, typename... BoundArgs>
struct bound_adaptor : adaptor_closure<bound_adaptor<Fn, BoundArgs...>> {
    [[no_unique_address]] Fn fn_;
    std::tuple<BoundArgs...>  bound_;

    constexpr bound_adaptor(Fn fn, std::tuple<BoundArgs...> bound)
        : fn_(std::move(fn)),
          bound_(std::move(bound)) {}

    template <typename T>
    constexpr auto operator()(T&& val) const& {
        return std::apply(
            [&](auto const&... args) { return fn_(std::forward<T>(val), args...); },
            bound_
        );
    }

    // Rvalue overload: moves stored arguments out so the resulting view takes ownership
    // rather than holding a ref_view into this closure's (about-to-be-destroyed) storage.
    template <typename T>
    constexpr auto operator()(T&& val) && {
        return std::apply(
            [&](auto&&... args) { return fn_(std::forward<T>(val), std::move(args)...); },
            bound_
        );
    }
};

/// Generic range adaptor factory, parameterized by the number of extra arguments (beyond the value).
/// Arity-based disambiguation (like libstdc++ internally):
///   - ExtraArgs arguments       → partial application, returns a pipeable bound_adaptor
///   - ExtraArgs + 1 arguments   → full call, first argument is the value
///
/// Usage:  inline constexpr adaptor<1, decltype([](auto&& r, MPI_Datatype t) { ... })> with_type{};
///         vec | with_type(MPI_INT)       // partial → bound_adaptor → pipe applies it
///         with_type(vec, MPI_INT)        // full call
template <std::size_t ExtraArgs, typename Fn>
struct adaptor {
    [[no_unique_address]] Fn fn_;

    /// Partial application: bind ExtraArgs arguments, return a pipeable closure.
    /// Range arguments are stored via all_t — lvalue ranges become ref_view (borrow the original),
    /// rvalue ranges become owning_view. Non-range arguments are stored by value (std::decay_t).
    template <typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(Args&&... args) const {
        auto store = []<typename Arg>(Arg&& arg) -> decltype(auto) {
            if constexpr (std::ranges::range<std::remove_cvref_t<Arg>>)
                return kamping::ranges::all(std::forward<Arg>(arg));
            else
                return std::decay_t<Arg>(std::forward<Arg>(arg));
        };
        return bound_adaptor<Fn, decltype(store(std::forward<Args>(args)))...>(
            fn_, std::tuple{store(std::forward<Args>(args))...}
        );
    }

    /// Full call: first argument is the value, remaining ExtraArgs are forwarded to fn_.
    template <typename T, typename... Args>
        requires(sizeof...(Args) == ExtraArgs)
    constexpr auto operator()(T&& val, Args&&... args) const {
        return fn_(std::forward<T>(val), std::forward<Args>(args)...);
    }
};

/// Variable-arity range adaptor factory, supporting MinExtraArgs to MaxExtraArgs extra arguments.
/// Disambiguates partial vs. full call by checking whether the first argument satisfies
/// std::ranges::range: if it does (and the remaining args are in [Min,Max]), it is a full call;
/// otherwise all args are treated as extra args for partial application.
///
/// This makes tag-based overloads natural: non-range tags count as extra args, not as the value.
///
/// Usage:  inline constexpr var_adaptor<1, 2, decltype([](auto&& r, auto&&... extra) {
///             if constexpr (sizeof...(extra) == 2) { /* resize path */ }
///             else { /* normal path */ }
///         })> with_auto_displs{};
template <std::size_t MinExtraArgs, std::size_t MaxExtraArgs, typename Fn>
struct var_adaptor {
    [[no_unique_address]] Fn fn_;

    /// Partial application: sizeof...(Args) in [Min, Max] and first arg (if any) is not a range.
    template <typename... Args>
        requires(sizeof...(Args) >= MinExtraArgs && sizeof...(Args) <= MaxExtraArgs
                 && !(sizeof...(Args) >= 1
                      && std::ranges::range<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Args...>>>>))
    constexpr auto operator()(Args&&... args) const {
        auto store = []<typename Arg>(Arg&& arg) -> decltype(auto) {
            if constexpr (std::ranges::range<std::remove_cvref_t<Arg>>)
                return kamping::ranges::all(std::forward<Arg>(arg));
            else
                return std::decay_t<Arg>(std::forward<Arg>(arg));
        };
        return bound_adaptor<Fn, decltype(store(std::forward<Args>(args)))...>(
            fn_, std::tuple{store(std::forward<Args>(args))...}
        );
    }

    /// Full call: first argument is the range value, remaining sizeof...(Args) in [Min, Max].
    template <typename T, typename... Args>
        requires(std::ranges::range<std::remove_cvref_t<T>> && sizeof...(Args) >= MinExtraArgs
                 && sizeof...(Args) <= MaxExtraArgs)
    constexpr auto operator()(T&& val, Args&&... args) const {
        return fn_(std::forward<T>(val), std::forward<Args>(args)...);
    }
};

} // namespace kamping::ranges
