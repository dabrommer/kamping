#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <mdspan>

#include "kamping/builtin_types.hpp"
#include "kamping/v2/ranges/adaptor_closure.hpp"

namespace kamping::ranges {

/// Wraps an std::mdspan and pack/unpacks it for MPI transport.
///
/// Send path: mpi_size()/mpi_data() lazily pack the wrapped mdspan into buffer_ on first
///            access, if the mdspan is not contigous already
/// Recv path:
///
template <typename span>
class mdspan_view {
    using value_type                = span::value_type;
    using extends                   = span::extents_type;
    using accessor                  = span::accessor_type;
    using layout_policy             = span::layout_policy;

    mutable span base_;

    mutable bool   packed_                  = false;
    mutable bool   needs_unpacking_         = false;
    static constexpr bool pack_not_needed_  = (std::same_as<layout_policy, std::layout_left> || std::same_as<layout_policy, std::layout_right>);

    std::mdspan<value_type, extends, std::layout_right, accessor> buffer_;

    // base_ is mutable, so this is safe from const methods.
    // The const-lvalue-ref case (value_type is const-qualified) is prevented from
    // reaching do_deserialize() by the set_recv_count requires-clause.
    value_type& base_ref() const noexcept {
        if constexpr (is_owning)
            return base_;
        else
            return *base_;
    }

    void do_pack() const {
        if constexpr (pack_not_needed_) {
            return;
        } else {
            // Do deep copy stuff
        }
        packed_ = true;
    }

    void do_unpack() const {
        if constexpr (pack_not_needed_) {
            return;
        } else {
            // Redo deep copy stuff
        }
        needs_unpacking_ = false;
    }

public:
    /// Non-owning constructor: stores a pointer to the referenced object.
    /// Handles both `T&` and `T const&` (value_type may be const-qualified).
    explicit mdspan_view(span& obj) requires(!is_owning) : base_(&obj) {}

    /// Owning constructor: takes ownership of a moved object.
    explicit mdspan_view(span&& obj) requires(is_owning) : base_(std::move(obj)) {}

    /// Dereference to the wrapped object, triggering unpacking if needed.
    value_type& operator*() {
        if (needs_unpacking_) do_unpack();
        return base_ref();
    }

    value_type const& operator*() const {
        if (needs_unpacking_) do_unpack();
        return base_ref();
    }

    value_type*       operator->() { return std::addressof(**this); }
    value_type const* operator->() const { return std::addressof(**this); }

    // ---- MPI protocol methods --------------------------------------------

    std::ptrdiff_t mpi_size() const {
        if (needs_unpacking_) return buffer_.size();
        if (!packed_) do_pack();
        return static_cast<std::ptrdiff_t>(buffer_.size());
    }

    MPI_Datatype mpi_type() const {
        // todo this does not need to be builtin
        return kamping::builtin_type<value_type>::data_type();
    }

    /// Returns a mutable pointer: satisfies send_buffer (void const* accepted) and
    /// recv_buffer (void* required). Pack lazily on the send side.
    void* mpi_data() const {
        if constexpr (pack_not_needed_) return base_.data_handle();
        if (!needs_unpacking_ && !packed_) do_pack();
        return buffer_.data_handle();
    }
};

// lvalue input (including const lvalue): non-owning.
// T deduced as U or U const → mdspan_view<U&> or mdspan_view<U const&>.
template <typename T>
mdspan_view(T&) -> mdspan_view<T&>;

// rvalue input: owning.
template <typename T>
    requires(!std::is_lvalue_reference_v<T>)
mdspan_view(T&&) -> mdspan_view<T>;

} // namespace kamping::ranges

namespace kamping::views {
inline constexpr struct mdspan_fn : kamping::ranges::adaptor_closure<mdspan_fn> {
    template <typename R>
    constexpr auto operator()(R&& r) const {
        return kamping::ranges::mdspan_view(std::forward<R>(r));
    }
} pack_mdspan{};

/// Returns an owning serialization_view<T> with a default-constructed T.
/// Use as a recv buffer when the object does not exist yet:
///   auto view = comm.recv(kamping::views::unpack<MyType>(), 0);
///   MyType& result = *view;
template <typename T, typename container>
    requires std::default_initializable<T> && std::default_initializable<container>
auto unpack() {
    using default_span = std::mdspan<T, std::extents<size_t, std::dynamic_extent>>;
    container c{};
    // todo need to store the container somewhere
}
} // namespace kamping::views
