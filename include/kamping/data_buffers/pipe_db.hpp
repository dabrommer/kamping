#pragma once

#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include "kamping/data_buffers/data_buffer_concepts.hpp"
#include "kamping/data_buffers/pipe_view_interface.hpp"


template <std::ranges::contiguous_range R, kamping::IntContiguousRange DisplRange = std::ranges::owning_view<std::vector<int>>, ResizePolicy>
struct auto_displs_view : pipe_view_interface<auto_displs_view<R, DisplRange>, R> {
    R    base_;
    bool displs_computed_ = false;
    bool displs_set_ = false;
    std::vector<int> internal_displs_;
    DisplRange displs_;


    explicit auto_displs_view(R base) requires kamping::HasSizeV<R>
        : base_(std::move(base)), displs_set_(false), internal_displs_(), displs_(internal_displs_) {}

   auto_displs_view(R base, DisplRange& empty_displs) requires kamping::HasSizeV<R>
           : base_(std::move(base)), displs_(empty_displs) {}


    // Move constructor
    auto_displs_view(auto_displs_view&& other) noexcept
            : base_(std::move(other.base_)),
              internal_displs_(std::move(other.internal_displs_)),
              displs_computed_(other.displs_computed_),
              displs_set_(other.displs_set_),
              displs_(other.use_external_ ? other.displs_ : std::ref(internal_displs_)) {}


    // Move assignment
    auto_displs_view& operator=(auto_displs_view&& other) noexcept {
        if (this != &other) {
            base_ = std::move(other.base_);
            internal_displs_ = std::move(other.internal_displs_);
            displs_computed_ = other.displs_computed_;
            displs_set_ = other.displs_set_;
            displs_ = other.use_external_ ? other.displs_ : std::ref(internal_displs_);
        }
        return *this;
    }

    auto displs() {
        if (!displs_computed_) {
            auto   counts = base_.size_v();
            // Counts have to be of correct size
            size_t ranks  = std::ranges::size(counts);
            if (!displs_set_) {
                internal_displs_.resize(ranks);
            }

            std::exclusive_scan(
                    counts.begin(),
                    counts.begin() + kamping::asserting_cast<int>(ranks),
                    displs_.begin(),
                    0
            );
            displs_computed_ = true;

        }
        return displs_;

    }
};

template <kamping::IntContiguousRange DisplRange = >
struct auto_displs : std::ranges::range_adaptor_closure<auto_displs<DisplRange>> {
    DisplRange internal_displs_;
    DisplRange empty_displs_;
    bool empty_displs_set_ = false;


    explicit auto_displs(DisplRange& empty_displs) : empty_displs_(empty_displs), empty_displs_set_(true) {}
    explicit auto_displs() : internal_displs_(), empty_displs_(internal_displs_), empty_displs_set_(false) {}

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return empty_displs_set_ ? auto_displs_view(std::forward<R>(r), empty_displs_) : auto_displs_view<R, DisplRange>(std::forward<R>(r));
    }
};

template <std::ranges::contiguous_range R>
auto_displs_view(R&&) -> auto_displs_view<std::ranges::views::all_t<R>, std::vector<int>>;

template <std::ranges::contiguous_range R, kamping::IntContiguousRange DisplRange>
auto_displs_view(R&&, DisplRange&) -> auto_displs_view<std::ranges::views::all_t<R>, DisplRange>;

template <std::ranges::contiguous_range R>
struct resize_ext_view : pipe_view_interface<resize_ext_view<R>, R> {
    R    base_;
    bool resized = false;

    explicit resize_ext_view(R base) requires kamping::HasDispls<R> && kamping::HasSetSize<R> && kamping::HasSizeV<R>
        : base_(std::move(base)) {}

    auto data() {
        resize();
        return std::ranges::data(base_);
    }
    auto size() {
        resize();
        return std::ranges::size(base_);
    }

    void resize() {
        if (!resized) {
            auto displs = base_.displs();
            auto counts = base_.size_v();

            auto counts_ptr = std::ranges::data(displs);
            auto displs_ptr = std::ranges::data(counts);

            size_t ranks = std::ranges::size(counts);

            int recv_buf_size = 0;
            for (size_t i = 0; i < ranks; ++i) {
                recv_buf_size = std::max(recv_buf_size, *(counts_ptr + i) + *(displs_ptr + i));
            }

            base_.set_size(kamping::asserting_cast<size_t>(recv_buf_size));
            resized = true;
        }
    }
};

struct resize_ext : std::ranges::range_adaptor_closure<resize_ext> {
    explicit resize_ext() = default;

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return resize_ext_view(std::forward<R>(r));
    }
};


template <std::ranges::contiguous_range R, kamping::IntContiguousRange DisplRange>
struct displs_view : pipe_view_interface<displs_view<R, DisplRange>, R> {
    R                base_;
    DisplRange& displs_;

    explicit displs_view(R&& base) : base_(std::move(base)) {}
    displs_view(R&& base, DisplRange& displs) : base_(std::move(base)), displs_(displs) {}

    auto displs() {
        return displs_;
    }

    void set_displs(DisplRange& displs) {
        displs_ = displs;
    }
};

template <std::ranges::contiguous_range R, kamping::IntContiguousRange DisplRange>
displs_view(R&&, DisplRange& displs) -> displs_view<std::ranges::views::all_t<R>, DisplRange>;


template <kamping::IntContiguousRange DisplRange>
struct add_displs : std::ranges::range_adaptor_closure<add_displs<DisplRange>> {
    DisplRange &displs_;

    explicit add_displs(DisplRange& displs) : displs_(displs) {}

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return displs_view(std::forward<R>(r), displs_);
    }
};

template <std::ranges::contiguous_range R>
struct size_v_view : pipe_view_interface<size_v_view<R>, R> {
    R                base_;
    std::vector<int> &size_v_;

    size_v_view(R&& base, std::vector<int>& size_v) : base_(std::move(base)), size_v_(size_v) {}

    auto size_v() {
        return size_v_;
    }
};

template <typename R>
size_v_view(R&&, std::vector<int>& size_v) -> size_v_view<std::ranges::views::all_t<R>>;

struct add_size_v : std::ranges::range_adaptor_closure<add_size_v> {
    std::vector<int> &size_v_;

    explicit add_size_v(std::vector<int>& size_v) : size_v_(size_v) {}

    template <std::ranges::contiguous_range R>
    auto operator()(R&& r) const {
        return size_v_view(std::forward<R>(r), size_v_);
    }
};
