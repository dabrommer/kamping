#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/tags.hpp"
#include "kamping/v2/views/auto_counts_view.hpp"
#include "kamping/v2/views/auto_displs_view.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

// ── Concept checks ────────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, SatisfiesExpectedConcepts) {
    std::vector<int> data{1, 2, 3};

    // auto_counts_view satisfies the deferred variadic buffer protocol
    auto view = data | kamping::views::auto_counts();
    static_assert(kamping::ranges::has_mpi_sizev<decltype(view)>);
    static_assert(kamping::ranges::has_set_comm_size<decltype(view)>);
    static_assert(kamping::ranges::has_commit_counts<decltype(view)>);
    static_assert(kamping::ranges::has_counts_accessor<decltype(view)>);
    static_assert(kamping::ranges::deferred_recv_buf_v<decltype(view)>);

    // data/size/type forwarded from base: satisfies recv_buffer
    static_assert(kamping::ranges::recv_buffer<decltype(view)>);
}

TEST(AutoCountsViewTest, FullPipelineSatisfiesRecvBufferV) {
    std::vector<int> data;
    auto             rbuf =
        data | kamping::views::auto_counts() | kamping::views::auto_displs() | kamping::views::resize_v;

    static_assert(kamping::ranges::deferred_recv_buf_v<decltype(rbuf)>);
    static_assert(kamping::ranges::recv_buffer_v<decltype(rbuf)>);
    static_assert(kamping::ranges::has_monotonic_displs<decltype(rbuf)>);
}

// ── set_comm_size ─────────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, SetCommSizeResizesCountsWhenResizeTrue) {
    std::vector<int> data{10, 20, 30};
    auto             view = data | kamping::views::auto_counts(); // resize=true (owned container)

    EXPECT_EQ(std::ranges::size(view.counts()), 0u);
    view.set_comm_size(3);
    EXPECT_EQ(std::ranges::size(view.counts()), 3u);
}

TEST(AutoCountsViewTest, SetCommSizeIsNoOpWhenResizeFalse) {
    std::vector<int> data{10, 20, 30};
    std::vector<int> user_counts(3, 0);
    auto             view = data | kamping::views::auto_counts(user_counts); // resize=false

    view.set_comm_size(99); // must not resize
    EXPECT_EQ(std::ranges::size(view.counts()), 3u);
}

// ── counts() accessor and mpi_sizev() ─────────────────────────────────────────

TEST(AutoCountsViewTest, CountsAccessorReflectsMpiWrites) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    auto             view = data | kamping::views::auto_counts();
    view.set_comm_size(3);

    // Simulate MPI writing counts directly into the buffer
    view.counts()[0] = 1;
    view.counts()[1] = 2;
    view.counts()[2] = 3;

    std::span<int const> sv = kamping::ranges::sizev(view);
    EXPECT_EQ((std::vector<int>(sv.begin(), sv.end())), (std::vector<int>{1, 2, 3}));
}

TEST(AutoCountsViewTest, UserProvidedCountsBuffer) {
    std::vector<int> data{10, 20, 30};
    std::vector<int> user_counts{4, 5, 6};
    auto             view = data | kamping::views::auto_counts(user_counts);

    std::span<int const> sv = kamping::ranges::sizev(view);
    EXPECT_EQ((std::vector<int>(sv.begin(), sv.end())), (std::vector<int>{4, 5, 6}));

    // Mutation through the view writes through to the original vector
    view.counts()[0] = 99;
    EXPECT_EQ(user_counts[0], 99);
}

TEST(AutoCountsViewTest, UserProvidedCountsBufferWithResize) {
    std::vector<int> data{1, 2};
    std::vector<int> user_counts; // empty, will be resized
    auto             view = data | kamping::views::auto_counts(kamping::v2::resize, user_counts);

    view.set_comm_size(2);
    EXPECT_EQ(std::ranges::size(view.counts()), 2u);
}

// ── commit_counts() ───────────────────────────────────────────────────────────

TEST(AutoCountsViewTest, CommitCountsIsCallable) {
    std::vector<int> data{1, 2};
    auto             view = data | kamping::views::auto_counts();
    view.set_comm_size(2);
    view.counts()[0] = 1;
    view.counts()[1] = 1;
    view.commit_counts(); // must not crash; currently a no-op
}

// ── base() and view_interface forwarding ─────────────────────────────────────

TEST(AutoCountsViewTest, BaseReturnsUnderlyingDataBuffer) {
    std::vector<int> data{10, 20, 30};
    auto             view = data | kamping::views::auto_counts();

    // begin/end iterate over the data buffer, not the counts
    EXPECT_EQ((std::vector<int>(view.begin(), view.end())), (std::vector<int>{10, 20, 30}));
}

TEST(AutoCountsViewTest, MpiSizeForwardsFromBase) {
    std::vector<int> data{1, 2, 3, 4};
    auto             view = data | kamping::views::auto_counts();

    EXPECT_EQ(kamping::ranges::size(view), 4u);
}

TEST(AutoCountsViewTest, MpiDataForwardsFromBase) {
    std::vector<int> data{1, 2, 3};
    auto             view = data | kamping::views::auto_counts();

    EXPECT_EQ(kamping::ranges::data(view), data.data());
}

// ── auto_displs integration ───────────────────────────────────────────────────

TEST(AutoCountsViewTest, AutoDisplsComputedFromCounts) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    auto             view = data | kamping::views::auto_counts();
    view.set_comm_size(3);
    view.counts()[0] = 1;
    view.counts()[1] = 2;
    view.counts()[2] = 3;
    view.commit_counts();

    auto chained = std::move(view) | kamping::views::auto_displs();
    std::span<int const> displs = kamping::ranges::displs(chained);
    EXPECT_EQ((std::vector<int>(displs.begin(), displs.end())), (std::vector<int>{0, 1, 3}));
}

TEST(AutoCountsViewTest, ResizeVResizesDataBufferFromCounts) {
    std::vector<int> data;
    auto             rbuf =
        data | kamping::views::auto_counts() | kamping::views::auto_displs() | kamping::views::resize_v;

    // Simulate infer(): set_comm_size → write counts → commit_counts
    rbuf.set_comm_size(3);
    rbuf.counts()[0] = 2;
    rbuf.counts()[1] = 3;
    rbuf.counts()[2] = 1;
    rbuf.commit_counts();

    // mpi_data() triggers resize to 2+3+1 = 6
    (void)rbuf.mpi_data();
    EXPECT_EQ(data.size(), 6u);
}
