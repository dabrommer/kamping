#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/allgatherv.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/resize_v_view.hpp"

using namespace ::testing;

// Each rank r sends r+1 copies of r: result is [0, 1,1, 2,2,2, ...].
TEST(V2AllgathervTest, VariableLengthSend) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(rank + 1, rank);
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::views::auto_counts() | kamping::views::auto_displs() | kamping::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), r + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
}

// All ranks send exactly one element — same result as allgather.
TEST(V2AllgathervTest, UniformSingleElementSend) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data{rank};
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::views::auto_counts() | kamping::views::auto_displs() | kamping::views::resize_v
    );

    std::vector<int> expected(size);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(recv_data, expected);
}

// Rank 0 sends nothing; rank r sends r elements of value r.
TEST(V2AllgathervTest, RankZeroSendsEmptyBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(rank, rank); // rank 0 → empty
    std::vector<int> recv_data;

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::views::auto_counts() | kamping::views::auto_displs() | kamping::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 1; r < size; ++r) {
        expected.insert(expected.end(), r, r);
    }
    EXPECT_EQ(recv_data, expected);
}

// User-provided counts buffer (no auto-resize of counts).
TEST(V2AllgathervTest, UserProvidedCountsBuffer) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(rank + 1, rank);
    std::vector<int> recv_data;
    std::vector<int> counts(size); // pre-sized, no auto-resize

    kamping::v2::allgatherv(
        send_data,
        recv_data | kamping::views::auto_counts(counts) | kamping::views::auto_displs() | kamping::views::resize_v
    );

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), r + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
    // counts should have been filled by infer()
    EXPECT_EQ(counts[rank], rank + 1);
}

// User-provided counts buffer (no auto-resize of counts).
TEST(V2AllgathervTest, ExplicitCountsAutoDisplsNoResize) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> send_data(rank + 1, rank);
    std::vector<int> recv_data(size * (size + 1) / 2);
    std::vector<int> counts(size); // pre-sized, no auto-resize
    std::ranges::iota(counts, 1);

    kamping::v2::allgatherv(send_data, recv_data | kamping::views::with_counts(counts) | kamping::views::auto_displs());

    std::vector<int> expected;
    for (int r = 0; r < size; ++r) {
        expected.insert(expected.end(), r + 1, r);
    }
    EXPECT_EQ(recv_data, expected);
}
