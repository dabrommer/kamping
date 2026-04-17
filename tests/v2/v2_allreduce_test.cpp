// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/tags.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::kamping;

class AllreduceTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

TEST_F(AllreduceTest, allreduce_with_mpi_sum) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::allreduce(send_data, recv_data, MPI_SUM);

    int expected_sum_ranks       = size_ * (size_ - 1) / 2;
    int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
    EXPECT_EQ(r[0], expected_sum_ranks);
    EXPECT_EQ(r[1], expected_sum_rank_plus_1);
}

TEST_F(AllreduceTest, allreduce_with_default_op) {
    std::vector<int> send_data = {rank_ + 1, rank_ + 2};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::allreduce(send_data, recv_data);

    int expected_0 = size_ * (size_ - 1) / 2 + size_;     // sum of (rank+1)
    int expected_1 = size_ * (size_ - 1) / 2 + 2 * size_; // sum of (rank+2)
    EXPECT_EQ(r[0], expected_0);
    EXPECT_EQ(r[1], expected_1);
}

TEST_F(AllreduceTest, allreduce_with_mpi_max) {
    std::vector<double> send_data = {1.0 * rank_, 2.0 * rank_};
    std::vector<double> recv_data(2);

    auto [s, r] = kamping::v2::allreduce(send_data, recv_data, MPI_MAX);

    EXPECT_EQ(r[0], 1.0 * (size_ - 1));
    EXPECT_EQ(r[1], 2.0 * (size_ - 1));
}

TEST_F(AllreduceTest, allreduce_with_std_plus) {
    std::vector<int> send_data = {rank_};
    std::vector<int> recv_data(1);

    auto [s, r] = kamping::v2::allreduce(send_data, recv_data, std::plus<>{});

    EXPECT_EQ(r[0], size_ * (size_ - 1) / 2);
}

TEST_F(AllreduceTest, allreduce_resize) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data;

    auto [s, r] = kamping::v2::allreduce(send_data, recv_data | kamping::views::resize, MPI_SUM);

    int expected_sum_ranks       = size_ * (size_ - 1) / 2;
    int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
    EXPECT_EQ(r[0], expected_sum_ranks);
    EXPECT_EQ(r[1], expected_sum_rank_plus_1);
}

TEST_F(AllreduceTest, allreduce_inplace) {
    std::vector<int> data = {rank_, rank_ + 1};

    kamping::v2::allreduce(kamping::v2::inplace, data, MPI_SUM);

    int expected_sum_ranks       = size_ * (size_ - 1) / 2;
    int expected_sum_rank_plus_1 = size_ * (size_ - 1) / 2 + size_;
    EXPECT_EQ(data[0], expected_sum_ranks);
    EXPECT_EQ(data[1], expected_sum_rank_plus_1);
}
