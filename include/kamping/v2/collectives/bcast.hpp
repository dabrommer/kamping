#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
namespace kamping::core {
template <
    ranges::send_recv_buffer                    SRBuf,
    bridge::mpi_rank                            Root = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bcast(
        kamping::ranges::data(send_recv_buf),
        static_cast<int>(kamping::ranges::size(send_recv_buf)),
        kamping::ranges::type(send_recv_buf),
        kamping::bridge::to_rank(root),
        kamping::bridge::native_handle(comm)

    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {
template <
    ranges::send_recv_buffer                    SRBuf,
    bridge::mpi_rank                            Root = int,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) -> SRBuf {
    infer(comm_op::bcast{}, send_recv_buf, kamping::bridge::to_rank(root), kamping::bridge::native_handle(comm));
    core::bcast(send_recv_buf, std::move(root), comm);
    return std::forward<SRBuf>(send_recv_buf);
}
} // namespace kamping::v2
