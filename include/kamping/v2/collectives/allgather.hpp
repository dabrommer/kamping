#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/result.hpp"

namespace kamping::core {
template <
    ranges::send_buffer                         SBuf,
    ranges::recv_buffer                         RBuf,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int comm_size = 0;
    MPI_Comm_size(kamping::bridge::native_handle(comm), &comm_size);
    int err = MPI_Allgather(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::ranges::data(rbuf),
        static_cast<int>(kamping::ranges::size(rbuf)) / comm_size,
        kamping::ranges::type(rbuf),
        kamping::bridge::native_handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {
template <
    ranges::send_buffer                         SBuf,
    ranges::recv_buffer                         RBuf,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto allgather(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgather{}, res.send, res.recv, kamping::bridge::native_handle(comm));
    core::allgather(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
