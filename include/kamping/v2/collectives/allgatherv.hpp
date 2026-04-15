#pragma once

#include <utility>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/infer.hpp"
#include "kamping/v2/native_handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/result.hpp"

namespace kamping::core {
template <
    ranges::send_buffer                         SBuf,
    ranges::recv_buffer_v                       RBuf,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void allgatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Allgatherv(
        kamping::ranges::data(sbuf),
        static_cast<int>(kamping::ranges::size(sbuf)),
        kamping::ranges::type(sbuf),
        kamping::ranges::data(rbuf),
        std::ranges::data(kamping::ranges::sizev(rbuf)),
        std::ranges::data(kamping::ranges::displs(rbuf)),
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
    ranges::recv_buffer_v                       RBuf,
    bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
auto allgatherv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) -> result<SBuf, RBuf> {
    result<SBuf, RBuf> res{std::forward<SBuf>(sbuf), std::forward<RBuf>(rbuf)};
    infer(comm_op::allgatherv{}, res.send, res.recv, kamping::bridge::native_handle(comm));
    core::allgatherv(res.send, res.recv, comm);
    return res;
}
} // namespace kamping::v2
