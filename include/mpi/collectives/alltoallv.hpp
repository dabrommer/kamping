#pragma once

#include <ranges>

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::send_buffer_v                       SBuf,
    mpi::experimental::recv_buffer_v                       RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void alltoallv(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Alltoallv(
        mpi::experimental::data(sbuf),
        std::ranges::data(mpi::experimental::sizev(sbuf)),
        std::ranges::data(mpi::experimental::displs(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::data(rbuf),
        std::ranges::data(mpi::experimental::sizev(rbuf)),
        std::ranges::data(mpi::experimental::displs(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
