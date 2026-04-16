#pragma once

#include <mpi.h>

#include "kamping/kassert/kassert.hpp"
#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::recv_buffer                         RBuf,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void alltoall(SBuf&& sbuf, RBuf&& rbuf, Comm const& comm = MPI_COMM_WORLD) {
    int comm_size = 0;
    MPI_Comm_size(mpi::experimental::handle(comm), &comm_size);
    KAMPING_ASSERT(mpi::experimental::count(sbuf) % comm_size == 0, "send buffer size must be divisible by comm size");
    KAMPING_ASSERT(mpi::experimental::count(rbuf) % comm_size == 0, "recv buffer size must be divisible by comm size");
    int err = MPI_Alltoall(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)) / comm_size,
        mpi::experimental::type(sbuf),
        mpi::experimental::data(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)) / comm_size,
        mpi::experimental::type(rbuf),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
