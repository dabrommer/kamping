#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::send_recv_buffer                    SRBuf,
    mpi::experimental::rank                                Root = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bcast(SRBuf&& send_recv_buf, Root root = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bcast(
        mpi::experimental::data(send_recv_buf),
        static_cast<int>(mpi::experimental::count(send_recv_buf)),
        mpi::experimental::type(send_recv_buf),
        mpi::experimental::to_rank(root),
        mpi::experimental::handle(comm)

    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
