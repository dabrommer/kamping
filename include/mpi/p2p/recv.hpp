#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::recv_buffer                               RBuf,
    mpi::experimental::rank                                      Source = int,
    mpi::experimental::tag                                       Tag    = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void recv(
    RBuf&&      rbuf,
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Recv(
        mpi::experimental::data(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
