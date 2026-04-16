#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::rank                                       Source  = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irecv(RBuf&& rbuf, Source source, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irecv(
        mpi::experimental::data(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
