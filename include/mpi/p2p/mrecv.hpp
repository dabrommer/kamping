#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::recv_buffer                                RBuf,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Message> Message,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status>  Status>
void mrecv(RBuf&& rbuf, Message&& message, Status&& status) {
    int err = MPI_Mrecv(
        mpi::experimental::data(rbuf),
        static_cast<int>(mpi::experimental::count(rbuf)),
        mpi::experimental::type(rbuf),
        mpi::experimental::handle_ptr(message),
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
