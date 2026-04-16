#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

template <
    mpi::experimental::rank                                   Source = int,
    mpi::experimental::tag                                    Tag    = int,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Message> Message,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm   = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status>  Status = MPI_Status*>
void mprobe(Source source, Tag tag, Comm const& comm, Message&& message, Status&& status = MPI_STATUS_IGNORE) {
    int err = MPI_Mprobe(
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(message),
        mpi::experimental::handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
