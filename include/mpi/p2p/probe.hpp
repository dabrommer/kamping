#pragma once

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

template <
    mpi::experimental::mpi_rank                                  Source = int,
    mpi::experimental::mpi_tag                                   Tag    = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>       Comm   = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Status> Status = MPI_Status*>
void probe(
    Source      source = MPI_ANY_SOURCE,
    Tag         tag    = MPI_ANY_TAG,
    Comm const& comm   = MPI_COMM_WORLD,
    Status&&    status = MPI_STATUS_IGNORE
) {
    int err = MPI_Probe(
        mpi::experimental::to_rank(source),
        mpi::experimental::to_tag(tag),
        mpi::experimental::native_handle(comm),
        mpi::experimental::native_handle_ptr(status)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
