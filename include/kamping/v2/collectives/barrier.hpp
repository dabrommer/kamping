#pragma once

#include <mpi.h>

#include "kamping/v2/error_handling.hpp"
#include "kamping/v2/native_handle.hpp"

namespace kamping::core {
template <bridge::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void barrier(Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Barrier(kamping::bridge::native_handle(comm));
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace kamping::core

namespace kamping::v2 {
using core::barrier;
}
