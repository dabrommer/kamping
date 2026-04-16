#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void send(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Send(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void bsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Bsend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void ssend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Ssend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                         SBuf,
    mpi::experimental::rank                                Dest = int,
    mpi::experimental::tag                                 Tag  = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm> Comm = MPI_Comm>
void rsend(SBuf&& sbuf, Dest dest, Tag tag = 0, Comm const& comm = MPI_COMM_WORLD) {
    int err = MPI_Rsend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
