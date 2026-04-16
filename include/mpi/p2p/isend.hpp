#pragma once

#include <mpi.h>

#include "mpi/buffer.hpp"
#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {
template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void isend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Isend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void ibsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Ibsend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void issend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Issend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}

template <
    mpi::experimental::send_buffer                                SBuf,
    mpi::experimental::rank                                       Dest    = int,
    mpi::experimental::tag                                        Tag     = int,
    mpi::experimental::convertible_to_mpi_handle<MPI_Comm>        Comm    = MPI_Comm,
    mpi::experimental::convertible_to_mpi_handle_ptr<MPI_Request> Request = MPI_Request*>
void irsend(SBuf&& sbuf, Dest dest, Tag tag, Comm const& comm, Request&& request) {
    int err = MPI_Irsend(
        mpi::experimental::data(sbuf),
        static_cast<int>(mpi::experimental::count(sbuf)),
        mpi::experimental::type(sbuf),
        mpi::experimental::to_rank(dest),
        mpi::experimental::to_tag(tag),
        mpi::experimental::handle(comm),
        mpi::experimental::handle_ptr(request)
    );
    if (err != MPI_SUCCESS) {
        throw mpi_error(err);
    }
}
} // namespace mpi::experimental
