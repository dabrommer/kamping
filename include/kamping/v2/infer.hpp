#pragma once

#include <cstddef>

#include <mpi.h>

#include "kamping/v2/comm_op.hpp"
#include "mpi/handle.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/status.hpp"

/// @file
/// infer() is a customization point that transfers metadata from the sending to the receiving
/// side before an MPI operation is issued. The default behavior sets the recv count on resizable
/// recv buffers. Users can provide their own overloads via ADL for custom buffer types or to
/// transfer additional metadata alongside the count.
///
/// Dispatch is on operation tag types (comm_op::recv, comm_op::allgather, ...) rather than an enum, so
/// users can add new tags without modifying this header.

namespace kamping {

// ---- Default infer() overloads ----------------------------------------------

template <mpi::experimental::recv_buffer RBuf>
auto infer(comm_op::recv, RBuf& rbuf, int source, int tag, MPI_Comm comm) {
    if constexpr (kamping::ranges::deferred_recv_buf<RBuf>) {
        v2::status  status;
        MPI_Message message = MPI_MESSAGE_NULL;
        MPI_Mprobe(source, tag, comm, &message, mpi::experimental::handle_ptr(status));
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(status.count(mpi::experimental::type(rbuf))));
        return message;
    }
}
template <mpi::experimental::send_recv_buffer SRBuf>
auto infer(comm_op::bcast, SRBuf& srbuf, int root, MPI_Comm comm) {
    if constexpr (kamping::ranges::deferred_recv_buf<SRBuf>) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        auto size_on_root = rank == root ? mpi::experimental::count(srbuf) : 0;
        auto size_view    = std::views::single(size_on_root);
        MPI_Bcast(&size_on_root, 1, MPI_INT, root, comm);
        if (rank != root) {
            srbuf.set_recv_count(size_on_root);
        }
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::allgather, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::ranges::deferred_recv_buf<RBuf>) {
        int comm_size = 0;
        MPI_Comm_size(comm, &comm_size);
        rbuf.set_recv_count(comm_size * static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::allgatherv, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::ranges::deferred_recv_buf_v<RBuf>) {
        int comm_size = 0;
        MPI_Comm_size(comm, &comm_size);
        rbuf.set_comm_size(comm_size);
        int send_count = static_cast<int>(mpi::experimental::count(sbuf));
        MPI_Allgather(&send_count, 1, MPI_INT, std::ranges::data(mpi::experimental::counts(rbuf)), 1, MPI_INT, comm);
        rbuf.commit_counts();
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(comm_op::alltoall, SBuf const& sbuf, RBuf& rbuf, MPI_Comm /* comm */) {
    if constexpr (kamping::ranges::deferred_recv_buf<RBuf>) {
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(mpi::experimental::count(sbuf)));
    }
}

template <mpi::experimental::send_buffer_v SBuf, mpi::experimental::recv_buffer_v RBuf>
void infer(comm_op::alltoallv, SBuf const& sbuf, RBuf& rbuf, MPI_Comm comm) {
    if constexpr (kamping::ranges::deferred_recv_buf_v<RBuf>) {
        int comm_size = 0;
        MPI_Comm_size(comm, &comm_size);
        rbuf.set_comm_size(comm_size);
        MPI_Alltoall(
            std::ranges::data(mpi::experimental::counts(sbuf)),
            1,
            MPI_INT,
            std::ranges::data(mpi::experimental::counts(rbuf)),
            1,
            MPI_INT,
            comm
        );
        rbuf.commit_counts();
    }
}

template <mpi::experimental::send_buffer SBuf, mpi::experimental::recv_buffer RBuf>
void infer(
    comm_op::sendrecv, SBuf const& sbuf, RBuf& rbuf, int dest, int send_tag, int source, int recv_tag, MPI_Comm comm
) {
    if constexpr (kamping::ranges::deferred_recv_buf<RBuf>) {
        int const send_count = static_cast<int>(mpi::experimental::count(sbuf));
        int       recv_count = 0;
        MPI_Sendrecv(
            &send_count,
            1,
            MPI_INT,
            dest,
            send_tag,
            &recv_count,
            1,
            MPI_INT,
            source,
            recv_tag,
            comm,
            MPI_STATUS_IGNORE
        );
        rbuf.set_recv_count(static_cast<std::ptrdiff_t>(recv_count));
    }
}

} // namespace kamping
