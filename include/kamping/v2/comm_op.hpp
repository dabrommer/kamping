#pragma once

namespace kamping::comm_op {

struct recv {};
struct bcast {};
struct allgather {};
struct allgatherv {};
struct alltoall {};
struct alltoallv {};
struct sendrecv {};
struct reduce {};

} // namespace kamping::comm_op
