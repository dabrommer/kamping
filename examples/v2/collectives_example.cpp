#include <print>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/v2/collectives/allgather.hpp"
#include "kamping/v2/collectives/allgatherv.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/views.hpp"
#include "kamping/v2/views/ref_single_view.hpp"
#include "kamping/v2/views/resize_view.hpp"

int main(int, char*[]) {
    kamping::Environment<> env;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    {
        int val = 0;
        if (kamping::world_rank() == 0) {
            val = 42;
        }
        kamping::v2::bcast(kamping::views::ref_single(val));
        std::println("[R{}] bcast result={}", kamping::world_rank(), val);
    }
    {
        std::vector<int> sbuf{kamping::world_rank_signed(), kamping::world_rank_signed()};
        auto             v = kamping::v2::allgather(sbuf, std::vector<int>{} | kamping::views::resize).recv;
        std::println("allgather v={}", v);
    }
    {
        std::vector<int> sbuf{kamping::world_rank_signed(), kamping::world_rank_signed()};
        auto             v = kamping::v2::allgatherv(
                             sbuf,
                             std::vector<int>{} | kamping::views::auto_counts() | kamping::views::auto_displs()
                                 | kamping::views::resize_v
                         )
                                 .recv;
        std::println("allgatherv v={}", v);
    }
    return 0;
}
