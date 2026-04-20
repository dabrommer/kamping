#include <print>

#include <Kokkos_Core.hpp>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/v2/contrib/kokkos_view.hpp"
#include "mpi/handle.hpp"
#include "kamping/v2/p2p/recv.hpp"
#include "kamping/v2/p2p/send.hpp"

template <>
struct mpi::experimental::handle_traits<kamping::Communicator<>> {
    static MPI_Comm handle(kamping::Communicator<> const& comm) {
        return comm.mpi_communicator();
    }
};

int main(int argc, char* argv[]) {
    kamping::Environment<>  env;
    kamping::Communicator<> comm;
    Kokkos::initialize(argc, argv);

    KAMPING_ASSERT(comm.size() == 2uz, "This example must be run with exactly 2 ranks.");
    using matrix_t = Kokkos::View<int**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

    // Send and recv subview on 4x5 matrix
    if (comm.rank() == 0) {
        matrix_t matrix("send_matrix", 4, 5);
        for (std::size_t i = 0; i < matrix.extent(0); ++i) {
            for (std::size_t j = 0; j < matrix.extent(1); ++j) {
                matrix(i, j) = static_cast<int>(100 + 10 * i + j);
            }
        }
        // Get a non-contiguous row view.
        auto row = Kokkos::subview(matrix, 1, Kokkos::ALL());
        kamping::v2::send(row | kamping::v2::views::kokkos, 1, 0, comm);

    } else if (comm.rank() == 1) {
        matrix_t matrix("recv_matrix", 4, 5);
        auto     row      = Kokkos::subview(matrix, 2, Kokkos::ALL());
        auto&    received = *(kamping::v2::recv(row | kamping::v2::views::kokkos, 0, 0, comm));
        std::print("rank {} received row = [", comm.rank());
        for (std::size_t j = 0; j < received.extent(0); ++j) {
            std::print("{}{}", received(j), (j + 1 < received.extent(0)) ? ", " : "");
        }
        std::println("]");
    }

    // Send vector and recv using kamping::views::unpack<int>
    if (comm.rank() == 0) {
        Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> v("send_unpack", 6);
        for (std::size_t i = 0; i < v.extent(0); ++i) {
            v(i) = static_cast<int>(200 + i);
        }
        kamping::v2::send(v | kamping::v2::views::kokkos, 1, 0, comm);
    } else if (comm.rank() == 1) {
        auto  received = kamping::v2::recv(kamping::v2::views::auto_kokkos_view<int>(), 0, 0, comm);
        auto& data     = *received;
        std::print("rank {} unpack recv = [", comm.rank());
        for (std::size_t i = 0; i < data.extent(0); ++i) {
            std::print("{}{}", data(i), (i + 1 < data.extent(0)) ? ", " : "");
        }
        std::println("]");
    }

    // Send 4x5 matrix and recv using kamping::views::unpack<int>
    if (comm.rank() == 0) {
        matrix_t matrix("send_matrix", 4, 5);
        for (std::size_t i = 0; i < matrix.extent(0); ++i) {
            for (std::size_t j = 0; j < matrix.extent(1); ++j) {
                matrix(i, j) = static_cast<int>(100 + 10 * i + j);
            }
        }
        kamping::v2::send(matrix | kamping::v2::views::kokkos, 1, 0, comm);

    } else if (comm.rank() == 1) {
        auto  received = kamping::v2::recv(kamping::v2::views::auto_kokkos_view<int>(), 0, 0, comm);
        auto& data     = *received;
        std::print("rank {} unpack recv = [", comm.rank());
        for (std::size_t i = 0; i < data.extent(0); ++i) {
            std::print("{}{}", data(i), (i + 1 < data.extent(0)) ? ", " : "");
        }
        std::println("]");
    }

    Kokkos::finalize();
    return 0;
}
