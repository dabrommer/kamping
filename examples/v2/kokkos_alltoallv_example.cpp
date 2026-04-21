#include <cstddef>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/v2/collectives/alltoallv.hpp"
#include "kamping/v2/contrib/kokkos_view.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/handle.hpp"

template <>
struct mpi::experimental::handle_traits<kamping::Communicator<>> {
    static MPI_Comm handle(kamping::Communicator<> const& comm) {
        return comm.mpi_communicator();
    }
};

using matrix_t = Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>;

auto matrix_to_string(int rank, int size, matrix_t const& matrix, int cols_to_print) -> std::string {
    std::string out = "rank " + std::to_string(rank) + "\n";
    for (int row = 0; row < size; ++row) {
        out += "  row " + std::to_string(row) + ": [";
        for (int col = 0; col < cols_to_print; ++col) {
            out += std::to_string(matrix(static_cast<std::size_t>(row), static_cast<std::size_t>(col)));
            out += (col + 1 < cols_to_print) ? ", " : "";
        }
        out += "]\n";
    }
    out += "\n";
    return out;
}

int main(int argc, char* argv[]) {
    kamping::Environment<>  env;
    kamping::Communicator<> comm;
    Kokkos::initialize(argc, argv);

    {
        int const rank = static_cast<int>(comm.rank());
        int const size = static_cast<int>(comm.size());

        matrix_t send_matrix("send_matrix", static_cast<std::size_t>(size), static_cast<std::size_t>(size));
        for (std::size_t i = 0; i < send_matrix.extent(0); ++i) {
            for (std::size_t j = 0; j < send_matrix.extent(1); ++j) {
                send_matrix(i, j) = rank;
            }
        }

        std::vector<int> send_counts(static_cast<std::size_t>(size));
        for (int dst = 0; dst < size; ++dst) {
            send_counts[static_cast<std::size_t>(dst)] = dst + 1;
        }

        {
            // Example 1: receive into size X (rank + 1) matrix
            matrix_t recv_matrix("recv_matrix", static_cast<std::size_t>(size), static_cast<std::size_t>(rank + 1));

            kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_matrix | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                comm
            );
        }

        {
            // Example 2: receive into a contiguous subview (all columns, first rank+1 rows).
            matrix_t recv_matrix_full(
                "recv_matrix_full",
                static_cast<std::size_t>(size),
                static_cast<std::size_t>(size)
            );
            for (std::size_t i = 0; i < recv_matrix_full.extent(0); ++i) {
                for (std::size_t j = 0; j < recv_matrix_full.extent(1); ++j) {
                    recv_matrix_full(i, j) = -1;
                }
            }

            auto recv_subview = Kokkos::subview(
                recv_matrix_full,
                std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(rank + 1)},
                Kokkos::ALL()
            );

            kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_subview | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                comm
            );
        }

        {
            // Example 3: receive into a non-contiguous subview (all rows, first rank+1 columns)
            matrix_t recv_matrix_full(
                "recv_matrix_full",
                static_cast<std::size_t>(size),
                static_cast<std::size_t>(size)
            );
            for (std::size_t i = 0; i < recv_matrix_full.extent(0); ++i) {
                for (std::size_t j = 0; j < recv_matrix_full.extent(1); ++j) {
                    recv_matrix_full(i, j) = -1;
                }
            }

            auto recv_subview = Kokkos::subview(
                recv_matrix_full,
                Kokkos::ALL(),
                std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(rank + 1)}
            );

            auto result = kamping::v2::alltoallv(
                send_matrix | kamping::v2::views::kokkos | kamping::v2::views::with_counts(send_counts)
                    | kamping::v2::views::auto_displs(),
                recv_subview | kamping::v2::views::kokkos | kamping::v2::views::auto_counts()
                    | kamping::v2::views::auto_displs(),
                comm
            );

            auto& sbuf = result.get<0>();
            auto& rbuf = result.get<1>();

            // Force unpack
            sbuf.base().base().base().unwrap();
            rbuf.base().base().base().unwrap();

            std::println("{}", matrix_to_string(rank, size, send_matrix, size));
            std::println("{}", matrix_to_string(rank, size, recv_matrix_full, size));
        }
    }

    Kokkos::finalize();
    return 0;
}
