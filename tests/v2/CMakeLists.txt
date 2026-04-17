kamping_register_test(test_v2_auto_counts_view FILES auto_counts_view_test.cpp)

kamping_register_test(test_v2_sentinels FILES sentinels_test.cpp)

kamping_register_mpi_test(
    test_v2_allgatherv
    FILES v2_allgatherv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_alltoall
    FILES v2_alltoall_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_alltoallv
    FILES v2_alltoallv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_reduce
    FILES v2_reduce_test.cpp
    CORES 1 2 4
)
