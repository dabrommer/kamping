add_executable(example_v2_send_recv send_recv_example.cpp)
target_link_libraries(example_v2_send_recv PRIVATE kamping)
target_compile_options(example_v2_send_recv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_v2_collectives collectives_example.cpp)
target_link_libraries(example_v2_collectives PRIVATE kamping)
target_compile_options(example_v2_collectives PRIVATE ${KAMPING_WARNING_FLAGS})

if (TARGET Kokkos::kokkos)
    add_executable(example_v2_kokkos_send_recv kokkos_send_recv_example.cpp)
    target_link_libraries(example_v2_kokkos_send_recv PRIVATE kamping Kokkos::kokkos)
    target_compile_options(example_v2_kokkos_send_recv PRIVATE ${KAMPING_WARNING_FLAGS})
endif ()
