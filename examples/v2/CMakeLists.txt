add_executable(example_v2_send_recv send_recv_example.cpp)
target_link_libraries(example_v2_send_recv PRIVATE kamping)
target_compile_options(example_v2_send_recv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_v2_collectives collectives_example.cpp)
target_link_libraries(example_v2_collectives PRIVATE kamping)
target_compile_options(example_v2_collectives PRIVATE ${KAMPING_WARNING_FLAGS})

FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
        URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
        OVERRIDE_FIND_PACKAGE
        SYSTEM
)
FetchContent_Declare(
        KokkosComm
        GIT_REPOSITORY https://github.com/kamping-site/kokkos-comm.git
        GIT_TAG develop
        SYSTEM
)
FetchContent_MakeAvailable(Kokkos KokkosComm)

if (TARGET Kokkos::kokkos)
    add_executable(example_v2_kokkos_send_recv kokkos_send_recv_example.cpp)
    target_link_libraries(example_v2_kokkos_send_recv PRIVATE kamping Kokkos::kokkos KokkosComm::KokkosComm)
    target_compile_options(example_v2_kokkos_send_recv PRIVATE ${KAMPING_WARNING_FLAGS})
endif ()
