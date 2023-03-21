set(CMAKE_CXX_STANDARD 20)

set(CPP_SOURCE_FILES
    src/reycode/reycode.cpp
    src/reycode/mesh/mesh.cpp
    src/reycode/mesh/hexcore.h
    src/reycode/mesh/polymesh.h
    src/reycode/mesh/detail/cube.h ../reycode_solver/src/reycode_solver/field/field.h
        ../reycode_solver/src/reycode_solver/fv/fvc.h
        ../reycode_solver/src/reycode_solver/fv/fvm.h ../reycode_solver/src/reycode_solver/field/field.h src/reycode/mesh/mixed_mesh.h)

add_library(reycode ${CPP_SOURCE_FILES})

target_include_directories(reycode PUBLIC src)
target_link_libraries(reycode glfw)

set(Kokkos_ENABLE_THREADS ON)
add_subdirectory(vendor/libmorton)
add_subdirectory(vendor/kokkos)
target_include_directories(reycode PUBLIC ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(reycode kokkos)
target_link_libraries(reycode libmorton)
