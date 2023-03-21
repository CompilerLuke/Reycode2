set(CMAKE_CXX_STANDARD 20)

set(CPP_SOURCE_FILES
    src/reycode_solver/solver.cpp
    src/reycode_solver/fv/fvm.h
        src/reycode_solver/fv/fvc.h
    src/reycode_solver/linear_solver/linear_solver.h
    src/reycode_solver/linear_solver/linear_solver.cpp
        src/reycode_solver/fv/bc.h)

add_library(reycode_solver ${CPP_SOURCE_FILES})

add_subdirectory(vendor/amgcl)
include_directories(vendor/amgcl)
target_include_directories(reycode_solver PUBLIC src)
target_link_libraries(reycode_solver reycode)
target_link_libraries(reycode_solver amgcl)