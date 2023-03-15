set(CMAKE_CXX_STANDARD 20)

set(CPP_SOURCE_FILES
    src/reycode_meshing/hexcore.h
    src/reycode_meshing/hexcore.cpp
)

add_library(reycode_meshing ${CPP_SOURCE_FILES})
target_include_directories(reycode_meshing PUBLIC src)
target_link_libraries(reycode_meshing reycode)