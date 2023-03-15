set(CMAKE_CXX_STANDARD 20)

set(CPP_SOURCE_FILES
    src/reycode_viewer/colormap.cpp
    src/reycode_viewer/fpv.cpp
    src/reycode_viewer/main.cpp
        src/reycode_viewer/mesh_viewer.cpp src/reycode_viewer/mesh_viewer.h)

add_executable(reycode_viewer ${CPP_SOURCE_FILES})
target_link_libraries(reycode_viewer reycode reycode_graphics reycode_meshing reycode_solver)

target_include_directories(reycode_viewer PRIVATE src)
target_include_directories(reycode_viewer PRIVATE ../reycode_graphics/vendor/glad/include)

add_dependencies(reycode_viewer reycode)
add_dependencies(reycode_viewer reycode_graphics)