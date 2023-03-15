set(CMAKE_CXX_STANDARD 20)

set(CPP_SOURCE_FILES
    src/reycode_graphics/rhi/draw.cpp
    src/reycode_graphics/rhi/shader.cpp
    src/reycode_graphics/rhi/vertex_buffer.cpp
    src/reycode_graphics/rhi/window.cpp

    vendor/glad/src/glad.c
)

if(${REY_PLATFORM_CUDA})
    set(CPP_SOURCE_FILES ${CPP_SOURCE_FILES}
            src/reycode_graphics/rhi/CUDA/vertex_buffer.h src/reycode_graphics/rhi/CUDA/vertex_buffer.cpp src/reycode_graphics/scene.h)
endif()

add_library(reycode_graphics ${CPP_SOURCE_FILES})

include_directories(vendor/glad/include)
target_include_directories(reycode_graphics PUBLIC src)

add_subdirectory(vendor/glfw)
target_include_directories(reycode_graphics PRIVATE vendor/glfw/include)
target_link_libraries(reycode_graphics reycode)