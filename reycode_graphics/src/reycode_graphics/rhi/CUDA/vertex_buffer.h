#pragma once

#include <cuda_runtime.h>

namespace reycode {
    struct Vertex_Buffer;

    struct Map_Vertex_Buffer_Cuda_Desc {
        Vertex_Buffer& vbuffer;
        cudaGraphicsResource_t vbo_resource;
        cudaGraphicsResource_t ibo_resource;
    };

    void vertex_buffer_cuda_unmap(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err);
    Vertex_Arena_Mapped vertex_buffer_map_cuda(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err);
}