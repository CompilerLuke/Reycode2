#include "vertex_buffer.h"

namespace reycode {
    Vertex_Arena_Mapped vertex_buffer_map_cuda(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err) {
        Vertex_Arena_Mapped mapped = {desc.vbuffer.arena};
        cudaGraphicsResource_t resources[2] = { desc.ibo_resource, desc.vbo_resource };
        err |= cudaGraphicsMapResources(2, resources, 0);

        size_t vertices_bytes, indices_bytes;
        err |= cudaGraphicsResourceGetMappedPointer((void**)(&mapped.vertices.data), &vertices_bytes, desc.vbo_resource);
        err |= cudaGraphicsResourceGetMappedPointer((void**)(&mapped.indices.data), &indices_bytes, desc.ibo_resource);

        mapped.vertices.length = (uint32_t)(vertices_bytes / sizeof(Vertex));
        mapped.indices.length = (uint32_t)(indices_bytes / sizeof(uint32_t));

        return mapped;
    }

    void vertex_buffer_cuda_unmap(Map_Vertex_Buffer_Cuda_Desc& desc, Cuda_Error err) {
        cudaGraphicsResource_t resources[2] = { desc.ibo_resource, desc.vbo_resource };
        err |= cudaGraphicsUnmapResources(2, resources, 0);
    }
}