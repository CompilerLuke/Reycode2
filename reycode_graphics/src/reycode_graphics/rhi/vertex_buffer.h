#pragma once

#include "reycode/reycode.h"
#include "reycode/mesh/detail/cube.h"
#include <glad/glad.h>

namespace reycode {
    struct RHI;

    struct Vertex_Buffer_Desc {
        size_t vertex_buffer_size = 0;
        size_t index_buffer_size = 0;
    };

    struct Vertex_Arena {
        uint32_t vertex_offset = 0;
        uint32_t vertex_count = 0;
        uint32_t vertex_capacity = 0;

        uint32_t index_offset = 0;
        uint32_t index_count = 0;
        uint32_t index_capacity = 0;
    };

    struct Vertex_Buffer {
        GLuint vao = 0;
        GLuint vertices = 0;
        GLuint indices = 0;

        Vertex_Arena arena;

        INLINE operator Vertex_Arena() { return arena; }

        Vertex_Buffer();
        Vertex_Buffer(RHI& RHI, const Vertex_Buffer_Desc&);
        Vertex_Buffer(Vertex_Buffer&&);
        Vertex_Buffer& operator=(Vertex_Buffer&&);
        ~Vertex_Buffer();
        void upload(slice<Vertex> vertices, slice<uint32_t> indices);
    };

    struct Vertex_Arena_Mapped {
        Vertex_Arena& arena;
        slice<Vertex> vertices;
        slice<uint32_t> indices;
    };

    Vertex_Arena_Mapped vertex_arena_submap(Vertex_Arena_Mapped& master, Vertex_Arena& child);
    Vertex_Arena vertex_arena_push(Vertex_Arena& arena, uint32_t vertices, uint32_t indices);
}