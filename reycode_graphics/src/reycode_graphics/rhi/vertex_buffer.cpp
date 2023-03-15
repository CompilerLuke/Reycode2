#include <glad/glad.h>
#include "reycode_graphics/rhi/vertex_buffer.h"

namespace reycode {
    Vertex_Buffer::Vertex_Buffer() {}

    Vertex_Buffer::Vertex_Buffer(RHI& rhi, const Vertex_Buffer_Desc& desc) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &indices);
        glGenBuffers(1, &vertices);

        glBindBuffer(GL_ARRAY_BUFFER, vertices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);

        enum Vertex_Attrib_Kind {
            VERTEX_ATTRIB_VEC2,
            VERTEX_ATTRIB_VEC3,
            VERTEX_ATTRIB_COUNT
        };

        GLenum TO_GL_VERTEX_ATTRIB_TYPE[VERTEX_ATTRIB_COUNT] = { GL_FLOAT, GL_FLOAT };
        uint32_t TO_GL_VERTEX_ATTRIB_COUNT[VERTEX_ATTRIB_COUNT] = { 2, 3 };

        struct Vertex_Attrib {
            Vertex_Attrib_Kind kind;
            uint64_t offset;
        };

        Vertex_Attrib attribs[4] = {
            {VERTEX_ATTRIB_VEC3, offsetof(Vertex, pos)},
            {VERTEX_ATTRIB_VEC3, offsetof(Vertex, normal)},
            {VERTEX_ATTRIB_VEC3, offsetof(Vertex, color)},
            {VERTEX_ATTRIB_VEC2, offsetof(Vertex, uv)},
        };
        uint32_t attrib_count = 4;

        for (uint32_t i = 0; i < attrib_count; i++) {
            Vertex_Attrib_Kind kind = attribs[i].kind;

            glEnableVertexAttribArray(i);
            glVertexAttribPointer(i, TO_GL_VERTEX_ATTRIB_COUNT[kind], TO_GL_VERTEX_ATTRIB_TYPE[kind], GL_FALSE, sizeof(Vertex), (void*)attribs[i].offset);
        }

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, desc.index_buffer_size, 0, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, desc.vertex_buffer_size, 0, GL_STATIC_DRAW);

        glBindVertexArray(0);

        arena.vertex_capacity = (uint32_t)(desc.vertex_buffer_size / sizeof(Vertex));
        arena.index_capacity = (uint32_t)(desc.index_buffer_size / sizeof(uint32_t));
    }

    void Vertex_Buffer::upload(slice<Vertex> vertices, slice<uint32_t> indices) {
        glBindVertexArray(vao);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * indices.length, indices.data, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.length, vertices.data, GL_STATIC_DRAW);
        arena.vertex_count = vertices.length;
        arena.index_count = indices.length;
    }

    Vertex_Buffer& Vertex_Buffer::operator=(Vertex_Buffer&& other) {
        this->~Vertex_Buffer();
        new (this) Vertex_Buffer(std::move(other));
        return *this;
    }

    Vertex_Buffer::Vertex_Buffer(Vertex_Buffer&& other) {
        arena = other.arena;
        vao = other.vao;
        vertices = other.vertices;
        indices = other.indices;
        other.vao = 0;
        other.vertices = 0;
        other.indices = 0;
        other.arena = {};
    }

    Vertex_Buffer::~Vertex_Buffer() {
        glDeleteBuffers(1, &indices);
        glDeleteBuffers(1, &vertices);
        glDeleteVertexArrays(1, &vao);
    }

    Vertex_Arena vertex_arena_push(Vertex_Arena& arena, uint32_t vertices, uint32_t indices) {
        Vertex_Arena sub = {};
        sub.vertex_offset = arena.vertex_offset + arena.vertex_count;
        sub.vertex_count = vertices;
        sub.vertex_capacity = vertices;
        sub.index_offset = arena.index_offset + arena.index_count;
        sub.index_count = indices;
        sub.index_capacity = indices;

        arena.vertex_count += vertices;
        arena.index_count += indices;

        assert(arena.vertex_count <= arena.vertex_capacity);
        assert(arena.index_count <= arena.index_capacity);

        return sub;
    }

    Vertex_Arena_Mapped vertex_arena_submap(Vertex_Arena_Mapped& master, Vertex_Arena& child) {
        uint32_t vertex_offset = child.vertex_offset - master.arena.vertex_offset;
        uint32_t index_offset = child.index_offset - master.arena.index_offset;

        Vertex_Arena_Mapped result = { child };
        result.vertices = subslice(master.vertices, vertex_offset, child.vertex_capacity);
        result.indices = subslice(master.indices, index_offset, child.index_capacity);
        return result;
    }
}