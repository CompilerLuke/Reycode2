#pragma once

namespace reycode {
    struct Vertex {
        vec3 pos;
        vec3 normal;
        vec3 color;
        vec2 uv;
    };

    constexpr uint32_t CUBE_FACE_COUNT = 6;
    constexpr uint32_t CUBE_VERTS = 8;

    enum Cube_Face {
        CUBE_FACE_NEG_Z,
        CUBE_FACE_POS_X,
        CUBE_FACE_POS_Z,
        CUBE_FACE_NEG_X,
        CUBE_FACE_POS_Y,
        CUBE_FACE_NEG_Y,
    };

    CGPU constexpr ivec3 cube_normals[CUBE_FACE_COUNT] = {
            {0,  0,  -1},
            {1,  0,  0},
            {0,  0,  1},
            {-1, 0,  0},
            {0,  1,  0},
            {0,  -1, 0}
    };

    CGPU constexpr uint32_t cube_face_axis[CUBE_FACE_COUNT] = {
            2,
            0,
            2,
            0,
            1,
            1
    };

    CGPU constexpr uint32_t cube_opposite_faces[CUBE_FACE_COUNT] = {
            2,
            3,
            0,
            1,
            5,
            4
    };

    CGPU constexpr vec3 cube_verts[CUBE_VERTS] = {
            {-1, -1, -1},
            {1,  -1, -1},
            {1,  1,  -1},
            {-1, 1,  -1},
            {-1, -1, 1},
            {1,  -1, 1},
            {1,  1,  1},
            {-1, 1,  1}
    };

    CGPU constexpr uint32_t cube_indices[CUBE_FACE_COUNT][4] = {
            {0, 1, 3, 2},
            {1, 5, 2, 6},
            {5, 4, 6, 7},
            {4, 0, 7, 3},
            {3, 2, 7, 6},
            {4, 5, 0, 1}
    };

    CGPU constexpr vec2 quad_face_uvs[4] = { vec2(0), vec2(1,0), vec2(0,1), vec2(1,1) };

    CGPU constexpr uint32_t quad_face_indices[6] = {
            0, 1, 2,
            2, 1, 3
    };

    CGPU constexpr uint32_t quad_line_indices[8] = {
            0, 1,
            1, 3,
            3, 2,
            2, 0
    };
}