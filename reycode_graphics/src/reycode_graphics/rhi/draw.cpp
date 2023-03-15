#include "draw.h"
#include "vertex_buffer.h"
#include <glad/glad.h>

namespace reycode {
    int DRAW_MODE_TO_GL[DRAW_MODE_COUNT] = { GL_TRIANGLES, GL_LINES, GL_POINTS };

    /*void cmd_buffer_bind(Command_Buffer& buffer, const Vertex_Buffer& vertex_buffer) {
        glBindVertexArray(vertex_buffer.vao);
    }

    void Command_Buffer::draw(Draw_Mode mode, const Vertex_Buffer& buffer, const Shader& shader) {
        int gl_mode = DRAW_MODE_TO_GL[mode];
        glBindVertexArray(buffer.vao);
        glDrawElements(gl_mode, buffer.arena.index_count, GL_UNSIGNED_INT, 0);
    }*/
}