#pragma once

#include <glad/glad.h>

namespace reycode {
	struct RHI;

    struct Shader {
        GLuint program;

        Shader();
        Shader(RHI& rhi, const char* vertex_shader_text, const char* fragment_shader);
        Shader(const Shader&) = delete;
        Shader& operator=(Shader&&);
        Shader(Shader&&);
        ~Shader();
    };
}