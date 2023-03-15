#include <glad/glad.h>
#include "reycode_graphics/rhi/shader.h"
#include <stdio.h>
#include <memory>

namespace reycode {
    Shader::Shader() {
        program = -1;
    }

    Shader::Shader(RHI& rhi, const char* vertex_shader_text, const char* fragment_shader_text) {
        program = -1;

        const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
        glCompileShader(vertex_shader);

        int  success;
        char infoLog[512];
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);

        if (!success) {
            glGetShaderInfoLog(vertex_shader, 512, NULL, infoLog);
            printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s", infoLog);
        }

        const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
        glCompileShader(fragment_shader);

        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);

        if (!success) {
            glGetShaderInfoLog(fragment_shader, 512, NULL, infoLog);
            printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s", infoLog);
        }

        program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);
    }

    Shader& Shader::operator=(Shader && other) {
        this->~Shader();
        new (this) Shader(std::move(other));
        return *this;
    }

    Shader::Shader(Shader && other) {
        program = other.program;
        other.program = 0;
    }

    Shader::~Shader() {
        glDeleteProgram(program);
    }
}