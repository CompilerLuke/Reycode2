#pragma once

#include "reycode/mesh/mesh.h"
#include "reycode/mesh/hexcore.h"

#include "reycode_graphics/scene.h"
#include "reycode_graphics/rhi/vertex_buffer.h"
#include "reycode_graphics/rhi/shader.h"

#include "reycode_viewer/colormap.h"

#include "glad/glad.h"

#include <Kokkos_Core.hpp>

namespace reycode {
    template<class Exec, class Mem, class Mesh>
    class Mesh_Viewer {
        Mesh &mesh;
        Colormap& colormap;
        Shader shader;
        Vertex_Buffer vertex_buffer;

        static constexpr const char *vertex_shader_face_text =
                "#version 330\n"
                "uniform mat4 MVP;\n"
                "in vec3 v_pos;\n"
                "in vec3 v_normal;\n"
                "in vec3 v_color;\n"
                "in vec2 v_uv;\n"
                "out vec3 f_pos;\n"
                "out vec3 f_normal;\n"
                "out vec3 f_color;\n"
                "out vec2 f_uv;\n"
                "void main()\n"
                "{\n"
                "    gl_PointSize = v_normal.x;\n"
                "    f_pos = v_pos;\n"
                "    f_color = v_color;\n"
                "    f_normal = v_normal;\n"
                "    f_uv = v_uv;\n"
                "    gl_Position = MVP * vec4(v_pos, 1.0);\n"
                "}\n";

        static constexpr const char *fragment_shader_face_text =
                "#version 330\n"
                "out vec4 fragment;\n"
                "in vec3 f_normal;\n"
                "in vec3 f_pos;\n"
                "in vec3 f_color;\n"
                "in vec2 f_uv;\n"
                "in float f_scaling;\n"
                "uniform vec3 g_dir_light;\n"
                "uniform vec3 g_view_pos;\n"
                "void main()\n"
                "{\n"
                "   float light = 1.0;"
                "   vec3 view_dir = f_pos - g_view_pos;"
                "   vec2 line_strength = vec2(1e-1)/fwidth(f_uv);"
                "   vec2 cutoff = vec2(2e-1)/line_strength;\n"
                "   float line_width = 0;"
                "   vec2 edge_mask = clamp(line_strength * smoothstep(vec2(0.5)-line_width-cutoff, vec2(0.5)-line_width, abs(f_uv-vec2(0.5))),vec2(0),vec2(1));"
                "   vec3 pixel = f_color * light;"
                "   fragment = vec4(mix(pixel, vec3(0), max(edge_mask.x,edge_mask.y)),1.0);\n"
                "}\n";

    public:
        Mesh_Viewer(RHI &rhi, Mesh &mesh, Colormap& colormap) : mesh(mesh), colormap(colormap) {
            Vertex_Buffer_Desc desc = {};
            vertex_buffer = Vertex_Buffer(rhi, desc);
            shader = Shader(rhi, vertex_shader_face_text, fragment_shader_face_text);
        }

        void update(Kokkos::View<real*, Mem> x, real min, real max) {
            Kokkos::View<bool *, Mem> cell_visibility("CELL VISIBLE", mesh.cell_count());

            vec3 center = 0.5*mesh.extent();

            mesh.for_each_cell("CELL VISIBILITY", KOKKOS_LAMBDA(auto &cell) {
                vec3 position = cell.center();
                bool visible = position.z <= center.z;
                cell_visibility(cell.id()) = visible;
            });

            Kokkos::View<bool *, Mem> face_visibility("CELL VISIBLE", mesh.face_count());

            mesh.for_each_face("FACE VISIBILITY", KOKKOS_LAMBDA(auto& face) {
                uint64_t neigh = face.neigh().id();
                bool face_visible = cell_visibility(face.cell().id()) && (neigh==UINT64_MAX || !cell_visibility(neigh));
                face_visibility(face.id()) = face_visible;
            });

            Kokkos::View<uint64_t *, Mem> visible_face_ids("CELL VISIBLE", mesh.face_count());

            printf("Mesh faces: %i, cells: %i\n", mesh.face_count(), mesh.cell_count());

            uint64_t visible_face_count = 0;
            Kokkos::parallel_scan("VISIBLE FACE SCAN", Kokkos::RangePolicy<Exec>(0, mesh.face_count()), KOKKOS_LAMBDA
                    (uint64_t i,
                     uint64_t &
                     partial_sum, bool is_final) {
                bool visible = face_visibility(i);
                if (is_final && visible) visible_face_ids(partial_sum) = i;
                partial_sum += visible;
            }, visible_face_count);

            Kokkos::View<uint64_t *, Mem> vertex_offset("VERTEX OFFSET", mesh.face_count());
            Kokkos::View<uint64_t *, Mem> index_offset("INDEX OFFSET", mesh.face_count());
            uint64_t vertex_count = 0;
            uint64_t index_count = 0;

            Kokkos::parallel_scan("VERTEX SCAN", Kokkos::RangePolicy<Exec>(0, visible_face_count),
                                  KOKKOS_LAMBDA(uint64_t i, uint64_t &partial_sum, bool is_final) {
                                      if (is_final) vertex_offset(i) = partial_sum;
                                      partial_sum += mesh.get_face(visible_face_ids(i)).vertices().size();
                                  }, vertex_count);

            Kokkos::parallel_scan("VERTEX SCAN", Kokkos::RangePolicy<Exec>(0, visible_face_count),
                                  KOKKOS_LAMBDA(uint64_t i, uint64_t &partial_sum, bool is_final) {
                                      if (is_final) index_offset(i) = partial_sum;
                                      partial_sum += mesh.get_face(visible_face_ids(i)).indices().size();
                                  }, index_count);

            printf("Vertex count : %i, index count : %i\n", vertex_count, index_count);

            Kokkos::View<Vertex *, Mem> vertices("VERTICES", vertex_count);
            Kokkos::View<uint32_t *, Mem> indices("INDICES", index_count);

            Kokkos::parallel_for("GENERATE VERTICES", Kokkos::RangePolicy<Exec>(0, visible_face_count),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     auto face = mesh.get_face(visible_face_ids(i));
                                     uint64_t v_offset = vertex_offset(i);
                                     uint64_t i_offset = index_offset(i);

                                     auto face_vertices = face.vertices();
                                     auto face_indices = face.indices();

                                     for (int i = 0; i < face_vertices.size(); i++) {
                                         Vertex vertex = face_vertices[i];
                                         vertex.color = color_map(colormap, x(face.cell().id()), min, max);
                                         vertices(v_offset + i) = vertex;
                                     }

                                     for (int i = 0; i < face_indices.size(); i++) {
                                         indices(i_offset + i) = v_offset + face_indices[i];
                                     }
                                 });

            vertex_buffer.upload({vertices.data(), uint32_t(vertex_count)}, {indices.data(), uint32_t(index_count)});
        }

        void cube() {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            for (uint32_t i = 0; i < 6; i++) {
                uint32_t offset = vertices.size();
                for (uint32_t j = 0; j < 4; j++) {
                    Vertex vertex = {};
                    vertex.pos = cube_verts[cube_indices[i][j]];
                    vertex.normal = vec3(cube_normals[i]);
                    vertex.color = vec3(0,1,0);
                    vertices.push_back(vertex);
                }

                for (uint32_t j = 0; j < 6; j++) {
                    indices.push_back(offset + quad_face_indices[j]);
                }
            }
            vertex_buffer.upload({vertices.data(), uint32_t(vertices.size())}, {indices.data(), uint32_t(indices.size
                    ())});
        }

        void render(Scene &scene) {
            glBindVertexArray(vertex_buffer.vao);
            glUseProgram(shader.program);
            glUniformMatrix4fv(glGetUniformLocation(shader.program, "MVP"), 1, GL_TRUE, (const GLfloat *) &scene.mvp);
            glUniform3fv(glGetUniformLocation(shader.program, "g_dir_light"), 1, &scene.dir_light.x);
            glDrawElements(GL_TRIANGLES, vertex_buffer.arena.index_count, GL_UNSIGNED_INT, 0);
        }
    };
}