#pragma once

#include "reycode/reycode.h"
#include "reycode/mesh/mesh.h"
#include "libmorton/morton3D.h"
#include "detail/cube.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_Vector.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <inttypes.h>

namespace reycode {
    template<class Exec, class Mem>
    class Hexcore {
    public:
        using Morton_Code = uint64_t;
        struct Refinement_Mask {
            uint32_t level: 8;
            bool ghost : 1;
            uint32_t up : 8;
            uint32_t down : 8;
        };

        using Device = Kokkos::Device<Exec, Mem>;

        static CGPU uvec3 morton_decode(Morton_Code code) {
            uvec3 result;
            libmorton::m3D_d_magicbits<Morton_Code, uint32_t>(code, result.x, result.y, result.z);
            return result;
        }

        static CGPU Morton_Code morton_encode(uvec3 pos) {
            Morton_Code code = libmorton::m3D_e_magicbits<Morton_Code, uint32_t>(pos.x, pos.y, pos.z);
            return code;
        }

        static constexpr uint32_t MAX_REFINEMENT = 10;
        static constexpr uint32_t CUBE_FACE_SUBDIVISION = 4;

    private:
        Kokkos::View<uint32_t *, Device> hash_bucket_count;
        Kokkos::View<uint64_t *, Device> hash_bucket_start;

        Kokkos::View<Morton_Code *, Device> morton_keys;
        Kokkos::View<Refinement_Mask *, Device> refinement_mask;

        struct Boundary_Patch {
            Kokkos::View<uint64_t *, Device> id;
            Kokkos::View<Cube_Face *, Device> face;

            uint64_t cell_count() const { return id.size(); }
        };

        std::vector<Boundary_Patch> m_patches;

        uint64_t m_cell_count;
        uint64_t m_cell_capacity;
        uint64_t m_cell_base = 0;

        struct Grid_Level_LUT {
            vec3 dx;
            vec3 idx;
            vec3 dx2;
            vec3 idx2;

            real cell_vol;
            real cell_ivol;

            vec3 face_normal[CUBE_FACE_COUNT];
            vec3 face_sf[CUBE_FACE_COUNT];

            real face_a[CUBE_FACE_COUNT];
            real face_dx[CUBE_FACE_COUNT];
            real face_idx[CUBE_FACE_COUNT];
        };

        vec3 m_extent;
        Grid_Level_LUT luts[MAX_REFINEMENT+1];

        uint64_t hash_buckets_count() const { return hash_bucket_count.size(); }

        CGPU uint64_t cell_id(Morton_Code morton) const {
            uint64_t hash = morton % hash_bucket_count.size();
            uint32_t count = hash_bucket_count(hash);
            uint32_t start = hash_bucket_start(hash);

            for (uint32_t i = 0; i < count; i++) {
                uint32_t dense_idx = start + i;
                if (morton_keys(dense_idx) == morton) return dense_idx;
            }

            return UINT64_MAX;
        }

        CGPU Morton_Code cell_neigh_morton(uint64_t id, Cube_Face face, uint32_t subcell) const {
            Refinement_Mask mask = refinement_mask(id);
            bool up = mask.up & (1<<face);
            bool down = mask.down & (1<<face);
            bool half = up && (face==CUBE_FACE_NEG_X || face==CUBE_FACE_NEG_Y || face==CUBE_FACE_NEG_Z);

            uint32_t level = mask.level;
            uvec3 pos = morton_decode(morton_keys(id));
            uvec3 pos_neigh = uvec3(pos + uvec3(1<<(MAX_REFINEMENT-(level+half)))*cube_normals[face]);

            uint32_t axis = cube_face_axis[face];
            uvec3 sub_offset = uvec3(0,0,0);
            sub_offset[(axis+1)%3] = subcell % 2;
            sub_offset[(axis+2)%3] = subcell / 2;

            uint64_t mask_down =  ~((1<<(MAX_REFINEMENT-(level+up-down)))-1);

            pos_neigh = pos_neigh & mask_down;
            pos_neigh = pos_neigh + (1u<<(MAX_REFINEMENT-(level+1)))*sub_offset;

            Morton_Code morton_neigh = morton_encode(pos_neigh); //todo: check refinement level
            return morton_neigh;
        }

        CGPU vec3 cell_position(Refinement_Mask mask, Morton_Code code) const {
            uvec3 pos = morton_decode(code);
            return luts[MAX_REFINEMENT].dx * (vec3(pos) - vec3(1<<MAX_REFINEMENT)) + 0.5_R * luts[mask.level].dx;
        }

    public:
        struct Init_Desc {
            uint32_t capacity = 0;
            vec3 dx;
            vec3 extent;
        };

        struct Update_Desc {
            uint64_t count;
            Kokkos::View<Morton_Code *, Device> morton_codes;
            Kokkos::View<Refinement_Mask *, Device> refinement_mask;
            struct Boundary_Patch {
                Kokkos::View<Morton_Code*, Device> morton_codes;
                Kokkos::View<Cube_Face*, Device> faces;
            };

            std::vector<Boundary_Patch> patches;

            Kokkos::View<uint32_t *, Device> refine_ids;
            Kokkos::View<uint32_t *, Device> coarsen_ids;
        };

        vec3 extent() const { return m_extent; }

        uint64_t cell_count() const { return m_cell_count; }

        uint64_t face_count() const { return m_cell_count * CUBE_FACE_COUNT * 4; }

        static constexpr uint32_t MAX_COEFFS = 1 + CUBE_FACE_COUNT*CUBE_FACE_SUBDIVISION;

        void resize(uint64_t size) {
            uint64_t hash_table_size = 8*size;
            hash_bucket_count = Kokkos::View<uint32_t *, Device>("HASH_BUCKET_COUNT", hash_table_size);
            hash_bucket_start = Kokkos::View<uint64_t *, Device>("HASH_BUCKET_START", hash_table_size);

            morton_keys = Kokkos::View<Morton_Code *, Device>("MORTON_KEYS", size);
            refinement_mask = Kokkos::View<Refinement_Mask *, Device>("REFINEMENT_MASK", size);

            m_cell_count = size;
            m_cell_capacity = size;
        }

        struct handle {
            uint64_t level: 8;
            uint64_t cell: 56;
        };


        class Face {
            const Hexcore* mesh = {};
            Cube_Face face;
            uint32_t subcell;
            uint64_t cell_id = 0;
            Refinement_Mask refinement;
            Morton_Code code;
            uint32_t stencil;
            Morton_Code neigh_morton;
            uint64_t m_neigh_id;
            uint32_t l;
        public:
            INL_CGPU Face() {}

            INL_CGPU Face(const Hexcore* level, uint64_t cell_id, Cube_Face face, uint32_t subcell, uint32_t stencil,
                          Morton_Code code, Refinement_Mask refinement)
                    : mesh(level), cell_id(cell_id), subcell(subcell), face(face), code(code), refinement(refinement),
                      stencil
                              (stencil) {
                assert(refinement.level == mesh->refinement_mask(cell_id).level);
                assert(code == mesh->morton_keys(cell_id));

                neigh_morton = mesh->cell_neigh_morton(cell_id, face, subcell);
                m_neigh_id = mesh->cell_id(neigh_morton);

                l = refinement.level+(refinement.up & (1 << face) ? 1 : 0);
            }

            INL_CGPU uint64_t id() const {
                return mesh->m_cell_base + cell_id * CUBE_FACE_COUNT * 4 + face * 4 + subcell;
                //cell_id * 4 * CUBE_FACE_COUNT + 4*face + subcell;
            }

            INL_CGPU vec3 center() const { return mesh->cell_position(refinement, code) + 0.5 * dx() * normal(); }

            INL_CGPU real vol() const { return mesh->luts[refinement.level].cell_vol; }

            INL_CGPU real ivol() const { return mesh->luts[refinement.level].cell_ivol; }

            INL_CGPU real dx() const { return mesh->luts[refinement.level].face_dx[face] * (0.5 + (refinement.up&
            (1<<face) ? 0.25 : 0.5)); }

            INL_CGPU real idx() const { return mesh->luts[l].face_idx[face]; }

            INL_CGPU real fa() const { return mesh->luts[l].face_a[face]; }

            INL_CGPU vec3 sf() const { return mesh->luts[l].face_sf[face]; }

            INL_CGPU vec3 normal() const { return vec3(cube_normals[face]); }

            INL_CGPU uint64_t cell_stencil() const { return 0; }
            INL_CGPU uint64_t neigh_stencil() const {
                assert(stencil < CUBE_FACE_COUNT* CUBE_FACE_SUBDIVISION + 1);
                return stencil;
            }

            INL_CGPU auto cell() const { return Cell(mesh, cell_id, code, refinement); }

            INL_CGPU auto neigh_id() const {
                return m_neigh_id;
            }

            INL_CGPU auto neigh() const {
                Refinement_Mask mask = refinement;
                mask.ghost = true;
                return m_neigh_id==UINT64_MAX ? Cell(mesh,m_neigh_id,neigh_morton,mask) : mesh->get_cell(m_neigh_id,false);
            }

            INL_CGPU array<Vertex, 4> vertices() const {
                vec3 cell_pos = mesh->cell_position(refinement,code);
                array<Vertex, 4> result(4);
                vec3 dx = mesh->luts[refinement.level].dx;
                for (uint32_t i = 0; i < 4; i++) {
                    Vertex vertex;
                    vertex.pos = cell_pos + dx * cube_verts[cube_indices[face][i]]/2;
                    vertex.uv = quad_face_uvs[i];
                    vertex.normal = vec3(cube_normals[face]);
                    result[i] = vertex;
                }
                return result;
            }

            INL_CGPU array<uint32_t, 6> indices() {
                array<uint32_t, 6> result(6);
                for (uint32_t i = 0; i < 6; i++) result[i] = quad_face_indices[i];
                return result;
            }

            // Hexcore-specific
            Cube_Face cube_face() const { return face; }
            uint32_t sub_cell() const { return subcell; }
        };

        class Cell {
            const Hexcore *mesh = {};
            uint64_t cell_id = 0;
            Morton_Code code;
            Refinement_Mask refinement;
            array<Face, CUBE_FACE_SUBDIVISION*CUBE_FACE_COUNT> face_cache;

            friend Hexcore;
        public:
            INL_CGPU Cell(const Hexcore *level, uint64_t cell_id, Morton_Code code, Refinement_Mask refinement) : mesh
                                                                                                                          (level), cell_id(cell_id), code(code), refinement(refinement) {}

            INL_CGPU bool is_ghost() { return refinement.ghost; }

            INL_CGPU real vol() const { return mesh->luts[refinement.level].cell_vol; }

            INL_CGPU void ivol() const { return mesh->luts[refinement.level].cell_ivol; }

            INL_CGPU vec3 center() const {
                return mesh->cell_position(refinement, code);
            }

            INL_CGPU uint64_t id() const { return mesh->m_cell_base + cell_id; }

            INL_CGPU uint32_t id_stencil() const { return 0; }

            INL_CGPU slice<Face> faces() const {
                return face_cache;
            }

            INL_CGPU array<uint64_t, CUBE_FACE_COUNT*CUBE_FACE_SUBDIVISION+1> stencil() const {
                if (refinement.ghost) {
                    return {0,1};
                }

                auto faces = this->faces();
                array<uint64_t, CUBE_FACE_COUNT*CUBE_FACE_SUBDIVISION+1> result(1+faces.size());
                result[0] = id();
                for (Face& face : faces) {
                    assert(face.neigh_stencil() != 0);
                    result[face.neigh_stencil()] = face.neigh_id();
                }
                return result;
            };

            // Hexcore specific attributes
            Refinement_Mask refinement_mask() { return refinement; }
            Morton_Code morton_code() { return code; }
            vec3 dx() { return mesh->luts[refinement.level].dx; }
        };


        Cell get_cell(uint64_t id, bool with_stencil = true) const {
            assert(id >= m_cell_base);

            Morton_Code code = morton_keys(id);
            Refinement_Mask refinement = refinement_mask(id);

            Cell result(this, id - m_cell_base, code, refinement);
            if (!with_stencil) return result;

            uint32_t stencil_offset = 1;

            for (uint32_t i = 0; i < CUBE_FACE_COUNT; i++) {
                bool refine = result.refinement.up & (1<<i);
                if (refine) {
                    result.face_cache.push_back(Face(this, id, Cube_Face(i), 0, stencil_offset, code, refinement));
                    result.face_cache.push_back(Face(this, id, Cube_Face(i), 1, stencil_offset+1, code, refinement));
                    result.face_cache.push_back(Face(this, id, Cube_Face(i), 2, stencil_offset+2, code, refinement));
                    result.face_cache.push_back(Face(this, id, Cube_Face(i), 3, stencil_offset+3, code, refinement));
                    stencil_offset += 4;
                } else {
                    result.face_cache.push_back(Face(this, id, Cube_Face(i), 0, stencil_offset, code, refinement));
                    stencil_offset++;
                }
            }
            return result;
        }

        Face get_face(uint64_t id) const {
            uint64_t cell_id = id / (CUBE_FACE_COUNT*4);
            Cube_Face face = Cube_Face(id%(CUBE_FACE_COUNT*4) / 4);
            uint32_t sub_id = id % 4;
            assert(cell_id >= m_cell_base);
            return Face(this, cell_id - m_cell_base, face, sub_id, -1, morton_keys(cell_id), refinement_mask(cell_id));
        }

        std::vector<Boundary_Patch> patches() {
            return m_patches;
        }

        template<class Func>
        void for_each_patch_cell(const char* name, Patch patch, Func func) const {
            const Boundary_Patch& ghost = m_patches[patch.id];
            Kokkos::parallel_for(name,
                                 Kokkos::RangePolicy<Exec>(0, ghost.cell_count()),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Cube_Face cube_face = ghost.face(i);
                                     uint64_t id = ghost.id(i);
                                     Refinement_Mask refinement = refinement_mask(id);
                                     Morton_Code code = morton_keys(id);
                                     Face face(this, id, cube_face, 0, 1, code, refinement);
                                     func(face);
                                 });
        }

        template<class Func>
        void for_each_cell(const char *name, Func func) const {
            Kokkos::parallel_for(name,
                                 Kokkos::RangePolicy<Exec>(0, cell_count()),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Cell cell = get_cell(i);
                                     if (cell.is_ghost()) return;
                                     func(cell);
                                 });
        }

        template<class Func>
        void for_each_face(const char *name, Func func) const {
            Kokkos::parallel_for(name,
                                 Kokkos::RangePolicy<Exec>(0, cell_count()),
                                 [=] CGPU(uint64_t i) {
                                     Cell cell = get_cell(i);
                                     if (cell.is_ghost()) return;
                                     for (Face face : cell.faces()) {
                                         func(face);
                                     }
                                 });
        }

        void init(const Init_Desc &init) {
            m_extent = init.extent;

            for (uint32_t i = 0; i <= MAX_REFINEMENT; i++) {
                Grid_Level_LUT &lut = luts[i];
                lut.dx = init.dx / (1 << i);
                lut.dx2 = lut.dx * lut.dx;
                lut.idx = vec3(1.0) / lut.dx;
                lut.idx2 = lut.idx * lut.idx;
                lut.cell_vol = lut.dx.x * lut.dx.y * lut.dx.z;
                lut.cell_ivol = 1.0_R / lut.cell_vol;

                for (uint32_t j = 0; j < CUBE_FACE_COUNT; j++) {
                    uint32_t axis = cube_face_axis[j];
                    lut.face_dx[j] = lut.dx[axis];
                    lut.face_idx[j] = 1.0 / lut.face_dx[j];
                    lut.face_normal[j] = vec3(cube_normals[j]);
                    lut.face_a[j] = lut.cell_vol / lut.face_dx[j];
                    lut.face_sf[j] = lut.face_a[j] * vec3(cube_normals[j]);
                }
            }

            resize(init.capacity);
        }

        void update(Update_Desc& init) {
            if (init.count > m_cell_capacity) {
                resize(max<uint64_t>(init.count, 2*m_cell_capacity));
            } else if (init.count <= m_cell_capacity) {
                resize(init.count); //todo: clear
            }

            m_cell_count = init.count;

            uint64_t morton_codes_size = init.count;
            uint64_t hash_buckets_count = this->hash_buckets_count();

            Kokkos::View<uint32_t *, Device> hash_position("HASH POSITION", morton_codes_size);

            Kokkos::parallel_for("HEXCORE HASH BUCKET COUNT",
                                 Kokkos::RangePolicy<Exec>(0, morton_codes_size),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Morton_Code code = init.morton_codes(i);
                                     uint32_t position = Kokkos::atomic_fetch_inc(
                                             &hash_bucket_count(code % hash_buckets_count));
                                     hash_position(i) = position;
                                 });

            Kokkos::parallel_scan("HEXCORE HASH BUCKET ASSIGNMENT", Kokkos::RangePolicy<Exec>(0, hash_buckets_count),
                                  KOKKOS_LAMBDA(uint64_t i, uint64_t &partial_sum, bool is_final) {
                                      if (is_final) hash_bucket_start(i) = partial_sum;
                                      partial_sum += hash_bucket_count(i);
                                  }
            );

            Kokkos::parallel_for("HEXCORE HASH MORTON KEYS", Kokkos::RangePolicy<Exec>(0, morton_codes_size),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Morton_Code code = init.morton_codes(i);
                                     uint64_t loc = hash_bucket_start(code % hash_buckets_count) + hash_position(i);
                                     morton_keys(loc) = code;
                                     refinement_mask(loc) = init.refinement_mask(i);
                                 });

            m_patches.resize(init.patches.size());
            for (uint32_t i = 0; i < init.patches.size(); i++) {
                auto& in_patch = init.patches[i];
                uint64_t count = in_patch.morton_codes.size();
                Boundary_Patch& out_patch = m_patches[i];
                out_patch.id = Kokkos::View<uint64_t*, Mem>("ids", count);
                out_patch.face = Kokkos::View<Cube_Face*, Mem>("face", count);

                Kokkos::parallel_for("HEXCORE BOUNDARY PATCH",
                                     Kokkos::RangePolicy<Exec>(0, count),
                                     KOKKOS_LAMBDA(uint64_t i) {
                                         out_patch.id(i) = cell_id(in_patch.morton_codes(i));
                                         out_patch.face(i) = in_patch.faces(i);
                                     });
            }
        }
    };
}
