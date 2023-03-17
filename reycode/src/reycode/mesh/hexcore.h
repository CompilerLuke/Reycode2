#pragma once

#include "reycode/reycode.h"
#include "libmorton/morton3D.h"
#include "detail/cube.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_Vector.hpp>
#include <inttypes.h>

namespace reycode {
    template<class Exec, class Mem>
    class Hexcore {
    public:
        using Morton_Code = uint32_t;
        struct Refinement_Mask {
            uint32_t level : 8 = 0;
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
    private:
        Kokkos::View<uint32_t *, Device> hash_bucket_count;
        Kokkos::View<uint64_t *, Device> hash_bucket_start;

        Kokkos::View<Morton_Code *, Device> morton_keys;
        Kokkos::View<Refinement_Mask *, Device> refinement_mask;

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
        Kokkos::vector<Grid_Level_LUT> luts;

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

        CGPU uint64_t cell_neigh(uint64_t id, Cube_Face face) const {
            uvec3 pos = morton_decode(morton_keys(id));
            uvec3 pos_neigh = uvec3(pos + cube_normals[face]);
            Morton_Code morton_neigh = morton_encode(pos_neigh); //todo: check refinement level
            return cell_id(morton_neigh);
        }

        CGPU vec3 cell_position(uint64_t id) const {
            uvec3 pos = morton_decode(morton_keys(id));
            uint32_t level = refinement_mask(id).level;
            return luts[level].dx * vec3(pos);
        }

    public:
        struct Init_Desc {
            uint32_t capacity = 0;
            vec3 dx;
            vec3 extent;
            uint32_t max_levels = 10;
            Kokkos::View<Morton_Code *, Device> morton_codes;
        };

        struct Update_Desc {
            Kokkos::View<uint32_t *, Device> refine_ids;
            Kokkos::View<uint32_t *, Device> coarsen_ids;
        };

        vec3 extent() const { return m_extent; }

        uint64_t cell_count() const { return m_cell_count; }

        uint64_t face_count() const { return m_cell_count * CUBE_FACE_COUNT; }

        static constexpr uint32_t MAX_COEFFS = 7;

        void resize(uint64_t size) {
            uint64_t hash_table_size = ceil_div(size,1024)*1024;
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
            const Hexcore *mesh = {};
            Cube_Face face;
            uint64_t cell_id = 0;
            Refinement_Mask refinement;
        public:
            INL_CGPU Face() {}

            INL_CGPU Face(const Hexcore *level, uint64_t cell_id, Cube_Face face, Refinement_Mask refinement)
                    : mesh(level), cell_id(cell_id), face(face), refinement(refinement) {}

            INL_CGPU uint64_t id() const { return mesh->m_cell_base + cell_id*CUBE_FACE_COUNT + face; }

            INL_CGPU vec3 center() const { return mesh->cell_position(cell_id, face) + dx()*normal(); }

            INL_CGPU real vol() const { return mesh->luts[refinement.level].cell_vol; }
            INL_CGPU real ivol() const { return mesh->luts[refinement.level].cell_ivol; }
            INL_CGPU real dx() const { return mesh->luts[refinement.level].face_dx[face]; }
            INL_CGPU real idx() const { return mesh->luts[refinement.level].face_idx[face]; }
            INL_CGPU real fa() const { return mesh->luts[refinement.level].face_a[face]; }

            INL_CGPU vec3 sf() const { return mesh->luts[refinement.level].face_sf[face]; }
            INL_CGPU vec3 normal() const { return vec3(cube_normals[face]); }

            INL_CGPU uint64_t cell_stencil() const { return 0; }
            INL_CGPU uint64_t neigh_stencil() const { return 1+face; }

            INL_CGPU uint64_t cell() const { return mesh->m_cell_base + cell_id; }
            INL_CGPU uint64_t neigh() const { return mesh->m_cell_base + mesh->cell_neigh(cell_id, face); }

            INL_CGPU array<Vertex, 4> vertices() const {
                vec3 cell_pos = mesh->cell_position(cell_id);
                array<Vertex, 4> result(4);
                vec3 dx = mesh->luts[refinement.level].dx;
                for (uint32_t i = 0; i < 4; i++) {
                    Vertex vertex;
                    vertex.pos = cell_pos + dx*cube_verts[cube_indices[face][i]]/2;
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
        };

        class Cell {
            const Hexcore *mesh = {};
            uint64_t cell_id = 0;
            Refinement_Mask refinement;
        public:
            INL_CGPU Cell(const Hexcore *level, uint64_t cell_id, Refinement_Mask refinement) : mesh(level), cell_id
            (cell_id), refinement(refinement) {}

            INL_CGPU real vol() const { return mesh->luts[refinement.level].cell_vol; }
            INL_CGPU void ivol() const { return mesh->luts[refinement.level].cell_ivol; }
            INL_CGPU vec3 center() const { return mesh->cell_position(cell_id); }

            INL_CGPU uint64_t id() const { return mesh->m_cell_base + cell_id; }

            INL_CGPU array<Face, CUBE_FACE_COUNT> faces() const {
                array<Face, CUBE_FACE_COUNT> result(CUBE_FACE_COUNT);
                for (uint32_t i = 0; i < CUBE_FACE_COUNT; i++) {
                    result[i] = Face(mesh, cell_id, Cube_Face(i), refinement);
                }
                return result;
            }

            INL_CGPU array<uint64_t, 7> stencil() const {
                array<uint64_t, 7> result(7);
                result[0] = mesh->m_cell_base + cell_id;
                for (uint32_t i = 0; i < CUBE_FACE_COUNT; i++) {
                    result[1+i] = mesh->cell_neigh(cell_id, Cube_Face(i));
                }
                return result;
            };
        };

        Cell get_cell(uint64_t id) const {
            assert(id >= m_cell_base);
            return Cell(this, id - m_cell_base, refinement_mask(id));
        }

        Face get_face(uint64_t id) const {
            uint64_t cell_id = id / CUBE_FACE_COUNT;
            Cube_Face face = Cube_Face(id % CUBE_FACE_COUNT);
            assert(cell_id >= m_cell_base);
            return Face(this, cell_id - m_cell_base, face, refinement_mask(cell_id));
        }

        template<class Func>
        void for_each_cell(const char *name, Func func) const {
            Kokkos::parallel_for(name,
                                 Kokkos::RangePolicy<Exec>(0, cell_count()),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Cell cell = get_cell(i);
                                     func(cell);
                                 });
        }

        template<class Func>
        void for_each_face(const char *name, Func func) const {
            Kokkos::parallel_for(name,
                                 Kokkos::RangePolicy<Exec>(0, face_count()),
                                 [=] CGPU(uint64_t i) {
                                    Face face = get_face(i);
                                     func(face);
                                 });
        }

        void init(const Init_Desc &init) {
            m_extent = init.extent;

            luts.resize(init.max_levels);
            for (uint32_t i = 0; i < init.max_levels; i++) {
                Grid_Level_LUT& lut = luts[i];
                lut.dx = init.dx;
                lut.dx2 = lut.dx*lut.dx;
                lut.idx = vec3(1.0)/lut.dx;
                lut.idx2 = lut.idx*lut.idx;
                lut.cell_vol = lut.dx.x*lut.dx.y*lut.dx.z;
                lut.cell_ivol = 1.0_R/lut.cell_vol;

                for (uint32_t j = 0; j < CUBE_FACE_COUNT; j++) {
                    uint32_t axis = cube_face_axis[j];
                    lut.face_dx[j] = lut.dx[axis];
                    lut.face_idx[j] = 1.0/lut.face_dx[j];
                    lut.face_normal[j] = vec3(cube_normals[j]);
                    lut.face_a[j] = lut.cell_vol/lut.face_dx[j];
                    lut.face_sf[j] = lut.face_a[j] * lut.face_sf[j];
                }
            }

            ASSERT_MESG(init.capacity >= init.morton_codes.size(), "Cells exceed capacity");
            resize(init.capacity);

            auto &morton_codes = init.morton_codes;
            uint64_t morton_codes_size = morton_codes.size();
            uint64_t hash_buckets_count = this->hash_buckets_count();

            Kokkos::View<uint32_t *, Device> hash_position("HASH POSITION", morton_codes_size);

            Kokkos::parallel_for("HEXCORE HASH BUCKET COUNT",
                                 Kokkos::RangePolicy<Exec>(0, morton_codes_size),
                                 KOKKOS_LAMBDA(uint64_t i) {
                                     Morton_Code code = morton_codes(i);
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
                                     Morton_Code code = morton_codes(i);
                                     uint64_t loc = hash_bucket_start(code % hash_buckets_count) + hash_position(i);
                                     morton_keys(loc) = code;
                                 });
        }

        void update(const Update_Desc &desc) {

        }
    };
}
