#pragma once

#include "reycode/reycode.h"
#include "reycode/mesh/hexcore.h"
#include <Kokkos_Core.hpp>

namespace reycode {
    template<class Exec, class Mem>
    class HexcoreAMR {
        Hexcore<Exec, Mem> &hexcore;

    public:
        explicit HexcoreAMR(Hexcore<Exec, Mem> &hexcore) : hexcore(hexcore) {}

        HexcoreAMR &uniform(vec3 extent, uvec3 dims) {
            uint32_t cell_count = dims.x * dims.y * dims.z;
            uint32_t ghost_count = 2*(dims.x*dims.y + dims.x*dims.z + dims.y*dims.z);

            using Hexcore = reycode::Hexcore<Exec, Mem>;

            typename Hexcore::Init_Desc init_desc;
            init_desc.capacity = cell_count + ghost_count;
            init_desc.extent = extent;
            init_desc.dx = extent / vec3(dims);
            init_desc.morton_codes = Kokkos::View<typename Hexcore::Morton_Code *, typename Hexcore::Device>("UNIFORM MORTON",
                                                                                                             cell_count + ghost_count);
            init_desc.refinement_mask = Kokkos::View<typename Hexcore::Refinement_Mask *, typename Hexcore::Device>
            ("UNIFORM MASK",
                                                                                                             cell_count + ghost_count);
            uvec3 ghost = uvec3(1);

            Kokkos::parallel_for("Uniform morton ",
                                 Kokkos::RangePolicy<Exec>(0, cell_count),
                                 KOKKOS_LAMBDA(uint64_t idx) {

                                     uvec3 pos;
                                     pos.x = (idx % dims.x);
                                     pos.y = (idx / dims.x) % dims.y;
                                     pos.z = (idx / dims.x / dims.y);
                                     pos = pos + ghost;

                                     typename Hexcore::Morton_Code code = Hexcore::morton_encode(pos);
                                     typename Hexcore::Refinement_Mask refinement = {};
                                     refinement.ghost = false;

                                     init_desc.morton_codes(idx) = code;
                                     init_desc.refinement_mask(idx) = refinement;
                                 });

            init_desc.patches.resize(CUBE_FACE_COUNT);

            uint64_t offset = cell_count;

            for (uint32_t face = 0; face < CUBE_FACE_COUNT; face++) {
                uint32_t axis = cube_face_axis[face];
                uint32_t axis_u = (axis + 1) % 3;
                uint32_t axis_v = (axis + 2) % 3;
                uint64_t n = dims[axis_u] * dims[axis_v];
                uvec3 corner = {}; uvec3 u = {}; uvec3 v = {};
                u[axis_u] = 1;
                v[axis_v] = 1;
                corner[axis] = cube_normals[face][axis] == 1 ? dims[axis]-1 : 0;
                corner = uvec3(ghost + corner + cube_normals[face]);

                typename Hexcore::Init_Desc::Boundary_Patch& patch = init_desc.patches[face];
                patch.morton_codes = Kokkos::View<typename Hexcore::Morton_Code*, Mem>("patch morton code", n);
                patch.faces = Kokkos::View<Cube_Face*, Mem>("patch faces", n);


                Kokkos::parallel_for("Cube face patch",
                                     Kokkos::RangePolicy<Exec>(0,n),
                                     KOKKOS_LAMBDA(uint64_t idx) {
                     uvec3 pos = corner + uint32_t(idx % dims.x) * u + uint32_t(idx / dims.y) * v;
                     typename Hexcore::Morton_Code code = Hexcore::morton_encode(pos);

                     patch.morton_codes(idx) = code;
                     patch.faces(idx) = Cube_Face(cube_opposite_faces[face]);

                     typename Hexcore::Refinement_Mask refinement = {};
                     refinement.ghost = true;

                     //assert(offset + idx < cell_count + ghost_count);
                     init_desc.morton_codes(offset + idx) = code;
                     init_desc.refinement_mask(offset + idx) = refinement;
                });

                offset += n;
            };
            assert(offset == cell_count + ghost_count);

            hexcore.init(init_desc);

            return *this;
        }
    };

}