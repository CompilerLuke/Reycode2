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

            using Hexcore = reycode::Hexcore<Exec, Mem>;

            static_assert(std::is_same_v<Exec, Kokkos::Serial>);
            static_assert(std::is_same_v<Mem, Kokkos::HostSpace>);

            typename Hexcore::Init_Desc init_desc;
            init_desc.capacity = cell_count;
            init_desc.extent = extent;
            init_desc.dx = extent / vec3(dims);
            init_desc.morton_codes = Kokkos::View<typename Hexcore::Morton_Code *, typename Hexcore::Device>("UNIFORM "
                                                                                                             "MORTON",
                                                                                                             cell_count);

            Kokkos::parallel_for("Uniform morton ",
                                 Kokkos::RangePolicy<Exec>(0, cell_count),
                                 KOKKOS_LAMBDA(uint64_t idx) {
                                     uvec3 pos;
                                     pos.x = idx % dims.x;
                                     pos.y = (idx / dims.x) % dims.y;
                                     pos.z = idx / dims.x / dims.y;

                                     typename Hexcore::Morton_Code code = Hexcore::morton_encode(pos);

                                     init_desc.morton_codes(idx) = code;
                                 });

            hexcore.init(init_desc);

            return *this;
        }
    };

}