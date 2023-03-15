#include <Kokkos_Core.hpp>

namespace reycode {
    template<class Space>
    class Polymesh {
        Kokkos::View<uint32_t*[3], Space> face_normals;
        Kokkos::View<uint32_t, Space> face_area;
        Kokkos::View<uint32_t, Space> face_neigh;

        uint64_t cell_base;
    };
}