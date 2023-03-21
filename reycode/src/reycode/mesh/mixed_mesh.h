#include <tuple>

namespace reycode {
    template<class... Mesh>
    class MixedMesh {
        std::tuple<Mesh...> meshes;

    public:
        template<class T>
        const T& get() const { return std::get<T>(meshes); }

        template<class T>
        T& get() { return std::get<T>(meshes); }
    };
}