#include <iostream>
#include "reycode/reycode.h"
#include "reycode/mesh/mesh.h"
#include "reycode/mesh/hexcore.h"
#include "reycode_solver/fv/fvm.h"
#include "reycode_solver/linear_solver.h"
#include "reycode_viewer/mesh_viewer.h"
#include "reycode_meshing/hexcore.h"
#include "reycode_graphics/rhi/rhi.h"
#include "reycode_graphics/rhi/window.h"
#include "reycode_viewer/fpv.h"
#include <glad/glad.h>
#include <Kokkos_Core.hpp>

using namespace reycode;

int main() {
    Kokkos::initialize();

    Arena arena = make_host_arena(mb(10)); //todo: make RAII
    DEFER(destroy_host_arena(arena));

    RHI rhi;

    using Mem = Kokkos::HostSpace;
    using Exec = Kokkos::Serial;
    using Mesh = MixedMesh<Hexcore<Exec,Mem>>;

    Mesh mesh = {};
    auto& hexcore = mesh.get<Hexcore<Exec,Mem>>();

    vec3 extent = vec3(1);
    uvec3 dims = uvec3(100);

    HexcoreAMR<Exec,Mem> hexcore_amr(hexcore);
    hexcore_amr.uniform(extent, dims);

    Window_Desc desc = {};
    desc.width = 1024;
    desc.height = 1024;
    desc.title = "Reycode";
    desc.validation = false;

    Window window(desc);

    Colormap cm = Colormap::viridis();

    Mesh_Viewer<Exec,Mem,Hexcore<Exec,Mem>> viewer(rhi, hexcore, cm);

    FPV fpv = {};
    fpv.view_pos.z = 2;

    constexpr uint32_t STENCIL_SIZE = 7;

#if 0
    Matrix<real,uint32_t,Mem> A;
    A.resize<Exec>(hexcore.cell_count(), KOKKOS_LAMBDA (uint32_t i) { return STENCIL_SIZE; });

    Kokkos::Timer timer;
    Kokkos::parallel_for("Build Matrix",
                         Kokkos::RangePolicy<Exec>(0,hexcore.cell_count()),
                         KOKKOS_LAMBDA (uint32_t i) {
            fvm::FVM<real, Hexcore<Exec,Mem>, STENCIL_SIZE> fv;

            auto cell = hexcore.get_cell(i);
            for (auto face : cell.faces()) {
                fv.laplace(face, 1.0);
            }

            auto stencil = cell.stencil();

            assert(stencil[0] == cell.id());
            assert(fv.stencil[0] != 0.0_R);

            uint64_t offset = A.rowBegin(i);
            for (uint32_t i = 0; i < STENCIL_SIZE; i++) {
                bool valid = stencil[i] != UINT64_MAX;
                A.coeffs[offset] = valid ? fv.stencil[i] : 0.0;
                A.cols[offset] = valid ? stencil[i] : 0;
                offset++;
            }
    });
    std::cout << "Matrix construction took : " << timer.seconds() << std::endl;

    Kokkos::View<real*,Mem> x("x", hexcore.cell_count());
    Kokkos::View<real*,Mem> b("b", hexcore.cell_count());

    Kokkos::parallel_for("fill b", Kokkos::RangePolicy<Exec>(0,hexcore.cell_count()), KOKKOS_LAMBDA(uint32_t i) {
        b(i) = -15.0;
    });

    timer.reset();

    auto solver = Linear_Solver<decltype(A)>::AMGCL_solver();

    std::cout << "Solving system of size " << A.n << std::endl;
    solver->solve(A,x,b);
    std::cout << "Solving system of size " << A.n << ", took " << timer.seconds() << std::endl;

    viewer.update(x, 0, 1);
#endif
    viewer.cube();

    while (window.is_open()) {
        Input_State input = window.poll();

        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        real dt = 1/60.0;
        fpv_update(fpv, input, dt);

        Scene scene = {};
        scene.mvp = fpv_proj_mat(fpv, {desc.width,desc.height}) * fpv_view_mat(fpv);
        scene.dir_light = vec3(-1,-1,-1);

        viewer.render(scene);

        window.draw();
    }

    Kokkos::finalize();

    return 0;
}
