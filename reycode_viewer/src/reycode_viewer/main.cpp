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
    using Mesh = Hexcore<Exec,Mem>;

    Exec exec;

    Mesh mesh = {};

    vec3 extent = vec3(1);
    uvec3 dims = uvec3(100);

    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(extent, dims);

    Window_Desc desc = {};
    desc.width = 1024;
    desc.height = 1024;
    desc.title = "Reycode";
    desc.validation = false;


    Window window(desc);

    Colormap cm = Colormap::viridis();

    Mesh_Viewer<Exec,Mem,Hexcore<Exec,Mem>> viewer(rhi, mesh, cm);

    FPV fpv = {};
    fpv.view_pos.z = 2;

    using Scheme = fvm::scheme::Central_Difference;

    Kokkos::View<real*,Mem> x("x", mesh.cell_count());

    auto solver = Linear_Solver<Matrix<real,uint32_t,Mem>>::AMGCL_solver();

    auto expr = fvm::laplace(1.0_R);
    fvm::solve(exec, *solver, mesh, expr, x, fvm::scheme::Central_Difference());

    viewer.update(x, 0, 1);

    real old_t = Window::get_time();

    while (window.is_open()) {
        real t = Window::get_time();
        real dt = t - old_t;

        Input_State input = window.poll();

        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        fpv_update(fpv, input, dt);

        Scene scene = {};
        scene.mvp = fpv_proj_mat(fpv, {desc.width,desc.height}) * fpv_view_mat(fpv);
        scene.dir_light = vec3(-1,-1,-1);

        viewer.render(scene);

        window.draw();
        old_t = t;
    }

    Kokkos::finalize();

    return 0;
}
