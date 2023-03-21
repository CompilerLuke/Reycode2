#include <iostream>
#include "reycode/reycode.h"
#include "reycode/mesh/mesh.h"
#include "reycode/mesh/hexcore.h"
#include "reycode_solver/field/field.h"
#include "reycode_solver/fv/fvm.h"
#include "reycode_solver/fv/bc.h"
#include "reycode_solver/linear_solver/linear_solver.h"
#include "reycode_viewer/mesh_viewer.h"
#include "reycode_meshing/hexcore.h"
#include "reycode_graphics/rhi/rhi.h"
#include "reycode_graphics/rhi/window.h"
#include "reycode_viewer/fpv.h"
#include <glad/glad.h>
#include <Kokkos_Core.hpp>

using namespace reycode;

struct PressureBC {};
struct VelocityBC {};

const real lid_velocity = 100;

template<class Mesh, class Exec, class Mem>
class FluidSolver {
    Exec& exec;
    Mesh& mesh;
    Boundary_Condition<real,Mesh,Mem> pressure_bc;
    Boundary_Condition<vec3,Mesh,Mem> velocity_bc;
    Boundary_Condition<vec3,Mesh,Mem> zero_grad_bc;

    struct Scheme : fvc::scheme::Central_Difference, fvc::scheme::Upwind_Flux {};
public:
    Field<real,Mesh,Mem> pressure;
    Field<vec3,Mesh,Mem> velocity;

    Field<vec3,Mesh,Mem> debug_vec;
    Field<real,Mesh,Mem> debug_scalar;

    std::unique_ptr<Linear_Solver<Matrix<real,uint64_t,Mem>>> solver;

    FluidSolver(Exec& exec, Mesh& mesh) : exec(exec), mesh(mesh), pressure_bc(mesh), velocity_bc(mesh), zero_grad_bc(mesh) {
        uint64_t n = mesh.cell_count();
        pressure = Field<real,Mesh,Mem>("pressure", mesh, pressure_bc);
        velocity = Field<vec3,Mesh,Mem>("velocity", mesh, velocity_bc);
        static_assert(is_constant<vec3>);

        Mem mem;
        Scheme scheme;

        solver = Linear_Solver<Matrix<real,uint64_t,Mem>>::AMGCL_solver();

        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_X}, scheme, bc::grad(1.0_R)));
        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_X}, scheme, bc::grad(0.0_R)));
        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Y}, scheme, bc::value(0.0_R)));
        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Y}, scheme, bc::grad(0.0_R)));
        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Z}, scheme, bc::grad(0.0_R)));
        pressure_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Z}, scheme, bc::grad(0.0_R)));

        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_X}, scheme, bc::value(vec3())));
        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_X}, scheme, bc::value(vec3())));
        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Y}, scheme, bc::value(vec3(lid_velocity,0,0))));
        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Y}, scheme, bc::value(vec3())));
        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Z}, scheme, bc::value(vec3())));
        velocity_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Z}, scheme, bc::value(vec3())));

        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_X}, scheme, bc::grad(vec3())));
        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_X}, scheme, bc::grad(vec3())));
        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Y}, scheme, bc::grad(vec3())));
        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Z}, scheme, bc::grad(vec3())));
        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_NEG_Y}, scheme, bc::grad(vec3())));
        zero_grad_bc.push_back(bc::Boundary_Patch_Expr(mem, mesh, {CUBE_FACE_POS_Z}, scheme, bc::grad(vec3())));
    }

    void advance(real dt) {
        auto& p = pressure;
        auto& u = velocity;

        auto scheme = Scheme();

        auto momentum_eq = fvm::ddt(u,dt) + fvm::conv(u,u) - fvm::laplace(u);

        fvm::Pseudo_Matrix<vec3, Mem> pseudo;
        fvm::solve(exec, *solver, mesh, momentum_eq, -fvc::grad(p), u, scheme, pseudo);

        Field<vec3,Mesh,Mem> H(mesh, pseudo.H, zero_grad_bc);
        Field<vec3,Mesh,Mem> A(mesh, pseudo.A, zero_grad_bc);

        auto iA = 1.0_R/fvc::avg(A);
        auto pressure_eq = fvm::ddt(p,dt) + fvm::laplace(iA,p) == fvc::div(iA, H);
        fvm::solve(exec, *solver, mesh, pressure_eq, p, scheme);
    }
};

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);

    Arena arena = make_host_arena(mb(10)); //todo: make RAII
    DEFER(destroy_host_arena(arena));

    RHI rhi;

    using Mem = Kokkos::HostSpace;
    using Exec = Kokkos::Threads;
    using Mesh = Hexcore<Exec,Mem>;

    Exec exec;
    Mem mem;

    Mesh mesh = {};

    vec3 extent = vec3(1);
    uvec3 dims = uvec3(100,100,100);//00);

    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(extent, dims);

    FluidSolver<Mesh,Exec,Mem> solver(exec,mesh);

    Window_Desc desc = {};
    desc.width = 1024;
    desc.height = 1024;
    desc.title = "Reycode";
    desc.validation = false;

    Window window(desc);

    Colormap cm = Colormap::viridis();

    Mesh_Viewer<Exec,Mem,Hexcore<Exec,Mem>> viewer(rhi, mesh, cm);

    Kokkos::View<real*,Mem> field("field", mesh.cell_count());

    FPV fpv = {};
    fpv.view_pos = vec3(0.5,0.5,2);
    fpv.mouse_sensitivity = 100.0/desc.width;

    real old_t = Window::get_time();

    std::mutex mutex;
    std::atomic<bool> flag;
    /*std::thread thread([&]() {
        while (true) {

            flag.store(true);
            //break;
        };
    });
    thread.detach();*/

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

        bool expected = true;

        {
            solver.advance(1e-2);
            auto velocity = solver.velocity;
            auto pressure = solver.pressure;
            {
                std::lock_guard<std::mutex> lock(mutex);
                Kokkos::parallel_for("length", Kokkos::RangePolicy<Exec>(0, mesh.cell_count()), KOKKOS_LAMBDA(uint64_t i) {
                    field(i) = length(velocity(i));
                });
                flag = true;
            }
        }

        if (flag.compare_exchange_strong(expected, false)) {
            printf("================\n");
            std::lock_guard<std::mutex> lock(mutex);
            viewer.update(field, 0, lid_velocity);
        }
        viewer.render(scene);

        window.draw();
        old_t = t;
    }

    Kokkos::finalize();

    return 0;
}
