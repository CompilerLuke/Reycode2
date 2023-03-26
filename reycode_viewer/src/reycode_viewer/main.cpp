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

const real lid_velocity = 200;

namespace reycode::fvc {
    namespace expr {
        template<uint32_t axis>
        struct AxisTag {};

        template<class T, class LHS, uint32_t axis>
        using Axis = Unary_Expr<struct AxisTag<axis>, T, LHS>;

        template<class LHS>
        using Pressure_Poisson = Unary_Expr<struct PressurePoissonTag, real, LHS>;
    };

    template<class LHS>
    expr::Pressure_Poisson<LHS> pressure_poisson(expr::Expr<vec3, LHS>& lhs) {
        return { lhs };
    }

    template<uint32_t AXIS, class T, class LHS>
    expr::Axis<to_scalar<T>, LHS, AXIS> axis(const expr::Expr<T,LHS>& lhs) {
        return {lhs};
    }

    namespace eval {
        template<class T, class LHS, uint32_t AXIS, class Mesh, class Scheme>
        class Evaluator<expr::Axis<T, LHS, AXIS>, Mesh, Scheme> {
            Evaluator<LHS, Mesh, Scheme> lhs;
        public:
            Evaluator(const expr::Axis<T,LHS,AXIS>& expr) : lhs(expr.lhs) {}

            T eval(const typename Mesh::Cell& cell) const {
                vec3 result = lhs.eval(cell);
                return result[AXIS];
            }
        };

        template<class LHS, class Mesh, class Scheme>
        class Evaluator<expr::Pressure_Poisson<LHS>, Mesh, Scheme> {
            Evaluator<expr::Grad<vec3,expr::Axis<real, LHS, 0>>, Mesh, Scheme> x_grad;
            Evaluator<expr::Grad<vec3,expr::Axis<real, LHS, 1>>, Mesh, Scheme> y_grad;
            Evaluator<expr::Grad<vec3,expr::Axis<real, LHS, 2>>, Mesh, Scheme> z_grad;
        public:
            Evaluator(const expr::Pressure_Poisson<LHS>& expr) :
                    x_grad(grad(axis<0>(expr.lhs))),
                    y_grad(grad(axis<1>(expr.lhs))),
                    z_grad(grad(axis<2>(expr.lhs))) {}

            real eval(const typename Mesh::Cell& cell) const {
                vec3 x = x_grad.eval(cell);
                vec3 y = y_grad.eval(cell);
                vec3 z = z_grad.eval(cell);

                //printf("(%f,%f,%f) (%f,%f,%f) (%f,%f,%f)\n", x.x,x.y,x.z, y.x,y.y,y.z, z.x,z.y,z.z);

                return -(x.x * x.x + y.y * y.y + z.z * z.z + 2 * (x.y * y.x + z.y * y.z + z.x * x.z));
            }
        };
    }
};

/*
            vec3 u_c = cell_data3(sm, local, {});
            vec3 u_dx = idx / 2 * (cell_data3(sm, local, DIR_X) - cell_data3(sm, local, -DIR_X));
            vec3 u_dy = idx / 2 * (cell_data3(sm, local, DIR_Y) - cell_data3(sm, local, -DIR_Y));
            vec3 u_dz = idx / 2 * (cell_data3(sm, local, DIR_Z) - cell_data3(sm, local, -DIR_Z));

            real A = -2 * (idx2.x + idx2.y + idx2.z);
            real source = -density * (u_dx.x * u_dx.x + u_dy.y * u_dy.y + u_dz.z * u_dz.z + 2 * (u_dy.x * u_dx.y + u_dy.z * u_dz.y + u_dz.x * u_dx.z));
*/


template<class Mesh, class Exec, class Mem>
class FluidSolver {
    Exec& exec;
    Mesh& mesh;
    Boundary_Condition<real,Mesh,Mem> pressure_bc;
    Boundary_Condition<vec3,Mesh,Mem> velocity_bc;

    struct Scheme : fvc::scheme::Central_Difference, fvc::scheme::Upwind_Flux {};
public:
    Field<real,Mesh,Mem> pressure;
    Field<vec3,Mesh,Mem> velocity;

    Field<vec3,Mesh,Mem> debug_vec;
    Field<real,Mesh,Mem> debug_scalar;

    std::unique_ptr<Linear_Solver<Matrix<real,uint64_t,Mem>>> solver;

    FluidSolver(Exec& exec, Mesh& mesh) : exec(exec), mesh(mesh), pressure_bc(mesh), velocity_bc(mesh) {
        uint64_t n = mesh.cell_count();
        pressure = Field<real,Mesh,Mem>("pressure", mesh, pressure_bc);
        velocity = Field<vec3,Mesh,Mem>("velocity", mesh, velocity_bc);
        debug_scalar = Field<real,Mesh,Mem>("debug", mesh, pressure_bc);
        static_assert(is_constant<vec3>);

        Mem mem;
        Scheme scheme;

        solver = Linear_Solver<Matrix<real,uint64_t,Mem>>::AMGCL_solver();

        pressure_bc.push_back({CUBE_FACE_POS_X},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_X},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_POS_Y},scheme, bc::value(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_Y},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_POS_Z},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_Z},scheme, bc::grad(0.0_R));

        velocity_bc.push_back({CUBE_FACE_POS_X}, scheme, bc::value(vec3()));
        velocity_bc.push_back({CUBE_FACE_NEG_X}, scheme, bc::value(vec3()));
        velocity_bc.push_back({CUBE_FACE_POS_Y}, scheme, bc::value(vec3(lid_velocity,0,0)));
        velocity_bc.push_back({CUBE_FACE_NEG_Y}, scheme, bc::value(vec3()));
        velocity_bc.push_back({CUBE_FACE_POS_Z}, scheme, bc::value(vec3()));
        velocity_bc.push_back({CUBE_FACE_NEG_Z}, scheme, bc::value(vec3()));
    }

    void advance(real dt) {
        auto& p = pressure;
        auto& u = velocity;

        auto scheme = Scheme();

        auto momentum_eq = fvm::ddt(u,dt) + fvm::conv(u,u) - fvm::laplace(u) == -fvc::grad(p);
        fvm::solve(exec, *solver, mesh, momentum_eq, u, scheme);

        fvc::compute(exec,mesh,debug_scalar.data(), fvc::pressure_poisson(u), scheme);

        auto pressure_eq = fvm::laplace(p) == debug_scalar;
        fvm::solve(exec, *solver, mesh, pressure_eq, p, scheme);
    }
};


using Mem = Kokkos::HostSpace;
using Exec = Kokkos::Threads;
using Mesh = Hexcore<Exec,Mem>;
using Scheme = fvm::scheme::Central_Difference;

void test_grad() {
    Hexcore<Exec,Mem> mesh;
    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(vec3(1), uvec3(10));

    Mem mem;
    Scheme scheme;
    Exec exec;

    Boundary_Condition<real,Mesh,Mem> bc(mesh);
    bc.push_back({CUBE_FACE_POS_X}, scheme, bc::grad(1.0_R));
    bc.push_back({CUBE_FACE_NEG_X}, scheme, bc::grad(-1.0_R));
    bc.push_back({CUBE_FACE_POS_Y}, scheme, bc::grad(0.0_R));
    bc.push_back({CUBE_FACE_NEG_Y}, scheme, bc::grad(0.0_R));
    bc.push_back({CUBE_FACE_POS_Z}, scheme, bc::grad(0.0_R));
    bc.push_back({CUBE_FACE_NEG_Z}, scheme, bc::grad(0.0_R));

    Field<real,Mesh,Mem> x("x", mesh, bc);
    auto view = x.data();
    mesh.for_each_cell("init", KOKKOS_LAMBDA(Mesh::Cell& cell) {
       view(cell.id()) = cell.center().x;
    });
    x.update_bc();
    Kokkos::View<vec3*,Mem> result("res",mesh.cell_count());
    fvc::compute(exec,mesh,result,fvc::grad(x),scheme);

    mesh.for_each_cell("init", KOKKOS_LAMBDA(Mesh::Cell& cell) {
        vec3 grad = result(cell.id());
        assert(length(grad - vec3(1,0,0)) < 1e-4);
    });
}


void test_div() {
    Hexcore<Exec,Mem> mesh;
    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(vec3(1), uvec3(10));

    Mem mem;
    Scheme scheme;
    Exec exec;

    Boundary_Condition<vec3,Mesh,Mem> bc(mesh);
    bc.push_back({CUBE_FACE_POS_X}, scheme, bc::grad(vec3(1,0,0)));
    bc.push_back({CUBE_FACE_NEG_X}, scheme, bc::grad(vec3(-1,0,0)));
    bc.push_back({CUBE_FACE_POS_Y}, scheme, bc::grad(vec3(0,1,0)));
    bc.push_back({CUBE_FACE_NEG_Y}, scheme, bc::grad(vec3(0,-1,0)));
    bc.push_back({CUBE_FACE_POS_Z}, scheme, bc::grad(vec3(0,0,1)));
    bc.push_back({CUBE_FACE_NEG_Z}, scheme, bc::grad(vec3(0,0,-1)));

    Field<vec3,Mesh,Mem> x("x", mesh, bc);
    mesh.for_each_cell("init", KOKKOS_LAMBDA(Mesh::Cell& cell) {
        x.data()(cell.id()) = cell.center();
    });
    x.update_bc();
    Kokkos::View<real*,Mem> result("res",mesh.cell_count());
    fvc::compute(exec,mesh,result,fvc::div(x),scheme);

    mesh.for_each_cell("init", KOKKOS_LAMBDA(Mesh::Cell& cell) {
        real div = result(cell.id());
        assert(fabs(div - 3) < 1e-4);
    });
}

void test_laplace() {
    Hexcore<Exec,Mem> mesh;
    HexcoreAMR<Exec,Mem> amr(mesh);

    uint32_t n = 100;
    amr.uniform(vec3(2,2,2.0/n), uvec3(n,n,1));

    Mem mem;
    Scheme scheme;
    Exec exec;

    Boundary_Condition<vec3,Mesh,Mem> bc(mesh);
    bc.push_back({CUBE_FACE_POS_X}, scheme, bc::value(vec3(0.0_R)));
    bc.push_back({CUBE_FACE_NEG_X}, scheme, bc::value(vec3(0.0_R)));
    bc.push_back({CUBE_FACE_POS_Y}, scheme, bc::value(vec3(0.0_R)));
    bc.push_back({CUBE_FACE_NEG_Y}, scheme, bc::value(vec3(0.0_R)));
    bc.push_back({CUBE_FACE_POS_Z}, scheme, bc::grad(vec3(0.0_R)));
    bc.push_back({CUBE_FACE_NEG_Z}, scheme, bc::grad(vec3(0.0_R)));

    Field<vec3,Mesh,Mem> x("x", mesh, bc);

    auto solver = Linear_Solver<Matrix<real,uint64_t,Mem>>::AMGCL_solver();

    vec3 scale = vec3(1,2,3);

    fvm::solve(exec,*solver,mesh,fvm::laplace(x)==-5*scale,x,scheme);

    for (uint32_t axis = 0; axis < 3; axis++) {
        Kokkos::View<real*> view("", mesh.cell_count());
        mesh.for_each_cell("", KOKKOS_LAMBDA(Mesh::Cell& cell) {
           view(cell.id()) = x(cell.id())[axis];
        });

        auto peak = Kokkos::Experimental::max_element(exec, view);
        assert(fabs(*peak - 1.47 * scale[axis]) < 1e-1);
    }
}


int main(int argc, char** argv) {
    Kokkos::InitializationSettings args;
    args.set_num_threads(8);

    Kokkos::initialize(args);
    DEFER(Kokkos::finalize());

    bool TEST = false;
    if (TEST) {
        test_grad();
        test_div();
        test_laplace();
        return 0;
    }

    RHI rhi;

    Exec exec;
    Mem mem;

    Mesh mesh = {};

    uint32_t n = 50;
    vec3 extent = vec3(1,1,1);
    uvec3 dims = uvec3(n,n,n);

    Kokkos::Timer timer;
    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(extent, dims);
    //amr.adapt(0);
    printf("Build mesh - %f ms", timer.seconds()*1e3);
    timer.reset();

    FluidSolver<Mesh,Exec,Mem> solver(exec,mesh);

    Window_Desc desc = {};
    desc.width = 1024;
    desc.height = 1024;
    desc.title = "Reycode";
    desc.validation = false;
    desc.vsync = true;

    Window window(desc);

    Colormap cm = Colormap::viridis();

    Mesh_Viewer<Exec,Mem,Hexcore<Exec,Mem>> viewer(rhi, mesh, cm);

    Kokkos::View<real*,Mem> field("field", mesh.cell_count());

    FPV fpv = {};
    fpv.view_pos = vec3(0.5,0.5,3);
    fpv.mouse_sensitivity = 100.0/desc.width;

    real old_t = Window::get_time();

    viewer.update(field, 0, 1);

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

        if (true) {
            solver.advance(1e-3);
            auto velocity = solver.velocity;
            auto pressure = solver.debug_scalar; //solver.pressure;//solver.debug_scalar; //solver.pressure;
            Kokkos::parallel_for("length", Kokkos::RangePolicy<Exec>(0, mesh.cell_count()), KOKKOS_LAMBDA(uint64_t i) {
                field(i) = length(velocity(i));
            });
        }

        viewer.update(field,0, lid_velocity);
        viewer.render(scene);

        window.draw();
        old_t = t;
    }

    return 0;
}

