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

template<class Mesh, class Exec, class Mem>
class FluidSolver {
    Exec& exec;
    Mesh& mesh;
    Boundary_Condition<real,Mesh,Mem> pressure_bc;
    Boundary_Condition<vec3,Mesh,Mem> velocity_bc;
    Boundary_Condition<vec3,Mesh,Mem> zero_bc;

    struct Scheme : fvc::scheme::Central_Difference, fvc::scheme::Upwind_Flux {};
public:
    Field<real,Mesh,Mem> pressure;
    Field<vec3,Mesh,Mem> velocity;
    Field<vec3,Mesh,Mem> velocity_flux;
    Field<vec3,Mesh,Mem> diagonal;

    Field<vec3,Mesh,Mem> debug_vec;
    Field<real,Mesh,Mem> debug_scalar;


    std::unique_ptr<Linear_Solver<Matrix<real,uint64_t,Mem>>> solver;

    FluidSolver(Exec& exec, Mesh& mesh) : exec(exec), mesh(mesh), pressure_bc(mesh), velocity_bc(mesh), zero_bc(mesh) {
        uint64_t n = mesh.cell_count();
        pressure = Field<real,Mesh,Mem>("pressure", mesh, pressure_bc);
        velocity = Field<vec3,Mesh,Mem>("velocity", mesh, velocity_bc);
        velocity_flux = Field<vec3,Mesh,Mem>("velocity flux", mesh, velocity_bc);
        diagonal = Field<vec3,Mesh,Mem>("diagonal", mesh, zero_bc);
        debug_scalar = Field<real,Mesh,Mem>("debug", mesh, pressure_bc);
        static_assert(is_constant<vec3>);

        mesh.for_each_cell("init", KOKKOS_LAMBDA(typename Mesh::Cell& cell) {
            velocity.view(cell.id()) = vec3(lid_velocity,0,0);
            velocity_flux.view(cell.id()) = vec3(lid_velocity,0,0);
        });
        velocity_flux.update_bc();
        velocity.update_bc();

        Mem mem;
        Scheme scheme;

        solver = Linear_Solver<Matrix<real,uint64_t,Mem>>::AMGCL_solver();

        pressure_bc.push_back({CUBE_FACE_POS_X},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_X},scheme, bc::value(0.0_R));
        pressure_bc.push_back({CUBE_FACE_POS_Y},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_Y},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_POS_Z},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_NEG_Z},scheme, bc::grad(0.0_R));
        pressure_bc.push_back({CUBE_FACE_COUNT},scheme, bc::grad(0.0_R));

        velocity_bc.push_back({CUBE_FACE_POS_X}, scheme, bc::grad(vec3()));
        velocity_bc.push_back({CUBE_FACE_NEG_X}, scheme, bc::value(vec3(lid_velocity,0,0)));
        velocity_bc.push_back({CUBE_FACE_POS_Y}, scheme, bc::grad(vec3()));
        velocity_bc.push_back({CUBE_FACE_NEG_Y}, scheme, bc::grad(vec3()));
        velocity_bc.push_back({CUBE_FACE_POS_Z}, scheme, bc::grad(vec3()));
        velocity_bc.push_back({CUBE_FACE_NEG_Z}, scheme, bc::grad(vec3()));
        velocity_bc.push_back({CUBE_FACE_COUNT}, scheme, bc::value(vec3()));

        zero_bc.push_back({CUBE_FACE_POS_X},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_NEG_X},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_POS_Y},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_NEG_Y},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_POS_Z},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_NEG_Z},scheme, bc::grad(vec3(0.0_R)));
        zero_bc.push_back({CUBE_FACE_COUNT},scheme, bc::grad(vec3(0.0_R)));
    }

    void advance() {
        auto& p = pressure;
        auto& u = velocity;
        auto& u_f = velocity_flux;
        auto& A = diagonal;

        auto scheme = Scheme();

        real mu = 0.1;

        fvm::Pseudo<vec3,Mesh,Mem> pseudo{A};

        auto momentum_eq = fvm::ddt(u,1e-4) + fvm::conv(u_f,u) - vec3(mu)*fvm::laplace(u) == -fvc::grad(p);
        fvm::solve(exec, *solver, mesh, momentum_eq, u, scheme, pseudo);

        auto inv_A = 1.0_R / fvc::avg(A);
        auto pressure_eq = fvm::laplace(p) == fvc::pressure_poisson(u);//fvc::div(u);
        fvm::solve(exec, *solver, mesh, pressure_eq, p, scheme);

        fvc::compute(exec,mesh,u - vec3(1)/A*fvc::grad(p),u_f.data(),scheme);
        u_f.update_bc();
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
    //fvc::compute(exec,mesh,result,fvc::grad(x),scheme);

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
    //fvc::compute(exec,mesh,result,fvc::div(x),scheme);

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

    //fvm::Pseudo<real,Mesh,Mem> pseudo{};
    //fvm::solve(exec,*solver,mesh,fvm::laplace(x)==-5*scale,x,scheme,pseudo);

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
    vec3 extent = vec3(5,5,5);
    uvec3 dims = uvec3(n,n,n);

    Kokkos::Timer timer;
    HexcoreAMR<Exec,Mem> amr(mesh);
    amr.uniform(extent, dims);
    amr.adapt(Mesh::MAX_REFINEMENT);
    printf("Buid mesh - %f ms", timer.seconds()*1e3);
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
    fpv.view_pos = vec3(2.5,2.5,10);
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

        bool solve = true;
        if (solve) {
            solver.advance();
            auto velocity = solver.velocity;
            auto pressure = solver.debug_scalar; //solver.pressure;//solver.debug_scalar; //solver.pressure;
            Kokkos::parallel_for("length", Kokkos::RangePolicy<Exec>(0, mesh.cell_count()), KOKKOS_LAMBDA(uint64_t i) {
                field(i) = length(velocity(i));
            });
        }

        viewer.update(field,0, 1.5*lid_velocity);
        viewer.render(scene);

        window.draw();
        old_t = t;
    }

    return 0;
}

