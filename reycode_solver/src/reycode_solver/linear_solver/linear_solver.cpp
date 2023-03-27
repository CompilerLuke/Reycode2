//#include <../../../vendor/amgcl/

#include "linear_solver.h"
#include "amgcl/backend/vexcl.hpp"
#include "amgcl/backend/builtin.hpp"
#include "amgcl/adapter/crs_tuple.hpp"
#include "amgcl/make_solver.hpp"
#include "amgcl/amg.hpp"
#include "amgcl/coarsening/smoothed_aggregation.hpp"
#include "amgcl/relaxation/spai0.hpp"
#include "amgcl/solver/bicgstab.hpp"

namespace reycode {
    //todo: maybe uint32_t makes more sense
    class AMGCL_Solver : public Linear_Solver<Matrix<real,uint64_t,Kokkos::HostSpace>> {
        using Backend = amgcl::backend::vexcl<float>;
        using Solver = amgcl::make_solver<
                amgcl::amg<Backend,amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
                amgcl::solver::bicgstab<Backend>
        >;

        vex::Context ctx;
    public:
        using Matrix = reycode::Matrix<real,uint64_t,Kokkos::HostSpace>;
        AMGCL_Solver();
        void solve(const Matrix& matrix, Kokkos::View<real*,Kokkos::HostSpace> x, Kokkos::View<real*,
                   Kokkos::HostSpace> b) override;
    };

    AMGCL_Solver::AMGCL_Solver() : ctx(vex::Filter::GPU && vex::Filter::Position(1)) {
        if (!ctx) {
            throw std::string("GPU Backend not initialized");
        }

        std::cout << ctx << std::endl;
    }

    void AMGCL_Solver::solve(const Matrix& matrix, Kokkos::View<real*,Kokkos::HostSpace> x, Kokkos::View<real*,
                             Kokkos::HostSpace> b) {

        uint64_t n = matrix.n;

        auto range = [](auto& view) { return amgcl::make_iterator_range(view.data(), view.data() + view.size()); };

        typename Solver::params s_params;
        //s_params.solver.tol = 1e-3;

        Backend::params b_params;
        b_params.q = ctx;

        Kokkos::Timer timer;
        Solver solver(std::tuple(n, range(matrix.rowBegin), range(matrix.cols), range(matrix.coeffs)), s_params,
                      b_params);
        printf("Build AMG hierarchy : %f\n", timer.seconds());
        timer.reset();


        vex::vector<real> vec_b(ctx, n);
        vex::vector<real> vec_x(ctx, n);
        vex::copy(x.data(), x.data()+n, vec_x.begin());
        vex::copy(b.data(), b.data()+n, vec_b.begin());
        printf("Copy data to device : %f\n", timer.seconds());
        timer.reset();
        solver(vec_b, vec_x);

        printf("Solve on device : %f\n", timer.seconds());
        timer.reset();

        vex::copy(vec_x, x.data());
    }

    template<>
    std::unique_ptr<Linear_Solver<AMGCL_Solver::Matrix>> Linear_Solver<AMGCL_Solver::Matrix>::AMGCL_solver() {
        return std::make_unique<AMGCL_Solver>();
    }
}