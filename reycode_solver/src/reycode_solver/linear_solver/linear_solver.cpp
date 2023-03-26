//#include <../../../vendor/amgcl/

#include "linear_solver.h"
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
        using Backend = amgcl::backend::builtin<float>;
        using Solver = amgcl::make_solver<
                amgcl::amg<Backend,amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
                amgcl::solver::bicgstab<Backend>
        >;
    public:
        using Matrix = reycode::Matrix<real,uint64_t,Kokkos::HostSpace>;
        void solve(const Matrix& matrix, Kokkos::View<real*,Kokkos::HostSpace> x, Kokkos::View<real*,
                   Kokkos::HostSpace> b) override;
    };

    void AMGCL_Solver::solve(const Matrix& matrix, Kokkos::View<real*,Kokkos::HostSpace> x, Kokkos::View<real*,
                             Kokkos::HostSpace> b) {

        uint64_t n = matrix.n;

        auto range = [](auto& view) { return amgcl::make_iterator_range(view.data(), view.data() + view.size()); };

        typename Solver::params params;
        params.solver.tol = 1e-2;

        Solver solver(std::tuple(n, range(matrix.rowBegin), range(matrix.cols), range(matrix.coeffs)), params);

        solver.apply(range(b), range(x));
    }

    template<>
    std::unique_ptr<Linear_Solver<AMGCL_Solver::Matrix>> Linear_Solver<AMGCL_Solver::Matrix>::AMGCL_solver() {
        return std::make_unique<AMGCL_Solver>();
    }
}