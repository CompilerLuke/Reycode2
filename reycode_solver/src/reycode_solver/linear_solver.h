#pragma once

#include "reycode/reycode.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace reycode {
    enum class MatrixFormat {
        CSR,
    };

    template<class C, class I, class M, MatrixFormat Format = MatrixFormat::CSR>
    struct Matrix {
        using Coeff = C;
        using Index = I;
        using Mem = M;

        uint64_t n;
        Kokkos::View<Coeff*, Mem> coeffs;
        Kokkos::View<Index*, Mem> rowBegin;
        Kokkos::View<Index*, Mem> cols;

        template<class Exec, class Func>
        void resize(uint32_t n, Func func) {
            this->n = n;
            rowBegin = Kokkos::View<Index*, Mem>("Row begin ", n+1);

            uint64_t coeff_count = 0;
            Kokkos::parallel_scan("Resize matrix",
                                  Kokkos::RangePolicy<Exec>(0,n),
                                  KOKKOS_LAMBDA(uint64_t i, uint64_t& partial_sum, bool is_final) {
                if (is_final) rowBegin(i) = partial_sum;
                partial_sum += func(i);
            }, coeff_count);
            rowBegin(n) = coeff_count;

            cols = Kokkos::View<Index*, Mem>("Cols", coeff_count);
            coeffs = Kokkos::View<Coeff*, Mem>("Coeffs", coeff_count);
        }


    };

    template<class Matrix>
    class Linear_Solver {
    public:
        using Coeff = typename Matrix::Coeff;
        using Mem = typename Matrix::Mem;
        virtual void solve(const Matrix& matrix, Kokkos::View<Coeff*,Mem> x, Kokkos::View<real*,Kokkos::HostSpace> b) = 0;
        virtual ~Linear_Solver() {}

        static std::unique_ptr<Linear_Solver<Matrix>> AMGCL_solver();
    };
}