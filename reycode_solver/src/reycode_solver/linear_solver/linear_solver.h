#pragma once

#include "reycode/reycode.h"
#include "Kokkos_Core.hpp"
#include <memory>

namespace reycode {
    //todo: does this belong here?
    template<class T, class Mesh>
    struct Stencil_Matrix {
        T data[Mesh::MAX_COEFFS] = {};
        uint32_t size() { return Mesh::MAX_COEFFS; }
        T &operator[](uint32_t i) { return data[i]; }
    };

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
        void resize(Exec& exec, uint32_t n, Func func) {
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

        template<class Exec, class Scalar = to_scalar<Coeff>>
        void split_diagonal(Exec& exec,
               Kokkos::View<Coeff*,Mem> out_diagonal,
               Kokkos::View<Coeff*,Mem> out_off_diagonal,
               Kokkos::View<Coeff*,Mem> source,
               const Kokkos::View<Coeff*,Mem>& values) {
            Kokkos::parallel_for("Diagonal splitting",
                                 Kokkos::RangePolicy<Exec>(0,n),
                                 KOKKOS_LAMBDA(uint64_t row) {
                Index start = rowBegin[row];
                Index end = rowBegin[row+1];
                Coeff diagonal = Coeff();
                Coeff off_diagonal = Coeff();

                for (uint32_t j = start; j < end; j++) {
                    Coeff coeff = coeffs(j);
                    Index index = cols(j);
                    if (index != row) off_diagonal += coeff * values(index);
                    else diagonal += coeff;
                }

                out_diagonal(row) = diagonal;
                out_off_diagonal(row) = source(row) - off_diagonal;
            });
        }

        void check() {
            return;
            for (int row = 0; row < n; row++) {
                int row_start = rowBegin[row];
                int row_end = rowBegin[row+1];
                assert(cols[row_start] == row);
                printf("[%i] ", row);
                for (int j = row_start; j < row_end; j++) {
                    printf("(%u) = %f, ", cols[j], coeffs[j]);
                }
                printf("\n");
            }
        }
    };

    template<class Matrix>
    class Linear_Solver {
    public:
        using Coeff = typename Matrix::Coeff;
        using Mem = typename Matrix::Mem;
        virtual void solve(const Matrix& matrix, Kokkos::View<Coeff*,Mem> x, Kokkos::View<Coeff*,Mem> b) = 0;
        virtual ~Linear_Solver() {}

        static std::unique_ptr<Linear_Solver<Matrix>> AMGCL_solver();
    };
}