#pragma once

#include "reycode/reycode.h"
#include "reycode_solver/linear_solver.h"

namespace reycode {
    namespace fvm {
        template<class T, class Mesh>
        struct Stencil_Matrix {
            T data[Mesh::MAX_COEFFS] = {};

            T &operator[](uint32_t i) { return data[i]; }
        };

        namespace expr {
            template<class T, class Expr>
            struct Div {
                using Elem = T;
                Expr lhs;
            };

            template<class T, class Expr>
            struct Laplace {
                using Elem = T;
                Expr lhs;
            };
        };

        template<class T>
        expr::Laplace<T, T> laplace(T factor) { return {factor}; }

        namespace scheme {
            struct Central_Difference {
            };
            struct Upwind_Flux {
            };
        };

        template<class Base, class Derived>
        using with_policy = std::enable_if_t<std::is_same_v<Base, Derived>>;

        namespace evaluator {
            template<class Expr, class Mesh, class Scheme, typename MATCHES = void>
            class Evaluator {
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Div<T, LHS>, Mesh, Scheme, with_policy<Scheme, scheme::Upwind_Flux>> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Div<T, LHS> expr) : lhs(expr.lhs) {}

                void face_flux(Stencil_Matrix<T, Mesh> stencil, typename Mesh::Face &face) const {
                    stencil[face.neigh_stencil()] += 0.5_R * face.sf() * face.ivol();
                    stencil[face.cell_stencil()] += 0.5_R * face.sf() * face.ivol();
                    return *this;
                }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Laplace<T, LHS>, Mesh, Scheme, with_policy<Scheme, scheme::Central_Difference>> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Laplace<T, LHS> expr) : lhs(expr.lhs) {}

                void face_flux(Stencil_Matrix<T, Mesh> &stencil, typename Mesh::Face &face) const {
                    stencil[face.neigh_stencil()] += face.idx() * face.fa() * face.ivol();
                    stencil[face.cell_stencil()] += -face.idx() * face.fa() * face.ivol();
                }
            };

            template<class T, class Mesh, class Scheme>
            class Evaluator<T, Mesh, Scheme, std::enable_if_t<std::is_arithmetic_v<T>>> {
                T value;
            public:
                Evaluator(T value) : value(value) {}
            };
        }

        template<class Exec, class Matrix, class Expr, class Mesh, class Scheme>
        void build_matrix(Exec& exec, const Mesh &mesh, Matrix& matrix, const Expr &expr, const Scheme& scheme) {
            using Elem = typename Expr::Elem;

            fvm::evaluator::Evaluator<Expr, Mesh, Scheme> eval(expr);

            matrix.template resize<Exec>(mesh.cell_count(), KOKKOS_LAMBDA(uint32_t i) {
                return mesh.get_cell(i).stencil().size();
            });

            mesh.for_each_cell("Assemble Matrix", KOKKOS_LAMBDA(const typename Mesh::Cell& cell) {
                fvm::Stencil_Matrix<Elem, Mesh> coeffs;

                for (auto face: cell.faces()) eval.face_flux(coeffs, face);

                auto stencil = cell.stencil();

                assert(stencil[0] == cell.id());
                assert(coeffs[0] != 0.0_R);

                uint64_t offset = matrix.rowBegin(cell.id());
                for (uint32_t i = 0; i < stencil.size(); i++) {
                    bool valid = stencil[i] != UINT64_MAX;
                    matrix.coeffs[offset] = valid ? coeffs[i] : 0.0;
                    matrix.cols[offset] = valid ? stencil[i] : 0;
                    offset++;
                }
            });
        }

        template<class Exec, class Elem, class Mem, class Expr, class Mesh, class Scheme>
        void build_source(Exec& exec,
                          const Mesh& mesh,
                          Kokkos::View<Elem,Mem>& source,
                          const Expr& expr,
                          const Scheme& scheme) {
            Kokkos::parallel_for("fill b", Kokkos::RangePolicy<Exec>(0,mesh.cell_count()), KOKKOS_LAMBDA(uint32_t i) {
                source(i) = -15.0;
            });
        }

        template<class Exec, class Matrix, class Expr, class Mesh, class Scheme>
        void solve(
                   Exec& exec,
                   Linear_Solver<Matrix>& solver,
                   const Mesh &mesh,
                   const Expr &expr,
                   const Kokkos::View<typename Expr::Elem*, typename Matrix::Mem>& x,
                   const Scheme& scheme
       ) {
            using Elem = typename Matrix::Coeff;
            using Mem = typename Matrix::Mem;

            Matrix matrix;
            Kokkos::View<float*,Kokkos::HostSpace> source("source", mesh.cell_count());

            printf("Building matrix : mesh %i\n", mesh.cell_count());
            if (mesh.cell_count() == 0) {
                fprintf(stderr, "MESH IS empty");
                return;
            }

            Kokkos::Timer timer;
            build_matrix(exec, mesh, matrix, expr, scheme);
            printf("Build matrix took : %f s\n", timer.seconds());
            timer.reset();
            build_source(exec, mesh, source, expr, scheme);
            printf("Build source took : %f s\n", timer.seconds());
            timer.reset();
            solver.solve(matrix, x, source);
            printf("Solve linear system took : %f s\n", timer.seconds());
        }
    }
}