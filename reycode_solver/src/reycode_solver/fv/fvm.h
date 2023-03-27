#pragma once

#include "reycode/reycode.h"
#include "reycode_solver/linear_solver/linear_solver.h"
#include "fvc.h"

namespace reycode {
    namespace fvm {
        namespace scheme {
            struct Central_Difference {};
            struct Upwind_Flux {};
        };

        namespace expr {
            template<class T, class Super>
            struct Expr {
                using Elem = T;
                operator const Super& () const { return *static_cast<const Super*>(this); }
                operator Super& () { return *static_cast<Super*>(this); }
            };

            template<class Tag, class T, class LHS>
            struct Unary_Expr : Expr<T, Unary_Expr<Tag,T,LHS>> {
                using Elem = T;

                LHS lhs;
                Unary_Expr(const LHS& lhs) : lhs(lhs) {}
            };

            template<class Tag, class T, class LHS, class RHS>
            struct Binary_Expr : Expr<T, Binary_Expr<Tag,T,LHS,RHS>> {
                using Elem = T;
                LHS lhs;
                RHS rhs;
                Binary_Expr(const LHS& lhs, const RHS& rhs) : lhs(lhs), rhs(rhs) {}
            };

            template<class T, class LHS>
            using Divergence = Unary_Expr<struct DivergenceTag, T,LHS>;

            template<class T, class LHS>
            using Laplace = Unary_Expr<struct LaplaceTag,T,LHS>;

            template<class T, class LHS>
            using Conv = Unary_Expr<struct ConvTag, T, LHS>;

            template<class T, class LHS>
            using DDt = Binary_Expr<struct DDtTag, T, LHS, real>;

            template<class T, class LHS, class RHS>
            using Add = Binary_Expr<struct AddTag,T,LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Sub = Binary_Expr<struct SubTag,T,LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Mul = Binary_Expr<struct MulTag,T,LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Eq = Binary_Expr<struct EqTag,T,LHS, RHS>;

            template<uint32_t axis>
            struct AxisTag {};

            template<class T, class LHS, uint32_t axis>
            using Axis = Unary_Expr<struct AxisTag<axis>, T, LHS>;

            template<class T, class LHS, class RHS>
            Add<T,LHS,RHS> operator+(const Expr<T,LHS>& lhs, const Expr<T,RHS>& rhs) { return {lhs,rhs}; }

            template<class T, class LHS, class RHS>
            Sub<T,LHS,RHS> operator-(const Expr<T,LHS>& lhs, const Expr<T,RHS>& rhs) { return {lhs,rhs}; }

            template<class T, class RHS>
            Mul<T,T,RHS> operator*(T lhs, const Expr<T,RHS>& rhs) { return {lhs,rhs}; }

            template<class T, class LHS, class RHS>
            Eq<T,LHS,RHS> operator==(const Expr<T,LHS>& lhs, const fvc::expr::Expr<T,RHS>& rhs) { return {lhs, rhs}; }

            template<class T, class LHS>
            Eq<T,LHS,T> operator==(const Expr<T,LHS>& lhs, const T& rhs) { return {lhs,rhs}; }
        };

        template<class T, class Mesh, class Mem>
        expr::DDt<T, Field<T,Mesh,Mem>> ddt(Field<T,Mesh,Mem>& field, real dt) {
            return {field,dt};
        }

        template<class T, class Mesh, class Mem>
        expr::Conv<T, Field<T,Mesh,Mem>> conv(const Field<T,Mesh,Mem>& lhs, const Field<T,Mesh,Mem>& rhs) {
            return {lhs};
        }

        template<class T, class Mesh, class Mem>
        expr::Laplace<T, T> laplace(const Field<T,Mesh,Mem>& field) {
            return {T(1.0)};
        }

        template<uint32_t AXIS, class T, class LHS>
        expr::Axis<to_scalar<T>, LHS, AXIS> axis(const expr::Expr<T,LHS>& lhs) {
            return {lhs};
        }

        template<class T, class Mesh, class Mem>
        expr::Laplace<T, Field<T,Mesh,Mem>> laplace(const Field<T,Mesh,Mem>& lhs, const Field<T,Mesh,Mem>& rhs) {
            return {lhs};
        }

        template<class T, class LHS, class Mesh, class Mem>
        expr::Laplace<T, LHS> laplace(const fvc::expr::Expr<T,LHS>& lhs, const Field<T,Mesh,Mem>&
        rhs) {
            return {lhs};
        }

        namespace eval {
            template<class Expr, class Mesh, class Scheme, typename MATCHES = void>
            class Evaluator {};

            template<class Expr,class Mesh,class Scheme>
            void coeff_face_sum(const Evaluator<Expr,Mesh,Scheme>& eval,
                                Stencil_Matrix<typename Expr::Elem, Mesh>& stencil,
                                const typename Mesh::Cell& cell,
                                typename Expr::Elem factor) {
                for (typename Mesh::Face& face : cell.faces()) eval.coeffs(stencil, face, factor);
            }

            template<class Expr,class Mesh,class Scheme>
            typename Expr::Elem source_face_sum(const Evaluator<Expr,Mesh,Scheme>& eval, typename Mesh::Cell& cell) {
                typename Expr::Elem sum = {};
                for (typename Mesh::Face& face : cell.faces()) sum += eval.source(face);
                return sum;
            }

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Divergence<T, LHS>, Mesh, Scheme, with_policy<Scheme, scheme::Upwind_Flux>> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Divergence<T, LHS> &expr) : lhs(expr.lhs) {}

                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Face &face, T factor) const {
                    stencil[face.neigh_stencil()] += factor * 0.5_R * face.sf() * face.ivol();
                    stencil[face.cell_stencil()] += factor * 0.5_R * face.sf() * face.ivol();
                    return *this;
                }
                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    coeff_face_sum(*this, stencil, cell, factor);
                }
                T source(const typename Mesh::Face& face) const { return T(); }
                T source(const typename Mesh::Cell& cell) const { return T(); }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::DDt<T, LHS>, Mesh, Scheme> {
                fvc::eval::Evaluator<LHS, Mesh, Scheme> lhs;
                real dt;
            public:
                Evaluator(const expr::DDt<T, LHS> &expr) : lhs(expr.lhs), dt(expr.rhs) {}

                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    stencil[cell.id_stencil()] += T(1)/dt;
                }
                T source(const typename Mesh::Cell& cell) const { return lhs.eval(cell) * T(1)/dt; }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Laplace<T, LHS>, Mesh, Scheme> {
                fvc::eval::Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Laplace<T, LHS> &expr) : lhs(expr.lhs) {}
                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    T alpha = factor * (lhs.eval(face.cell()) + lhs.eval(face.neigh())) / 2;
                    stencil[face.neigh_stencil()] += alpha * face.idx() * face.fa() * face.ivol();
                    stencil[face.cell_stencil()] += -alpha * face.idx() * face.fa() * face.ivol();
                }
                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    coeff_face_sum(*this, stencil, cell, factor);
                }
                T source(const typename Mesh::Face& face) const { return T(); }
                T source(const typename Mesh::Cell& cell) const { return T(); }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Conv<T, LHS>, Mesh, Scheme> {
                fvc::eval::Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Conv<T, LHS> &expr) : lhs(expr.lhs) {}
                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    T cell = lhs.eval(face.cell());
                    T neigh = lhs.eval(face.neigh());
                    T flux = 0.5_R * (cell + neigh);

                    stencil[face.neigh_stencil()] += 0.5_R * factor * dot(face.sf(), flux) * face.ivol();
                    stencil[face.cell_stencil()] += 0.5_R * factor * dot(face.sf(), flux) * face.ivol();
                }
                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    coeff_face_sum(*this, stencil, cell, factor);
                }
                T source(const typename Mesh::Face& face) const { return T(); }
                T source(const typename Mesh::Cell& cell) const { return T(); }
            };

            template<class T, class Mesh, class Scheme>
            class Evaluator<T, Mesh, Scheme, std::enable_if_t<std::is_arithmetic_v<T>>> {
            public:
                Evaluator(T value) : value(value) {}
                T value;
            };

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Add<T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Add<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    lhs.coeffs(stencil, face, factor);
                    rhs.coeffs(stencil, face, factor);
                }
                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    lhs.coeffs(stencil, cell, factor);
                    rhs.coeffs(stencil, cell, factor);
                }
                T source(const typename Mesh::Cell& cell) const { return lhs.source(cell) + rhs.source(cell); }
                T source(const typename Mesh::Face& face) const { return lhs.source(face) + rhs.source(face); }
            };

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Sub<T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Sub<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    lhs.coeffs(stencil, face, factor);
                    rhs.coeffs(stencil, face, -factor);
                }
                void coeffs(Stencil_Matrix<T, Mesh>& stencil, const typename Mesh::Cell &cell, T factor) const {
                    lhs.coeffs(stencil, cell, factor);
                    rhs.coeffs(stencil, cell, -factor);
                }
                T source(const typename Mesh::Cell& cell) const { return lhs.source(cell) - rhs.source(cell); }
                T source(const typename Mesh::Face& face) const { return lhs.source(face) - rhs.source(face); }
            };

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Mul<T, LHS, RHS>, Mesh, Scheme> {
                fvc::eval::Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Mul<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face& face, T factor) const {
                    rhs.coeffs(stencil, face, factor * lhs.eval(face));
                }
                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Cell& cell, T factor) const {
                    rhs.coeffs(stencil, cell, factor * lhs.eval(cell));
                }
                T source(const typename Mesh::Face& face) const { return lhs.eval(face) * rhs.source(face); }
                T source(const typename Mesh::Cell& cell) const { return lhs.eval(cell) * rhs.source(cell); }
            };

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Eq<T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                fvc::eval::Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Eq<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    lhs.coeffs(stencil, face, factor);
                }
                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Cell &cell, T factor) const {
                    lhs.coeffs(stencil, cell, factor);
                }
                T source(const typename Mesh::Face& face) const { return lhs.source(face) + rhs.eval(face); }
                T source(const typename Mesh::Cell& cell) const { return lhs.source(cell) + rhs.eval(cell); }
            };

            template<class T, class LHS, uint32_t AXIS, class Mesh, class Scheme>
            class Evaluator<expr::Axis<T, LHS, AXIS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Axis<T, LHS, AXIS> &expr) : lhs(expr.lhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Cell &cell, T factor) const {
                    using Vec = typename LHS::Elem;
                    Stencil_Matrix<Vec,Mesh> stencil_vec;
                    lhs.coeffs(stencil_vec, cell, Vec(factor));
                    for (uint32_t i = 0; i < stencil.size(); i++) stencil[i] = stencil_vec[i][AXIS];
                }

                T source(const typename Mesh::Cell& cell) const {
                    return lhs.source(cell)[AXIS];
                }
            };
        }

        template<class Exec, class Elem, class Mem, class Expr, class Mesh, class Scheme>
        void build_matrix(Exec& exec,
                          const Mesh &mesh,
                          Matrix<Elem,uint64_t,Mem>& matrix,
                          const Expr &expr,
                          const Boundary_Condition<Elem, Mesh, Mem>& bc,
                          const Scheme& scheme) {
            fvm::eval::Evaluator<Expr, Mesh, Scheme> eval(expr);

            matrix.resize(exec, mesh.cell_count(), KOKKOS_LAMBDA(uint32_t i) {
                return mesh.get_cell(i).stencil().size();
            });

            mesh.for_each_cell("Assemble Matrix", KOKKOS_LAMBDA(const typename Mesh::Cell& cell) {
                Stencil_Matrix<Elem, Mesh> coeffs;
                eval.coeffs(coeffs, cell, Elem(1.0));

                auto stencil = cell.stencil();

                assert(stencil[0] == cell.id());
                assert(coeffs[0] != 0.0_R);

                uint64_t offset = matrix.rowBegin(cell.id());
                for (uint32_t i = 0; i < stencil.size(); i++) {
                    bool valid = stencil[i] != UINT64_MAX;
                    matrix.coeffs[offset] = valid ? coeffs[i] : Elem();
                    matrix.cols[offset] = valid ? stencil[i] : 0;
                    offset++;
                }
            });
            bc.implicit_bc_matrix(matrix);
        }

        template<class Exec, class Elem, class... Layout, class Mem, class Expr, class Mesh, class Scheme>
        void build_source(Exec& exec,
                          const Mesh& mesh,
                          Kokkos::View<Elem*,Layout...>& source,
                          const Expr& expr,
                          const Boundary_Condition<Elem, Mesh, Mem>& bc,
                          const Scheme& scheme) {
            fvm::eval::Evaluator<Expr, Mesh, Scheme> eval(expr);

            mesh.for_each_cell("fill b", KOKKOS_LAMBDA(const typename Mesh::Cell& cell) {
                source(cell.id()) = eval.source(cell);
            });
            bc.implicit_bc_source(source);
        }

        template<class Exec, class Elem, class... Layout,class... Layout2,  class Mem, class Expr, class Mesh, class
        Scheme>
        void solve(
                Exec& exec,
                Linear_Solver<Matrix<Elem,uint64_t,Mem>>& solver,
                const Mesh &mesh,
                const Expr &expr,
                Kokkos::View<Elem*, Layout...>& x,
                const Boundary_Condition<Elem, Mesh, Mem>& bc,
                const Scheme& scheme,
                Matrix<Elem,uint64_t,Mem>& matrix,
                Kokkos::View<float*, Layout2...>& source
        ) {
            printf("Building matrix : mesh %i\n", mesh.cell_count());
            if (mesh.cell_count() == 0) {
                fprintf(stderr, "MESH IS empty");
                return;
            }

            Kokkos::Timer timer;
            build_matrix(exec, mesh, matrix, expr, bc, scheme);
            matrix.check();
            printf("Build matrix took : %f s\n", timer.seconds());
            timer.reset();
            build_source(exec, mesh, source, expr, bc, scheme);
            printf("Build source took : %f s\n", timer.seconds());
            timer.reset();
            solver.solve(matrix, x, source);
            printf("Solve linear system took : %f s\n", timer.seconds());
        }

        template<class Exec, class Elem, class Mem, class Expr, class Mesh, class Scheme>
        std::enable_if_t<is_scalar<Elem>> solve(
                Exec& exec,
                Linear_Solver<Matrix<Elem,uint64_t,Mem>>& solver,
                const Mesh &mesh,
                const Expr &expr,
                Field<Elem, Mesh, Mem>& x,
                const Scheme& scheme
        ) {
            Matrix<Elem,uint64_t,Mem> matrix;
            Kokkos::View<Elem*, Kokkos::HostSpace> source("source", mesh.cell_count());
            solve(exec,solver,mesh,expr,x.data(),x.bc(),scheme,matrix,source);
        }

        template<class T, class Mesh, class Mem>
        struct Pseudo {
            Field<T,Mesh,Mem>& A;
        };

        template<class Exec, class Elem, class Mem, class Expr, class Mesh, class Scheme>
        std::enable_if_t<!is_scalar<Elem>> solve(
                Exec& exec,
                Linear_Solver<Matrix<to_scalar<Elem>,uint64_t,Mem>>& solver,
                const Mesh &mesh,
                const Expr &expr,
                Field<Elem, Mesh, Mem>& x_vec,
                const Scheme& scheme,
                const Pseudo<Elem,Mesh,Mem> pseudo
        ) {
            Matrix<to_scalar<Elem>,uint64_t,Mem> matrix;
            uint64_t n = mesh.cell_count();

            Kokkos::View<to_scalar<Elem>*, Kokkos::HostSpace> source("source", n);
            Kokkos::View<to_scalar<Elem>*, Kokkos::HostSpace> x("source", n);
            Kokkos::View<to_scalar<Elem>*,Mem> A("A",n);

            auto solve_axis = [&](auto AXIS) {
                assert(x_vec.size() >= x.size());

                constexpr uint32_t axis = uint32_t(decltype(AXIS)());

                Kokkos::View<to_scalar<Elem>*,Mem> x("x",n);

                auto axis_dst = [&](auto& dst, auto& src) {
                    Kokkos::parallel_for("axis_dst", Kokkos::RangePolicy<Exec>(0,n), KOKKOS_LAMBDA(uint32_t i) {
                        dst(i) = src(i)[axis];
                    });
                };

                auto axis_src = [&](auto& dst, auto& src){
                    Kokkos::parallel_for("axis_src", Kokkos::RangePolicy<Exec>(0,n), KOKKOS_LAMBDA(uint32_t i) {
                        dst.data()[i][axis] = src(i);
                    });
                };

                axis_dst(x,x_vec);

                auto bc = x_vec.bc().segregated(axis);
                solve(exec,solver,mesh,fvm::axis<axis>(expr), x,bc,scheme, matrix,source);
                matrix.diagonal(exec,A);
                axis_src(pseudo.A, A);
                axis_src(x_vec, x);
            };

            solve_axis(std::integral_constant<uint32_t,0>());
            solve_axis(std::integral_constant<uint32_t,1>());
            solve_axis(std::integral_constant<uint32_t,2>());
        }
    }
}