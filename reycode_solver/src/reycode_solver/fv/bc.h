#include "fvm.h"
#include "reycode/mesh/mesh.h"
#include "reycode_solver/field/field.h"

namespace reycode {
    namespace bc {
        namespace expr {
            template<class T, class Super>
            struct Expr {
                using Elem = T;

                operator const Super &() const { return *static_cast<const Super *>(this); }
                operator Super &() { return *static_cast<Super *>(this); }
            };

            template<class Tag, class T, class LHS>
            struct Unary_Expr : Expr<T, Unary_Expr<Tag,T,LHS>> {
                using Elem = T;
                LHS lhs;

                Unary_Expr(const LHS& lhs) : lhs(lhs) {}
            };

            template<uint32_t AXIS>
            struct AxisTag;

            template<class T, class LHS, uint32_t axis>
            using Axis = Unary_Expr<struct AxisTag<axis>, T, LHS>;

            template<class T, class LHS>
            using Grad = Unary_Expr<struct GradTag, T, LHS>;

            template<class T, class LHS>
            using Value = Unary_Expr<struct ValueTag, T, LHS>;
        };

        template<class T>
        expr::Grad<T,T> grad(T value) {
            return {value};
        }

        template<class T>
        expr::Value<T,T> value(T value) {
            return {value};
        }

        template<uint32_t AXIS, class T, class LHS>
        expr::Axis<to_scalar<T>, LHS, AXIS> axis(const expr::Expr<T,LHS>& lhs) {
            return {lhs};
        }

        namespace eval {
            template<class Expr, class Mesh, class Scheme, typename MATCHES = void>
            class Evaluator {
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Grad< T, LHS>, Mesh, Scheme> {
                fvc::eval::Evaluator <LHS, Mesh, Scheme> lhs;
            public:

                Evaluator(const expr::Grad <T, LHS> &expr) : lhs(expr.lhs) {}
                void coeffs(Stencil_Matrix <T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    stencil[face.neigh_stencil()] += T(1);//factor * face.idx() * face.fa() * face.ivol();
                    stencil[face.cell_stencil()] += -T(1); //factor * face.idx() * face.fa() * face.ivol();
                }

                T source(const typename Mesh::Face &face) const { return T();
                    //face.fa() * face.ivol() * face.dx() * lhs.eval(face);
                }

                template<class... Layout>
                T eval(const typename Mesh::Face& face, Kokkos::View<kokkos_ptr<T>,Layout...> prev) const {
                    return prev(face.neigh().id()) + lhs.eval(face)*face.dx();
                }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Value<T, LHS>, Mesh, Scheme> {
                fvc::eval::Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Value<T, LHS> &expr) : lhs(expr.lhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    stencil[face.neigh_stencil()] += T(0.5);//factor / 2 * face.fa() * face.ivol();
                    stencil[face.cell_stencil()] += T(0.5);//factor / 2 * face.fa() * face.ivol();
                }

                //face.fa() * face.ivol() *
                T source(const typename Mesh::Face &face) const { return lhs.eval(face); }

                template<class... Layout>
                T eval(const typename Mesh::Face& face, Kokkos::View<kokkos_ptr<T>, Layout...> prev) const {
                    return 2*lhs.eval(face) - prev(face.neigh().id());
                }
            };

            template<class T, class LHS, uint32_t AXIS, class Mesh, class Scheme>
            class Evaluator<expr::Axis<T, LHS, AXIS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Axis<T, LHS, AXIS> &expr) : lhs(expr.lhs) {}

                void coeffs(Stencil_Matrix<T, Mesh> &stencil, const typename Mesh::Face &face, T factor) const {
                    using Vec = typename LHS::Elem;
                    Stencil_Matrix<Vec,Mesh> stencil_vec;
                    lhs.coeffs(stencil_vec, face, Vec(factor));
                    for (uint32_t i = 0; i < stencil.size(); i++) stencil[i] = stencil_vec[i][AXIS];
                }

                T source(const typename Mesh::Face &face) const { return lhs.source(face)[AXIS]; }

                template<class... Layout>
                T eval(const typename Mesh::Face& face, Kokkos::View<kokkos_ptr<T>, Layout...> prev) const {
                    throw new std::logic_error("Eval cannot be called in segregated mode");
                }
            };
        }

        template<class T, class Expr, class Mesh, class Mem, class Scheme, bool SEGREGATED>
        class Boundary_Patch_Expr : public Boundary_Patch<T,Mesh,Mem> {
            Mesh &mesh;
            Expr expr;
            Patch patch;
            Scheme scheme;
        public:
            Boundary_Patch_Expr(Mem &mem,
                                Mesh &mesh,
                                Patch patch,
                                Scheme scheme,
                                const expr::Expr<T, Expr> &expr) : mesh(mesh),
                                                                   patch(patch), expr(expr), scheme(scheme) {}

            virtual void implicit_bc_matrix(Matrix<T, uint64_t, Mem> &matrix) const override {
                eval::Evaluator<Expr, Mesh, Scheme> eval(expr);
                mesh.for_each_patch_cell("implicit bc matrix", patch, KOKKOS_LAMBDA(const typename Mesh::Face &face) {
                    uint64_t begin = matrix.rowBegin(face.cell().id());
                    Stencil_Matrix<T, Mesh> stencil;
                    eval.coeffs(stencil, face, T(1.0_R));

                    face.neigh().id();
                    assert(face.neigh().id() != UINT64_MAX);

                    matrix.cols[begin + face.cell_stencil()] = face.cell().id();
                    matrix.cols[begin + face.neigh_stencil()] = face.neigh().id();
                    matrix.coeffs(begin + face.cell_stencil()) = stencil[face.cell_stencil()];
                    matrix.coeffs(begin + face.neigh_stencil()) = stencil[face.neigh_stencil()];
                });
            }

            virtual void implicit_bc_source(Kokkos::View<kokkos_ptr<T>, Mem> result) const override {
                eval::Evaluator<Expr, Mesh, Scheme> eval(expr);
                mesh.for_each_patch_cell("implicit bc source", patch, KOKKOS_LAMBDA(const typename Mesh::Face &face) {
                    result(face.cell().id()) = eval.source(face);
                });
            }

            virtual void explicit_bc(Kokkos::View<kokkos_ptr<T>, Mem> result, Kokkos::View<kokkos_ptr<T>, Mem> prev)
            const override {
                eval::Evaluator<Expr, Mesh, Scheme> eval(expr);
                mesh.for_each_patch_cell("explicit bc", patch, KOKKOS_LAMBDA(const typename Mesh::Face &face) {
                    result(face.cell().id()) = eval.eval(face, prev);
                });
            }

            virtual std::unique_ptr<Boundary_Patch<to_scalar<T>,Mesh,Mem>> segregated(uint32_t axis) override {
                throw std::string("no segregated bc condition");
            }
        };

        template<class T, class Expr, class Mesh, class Mem, class Scheme>
        class Boundary_Patch_Expr<T,Expr,Mesh,Mem,Scheme,true> : public Boundary_Patch<T,Mesh,Mem> {
            Mesh& mesh;
            Patch patch;
            Scheme scheme;
            Expr expr;
            Boundary_Patch_Expr<T, Expr, Mesh, Mem, Scheme, false> m_coupled;

            template<uint32_t axis>
            std::unique_ptr<Boundary_Patch<to_scalar<T>,Mesh,Mem>> segregated_axis() {
                Mem mem;
                Boundary_Patch_Expr<to_scalar<T>,
                                    expr::Axis<to_scalar<T>,Expr,axis>,
                                    Mesh,
                                    Mem,
                                    Scheme
                > bc{mem, mesh, patch, scheme, bc::axis<axis>(expr)};
                return std::make_unique<decltype(bc)>(bc);
            };
        public:
            Boundary_Patch_Expr(Mem &mem,
                                Mesh &mesh,
                                Patch patch,
                                Scheme scheme,
                                const expr::Expr<T, Expr> &expr)
                    : expr(expr), mesh(mesh), patch(patch), scheme(scheme), m_coupled(mem,mesh,patch,scheme,expr) {}

            virtual void implicit_bc_matrix(Matrix<T, uint64_t, Mem> &matrix) const override {
                m_coupled.implicit_bc_matrix(matrix);
            }

            virtual void implicit_bc_source(Kokkos::View<kokkos_ptr<T>, Mem> result) const override {
                m_coupled.implicit_bc_source(result);
            }

            virtual void explicit_bc(Kokkos::View<kokkos_ptr<T>, Mem> result, Kokkos::View<kokkos_ptr<T>, Mem> prev)
            const override {
                m_coupled.explicit_bc(result, prev);
            }

            virtual std::unique_ptr<Boundary_Patch<to_scalar<T>,Mesh,Mem>> segregated(uint32_t axis) override {
                Mem mem;
                if (axis==0) return segregated_axis<0>();
                if (axis==1) return segregated_axis<1>();
                if (axis==2) return segregated_axis<2>();
                throw std::string("Expected axis<3");
            }
        };
    }
}