
namespace reycode {
    namespace fvc {
        namespace scheme {
            struct Central_Difference {
            };
            struct Upwind_Flux {
            };
        };

        namespace expr {
            template<class T, class Super>
            struct Expr {
                using Elem = T;

                operator const Super &() const { return *static_cast<const Super *>(this); }
                operator Super &() { return *static_cast<Super *>(this); }
            };

            template<class Tag, class T, class LHS>
            struct Unary_Expr : Expr<T, Unary_Expr<Tag, T, LHS>> {
                using Elem = T;

                LHS lhs;

                Unary_Expr(const LHS &lhs) : lhs(lhs) {}
            };

            template<class Tag, class T, class LHS, class RHS>
            struct Binary_Expr : Expr<T, Binary_Expr<Tag, T, LHS, RHS>> {
                using Elem = T;
                LHS lhs;
                RHS rhs;

                Binary_Expr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}
            };

            template<class T, class LHS, class RHS>
            using Divergence = Binary_Expr<struct DivergenceTag, T, LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Laplace = Binary_Expr<struct LaplaceTag, T, LHS, RHS>;

            template<class T, class RHS>
            using Grad = Unary_Expr<struct GradTag, T, RHS>;

            template<class T, class LHS, class RHS>
            using Add = Binary_Expr<struct AddTag, T, LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Sub = Binary_Expr<struct SubTag, T, LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Mul = Binary_Expr<struct MulTag, T, LHS, RHS>;

            template<class T, class LHS, class RHS>
            using Div = Binary_Expr<struct DivTag, T, LHS, RHS>;

            template<class T, class LHS>
            using Avg = Unary_Expr<struct AvgTag, T, LHS>;

            template<class T, class LHS, class RHS>
            Add<T, LHS, RHS> operator+(const Expr<T, LHS> &lhs, const Expr<T, RHS> &rhs) { return {lhs, rhs}; }

            template<class T, class RHS>
            Sub<T, T, RHS> operator-(const Expr<T, RHS> &rhs) { return {T(), rhs}; }

            template<class T, class LHS, class RHS>
            Sub<T, LHS, RHS> operator-(const Expr<T, LHS> &lhs, const Expr<T, RHS> &rhs) { return {lhs, rhs}; }

            template<class T, class RHS>
            Mul<T, T, RHS> operator*(T lhs, const Expr<T, RHS> &rhs) { return {lhs, rhs}; }

            template<class T, class RHS>
            Div<T, T, RHS> operator/(T lhs, const Expr<T, RHS> &rhs) { return {lhs, rhs}; }
        };

        template<class T, class Mesh, class Mem>
        expr::Avg<to_scalar<T>, Field<T,Mesh,Mem>> avg(const Field<T,Mesh,Mem>& rhs) {
            return {rhs};
        }

        template<class T, class Mesh, class Mem>
        expr::Grad<to_vector<T>, Field<T,Mesh,Mem>> grad(const Field<T,Mesh,Mem>& rhs) {
            return {rhs};
        }

        template<class T, class Mesh, class Mem>
        expr::Divergence<to_scalar<T>, T, Field<T,Mesh,Mem>> div(const Field<T,Mesh,Mem>& rhs) {
            return {T(),rhs};
        }

        template<class T, class LHS, class Mesh, class Mem>
        expr::Divergence<to_scalar<T>, LHS, Field<T,Mesh,Mem>> div(const expr::Expr<to_scalar<T>,LHS>& lhs, const
        Field<T,Mesh,
                                                            Mem>&
                rhs) {
            return {lhs,rhs};
        }

        template<class T, class LHS, class Mesh, class Mem>
        expr::Laplace<T, LHS, Field<T,Mesh,Mem>> laplace(
                const expr::Expr<T,LHS>& lhs,
                const Field<T,Mesh,Mem>& rhs) {
            return {lhs,rhs};
        }

        template<class T, class Mesh, class Mem>
        expr::Divergence<to_scalar<T>, Field<to_scalar<T>,Mesh,Mem>, Field<T,Mesh,Mem>> div(const Field<to_scalar<T>,
                Mesh,
                                                                                     Mem>& lhs,
                                                                                     const Field<T,Mesh,Mem>& rhs) {
            return {lhs,rhs};
        }

        namespace eval {
            template<class Expr, class Mesh, class Scheme, typename MATCHES = void>
            class Evaluator {
            };

            template<class Expr,class Mesh,class Scheme>
            typename Expr::Elem face_sum(const Evaluator<Expr,Mesh,Scheme>& eval,
                                const typename Mesh::Cell& cell) {
                typename Expr::Elem sum = {};
                for (typename Mesh::Face& face : cell.faces()) sum += eval.eval(face);
                return sum;
            }

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Divergence<T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Divergence<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                T eval(const typename Mesh::Face &face) const {
                    auto interp_lhs = (lhs.eval(face.neigh()) + lhs.eval(face.cell())) / T(2.0);
                    auto interp_rhs = (rhs.eval(face.neigh()) + rhs.eval(face.cell())) / T(2.0);

                    return face.ivol() * interp_lhs * dot(face.sf(), interp_rhs);
                }
                T eval(const typename Mesh::Cell& cell) const { return face_sum(*this, cell); }
            };

            template<class T, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Laplace<T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Laplace<T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                void eval(const typename Mesh::Face &face) const {
                    return face.idx() * face.fa() * face.ivol() * lhs.eval(face) * (
                            rhs.eval(face.cell()) + rhs.eval(face.neigh())
                    );
                }
                void eval(const typename Mesh::Cell& cell) const { return face_sum(*this, cell); }
            };

            template<class T, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Grad<T, RHS>, Mesh, Scheme> {
                Evaluator<RHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Grad<T, RHS> &expr) : lhs(expr.lhs) {}

                T eval(const typename Mesh::Face &face) const {
                    auto cell = lhs.eval(face.cell());
                    auto neigh = lhs.eval(face.neigh());
                    return face.ivol() * face.sf() * (cell + neigh) / 2;
                }
                T eval(const typename Mesh::Cell& cell) const { return face_sum(*this, cell); }
            };

            template<class T, class LHS, class Mesh, class Scheme>
            class Evaluator<expr::Avg<T, LHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
            public:
                Evaluator(const expr::Avg<T, LHS> &expr) : lhs(expr.lhs) {}

                T eval(const typename Mesh::Face &face) const { return avg(lhs.eval(face)); }
                T eval(const typename Mesh::Cell& cell) const { return avg(lhs.eval(cell)); }
            };

            template<class T, class Mesh, class Scheme>
            class Evaluator<T, Mesh, Scheme, std::enable_if_t<is_constant<T>>> {
                T value;
            public:
                Evaluator(T value) : value(value) {}

                T eval(const typename Mesh::Face& face) const { return value; }
                T eval(const typename Mesh::Cell& face) const { return value; }
            };

            template<class T, class Mem, class Mesh, class Scheme>
            class Evaluator<Field<T, Mesh, Mem>, Mesh, Scheme> {
                Field<T, Mesh, Mem> view;
            public:
                Evaluator(Field<T, Mesh, Mem> view) : view(view) {}

                T eval(const typename Mesh::Cell &cell) const { return view(cell.id()); }
            };

            template<class T, class Tag, class LHS, class RHS, class Mesh, class Scheme>
            class Evaluator<expr::Binary_Expr<Tag, T, LHS, RHS>, Mesh, Scheme> {
                Evaluator<LHS, Mesh, Scheme> lhs;
                Evaluator<RHS, Mesh, Scheme> rhs;
            public:
                Evaluator(const expr::Binary_Expr<Tag, T, LHS, RHS> &expr) : lhs(expr.lhs), rhs(expr.rhs) {}

                T eval(const typename Mesh::Face &face) const {
                    if constexpr (std::is_same_v<Tag, expr::AddTag>) return lhs.eval(face) + rhs.eval(face);
                    else if constexpr (std::is_same_v<Tag, expr::SubTag>) return lhs.eval(face) - rhs.eval(face);
                    else if constexpr (std::is_same_v<Tag, expr::MulTag>) return lhs.eval(face) * rhs.eval(face);
                    else if constexpr (std::is_same_v<Tag, expr::DivTag>) return lhs.eval(face) / rhs.eval(face);
                };

                T eval(const typename Mesh::Cell &cell) const {
                    if constexpr (std::is_same_v<Tag, expr::AddTag>) return lhs.eval(cell) + rhs.eval(cell);
                    else if constexpr (std::is_same_v<Tag, expr::SubTag>) return lhs.eval(cell) - rhs.eval(cell);
                    else if constexpr (std::is_same_v<Tag, expr::MulTag>) return lhs.eval(cell) * rhs.eval(cell);
                    else if constexpr (std::is_same_v<Tag, expr::DivTag>) return lhs.eval(cell) / rhs.eval(cell);
                };
            };
        }

        template<class Exec, class Mem, class Expr, class Mesh, class Scheme>
        void compute(Exec& exec,
                     const Mesh& mesh,
                     const Kokkos::View<typename Expr::Elem*, Mem>& result,
                     const Expr& expr,
                     const Scheme& scheme) {
            using Elem = typename Expr::Elem;
            eval::Evaluator<Expr, Mesh, Scheme> eval(expr);

            mesh.for_each_cell("Eval expr", KOKKOS_LAMBDA(const typename Mesh::Cell& cell) {
                result(cell.id()) = eval.eval(cell);
            });
        }
    }
}