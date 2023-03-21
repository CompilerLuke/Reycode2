#pragma once

#include "Kokkos_Core.hpp"
#include "reycode_solver/linear_solver/linear_solver.h"

namespace reycode {
    template<class T, class Mesh, class Mem>
    class Boundary_Patch {
    public:
        virtual void implicit_bc_matrix(Matrix<T,uint64_t,Mem>& matrix) const = 0;
        virtual void implicit_bc_source(Kokkos::View<kokkos_ptr<T>,Mem> result) const = 0;
        virtual void explicit_bc(Kokkos::View<kokkos_ptr<T>,Mem> result, Kokkos::View<kokkos_ptr<T>,Mem> prev) const= 0;
        virtual std::unique_ptr<Boundary_Patch<to_scalar<T>,Mesh,Mem>> segregated(uint32_t axis)= 0;

        Boundary_Patch() {}
        virtual ~Boundary_Patch() {}
    };

    template<class T, class Mesh, class Mem>
    class Boundary_Condition {
        Mesh& mesh;
        std::vector<std::unique_ptr<Boundary_Patch<T,Mesh,Mem>>> patches;
    public:
        Boundary_Condition(Mesh& mesh) : mesh(mesh) {}
        Boundary_Condition(std::vector<std::unique_ptr<Boundary_Patch<T,Mesh,Mem>>>&& patches) {
            this->patches = std::move(patches);
        }

        //todo: avoid copying
        Boundary_Condition<to_scalar<T>,Mesh,Mem> segregated(uint32_t axis) const {
            Boundary_Condition<to_scalar<T>,Mesh,Mem> result(mesh);
            for (auto& patch : patches) result.push_back(patch->segregated(axis));
            return result;
        }

        template<class BC>
        void push_back(std::unique_ptr<BC>&& bc) {
            patches.push_back(std::move(bc));
        }

        template<class BC>
        void push_back(BC&& bc) {
            Boundary_Patch<T,Mesh,Mem>& compatibility_check = bc;
            patches.push_back(std::make_unique<BC>(std::move(bc)));
        }

        void implicit_bc_matrix(Matrix<to_scalar<T>*,uint64_t,Mem>& matrix, uint32_t axis) const {
            for (auto& patch : patches) patch->implicit_bc_matrix(matrix,axis);
        }

        void implicit_bc_source(Kokkos::View<to_scalar<T>*,Mem> result, uint32_t axis) const {
            for (auto& patch : patches) patch->implicit_bc_source(result,axis);
        }

        void implicit_bc_matrix(Matrix<T,uint64_t,Mem>& matrix) const {
            for (auto& patch : patches) patch->implicit_bc_matrix(matrix);
        }

        void implicit_bc_source(Kokkos::View<kokkos_ptr<T>,Mem> result) const {
            for (auto& patch : patches) patch->implicit_bc_source(result);
        }

        void explicit_bc(Kokkos::View<kokkos_ptr<T>,Mem> result, Kokkos::View<kokkos_ptr<T>,Mem> prev) const {
            for (auto& patch : patches) patch->explicit_bc(result, prev);
        }
    };

    namespace fvc::expr {
        template<class T, class Super>
        struct Expr;
    }

    template<class T, class Mesh, class Mem, class Tag = void>
    struct Field : fvc::expr::Expr<T,Field<T,Mesh,Mem,Tag>> {
        Kokkos::View<kokkos_ptr<T>,Mem> view;
        Boundary_Condition<T,Mesh,Mem>* bc_patches = nullptr;
    public:
        Field() {}
        Field(Mesh& mesh, Kokkos::View<kokkos_ptr<T>,Mem> view, Boundary_Condition<T,Mesh,Mem>& bc_patches) : view
        (view), bc_patches(&bc_patches) {
            update_bc();
        }

        Field(const std::string& name,
              Mesh& mesh,
              Boundary_Condition<T,Mesh,Mem>& bc_patches
          ) : bc_patches(&bc_patches) {
            view = Kokkos::View<kokkos_ptr<T>,Mem>(name,mesh.cell_count());
        }

        const Boundary_Condition<T,Mesh,Mem>& bc() {
            assert(bc_patches);
            return *bc_patches;
        }

        const Kokkos::View<kokkos_ptr<T>,Mem>& data() const { return view; }
        Kokkos::View<kokkos_ptr<T>,Mem>& data() { return view; }

        size_t size() const { return view.size(); }

        T operator()(uint64_t id) const {
            return view(id);
        }

        void update_bc() {
            assert(bc_patches);
            bc_patches->explicit_bc(view,view);
        }
    };
}