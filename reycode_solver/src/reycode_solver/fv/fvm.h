#pragma once

#include "reycode/reycode.h"

namespace reycode {
    namespace fvm {
        template<class T, uint32_t MAX_COEFFS>
        struct Stencil_Matrix {
            T data[MAX_COEFFS] = {};
            T& operator[](uint32_t i) { return data[i]; }
        };

        template<class T, class Mesh, uint32_t MAX_COEFFS>
        struct FVM {
            Stencil_Matrix<T, MAX_COEFFS> stencil;

            FVM() {}

            FVM& div(typename Mesh::Face& face, T alpha) {
                stencil[face.neigh_stencil()] += alpha * 0.5_R * face.fa() * face.ivol();
                stencil[face.cell_stencil()] += alpha * 0.5_R * face.fa() * face.ivol();
                return *this;
            }

            FVM& laplace(typename Mesh::Face& face, T alpha) {
                stencil[face.neigh_stencil()] += alpha * face.idx() * face.fa() * face.ivol();
                stencil[face.cell_stencil()] += -alpha * face.idx() * face.fa() * face.ivol();
                return *this;
            };
        };



    }
}