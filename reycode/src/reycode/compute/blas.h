#pragma once

#include <reycode/reycode.h>
#include <cublas_v2.h>

namespace reycode {
    class BLAS {
        cublasHandle_t handle;
    
        void handle_status(cublasStatus_t status);
    public:
        void copy(slice<real> dst, slice<real> src);
        void axpy(slice<real> dst, real alpha, slice<real> x, slice<real> y);
        real dot(slice<real> x, slice<real> y);
    
        BLAS();
        BLAS(BLAS&) = delete;
        BLAS(BLAS&&) = delete;
        ~BLAS();
    };
}