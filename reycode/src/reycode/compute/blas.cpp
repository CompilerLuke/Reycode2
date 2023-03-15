#include "reycode/compute/blas.h"
#include "cuda_runtime.h"

namespace reycode {
    void BLAS::handle_status(cublasStatus_t status) {
        if (status != 0) {
            printf("BLAS Error : %i", status);
            abort();
        }
    }

    void BLAS::copy(slice<real> dst, slice<real> src) {
        assert(dst.length == src.length);
        handle_status(cublasScopy(handle, dst.length, src.data, 1, dst.data, 1));
    }

    void BLAS::axpy(slice<real> dst, real alpha, slice<real> x, slice<real> y) {
        assert(dst.length == x.length && x.length == y.length);
        if (dst.data != y.data) copy(dst, y);

        handle_status(cublasSaxpy(handle, dst.length, &alpha, x.data, 1, y.data, 1));
    }

    real BLAS::dot(slice<real> x, slice<real> y) {
        assert(x.length == y.length);

        real result = -10.0_R;
        cublasSdot(handle, x.length, x.data, 1, y.data, 1, &result);
        cudaDeviceSynchronize();
        return result;
    }

    BLAS::BLAS() {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != 0) {
            printf("Could not initialize cublas!");
            abort();
        }
    }

    BLAS::~BLAS() {
        cublasDestroy(handle);
    }
}