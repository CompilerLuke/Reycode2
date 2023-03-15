#pragma once

#include "reycode/reycode.h"
#include <driver_types.h>

namespace reycode {
    struct Cuda_Error {
        cudaError_t error;

        operator bool() {
            return error;
        }
    };

    INLINE bool operator!(Cuda_Error& err) { return !err; }

    INLINE void operator|=(Cuda_Error& err, cudaError_t error) {
        err.error = error;
        if (error) {
            fprintf(stderr, "CUDA ERROR: %i", error);
            abort();
        }
    }

    Arena make_cuda_device_arena(uint64_t size, Cuda_Error&);
    void destroy_cuda_device_arena(Arena& arena);
}