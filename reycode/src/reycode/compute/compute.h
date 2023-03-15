#pragma once

#include "reycode/reycode.h"
#include <cuda/atomic>
#include <cub/block/block_scan.cuh>

namespace reycode {
    struct Stream_Compaction_Result {
        uint32_t block_base;
        uint32_t block_id;
        uint32_t block_count;
        uint32_t id;
    };

    using cuda_atomic_uint32_t = cuda::atomic<uint32_t, cuda::thread_scope_device>;

    template<uint32_t DIM_X, uint32_t DIM_Y, uint32_t DIM_Z>
    union Stream_Compaction_Temp {
        using Block_Scan = cub::BlockScan<uint32_t, DIM_X, cub::BLOCK_SCAN_RAKING, DIM_Y, DIM_Z>;
        
        typename Block_Scan::TempStorage scan;
        uint32_t block_base;

        static_assert(sizeof(scan) < kb(10), "Out of shared memory");
    };


    template<uint32_t DIM_X, uint32_t DIM_Y, uint32_t DIM_Z>
    __device__ Stream_Compaction_Result stream_compaction(Stream_Compaction_Temp<DIM_X, DIM_Y, DIM_Z>& temp, cuda_atomic_uint32_t& offset, uint32_t count) {
        uint32_t block_offset;
        uint32_t block_count;

        using Block_Scan = typename Stream_Compaction_Temp<DIM_X, DIM_Y, DIM_Z>::Block_Scan;

        __syncthreads();

        Block_Scan(temp.scan).ExclusiveSum(count, block_offset, block_count);

        if (uvec3(threadIdx) == uvec3(0)) temp.block_base = offset.fetch_add(block_count);
        __syncthreads();

        uint32_t block_base = temp.block_base;

        Stream_Compaction_Result result;
        result.block_base = block_base;
        result.block_id = block_offset;
        result.block_count = block_count;
        result.id = block_base + block_offset;

        return result;
    }

    template<class T>
    void memcpy_t(T* dst, T* src, cudaMemcpyKind kind) {
        cudaMemcpy(dst, src, sizeof(T), kind);
    }

    template<class T>
    void memcpy_t(slice<T> dst, slice<T> src, cudaMemcpyKind kind) {
        assert(src.length == dst.length);
        cudaMemcpy(dst.data, src.data, sizeof(T) * dst.length, kind);
    }
};