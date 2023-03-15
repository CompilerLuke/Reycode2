#include "reycode/reycode.h"

#ifdef REY_PLATFORM_CUDA
#include <cuda_runtime.h>
#endif

namespace reycode {
	Arena make_host_arena(uint64_t size) {
		Arena arena = {};
		arena.capacity = size;
		arena.data = (uint8_t*)malloc(arena.capacity);
		return arena;
	}

	void destroy_host_arena(Arena& arena) {
		free(arena.data);
	}

#ifdef REY_PLATFORM_CUDA
	Arena make_device_arena(uint64_t size, Cuda_Error& err) {
		Arena arena = {};
		arena.capacity = size;
		cudaMalloc((void**)(&arena.data), size);
		return arena;
	}

	void destroy_device_arena(Arena& arena) {
		cudaFree(arena.data);
	}

	Memory_Ctx make_memory_ctx(uint64_t host_perm, uint64_t host_frame, uint64_t device_perm, uint64_t device_frame, Cuda_Error& err) {
		Memory_Ctx ctx = {};
		ctx.host_perm_arena = make_host_arena(host_perm);
		ctx.host_temp_arena = make_host_arena(host_frame);

		ctx.device_perm_arena = make_device_arena(device_perm, err);
		ctx.device_temp_arena = make_device_arena(device_frame, err);

		return ctx;
	}

	void destroy_memory_ctx(Memory_Ctx& ctx) {
		destroy_host_arena(ctx.host_perm_arena);
		destroy_host_arena(ctx.host_temp_arena);

		destroy_device_arena(ctx.device_temp_arena);
		destroy_device_arena(ctx.host_temp_arena);
	}
#endif
}