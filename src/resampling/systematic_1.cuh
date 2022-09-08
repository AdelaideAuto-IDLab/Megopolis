#pragma once

#include <cooperative_groups.h>  // cooperative groups::this_thread_block, cooperative groups::tiled_partition
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <random>
#include <stdio.h>
#include <assert.h>
#include "helper.cuh"
#include "resampler.cuh"
#include "random.cuh"

// Size of warp shared storage
constexpr unsigned int kTRI { 64 }; 
constexpr unsigned int kWarpSize { 32 }; 

namespace resampling {
    namespace cg = cooperative_groups;

    __device__ void system_resample_up_warp(
        cg::thread_block_tile<kWarpSize> const &tile_32,
        size_t const tid,
        float u,
        size_t *indices,
        float *prefix_sum,
        size_t const indices_count,
        size_t const prefix_count,
        float *shared
    ) {

        auto const t = tile_32.thread_rank(); 

        int l = 0;
        bool mask = true; 
        float weight_norm = 1.0 / prefix_sum[prefix_count - 1];
        float norm = 1.0 / __uint2float_rn(indices_count);
        float prefix_float = __uint2float_rn(prefix_count); 

        // Compute the correct offset into the prefix sum
        float idx_float = __uint2float_rn(tid) * norm * prefix_float;
        // Static idx to remember the starting position of the index value
        size_t idx = __float2int_rn(idx_float);
        // Mutable idx to modify the ancestor of this particle
        size_t mut_idx = idx; 
        float draw = (u + idx_float) / prefix_float;

        // Load the first 64 values into shared memory, normalising the weights
        if (idx < prefix_count - kWarpSize && prefix_count >= kTRI) {
            shared[t] = prefix_sum[idx] * weight_norm; 
            shared[t + kWarpSize] = prefix_sum[idx + kWarpSize] * weight_norm;
        }

        tile_32.sync();

        while (tile_32.any( mask )) {
            // If 'idx + l' is not in the final 'kTRI' values of the prefix sum
            // we can use the values stored in shared memory
            if (idx < prefix_count - (kTRI) - l) {
        
    #pragma unroll kWarpSize
                for (int i = 0; i < kWarpSize; i++) {
                    mask = shared[t + i] < draw;

                    if (mask) {
                        // If there are more particles to the left of this particle that have been
                        // cloned then we shift this particles ancestor to the right
                        mut_idx += 1; 
                    }
                }

                l += kWarpSize;
                // First shift the [(l + 1) * kWarpSize, (l + 2) * kWarpSize] elements to the left 
                shared[t] = shared[t + kWarpSize];
                // Load the new [(l + 1) * kWarpSize, (l + 2) * kWarpSize] elements from global memory
                shared[t + kWarpSize] = prefix_sum[idx + kWarpSize + l] * weight_norm;
            }
            else {
                while (mask) {
                    if (idx > (prefix_count - l)) {
                        mask = false; 
                    }
                    else {
                        mask = prefix_sum[idx + l] * weight_norm < draw; 
                    }

                    if (mask) {
                        // If there are more particles to the left of this particle that have been
                        // cloned then we shift this particles ancestor to the right
                        mut_idx += 1;
                    }

                    l += 1;
                }
            }

            tile_32.sync( );
        }

        indices[tid] = mut_idx;
    }

    __global__ void systematic_resample_up_shared_prefetch(
        size_t *indices,
        float *prefix_sum,
        size_t indices_count, // size of indices
        size_t prefix_count, // size of prefix sum
        float u
        // curandState *states
    ) { 
        auto const tile_32 = cg::tiled_partition<kWarpSize>( cg::this_thread_block() );

        // Allocate shared memory for 2 warps worth of threads
        // Requires the number of threads in a block to be 64
        __shared__ float s_warp_0[kTRI];
        __shared__ float s_warp_1[kTRI];

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        // curandState rng = states[idx];
        
        for (size_t tid = idx; tid < indices_count; tid += stride) {
            // float u = curand_uniform(&rng);

            // If this thread is apart of the first warp use the first block of memory
            if (threadIdx.x < kWarpSize) {
                system_resample_up_warp(
                    tile_32,
                    tid,
                    u,
                    indices,
                    prefix_sum,
                    indices_count,
                    prefix_count, 
                    s_warp_0
                );
            }
            // If this thread is apart of the second warp use the second block of memory
            else {
                system_resample_up_warp(
                    tile_32,
                    tid,
                    u,
                    indices,
                    prefix_sum,
                    indices_count,
                    prefix_count, 
                    s_warp_1
                );
            }
        }
    }

    template <typename T>
    __device__ void system_resample_down_warp(
        T * into,
        T * from, 
        cg::thread_block_tile<kWarpSize> const &tile_32,
        size_t const tid,
        float u,
        size_t *indices,
        float *prefix_sum,
        size_t const into_count,
        size_t const from_count,
        float *shared
    ) {
        auto const t = tile_32.thread_rank(); 

        int l = 0;
        bool mask = false; 
        float const weight_norm = 1.0 / prefix_sum[from_count - 1];
        float const norm = 1.0 / static_cast<float>(into_count);

        // Compute the correct offset into the prefix sum
        float const idx_float = static_cast<float>(tid) * norm * static_cast<float>(from_count);
        // Static idx to remember the starting position of the index value
        size_t const idx = static_cast<size_t>(idx_float);
        // Mutable idx to modify the ancetor of this particle
        size_t mut_idx = indices[tid];
        float const draw = (u + idx_float) / static_cast<float>(from_count);

        // Load the first 64 values into shared memory, normalising the weights
        if (idx > kWarpSize) {
            shared[t] = prefix_sum[idx - kWarpSize] * weight_norm; 
            shared[t + kWarpSize] = prefix_sum[idx] * weight_norm;
        }

        tile_32.sync();

        while (!tile_32.all( mask )) {
            // If 'idx - l' is not in the first 'kTRI' values of the prefix sum
            // we can use the values stored in shared memory
            if (idx > kTRI + l) {
        
    #pragma unroll kWarpSize
                // Skip 'i = 0' as we already checked that in the 'up' kernel
                for (int i = 1; i < kWarpSize + 1; i++) {
                    // Iterate backwards over the shared values
                    mask = shared[t + kWarpSize - i] < draw;

                    if (!mask) {
                        // If there are more particles to the right of this particle that have been
                        // cloned then we shift this particle's ancestor to the left
                        mut_idx -= 1; 
                    }
                }

                l += kWarpSize;
                // First shift the [-(l + 1) *kWarpSize, -l] elements to the right 
                shared[t + kWarpSize] = shared[t];
                // Load the new [-(l + 2) * kWarpSize, -(l + 1) * kWarpSize] elements from global memory
                shared[t] = prefix_sum[idx - kWarpSize - l] * weight_norm; 
            }
            else {
                while (!mask) {
                    if (idx < l) {
                        mask = true;
                    }
                    else {
                        mask = prefix_sum[idx - (l + 1)] * weight_norm < draw; 
                    }

                    if (!mask) {
                        // If there are more particles to the right of this particle that have been
                        // cloned then we shift this particle's ancestor to the left
                        mut_idx -= 1;
                    }

                    l += 1;
                }

                mask = true; 
            }

            tile_32.sync( );
        }

        indices[tid] = mut_idx;
        into[tid] = from[mut_idx];
    }

    template <typename T>
    __global__ void systematic_resample_down_shared_prefetch(
        T * into,
        T * from, 
        size_t *indices,
        float *prefix_sum,
        size_t into_count, // size of indices
        size_t from_count, // size of prefix sum
        float u
        // curandState *states
    ) { 
        auto const tile_32 = cg::tiled_partition<kWarpSize>( cg::this_thread_block() );

        // Allocate shared memory for 2 warps worth of threads
        // Requires the number of threads in a block to be 64
        __shared__ float s_warp_0[kTRI];
        __shared__ float s_warp_1[kTRI];

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        // curandState rng = states[idx];
        
        for (size_t tid = idx; tid < into_count; tid += stride) {
            // float u = curand_uniform(&rng);

            // If this thread is apart of the first warp use the first block of memory
            if (threadIdx.x < kWarpSize) {
                system_resample_down_warp(
                    into,
                    from,
                    tile_32,
                    tid,
                    u,
                    indices,
                    prefix_sum,
                    into_count,
                    from_count, 
                    s_warp_0
                );
            }
            // If this thread is apart of the second warp use the second block of memory
            else {
                system_resample_down_warp(
                    into,
                    from,
                    tile_32,
                    tid,
                    u,
                    indices,
                    prefix_sum,
                    into_count,
                    from_count, 
                    s_warp_1
                );
            }
        }

        // states[idx] = rng;
    }


    __global__ void naive_systematic_resample_loop_1(
        size_t *indices,
        float *weights,
        float u,
        size_t into_count,
        size_t from_count
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float weight_norm = 1.0 / weights[from_count - 1];
        float norm = 1.0 / __uint2float_rn(into_count);

        for (size_t i = idx; i < into_count; i += stride) {
            size_t idx = i;
            size_t l = 0;
            float u_i = (__uint2float_rn(i) + u) * norm; 
            bool mask = true;

            while (mask) {
                if (i > (into_count - l)) {
                    mask = false; 
                }
                else {
                    mask = weights[i + l] * weight_norm < u_i;
                }

                if (mask) {
                    idx += 1;
                }

                l += 1;
            }

            indices[i] = idx;
        }
    }

    template<typename T>
    __global__ void naive_systematic_resample_loop_2(
        T * into,
        T * from, 
        size_t *indices,
        float *weights,
        float u,
        size_t into_count,
        size_t from_count
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float weight_norm = 1.0 / weights[from_count - 1];
        float norm = 1.0 / __uint2float_rn(into_count);

        for (size_t i = idx; i < into_count; i += stride) {
            size_t idx = indices[i];
            size_t l = 1;
            float u_i = (__uint2float_rn(i) + u) * norm; 
            bool mask = false;

            while (!mask) {
                if (i < l) {
                    mask = true; 
                }
                else {
                    mask = weights[i - l] * weight_norm < u_i;
                }

                if (!mask) {
                    idx -= 1;
                }

                l += 1;
            }
            indices[i] = idx;
            into[i] = from[idx];
        }
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class NicelySystematicResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t size;
        thrust::device_ptr<size_t> indices;

    public:
        NicelySystematicResampler() : size(0)
        {
            size_t *ptr = nullptr;
            indices = thrust::device_ptr<size_t>(ptr);
        }

        NicelySystematicResampler(size_t size) : size(size)
        {
            size_t *ptr;

            gpuErrchk(cudaMalloc(&ptr, size * sizeof(size_t)));
            indices = thrust::device_ptr<size_t>(ptr);
        }

        ~NicelySystematicResampler()
        {
            gpuErrchk(cudaFree(to_raw(indices)));
        }

        NicelySystematicResampler(const NicelySystematicResampler &other) : size(other.size)
        {
            // Other had not allocated any buffer so do not do anything
            if (size == 0)
            {
                indices = thrust::device_ptr<size_t>(nullptr);
                return;
            }

            // Allocate a buffer of correct size
            size_t *ptr;
            gpuErrchk(cudaMalloc(&ptr, other.size * sizeof(size_t)));

            // Copy other indices to this
            gpuErrchk(cudaMemcpy(
                to_raw(indices),
                to_raw(other.indices),
                size * sizeof(size_t),
                cudaMemcpyDeviceToDevice));

            indices = thrust::device_ptr<size_t>(ptr);
        }

        NicelySystematicResampler &operator=(const NicelySystematicResampler &other)
        {
            // Other did not have any allocated buffer
            if (other.size == 0)
            {
                gpuErrchk(cudaFree(to_raw(indices)));
                indices = thrust::device_ptr<size_t>(nullptr);
                size = 0;
                return *this;
            }
            // Buffer sizes did not match so have to reallocate
            else if (size != other.size)
            {
                gpuErrchk(cudaFree(to_raw(indices)));
                size_t *ptr;
                gpuErrchk(cudaMalloc(&ptr, other.size * sizeof(size_t)));
                indices = thrust::device_ptr<size_t>(ptr);
            }

            size = other.size;

            // Copy the other indices to this
            gpuErrchk(cudaMemcpy(
                to_raw(indices),
                to_raw(other.indices),
                size * sizeof(size_t),
                cudaMemcpyDeviceToDevice));

            return *this;
        }

        NicelySystematicResampler(NicelySystematicResampler &&other) : indices(other.indices)
        {
            other.indices = thrust::device_ptr<size_t>(nullptr);
            other.size = 0;
        }

        NicelySystematicResampler &operator=(NicelySystematicResampler &&other)
        {
            gpuErrchk(cudaFree(to_raw(indices)));
            indices = other.indices;
            size = other.size;
            other.indices = thrust::device_ptr<size_t>(nullptr);
            other.size = 0;

            return this;
        }

        // If count > size then it reallocates enough space on the gpu to hold the indices
        // otherwise it does nothing. Does not copy any existing indices to the new buffer
        void reserve(size_t count)
        {
            if (count <= size)
            {
                return;
            }

            gpuErrchk(cudaFree(to_raw(indices)));
            size_t *ptr;
            gpuErrchk(cudaMalloc(&ptr, count * sizeof(size_t)));
            indices = thrust::device_ptr<size_t>(ptr);
            size = count;
        }

        size_t *raw_indices()
        {
            return to_raw(indices);
        }

        void resample(
            T *into,
            T *from,
            float *weights,
            size_t *ancestors,
            size_t into_count,
            size_t from_count,
            curandState *states,
            RNG &rng)
        {
            // this->reserve(into_count);

            std::uniform_real_distribution<float> dist(0.0, 1.0);
            thrust::device_ptr<float> start(weights);
            thrust::device_ptr<float> end = start + from_count;

            thrust::inclusive_scan(start, end, start);
            gpuErrchk( cudaDeviceSynchronize() );
            float u = dist(rng);
            // float u = 0.0;

            systematic_resample_up_shared_prefetch<<<BLOCKS, 64>>>(
                ancestors,
                weights,
                into_count,
                from_count,
                u
                // states
            );

            // naive_systematic_resample_loop_1<<<BLOCKS, THREADS>>>(
            //     ancestors,
            //     weights,
            //     u,
            //     into_count, 
            //     from_count
            // );

            
            gpuErrchk( cudaDeviceSynchronize() );

            systematic_resample_down_shared_prefetch<<<BLOCKS, 64>>>(
                into,
                from,
                ancestors,
                weights,
                into_count,
                from_count,
                u
                // states
            );

            // naive_systematic_resample_loop_2<<<BLOCKS, THREADS>>>(
            //     into,
            //     from, 
            //     ancestors,
            //     weights,
            //     u,
            //     into_count, 
            //     from_count
            // );

            gpuErrchk( cudaDeviceSynchronize() );
        }

        void resample(
            T *into,
            T *from,
            float *weights,
            size_t into_count,
            size_t from_count,
            curandState *states,
            RNG &rng)
        {
            this->reserve(into_count);

            this->resample(
                into, 
                from,
                weights,
                to_raw(indices),
                into_count,
                from_count,
                states,
                rng
            );
        }
    };
}