#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <random>
#include <stdio.h>
#include "helper.cuh"
#include "resampler.cuh"
#include "random.cuh"

namespace resampling
{
    __device__ size_t metropolis_c2_find_ancestor(
        size_t start,
        float *weights,
        size_t iters,
        size_t segments,
        size_t segment_size,
        curandState *segment_state,
        curandState *rand_state)
    {
        size_t k = start;
        float denom = weights[k];

        for (size_t i = 0; i < iters; i++)
        {
            float u = curand_uniform(rand_state);
            float j = curand_uniform(rand_state);

            // Calculate the segment to use
            float s_val = curand_uniform(segment_state);
            int segment = __float2int_ru(s_val * segments) - 1;
            size_t offset = segment * segment_size;

            // Find the index of a particle in a given segment
            int j_idx = offset + __float2int_ru(j * segment_size) - 1;

            float num = weights[j_idx];

            if (u < (num / denom))
            {
                k = j_idx;
                denom = num;
            }
        }

        return k;
    }

    template <typename P>
    __global__ void metropolis_c2_resample_ancestors_kernel(
        P *into,
        P *from,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        size_t iters,
        size_t segments,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;

        // Calculate segment size. 'from_count' must be a multiple of segments
        size_t segment_size = from_count / segments;

        // Get the warp local rng state so each thread in a warp uses the same rng state to calculate
        // the same random segment
        size_t warp_id = idx / 32;
        curandState segment_state = segment_states[warp_id];

        // Get thread local rng state
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {

            size_t k = metropolis_c2_find_ancestor(
                i,
                weights,
                iters,
                segments,
                segment_size,
                &segment_state,
                &rand_state);
            into[i] = from[k];
            ancestors[i] = k;
        }

        // Save thread local rng state
        states[idx] = rand_state;

        // Save the warp local rng state only if this thread is the first thread in a warp
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <typename P>
    __global__ void metropolis_c2_resample_kernel(
        P *into,
        P *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        size_t iters,
        size_t segments,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;

        // Calculate segment size. 'from_count' must be a multiple of segments
        size_t segment_size = from_count / segments;

        // Get the warp local rng state so each thread in a warp uses the same rng state to calculate
        // the same random segment
        size_t warp_id = idx / 32;
        curandState segment_state = segment_states[warp_id];

        // Get thread local rng state
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t k = metropolis_c2_find_ancestor(
                i,
                weights,
                iters,
                segments,
                segment_size,
                &segment_state,
                &rand_state);
            into[i] = from[k];
        }

        // Save thread local rng state
        states[idx] = rand_state;

        // Save the warp local rng state only if this thread is the first thread in a warp
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class MetropolisC2Resampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t segment_size;
        Box<curandState> segment_rng;
        size_t iters;

    public:
        MetropolisC2Resampler(int seed)
        {
            curandState *ptr;
            gpuErrchk(cudaMalloc(&ptr, BLOCKS * THREADS / 32 * sizeof(curandState)));
            setup_curand<<<BLOCKS, THREADS>>>(
                seed,
                ptr,
                BLOCKS * THREADS / 32);
            gpuErrchk(cudaDeviceSynchronize());
            segment_rng = Box<curandState>(ptr);
        }

        void set_iters(size_t iters)
        {
            this->iters = iters;
        }

        void set_segment_size(size_t size)
        {
            this->segment_size = size;
        }

        size_t determine_iters(float *weights, size_t count, size_t desired, float epsilon)
        {
            float sum = thrust::reduce(thrust::device, weights, weights + count);
            float *max_ptr = thrust::max_element(thrust::device, weights, weights + count);
            float max;

            gpuErrchk(cudaMemcpy(&max, max_ptr, sizeof(float), cudaMemcpyDeviceToHost));

            float avg_weight = sum / (float)count;
            return log10(epsilon) / log10(1.0 - avg_weight / max);
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
            //
            size_t ssize = std::min(segment_size, from_count);
            size_t segments = from_count / ssize;
            metropolis_c2_resample_kernel<<<BLOCKS, THREADS>>>(
                into,
                from,
                weights,
                into_count,
                from_count,
                iters,
                segments,
                states,
                to_raw(segment_rng.get()));

            gpuErrchk(cudaDeviceSynchronize());
        }

        void resample_save_ancestors(
            T *into,
            T *from,
            float *weights,
            size_t *ancestors,
            size_t into_count,
            size_t from_count,
            curandState *states,
            RNG &rng)
        {
            size_t ssize = std::min(segment_size, from_count);
            size_t segments = from_count / ssize;
            metropolis_c2_resample_ancestors_kernel<<<BLOCKS, THREADS>>>(
                into,
                from,
                weights,
                ancestors,
                into_count,
                from_count,
                iters,
                segments,
                states,
                to_raw(segment_rng.get()));

            gpuErrchk(cudaDeviceSynchronize());
        }
    };
}
