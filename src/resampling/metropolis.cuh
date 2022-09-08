#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <stdio.h>
#include "helper.cuh"
#include "resampler.cuh"
#include "random.cuh"

namespace resampling
{
    __device__ size_t metropolis_find_ancestor(
        size_t iters,
        size_t start,
        float *weights,
        size_t count,
        curandState *rand_state)
    {
        size_t k = start;

        for (size_t i = 0; i < iters; i++)
        {
            float u = curand_uniform(rand_state);
            float j = curand_uniform(rand_state);

            int j_idx = __float2int_ru(j * count) - 1;

            float a = weights[j_idx];
            float b = weights[k];

            if (u < (a / b))
            {
                k = j_idx;
            }
        }

        return k;
    }

    template <typename P>
    __global__ void metropolis_resample_ancestors_kernel(
        P *into,
        P *from,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        size_t iters,
        curandState *states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = *(states + idx);

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t k = metropolis_find_ancestor(iters, i, weights, from_count, &rand_state);
            into[i] = from[k];
            ancestors[i] = k;
        }

        *(states + idx) = rand_state;
    }

    template <typename P>
    __global__ void metropolis_resample_kernel(
        P *into,
        P *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        size_t iters,
        curandState *states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = *(states + idx);

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t k = metropolis_find_ancestor(iters, i, weights, from_count, &rand_state);
            into[i] = from[k];
        }

        *(states + idx) = rand_state;
    }

    template <size_t BLOCKS, size_t THREADS, typename T>
    void metropolis_resample_ancestors(
        T *into,
        T *from,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        size_t iters,
        curandState *states)
    {
        metropolis_resample_ancestors_kernel<<<BLOCKS, THREADS>>>(
            into,
            from,
            weights,
            ancestors,
            into_count,
            from_count,
            iters,
            states);
        gpuErrchk(cudaDeviceSynchronize());
    }

    template <size_t BLOCKS, size_t THREADS, typename T>
    void metropolis_resample(
        T *into,
        T *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        size_t iters,
        curandState *states)
    {
        metropolis_resample_kernel<<<BLOCKS, THREADS>>>(
            into,
            from,
            weights,
            into_count,
            from_count,
            iters,
            states);
        gpuErrchk(cudaDeviceSynchronize());
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class MetropolisResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t iters;

    public:
        void set_iters(size_t iters)
        {
            this->iters = iters;
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
            metropolis_resample<BLOCKS, THREADS>(
                into,
                from,
                weights,
                into_count,
                from_count,
                iters,
                states);
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
            metropolis_resample_ancestors<BLOCKS, THREADS>(
                into,
                from,
                weights,
                ancestors,
                into_count,
                from_count,
                iters,
                states);
        }
    };
}
