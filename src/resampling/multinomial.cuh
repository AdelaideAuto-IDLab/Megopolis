#pragma once

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <stdio.h>
#include "resampler.cuh"
#include "random.cuh"

namespace resampling
{
    // Generates a random value in the range [0, 1) and Performs a binary search on 'weights'
    // to find an ancestor where the ancestors weight is the likelihood of choosing that particle
    __device__ size_t multinomial_find_ancestor(
        float *weights,
        size_t count,
        float max,
        curandState *state)
    {
        size_t l = 0;
        size_t r = count - 1;
        float u = curand_uniform(state) * max;

        // Perform a binary search to find the index of the ancestor
        while (l != r)
        {
            size_t j = (l + r) / 2;
            float weight = weights[j];
            if (u < weight)
            {
                r = j;
                continue;
            }

            weight = weights[j + 1];
            if (u < weight)
            {
                l = j;
                break;
            }
            else
            {
                l = j + 1;
            }
        }

        return l;
    }

    template <typename T>
    __global__ void multinomial_resample_ancestors_kernel(
        T *from,
        T *into,
        float *weights,
        float end,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        curandState *states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = *(states + idx);

        // Calculate Wb + wb
        float max = weights[from_count - 1] + end;

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t l = multinomial_find_ancestor(weights, from_count, max, &rand_state);
            into[i] = from[l];
            ancestors[i] = l;
        }

        *(states + idx) = rand_state;
    }

    template <typename T>
    __global__ void multinomial_resample_kernel(
        T *from,
        T *into,
        float *weights,
        float end,
        size_t into_count,
        size_t from_count,
        curandState *states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = *(states + idx);

        // Calculate Wb + wb
        float max = weights[from_count - 1] + end;

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t l = multinomial_find_ancestor(weights, from_count, max, &rand_state);
            into[i] = from[l];
        }

        *(states + idx) = rand_state;
    }

    template <size_t BLOCKS, size_t THREADS, typename T>
    void multinomial_resample_ancestors(
        T *from,
        T *into,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        curandState *states)
    {
        float last = 0.0;
        gpuErrchk(cudaMemcpy(&last, &weights[from_count - 1], sizeof(float), cudaMemcpyDeviceToHost));
        thrust::device_ptr<float> start(weights);
        thrust::device_ptr<float> end = start + from_count;
        thrust::exclusive_scan(start, end, start);
        gpuErrchk(cudaDeviceSynchronize());

        multinomial_resample_ancestors_kernel<<<BLOCKS, THREADS>>>(
            from,
            into,
            weights,
            last,
            ancestors,
            into_count,
            from_count,
            states);
        gpuErrchk(cudaDeviceSynchronize());
    }

    template <size_t BLOCKS, size_t THREADS, typename T>
    void multinomial_resample(
        T *into,
        T *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        curandState *states)
    {
        float last = 0.0;
        gpuErrchk(cudaMemcpy(&last, &weights[from_count - 1], sizeof(float), cudaMemcpyDeviceToHost));
        thrust::device_ptr<float> start(weights);
        thrust::device_ptr<float> end = start + from_count;
        thrust::exclusive_scan(start, end, start);

        multinomial_resample_kernel<<<BLOCKS, THREADS>>>(
            from,
            into,
            weights,
            last,
            into_count,
            from_count,
            states);
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Resampler that uses prefix sum of weight to perform binary searches to find particle ancestors
    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class MultinomialResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
    public:
        void resample(
            T *into,
            T *from,
            float *weights,
            size_t into_count,
            size_t from_count,
            curandState *states,
            RNG &rng)
        {
            multinomial_resample<BLOCKS, THREADS>(
                into,
                from,
                weights,
                into_count,
                from_count,
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
            multinomial_resample_ancestors<BLOCKS, THREADS>(
                into,
                from,
                weights,
                ancestors,
                into_count,
                from_count,
                states);
        }
    };
}