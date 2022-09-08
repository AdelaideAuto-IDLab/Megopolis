#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <array>
#include <random>
#include "helper.cuh"
#include "random.cuh"
#include "resampler.cuh"

namespace resampling
{
    __device__ size_t megopolis_aligned_ancestor(
        size_t i,
        size_t iters,
        size_t *offsets,
        size_t from_count,
        float *weights,
        curandState *rand_state)
    {
        float weight = weights[i];
        size_t selected = i;

        for (size_t j = 0; j < iters; j++)
        {
            size_t offset = offsets[j];
            float u = curand_uniform(rand_state);

            // Fast mod 8
            size_t overshoot = offset & 0x7;
            size_t index = i + offset;
            // Finding the 32-byte aligned block
            size_t block = (index - overshoot) >> 3 << 3;
            index = (index - block) & 0x7;
            index = index + block;
            index = index % from_count;
            float test_weight = weights[index];

            if (u < (test_weight / weight))
            {
                weight = test_weight;
                selected = index;
            }
        }

        return selected;
    }

    template <typename T>
    __global__ void megopolis_aligned_resample(
        size_t *offsets,
        T *into,
        T *from,
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
            size_t selected = megopolis_aligned_ancestor(
                i,
                iters,
                offsets,
                from_count,
                weights,
                &rand_state);

            into[i] = from[selected];
        }

        *(states + idx) = rand_state;
    }

    template <typename T>
    __global__ void megopolis_aligned_resample_save_ancestors(
        size_t *offsets,
        T *into,
        T *from,
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
            size_t selected = megopolis_aligned_ancestor(
                i,
                iters,
                offsets,
                from_count,
                weights,
                &rand_state);

            ancestors[i] = selected;
            into[i] = from[selected];
        }

        *(states + idx) = rand_state;
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class MegopolisAlignedResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t buffer_size;
        std::vector<size_t> host_offsets;
        thrust::device_ptr<size_t> offsets;

    public:
        size_t iters;

        MegopolisAlignedResampler() : iters(0), buffer_size(0)
        {
            size_t *ptr = nullptr;
            offsets = thrust::device_ptr<size_t>(ptr);
        }

        ~MegopolisAlignedResampler()
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
        }

        MegopolisAlignedResampler(const MegopolisAlignedResampler<BLOCKS, THREADS, T, RNG> &other) : iters(other.iters),
                                                                                                     buffer_size(other.buffer_size),
                                                                                                     host_offsets(other.host_offsets)
        {
            size_t *ptr = nullptr;
            if (buffer_size > 0)
            {
                gpuErrchk(cudaMalloc(&ptr, buffer_size * sizeof(size_t)));
                gpuErrchk(cudaMemcpy(
                    ptr,
                    to_raw(other.offsets),
                    buffer_size * sizeof(size_t),
                    cudaMemcpyDeviceToDevice));
            }

            offsets = thrust::device_ptr<size_t>(ptr);
        }

        MegopolisAlignedResampler &operator=(const MegopolisAlignedResampler<BLOCKS, THREADS, T, RNG> &other)
        {
            host_offsets = other.host_offsets;
            iters = other.iters;

            if (buffer_size < other.iters)
            {
                buffer_size = other.iters;
                size_t *ptr;
                gpuErrchk(cudaFree(to_raw(offsets)));
                gpuErrchk(cudaMalloc(&ptr, buffer_size * sizeof(size_t)));
                offsets = thrust::device_ptr<size_t>(ptr);
            }

            if (buffer_size > 0)
            {
                gpuErrchk(cudaMemcpy(
                    to_raw(offsets),
                    to_raw(other.offsets),
                    other.iters * sizeof(size_t),
                    cudaMemcpyDeviceToDevice));
            }

            return *this;
        }

        MegopolisAlignedResampler(MegopolisAlignedResampler &&other) : iters(other.iters),
                                                                       buffer_size(other.buffer_size),
                                                                       host_offsets(other.host_offsets),
                                                                       offsets(other.offsets)
        {
            other.offsets = thrust::device_ptr<size_t>(nullptr);
        }

        MegopolisAlignedResampler &operator=(MegopolisAlignedResampler &&other)
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
            iters = other.iters;
            buffer_size = other.buffer_size;
            host_offsets = other.host_offsets;
            offsets = other.offsets;
            other.offsets = thrust::device_ptr<size_t>(nullptr);

            return this;
        }

        void reserve(size_t amount)
        {
            host_offsets.reserve(amount);

            if (buffer_size < amount)
            {
                buffer_size = amount;
                size_t *ptr;
                gpuErrchk(cudaFree(to_raw(offsets)));
                gpuErrchk(cudaMalloc(&ptr, buffer_size * sizeof(size_t)));
                offsets = thrust::device_ptr<size_t>(ptr);
            }
        }

        // Returns w_max / w_sum * desired as at each iteration you would expect the particle of w_max
        // to be cloned and there should be w_max / w_sum * desired of those particles
        // desired is the amount of particles you are sampling into
        size_t determine_iters(float *weights, size_t count, size_t desired, bool method)
        {
            float sum = thrust::reduce(thrust::device, weights, weights + count);
            float *max_ptr = thrust::max_element(thrust::device, weights, weights + count);
            float max;

            gpuErrchk(cudaMemcpy(&max, max_ptr, sizeof(float), cudaMemcpyDeviceToHost));

            if (method)
            {
                float max_copies = max / sum * float(desired);
                float size = desired;
                float size_m1 = size - 1;
                float upper = std::log((size - max_copies) / size_m1);
                float lower = std::log(size_m1 / size);
                return upper / lower;
            }
            else
            {
                float avg_weight = sum / (float)count;

                return log10(0.01) / log10(1.0 - avg_weight / max);
            }
        }

        void set_iters(size_t iters)
        {
            this->iters = iters;
        }

        void generate_offsets(curandState *rng, RNG &other, size_t max)
        {
            this->reserve(iters);
            std::uniform_int_distribution<size_t> dist(0, max);

            for (size_t i = 0; i < iters; i++)
            {
                host_offsets.push_back(dist(other));
            }

            gpuErrchk(cudaMemcpy(
                to_raw(offsets),
                host_offsets.data(),
                iters * sizeof(size_t),
                cudaMemcpyHostToDevice));
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
            this->generate_offsets(states, rng, from_count);
            megopolis_aligned_resample<<<BLOCKS, THREADS>>>(
                to_raw(offsets),
                into,
                from,
                weights,
                into_count,
                from_count,
                iters,
                states);
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
            this->generate_offsets(states, rng, from_count);
            megopolis_aligned_resample_save_ancestors<<<BLOCKS, THREADS>>>(
                to_raw(offsets),
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
    };
}