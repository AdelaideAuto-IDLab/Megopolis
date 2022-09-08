#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <array>
#include <random>
#include <stdexcept>

#include "helper.cuh"
#include "random.cuh"
#include "resampler.cuh"

namespace resampling
{
    // Ouputs a random offset in range [0, 'max') for value in 'offsets'
    __global__ void generate_offsets_kernel(
        size_t *offsets,
        size_t size,
        size_t max,
        curandState *states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = *(states + idx);

        for (size_t i = idx; i < size; i += stride)
        {
            float j = curand_uniform(&rand_state);

            offsets[i] = __float2int_ru(j * max) - 1;
        }

        *(states + idx) = rand_state;
    }

    template <typename T>
    __global__ void megopolis_resample(
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

        // Get thread local rng state
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {
            float weight = weights[i];
            size_t selected = i;

            for (size_t j = 0; j < iters; j++)
            {
                size_t offset = offsets[j];
                float u = curand_uniform(&rand_state);

                // Get next index from current offset
                size_t index = (i + offset) % from_count;
                float test_weight = weights[index];

                if (u < (test_weight / weight))
                {
                    weight = test_weight;
                    selected = index;
                }
            }

            into[i] = from[selected];
        }

        // Save thread local rng state
        states[idx] = rand_state;
    }

    template <typename T>
    __global__ void megopolis_resample_save_ancestors(
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
        // Get thread local rng state
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {
            float weight = weights[i];
            size_t selected = i;

            for (size_t j = 0; j < iters; j++)
            {
                size_t offset = offsets[j];
                float u = curand_uniform(&rand_state);

                size_t index = (i + offset) % from_count;
                float test_weight = weights[index];

                if (u < (test_weight / weight))
                {
                    weight = test_weight;
                    selected = index;
                }
            }

            ancestors[i] = selected;
            into[i] = from[selected];
        }

        // Save thread local rng state
        states[idx] = rand_state;
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class MegopolisResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t buffer_size;
        std::vector<size_t> host_offsets;
        thrust::device_ptr<size_t> offsets;

    public:
        size_t iters;

        MegopolisResampler() : iters(0), buffer_size(0)
        {
            size_t *ptr = nullptr;
            offsets = thrust::device_ptr<size_t>(ptr);
        }

        ~MegopolisResampler()
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
        }

        MegopolisResampler(const MegopolisResampler<BLOCKS, THREADS, T, RNG> &other) : iters(other.iters),
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

        MegopolisResampler &operator=(const MegopolisResampler<BLOCKS, THREADS, T, RNG> &other)
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

        MegopolisResampler(MegopolisResampler &&other) : iters(other.iters),
                                                         buffer_size(other.buffer_size),
                                                         host_offsets(std::move(other.host_offsets)),
                                                         offsets(other.offsets)
        {
            other.buffer_size = 0;
            other.offsets = thrust::device_ptr<size_t>(nullptr);
        }

        MegopolisResampler &operator=(MegopolisResampler &&other)
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
            iters = other.iters;
            buffer_size = other.buffer_size;
            host_offsets = std::move(other.host_offsets);
            offsets = other.offsets;

            other.buffer_size = 0;
            other.offsets = thrust::device_ptr<size_t>(nullptr);

            return this;
        }

        // If 'amount' > 'buffer_size' then the buffers are reallocated.
        // The device offsets are not copied to the newly allocated buffer
        void reserve(size_t amount)
        {
            try
            {
                host_offsets.reserve(amount);
            }
            catch (const std::length_error &e)
            {
                // @Polish: Improve error handling here
                printf("Reserve length error\n");

                throw e;
            }

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
        // to be cloned and there should be w_max / w_sum * desired of those particles.
        // Desired is the amount of particles you are resampling into
        size_t determine_iters(float *weights, size_t count, size_t desired, float est_ratio, float epsilon)
        {
            size_t to_sum = (size_t)(est_ratio * (float)count);
            to_sum = std::min(to_sum, count);

            float sum = thrust::reduce(thrust::device, weights, weights + to_sum);
            float *max_ptr = thrust::max_element(thrust::device, weights, weights + to_sum);
            float max;

            gpuErrchk(cudaMemcpy(&max, max_ptr, sizeof(float), cudaMemcpyDeviceToHost));

            float avg_weight = sum / (float)to_sum;

            return log10(epsilon) / log10(1.0 - avg_weight / max);
        }

        void set_iters(size_t iters)
        {
            this->iters = iters;
        }

        // Generates the list of random offsets in the range [0, max)
        void generate_offsets(RNG &rng, size_t max)
        {
            this->reserve(iters);
            host_offsets.clear();
            std::uniform_int_distribution<size_t> dist(0, max);

            for (size_t i = 0; i < iters; i++)
            {
                host_offsets.push_back(dist(rng));
            }

            gpuErrchk(cudaMemcpy(
                to_raw(offsets),
                host_offsets.data(),
                iters * sizeof(size_t),
                cudaMemcpyHostToDevice));
        }

        void generate_offsets_device(curandState *rng, size_t max)
        {
            this->reserve(iters);
            host_offsets.clear();

            std::pair<size_t, size_t> bt = determine_bt(BLOCKS, 5, THREADS, iters);
            size_t blocks = std::get<0>(bt);
            size_t threads = std::get<1>(bt);
            generate_offsets_kernel<<<blocks, threads>>>(
                to_raw(offsets),
                iters,
                max,
                rng);

            gpuErrchk(cudaDeviceSynchronize());
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
            this->generate_offsets(rng, from_count);
            megopolis_resample<<<BLOCKS, THREADS>>>(
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
            this->generate_offsets(rng, from_count);
            megopolis_resample_save_ancestors<<<BLOCKS, THREADS>>>(
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