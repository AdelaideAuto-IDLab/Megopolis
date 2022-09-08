#pragma once

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <random>
#include <stdio.h>
#include <assert.h>
#include "helper.cuh"
#include "resampler.cuh"
#include "random.cuh"

namespace resampling
{
    __device__ size_t get_start(
        size_t idx,
        size_t *indices)
    {
        if (idx == 0)
        {
            return 0;
        }
        else
        {
            return indices[idx - 1];
        }
    }

    // Copies 'value' into 'into' 'amount' times starting at position 'start'.
    // If it would copy into a position greater than 'maximum' then it stops copying
    template <typename T>
    __device__ void duplicate_value(
        T value,
        T *into,
        size_t start,
        size_t amount,
        size_t maximum)
    {
        for (size_t i = 0; i < amount; i++)
        {
            size_t idx = start + i;
            if (idx >= maximum)
            {
                break;
            }

            into[idx] = value;
        }
    }

    // For all i in 0 to 'from_count', the i-th value in 'from' is copied to 'into' at positions
    // [indices[i], indices[i + 1])
    // For each value in 'from' the number of copies made is stored in 'offspring[i]'
    template <typename T>
    __global__ void systematic_resample_copy_offspring_kernel(
        T *from,
        T *into,
        size_t *indices,
        unsigned long long *offspring,
        size_t into_count,
        size_t from_count)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < from_count; i += stride)
        {
            int start = get_start(i, indices);
            int current = indices[i];
            int copies = max(0, current - start);

            offspring[i] = copies;
            T to_copy = from[i];

            duplicate_value(to_copy, into, start, copies, into_count);
        }
    }

    // For all i in 0 to 'from_count', the i-th value in 'from' is copied to 'into' at positions
    // [indices[i], indices[i + 1])
    template <typename T>
    __global__ void systematic_resample_copy_kernel(
        T *from,
        T *into,
        size_t *indices,
        size_t into_count,
        size_t from_count)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < from_count; i += stride)
        {
            int start = get_start(i, indices);
            int current = indices[i];
            int copies = max(0, current - start);
            T to_copy = from[i];
            duplicate_value(to_copy, into, start, copies, into_count);
        }
    }

    // 'weights' must be an input buffer containing a prefix sum of weights.
    // 'indices' stores the start position in the output buffer that a given particle should
    // be copied in to.
    // 'u' is a random offset
    __global__ void systematic_resample_indices_kernel(
        float *weights,
        size_t *indices,
        float u,
        size_t into_count,
        size_t from_count)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float w_p = 1.0 / weights[from_count - 1] * __uint2float_rn(into_count);

        for (size_t i = idx; i < from_count; i += stride)
        {
            float weight = weights[i];
            int value = __float2int_rd(weight * w_p - u) + 1;
            indices[i] = (size_t)value;
        }
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG>
    void systematic_resample_offspring(
        T *from,
        T *into,
        float *weights,
        size_t *indices,
        unsigned long long *offspring,
        size_t into_count,
        size_t from_count,
        RNG &rng)
    {
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        thrust::device_ptr<float> start(weights);
        thrust::device_ptr<float> end = start + from_count;

        thrust::inclusive_scan(start, end, start);
        gpuErrchk(cudaDeviceSynchronize());
        float u = dist(rng);

        systematic_resample_indices_kernel<<<BLOCKS, THREADS>>>(
            weights,
            indices,
            u,
            into_count,
            from_count);

        gpuErrchk(cudaDeviceSynchronize());

        systematic_resample_copy_offspring_kernel<<<BLOCKS, THREADS>>>(
            from,
            into,
            indices,
            offspring,
            into_count,
            from_count);

        gpuErrchk(cudaDeviceSynchronize());
    }

    // Size of into must be >= into count
    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG>
    void systematic_resample(
        T *from,
        T *into,
        float *weights,
        size_t *indices,
        size_t into_count,
        size_t from_count,
        RNG &rng)
    {
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        thrust::device_ptr<float> start(weights);
        thrust::device_ptr<float> end = start + from_count;

        thrust::inclusive_scan(start, end, start);
        gpuErrchk(cudaDeviceSynchronize());
        float u = dist(rng);
        u = 0.0;

        // For each particle calculate the number of copies to output
        systematic_resample_indices_kernel<<<BLOCKS, THREADS>>>(
            weights,
            indices,
            u,
            into_count,
            from_count);

        gpuErrchk(cudaDeviceSynchronize());

        // Create particle copies from the indices buffer
        systematic_resample_copy_kernel<<<BLOCKS, THREADS>>>(
            from,
            into,
            indices,
            into_count,
            from_count);

        gpuErrchk(cudaDeviceSynchronize());
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class SystematicResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t size;
        thrust::device_ptr<size_t> indices;

    public:
        SystematicResampler() : size(0)
        {
            size_t *ptr = nullptr;
            indices = thrust::device_ptr<size_t>(ptr);
        }

        SystematicResampler(size_t size) : size(size)
        {
            size_t *ptr;

            gpuErrchk(cudaMalloc(&ptr, size * sizeof(size_t)));
            indices = thrust::device_ptr<size_t>(ptr);
        }

        ~SystematicResampler()
        {
            gpuErrchk(cudaFree(to_raw(indices)));
        }

        SystematicResampler(const SystematicResampler &other) : size(other.size)
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

        SystematicResampler &operator=(const SystematicResampler &other)
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

        SystematicResampler(SystematicResampler &&other) : indices(other.indices)
        {
            other.indices = thrust::device_ptr<size_t>(nullptr);
            other.size = 0;
        }

        SystematicResampler &operator=(SystematicResampler &&other)
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

        // Overloaded resample function without gpu curandStates
        void resample(
            T *into,
            T *from,
            float *weights,
            size_t into_count,
            size_t from_count,
            RNG &rng)
        {
            this->reserve(from_count);

            systematic_resample<BLOCKS, THREADS>(
                from,
                into,
                weights,
                to_raw(indices),
                into_count,
                from_count,
                rng);
        }

        void resample(
            T *into,
            T *from,
            float *weights,
            size_t into_count,
            size_t from_count,
            curandState *rand,
            RNG &rng)
        {
            this->resample(
                into,
                from,
                weights,
                into_count,
                from_count,
                rng);
        }

        void resample_save_offspring(
            T *into,
            T *from,
            float *weights,
            unsigned long long *offspring,
            size_t into_count,
            size_t from_count,
            RNG &rng)
        {
            this->reserve(from_count);

            systematic_resample_offspring<BLOCKS, THREADS>(
                from,
                into,
                weights,
                to_raw(indices),
                offspring,
                into_count,
                from_count,
                rng);
        }
    };
}
