#pragma once

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <random>
#include <stdio.h>
#include <assert.h>
#include "helper.cuh"
#include "resampler.cuh"
#include "random.cuh"

namespace resampling {
    __global__ void naive_stratified_resample_loop_1(
        float *weights,
        size_t *indices,
        size_t into_count,
        size_t from_count,
        curandState *states
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float weight_norm = 1.0 / weights[from_count - 1];
        float norm = 1.0 / __uint2float_rn(into_count);
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride) {
            size_t idx = i;
            size_t l = 0;
            float rng = curand_uniform(&rand_state);
            float u_i = (__uint2float_rn(i) + rng) * norm; 
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
    __global__ void naive_stratified_resample_loop_2(
        T * into,
        T * from, 
        float *weights,
        size_t *indices,
        size_t into_count,
        size_t from_count,
        curandState *states
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        float weight_norm = 1.0 / weights[from_count - 1];
        float norm = 1.0 / __uint2float_rn(into_count);
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride) {
            size_t idx = indices[i];
            size_t l = 1;
            float rng = curand_uniform(&rand_state);
            float u_i = (__uint2float_rn(i) + rng) * norm; 
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

        states[idx] = rand_state;
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class NaiveStratifiedResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t size;
        thrust::device_ptr<size_t> indices;

    public:
        NaiveStratifiedResampler() : size(0)
        {
            size_t *ptr = nullptr;
            indices = thrust::device_ptr<size_t>(ptr);
        }

        NaiveStratifiedResampler(size_t size) : size(size)
        {
            size_t *ptr;

            gpuErrchk(cudaMalloc(&ptr, size * sizeof(size_t)));
            indices = thrust::device_ptr<size_t>(ptr);
        }

        ~NaiveStratifiedResampler()
        {
            gpuErrchk(cudaFree(to_raw(indices)));
        }

        NaiveStratifiedResampler(const NaiveStratifiedResampler &other) : size(other.size)
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

        NaiveStratifiedResampler &operator=(const NaiveStratifiedResampler &other)
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

        NaiveStratifiedResampler(NaiveStratifiedResampler &&other) : indices(other.indices)
        {
            other.indices = thrust::device_ptr<size_t>(nullptr);
            other.size = 0;
        }

        NaiveStratifiedResampler &operator=(NaiveStratifiedResampler &&other)
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
            thrust::device_ptr<float> start(weights);
            thrust::device_ptr<float> end = start + from_count;

            thrust::inclusive_scan(start, end, start);
            gpuErrchk( cudaDeviceSynchronize() );

            naive_stratified_resample_loop_1<<<BLOCKS, THREADS>>>(
                weights,
                ancestors,
                into_count,
                from_count,
                states
            );

            naive_stratified_resample_loop_2<<<BLOCKS, THREADS>>>(
                into,
                from,
                weights,
                ancestors,
                into_count,
                from_count,
                states
            );

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
