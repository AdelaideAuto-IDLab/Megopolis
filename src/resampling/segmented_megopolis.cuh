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
#include "resampler.cuh"
#include "random.cuh"

namespace resampling
{
    __device__ size_t dyn_seg_megopolis_find_ancestor(
        size_t start,
        size_t *offsets,
        float *weights,
        size_t segment,
        size_t segment_size,
        size_t iters,
        curandState *rand_state)
    {
        size_t k = start;
        size_t offset = segment * segment_size;

        for (size_t i = 0; i < iters; i++)
        {
            float u = curand_uniform(rand_state);
            float j = (start + offsets[i]) % segment_size;

            int j_idx = offset + j;

            float a = weights[j_idx];
            float b = weights[k];

            if (u < (a / b))
            {
                k = j_idx;
            }
        }

        return k;
    }

    template <size_t LOOPS>
    __device__ size_t seg_megopolis_find_ancestor(
        size_t start,
        size_t *offsets,
        float *weights,
        size_t segment,
        size_t segment_size,
        curandState *rand_state)
    {
        size_t k = start;
        size_t offset = segment * segment_size;

        for (size_t i = 0; i < LOOPS; i++)
        {
            float u = curand_uniform(rand_state);
            float j = (start + offsets[i]) % segment_size;

            int j_idx = offset + j;

            float a = weights[j_idx];
            float b = weights[k];

            if (u < (a / b))
            {
                k = j_idx;
            }
        }

        return k;
    }

    template <typename T>
    __global__ void dyn_seg_megopolis_resample_save_ancestor(
        size_t *offsets,
        T *into,
        T *from,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        size_t segments,
        size_t iters,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t warp_id = idx / 32;
        size_t segment_size = from_count / segments;
        curandState segment_state = segment_states[warp_id];
        curandState rand_state = states[idx];
        float s_val = curand_uniform(&segment_state);
        int segment = __float2int_ru(s_val * segments) - 1;

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t k = dyn_seg_megopolis_find_ancestor(
                i,
                offsets,
                weights,
                segment,
                segment_size,
                iters,
                &rand_state);

            into[i] = from[k];
            ancestors[i] = k;
        }

        *(states + idx) = rand_state;
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <typename T>
    __global__ void dyn_seg_megopolis_resample(
        size_t *offsets,
        T *into,
        T *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        size_t segments,
        size_t iters,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t warp_id = idx / 32;
        size_t segment_size = from_count / segments;
        curandState segment_state = segment_states[warp_id];
        curandState rand_state = states[idx];
        float s_val = curand_uniform(&segment_state);
        int segment = __float2int_ru(s_val * segments) - 1;

        for (size_t i = idx; i < into_count; i += stride)
        {
            size_t k = dyn_seg_megopolis_find_ancestor(
                i,
                offsets,
                weights,
                segment,
                segment_size,
                iters,
                &rand_state);

            into[i] = from[k];
        }

        *(states + idx) = rand_state;
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <size_t I, typename T>
    __global__ void seg_megopolis_resample(
        size_t *offsets,
        T *into,
        T *from,
        float *weights,
        size_t into_count,
        size_t from_count,
        size_t segments,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t warp_id = idx / 32;
        size_t segment_size = from_count / segments;
        curandState segment_state = segment_states[warp_id];
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {
            float s_val = curand_uniform(&segment_state);
            int segment = __float2int_ru(s_val * segments) - 1;
            size_t k = seg_megopolis_find_ancestor<I>(
                i,
                offsets,
                weights,
                segment,
                segment_size,
                &rand_state);

            into[i] = from[k];
        }

        *(states + idx) = rand_state;
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <size_t I, typename T>
    __global__ void seg_megopolis_resample_save_ancestor(
        size_t *offsets,
        T *into,
        T *from,
        float *weights,
        size_t *ancestors,
        size_t into_count,
        size_t from_count,
        size_t segments,
        curandState *states,
        curandState *segment_states)
    {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        size_t warp_id = idx / 32;
        size_t segment_size = from_count / segments;
        curandState segment_state = segment_states[warp_id];
        curandState rand_state = states[idx];

        for (size_t i = idx; i < into_count; i += stride)
        {
            float s_val = curand_uniform(&segment_state);
            int segment = __float2int_ru(s_val * segments) - 1;
            size_t k = seg_megopolis_find_ancestor<I>(
                i,
                offsets,
                weights,
                segment,
                segment_size,
                &rand_state);

            into[i] = from[k];
            ancestors[i] = k;
        }

        *(states + idx) = rand_state;
        if (threadIdx.x % 32 == 0)
        {
            segment_states[warp_id] = segment_state;
        }
    }

    template <size_t BLOCKS, size_t THREADS, typename T, typename RNG = xorshift>
    class SegMegopolisResampler : public Resampler<RNG, BLOCKS, THREADS, T>
    {
        size_t buffer_size;
        std::vector<size_t> host_offsets;
        thrust::device_ptr<size_t> offsets;
        size_t segment_size;
        Box<curandState> segment_rng;

    public:
        size_t iters;

        SegMegopolisResampler(int seed) : iters(0), buffer_size(0)
        {
            size_t *ptr = nullptr;
            offsets = thrust::device_ptr<size_t>(ptr);

            curandState *seg_ptr;
            gpuErrchk(cudaMalloc(&seg_ptr, BLOCKS * THREADS / 32 * sizeof(curandState)));
            setup_curand<<<BLOCKS, THREADS>>>(
                seed,
                seg_ptr,
                BLOCKS * THREADS / 32);
            gpuErrchk(cudaDeviceSynchronize());
            segment_rng = Box<curandState>(seg_ptr);
        }

        ~SegMegopolisResampler()
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
        }

        SegMegopolisResampler(SegMegopolisResampler &&other) : host_offsets(std::move(other.host_offsets)),
                                                               offsets(other.offsets),
                                                               segment_size(other.segment_size),
                                                               segment_rng(std::move(other.segment_rng)),
                                                               buffer_size(other.buffer_size),
                                                               iters(other.iters)
        {
            other.offsets = thrust::device_ptr<size_t>(nullptr);
        }

        SegMegopolisResampler &operator=(SegMegopolisResampler &&other)
        {
            gpuErrchk(cudaFree(to_raw(offsets)));
            iters = other.iters;
            buffer_size = other.buffer_size;
            offsets = other.offsets;
            other.offsets = thrust::device_ptr<size_t>(nullptr);
            segment_size = other.segment_size;
            segment_rng = std::move(other.segment_rng);

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

        size_t determine_iters(float *weights, size_t count, size_t desired, float epsilon)
        {
            float sum = thrust::reduce(thrust::device, weights, weights + count);
            float *max_ptr = thrust::max_element(thrust::device, weights, weights + count);
            float max;

            gpuErrchk(cudaMemcpy(&max, max_ptr, sizeof(float), cudaMemcpyDeviceToHost));

            float avg_weight = sum / (float)count;
            return log10(epsilon) / log10(1.0 - avg_weight / max);
        }

        void set_iters(size_t iters)
        {
            this->iters = iters;
        }

        void generate_offsets(RNG &rng, size_t min, size_t max)
        {
            this->reserve(iters);
            host_offsets.clear();
            std::uniform_int_distribution<size_t> dist(min, max);

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

        void set_segment_size(size_t size)
        {
            this->segment_size = size;
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
            size_t ssize = std::min(segment_size, from_count);

            size_t segments = from_count / ssize;
            this->generate_offsets(rng, 0, from_count);
            dyn_seg_megopolis_resample<<<BLOCKS, THREADS>>>(
                to_raw(offsets),
                into,
                from,
                weights,
                into_count,
                from_count,
                segments,
                iters,
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
            this->generate_offsets(rng, 0, from_count);
            dyn_seg_megopolis_resample_save_ancestor<<<BLOCKS, THREADS>>>(
                to_raw(offsets),
                into,
                from,
                weights,
                ancestors,
                into_count,
                from_count,
                segments,
                iters,
                states,
                to_raw(segment_rng.get()));
            gpuErrchk(cudaDeviceSynchronize());
        }
    };
}