#pragma once

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cstddef>
#include <stdio.h>
#include "../resampling/helper.cuh"

template <typename T>
__global__ void random_copy(
    T *into,
    T *from,
    size_t into_count,
    size_t from_count,
    curandState *states)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    curandState rand_state = states[idx];

    for (size_t i = idx; i < into_count; i += stride)
    {
        float u = curand_uniform(&rand_state);
        size_t index = __float2int_ru(u * from_count) - 1;
        into[i] = from[index];
    }

    states[idx] = rand_state;
}

// Object used to store information about particles on the gpu
// Will reallocate to correct size to fit data in
// @Changes: be able to choose device for particles
template <typename T>
class ParticleBuffer
{
    size_t capacity;
    size_t length;
    thrust::device_ptr<T> current;
    thrust::device_ptr<T> previous;
    thrust::device_ptr<float> weights;

public:
    ParticleBuffer() : capacity(0), length(0) {}

    ParticleBuffer(size_t cap) : capacity(cap), length(0)
    {
        size_t size = cap * sizeof(T);
        T *current_ptr;
        T *previous_ptr;
        float *weights_ptr;

        gpuErrchk(cudaMalloc(&current_ptr, size));
        gpuErrchk(cudaMalloc(&previous_ptr, size));
        gpuErrchk(cudaMalloc(&weights_ptr, cap * sizeof(float)));

        current = thrust::device_ptr<T>(current_ptr);
        previous = thrust::device_ptr<T>(previous_ptr);
        weights = thrust::device_ptr<float>(weights_ptr);
    }

    ParticleBuffer(thrust::device_ptr<T> particles, size_t count) : capacity(count), length(count), current(particles)
    {
        T *previous_ptr;
        float *weights_ptr;

        gpuErrchk(cudaMalloc(&previous_ptr, count * sizeof(T)));
        gpuErrchk(cudaMalloc(&weights_ptr, count * sizeof(float)));
        previous = thrust::device_ptr<T>(previous_ptr);
        weights = thrust::device_ptr<float>(weights_ptr);
    }

    ~ParticleBuffer()
    {
        gpuErrchk(cudaFree(resampling::to_raw(current)));
        gpuErrchk(cudaFree(resampling::to_raw(previous)));
        gpuErrchk(cudaFree(resampling::to_raw(weights)));
    }

    ParticleBuffer(const ParticleBuffer<T> &other) : capacity(other.capacity), length(other.length)
    {
        size_t size = other.capacity * sizeof(T);
        T *current_ptr;
        T *previous_ptr;
        float *weights_ptr;

        gpuErrchk(cudaMalloc(&current_ptr, size));
        gpuErrchk(cudaMalloc(&previous_ptr, size));
        gpuErrchk(cudaMalloc(&weights_ptr, other.capacity * sizeof(float)));
        size_t copy_size = other.length * sizeof(T);

        gpuErrchk(cudaMemcpy(current_ptr, resampling::to_raw(other.current), copy_size, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(previous_ptr, resampling::to_raw(other.previous), copy_size, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(weights_ptr, resampling::to_raw(other.weights), other.length * sizeof(float), cudaMemcpyDeviceToDevice));

        current = thrust::device_ptr<T>(current_ptr);
        previous = thrust::device_ptr<T>(previous_ptr);
        weights = thrust::device_ptr<float>(weights_ptr);
    }

    ParticleBuffer(ParticleBuffer<T> &&other) : capacity(other.capacity),
                                                length(other.length),
                                                current(other.current),
                                                previous(other.previous),
                                                weights(other.weights)
    {
        other.current = thrust::device_ptr<T>(nullptr);
        other.previous = thrust::device_ptr<T>(nullptr);
        other.weights = thrust::device_ptr<T>(nullptr);
    }

    ParticleBuffer &operator=(const ParticleBuffer<T> &other)
    {
        size_t copy_size = other.length * sizeof(T);

        if (other.length > capacity)
        {
            size_t size = other.capacity * sizeof(T);
            T *new_current;
            T *new_previous;
            float *new_weights;

            gpuErrchk(cudaMalloc(&new_current, size));
            gpuErrchk(cudaMalloc(&new_previous, size));
            gpuErrchk(cudaMalloc(&new_weights, other.capacity * sizeof(float)));

            gpuErrchk(cudaMemcpy(new_current, resampling::to_raw(other.current), copy_size, cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(new_previous, resampling::to_raw(other.previous), copy_size, cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(new_weights, resampling::to_raw(other.weights), other.length * sizeof(float), cudaMemcpyDeviceToDevice));

            gpuErrchk(cudaFree(resampling::to_raw(current)));
            gpuErrchk(cudaFree(resampling::to_raw(previous)));
            gpuErrchk(cudaFree(resampling::to_raw(weights)));

            current = thrust::device_ptr<T>(new_current);
            previous = thrust::device_ptr<T>(new_previous);
            weights = thrust::device_ptr<float>(new_weights);
        }
        else
        {
            gpuErrchk(cudaMemcpy(resampling::to_raw(current), resampling::to_raw(other.current), copy_size, cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(resampling::to_raw(previous), resampling::to_raw(other.previous), copy_size, cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(resampling::to_raw(weights), resampling::to_raw(other.weights), other.length * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        return this;
    }

    ParticleBuffer &operator=(ParticleBuffer<T> &&other)
    {
        gpuErrchk(cudaFree(resampling::to_raw(current)));
        gpuErrchk(cudaFree(resampling::to_raw(previous)));
        gpuErrchk(cudaFree(resampling::to_raw(weights)));

        capacity = other.capacity;
        length = other.length;
        current = other.current;
        previous = other.previous;
        weights = other.weights;

        other.current = thrust::device_ptr<T>(nullptr);
        other.previous = thrust::device_ptr<T>(nullptr);
        other.weights = thrust::device_ptr<T>(nullptr);

        return this;
    }

    // Forces the object to reallocate to size 'cap'
    void set_capacity(size_t cap)
    {
        size_t size = cap * sizeof(T);
        size_t copy_len = length;

        if (length > cap)
        {
            copy_len = cap;
        }

        T *new_current;
        T *new_previous;
        float *new_weights;

        gpuErrchk(cudaMalloc(&new_current, size));
        gpuErrchk(cudaMalloc(&new_previous, size));
        gpuErrchk(cudaMalloc(&new_weights, cap * sizeof(float)));

        gpuErrchk(cudaMemcpy(new_current, resampling::to_raw(current), copy_len * sizeof(T), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(new_previous, resampling::to_raw(previous), copy_len * sizeof(T), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(new_weights, resampling::to_raw(weights), copy_len * sizeof(float), cudaMemcpyDeviceToDevice));

        gpuErrchk(cudaFree(resampling::to_raw(current)));
        gpuErrchk(cudaFree(resampling::to_raw(previous)));
        gpuErrchk(cudaFree(resampling::to_raw(weights)));

        current = thrust::device_ptr<T>(new_current);
        previous = thrust::device_ptr<T>(new_previous);
        weights = thrust::device_ptr<float>(new_weights);

        capacity = cap;
        length = copy_len;
    }

    size_t len() const
    {
        return length;
    }

    size_t cap() const
    {
        return capacity;
    }

    bool can_fit(size_t amount) const
    {
        return capacity - length >= amount;
    }

    // Length must be less than or equal to capacity
    void set_length(size_t len)
    {
        assert(len <= capacity);
        length = len;
    }

    void clear()
    {
        length = 0;
    }

    // Appends particles on device to 'current' array
    // Increases length by count
    void append_device_data(thrust::device_ptr<T> particles, size_t count)
    {
        size_t new_length = length + count;
        size_t cap = capacity;

        if (new_length > capacity)
        {
            while (new_length > cap)
            {
                cap *= 2;
            }

            this->set_capacity(cap);
        }

        thrust::device_ptr<T> after = current + length;
        gpuErrchk(cudaMemcpy(resampling::to_raw(after), resampling::to_raw(particles), count * sizeof(T), cudaMemcpyDeviceToDevice));
        length = new_length;
    }

    void append_host_data(T *particles, size_t count)
    {
        size_t new_length = length + count;
        size_t cap = capacity;

        if (new_length > capacity)
        {
            while (new_length > cap)
            {
                cap *= 2;
            }

            this->set_capacity(cap);
        }

        thrust::device_ptr<T> after = current + length;
        gpuErrchk(cudaMemcpy(resampling::to_raw(after), particles, count * sizeof(T), cudaMemcpyHostToDevice));
        length = new_length;
    }

    size_t calculate_maximum_offset(size_t len) const
    {
        if (len >= this->length)
        {
            return 0;
        }

        return this->length - len;
    }

    void copy_device_data_to_previous(thrust::device_ptr<T> particles, size_t count)
    {
        size_t to_copy = count;

        if (count > length)
        {
            to_copy = length;
        }

        gpuErrchk(cudaMemcpy(resampling::to_raw(previous), resampling::to_raw(particles), to_copy * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    // Will copy 'count' number of particles
    void copy_to_host(T *host_ptr, size_t count, size_t offset) const
    {
        assert(count + offset <= length);

        thrust::device_ptr<T> ptr = current + offset;

        gpuErrchk(cudaMemcpy(host_ptr, resampling::to_raw(ptr), count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copy_to_device(thrust::device_ptr<T> ptr, size_t count, size_t offset) const
    {
        assert(count <= length);

        gpuErrchk(cudaMemcpy(
            resampling::to_raw(ptr),
            resampling::to_raw(current) + offset,
            count * sizeof(T),
            cudaMemcpyDeviceToDevice));
    }

    template <size_t BLOCKS, size_t THREADS>
    void rand_copy_to_device(thrust::device_ptr<T> ptr, size_t count, curandState *states) const
    {
        random_copy<<<BLOCKS, THREADS>>>(
            resampling::to_raw(ptr),
            resampling::to_raw(current),
            count,
            length,
            states);

        gpuErrchk(cudaDeviceSynchronize());
    }

    // Replaces current particles with particles in ptr
    void copy_replace(thrust::device_ptr<T> ptr, size_t count)
    {
        this->clear();
        this->append_device_data(ptr, count);
    }

    size_t available()
    {
        return capacity - length;
    }

    void swap()
    {
        thrust::device_ptr<T> temp = current;
        current = previous;
        previous = temp;
    }

    void remove_last(size_t n)
    {
        if (n > length)
        {
            length = 0;
        }
        else
        {
            length = length - n;
        }
    }

    // Sets all weights in [0, length) to be 'weight'
    void set_weights(float weight)
    {
        thrust::device_ptr<float> end = weights + length;

        thrust::fill(thrust::device, weights, end, weight);
    }

    void set_weights_range(float weight, size_t start, size_t end)
    {
        assert(start < end);
        assert(start < length);
        assert(end <= length);

        thrust::device_ptr<float> start_ptr = weights + start;
        thrust::device_ptr<float> end_ptr = weights + end;
        thrust::fill(thrust::device, start_ptr, end_ptr, weight);
    }

    void copy_previous_to_current()
    {
        gpuErrchk(cudaMemcpy(resampling::to_raw(current), resampling::to_raw(previous), length * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    float sum_weights()
    {
        return thrust::reduce(thrust::device, weights, weights + length);
    }

    thrust::device_ptr<T> particles_begin()
    {
        return current;
    }

    thrust::device_ptr<T> particles_end()
    {
        return current + length;
    }

    thrust::device_ptr<T> previous_begin()
    {
        return previous;
    }

    thrust::device_ptr<T> previous_end()
    {
        return previous + length;
    }

    thrust::device_ptr<float> weights_begin()
    {
        return weights;
    }

    thrust::device_ptr<float> weights_end()
    {
        return weights + length;
    }
};