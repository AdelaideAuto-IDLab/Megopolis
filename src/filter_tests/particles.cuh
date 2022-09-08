#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <random>

#include "../resampling/resampling.cuh"
#include "../resampling/random.cuh"
#include "../resampling/shared_ptr.cuh"
#include "../resampling/helper.cuh"
#include "../util/gaussian.cuh"
#include "../filter/filter.cuh"
#include "../filter/buffer.cuh"

namespace filter_ex {
    __host__ __device__ float propagate_particle(float k, float x_k, float noise) {
        float x_kp1 = 
            x_k * 0.5 +
            25.0 * x_k / (1.0 + x_k * x_k) +
            8.0 * cos(1.2 * k) +
            noise;

        return x_kp1;
    }

    __global__ void filter_propagate(
        float k,
        float measurement,
        Gaussian measurement_model,
        float prop_noise,
        float * particles,
        float * weights,
        size_t size,
        curandState * rand_states
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = rand_states[idx];

        for (size_t i = idx; i < size; i += stride) {
            float noise = curand_normal(&rand_state);
            float x_k = particles[i];
            x_k = propagate_particle(k, x_k, noise * prop_noise);
            particles[i] = x_k;
            float expected_measurement = x_k * x_k / 20.0;
            weights[i] = measurement_model.density(expected_measurement - measurement);
        }

        rand_states[idx] = rand_state;
    }

    __global__ void generate_particles(
        float * buffer,
        size_t count,
        float stddev,
        curandState * rand_states
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;
        curandState rand_state = rand_states[idx];

        for (size_t i = idx; i < count; i += stride) {
            buffer[i] = curand_normal(&rand_state) * stddev;
        }
        rand_states[idx] = rand_state;
    }

    __global__ void multiply_particles_weights(
        float * particles,
        float * weights,
        size_t count,
        float norm
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < count; i += stride) {
            particles[i] = weights[i] * particles[i] * norm; 
        }
    }



    template<size_t BLOCKS, size_t THREADS>
    class Filter {
        ParticleBuffer<float> buffer;
        size_t iter;
        Gaussian measurement_model;
        float process_noise;
        resampling::Box<curandState> rand_states;
    
    public:
        Filter(size_t size, float noise, float process_noise) : 
            buffer(size),
            iter(0),
            measurement_model(0.0, noise),
            process_noise(process_noise)
        {
            srand(time(0));

            rand_states = resampling::generate_curand_states<BLOCKS, THREADS>(rand());
        }

        void initialize_particles(float stddev) {
            buffer.set_length(buffer.cap());
            iter = 0;

            generate_particles<<<BLOCKS, THREADS>>>(
                resampling::to_raw(buffer.particles_begin()),
                buffer.len(),
                stddev,
                rand_states.get_raw()
            );
            gpuErrchk( cudaDeviceSynchronize() );
        }

        float * get_weights() {
            return resampling::to_raw(buffer.weights_begin());
        }

        size_t get_size() {
            return buffer.len(); 
        }

        void propagate(float measurement) {
            filter_propagate<<<BLOCKS, THREADS>>>(
                (float) iter,
                measurement,
                measurement_model,
                process_noise,
                resampling::to_raw(buffer.particles_begin()),
                resampling::to_raw(buffer.weights_begin()),
                buffer.len(),
                rand_states.get_raw()
            );
            iter++;
            gpuErrchk( cudaDeviceSynchronize() );
        }

        float estimate_uniform_weights() {
            float sum = thrust::reduce(buffer.particles_begin(), buffer.particles_end(), 0.0);
            return sum / (float) buffer.len();
        } 

        float estimate_non_uniform_weights() {
            float weight_sum = thrust::reduce(buffer.weights_begin(), buffer.weights_end(), 0.0);

            multiply_particles_weights<<<BLOCKS, THREADS>>>(
                resampling::to_raw(buffer.previous_begin()),
                resampling::to_raw(buffer.weights_begin()),
                buffer.len(),
                1.0 / weight_sum
            );
            gpuErrchk( cudaDeviceSynchronize() );
            float sum = thrust::reduce(buffer.previous_begin(), buffer.previous_end(), 0.0);
            if (isnan(sum)) {
                printf("SLDJFLSDJF avg: %f, sum: %f\n\n\n", sum, weight_sum);
            }
            return sum;
        } 

        template<typename RNG>
        void resample(resampling::Resampler<RNG, BLOCKS, THREADS, float> * resampler, RNG& rng) {
            buffer.swap();

            // We let the resamplers decide whether or not to normalize weights
            resampler->resample(
                resampling::to_raw(buffer.particles_begin()),
                resampling::to_raw(buffer.previous_begin()),
                resampling::to_raw(buffer.weights_begin()),
                buffer.len(),
                buffer.len(),
                rand_states.get_raw(),
                rng
            );
            
            // We dont need to reset weights as the propagation stage resets thems
        }
    };
}
