#pragma once

#include <curand.h>
#include <curand_kernel.h>

namespace resampling
{
    // A gpu resampling interface
    template <typename RNG, size_t BLOCKS, size_t THREADS, typename T>
    class Resampler
    {
    public:
        // into: A buffer to hold the output of the resample
        // from: A buffer that holds the input particles
        // weights: A buffer that holds the weights of the input particles
        // into_count: The size, in particles, of the 'into' buffer
        // from_count: The size, in particles, of the 'from' buffer and the 'weight' buffer
        // states: A buffer of GPU random number generator states, should be of size BLOCKS * THREADS
        // cpu_rng: A reference to some cpu random number generator
        virtual void resample(
            T *into,
            T *from,
            float *weights,
            size_t into_count,
            size_t from_count,
            curandState *states,
            RNG &cpu_rng) = 0;
    };
}