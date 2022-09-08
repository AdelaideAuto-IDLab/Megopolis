#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include <random>
#include "shared_ptr.cuh"

class xorshift
{
public:
    using result_type = uint32_t;
    static constexpr result_type(min)() { return 0; }
    static constexpr result_type(max)() { return UINT32_MAX; }
    friend bool operator==(xorshift const &, xorshift const &);
    friend bool operator!=(xorshift const &, xorshift const &);

    xorshift() : m_seed(0xc1f651c67c62c6e0ull) {}
    explicit xorshift(std::random_device &rd)
    {
        seed(rd);
    }

    void seed(std::random_device &rd)
    {
        m_seed = uint64_t(rd()) << 31 | uint64_t(rd());
    }

    result_type operator()()
    {
        uint64_t result = m_seed * 0xd989bcacc137dcd5ull;
        m_seed ^= m_seed >> 11;
        m_seed ^= m_seed << 31;
        m_seed ^= m_seed >> 18;
        return uint32_t(result >> 32ull);
    }

    void discard(unsigned long long n)
    {
        for (unsigned long long i = 0; i < n; ++i)
            operator()();
    }

private:
    uint64_t m_seed;
};

namespace resampling
{
    __global__ void setup_curand(int seed, curandState *states, size_t count)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if (id < count)
        {
            curandState initialize;
            curand_init(seed + id, 0, 0, &initialize);
            states[id] = initialize;
        }
    }

    template <size_t BLOCKS, size_t THREADS>
    Box<curandState> generate_curand_states(int seed)
    {
        curandState *states;
        gpuErrchk(cudaMalloc(&states, BLOCKS * THREADS * sizeof(curandState)));

        setup_curand<<<BLOCKS, THREADS>>>(seed, states, BLOCKS * THREADS);
        gpuErrchk(cudaDeviceSynchronize());

        return Box<curandState>(states);
    }
}
