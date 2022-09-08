#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <cstdio>
#include <utility>
#include <algorithm>
#include <assert.h>

// Basic gpu error debugging function. Errors should be handled properly in production code
#define gpuErrchk(ans)                        \
{                                         \
    resampling::gpuAssert((ans), __FILE__, __LINE__); \
}
namespace resampling
{
    enum class Device {
        Cpu,
        Gpu
    };

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort)
                exit(code);
        }
    }

    // TODO: Find a better home for to_raw
    template <typename T>
    inline T *to_raw(thrust::device_ptr<T> ptr)
    {
        return thrust::raw_pointer_cast(ptr);
    }

    template <typename T>
    inline T *to_raw(thrust::device_reference<T> ptr)
    {
        return thrust::raw_pointer_cast(ptr);
    }

    // Max threads should be a multiple of 2
    // returns number of blocks and threads needed to complete 'jobs'
    std::pair<size_t, size_t> determine_bt(
        size_t max_blocks,
        size_t log_min_threads,
        size_t max_threads,
        size_t jobs)
    {
        if (jobs == max_threads)
        {
            return std::make_pair(1, jobs);
        }

        size_t min_threads = 1 << log_min_threads;
        size_t total_threads = jobs >> log_min_threads << log_min_threads;

        if (jobs % min_threads != 0)
        {
            total_threads += min_threads;
        }

        size_t threads = min(total_threads, max_threads);
        size_t blocks = jobs / threads;

        if (jobs % threads != 0)
        {
            blocks += 1;
        }
        blocks = std::min(max_blocks, blocks);
        return std::make_pair(blocks, threads);
    }

}