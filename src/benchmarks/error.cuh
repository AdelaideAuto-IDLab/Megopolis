#pragma once

#include "../resampling/helper.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <random>
#include <functional>
#include <math.h>

// Calculates (a - b)^2
struct DiffSqrd : public std::binary_function<float, float, float> {
    __host__ __device__
    float operator()(const float &a, const float &b) const {
        float diff = a - b;
        return diff * diff;
    }
};

// Casts ULL to float and adds them
struct FloatAddAssignULL : public std::binary_function<float, unsigned long long, float> {
    __host__ __device__
    float operator()(const float &a, const float &b) const {
        return a + (float) b;
    }
};

// Ouputs 'multiplier * a'
template<typename T>
struct Multiply : public std::unary_function<T, T> {
    T multiplier; 
    
    Multiply(T multiplier) : multiplier(multiplier) {}

    __host__ __device__
    float operator()(const T &a) const {
        return a * multiplier;
    }
};

// Output[i] = BinOp(left[i], right[i])
template<typename L, typename R, typename Result, typename BinOp>
__global__ void binop_kernel(
    L * left,
    R * right,
    Result * output,
    size_t size,
    BinOp op
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < size; i += stride) {
        Result value = op(left[i], right[i]);
        output[i] = value;
    }
}

// Output[i] = UnaryOp(Input[i])
template<typename T, typename Result, typename UnaryOp>
__global__ void tranform_into_kernel(
    T * input,
    Result * output,
    size_t size,
    UnaryOp op
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < size; i += stride) {
        output[i] = op(input[i]);
    }
}

// Calculates the minimum required blocks and threads using with the maximum number of threads per 
// block being 32 and the maximum number of blocks being 256 
template<typename L, typename R, typename Result, typename BinOp>
void device_binop(L * left, R * right, Result * output, size_t size, BinOp op) {
    auto bt = resampling::determine_bt(32, 5, 256, size);
    size_t blocks = std::get<0>(bt);
    size_t threads = std::get<1>(bt);

    binop_kernel<<<blocks, threads>>>(left, right, output, size, op);
    gpuErrchk( cudaDeviceSynchronize() );
}

// For each value in 'input' perform the save the output from the 'op' into 'ouput'
template<typename T, typename Result, typename UnaryOp>
void transform_into(T * input, Result * output, size_t size, UnaryOp op) {
    auto bt = resampling::determine_bt(32, 5, 256, size);
    size_t blocks = std::get<0>(bt);
    size_t threads = std::get<1>(bt);

    tranform_into_kernel<<<blocks, threads>>>(input, output, size, op);
    gpuErrchk( cudaDeviceSynchronize() );
}

// Counts the total number of offspring for each ancestor 
__global__ void count_offspring(
    size_t * ancestors,
    unsigned long long int * offspring,
    size_t count
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < count; i += stride) {
        size_t ancestor = ancestors[i];
        atomicAdd(&offspring[ancestor], 1);
    }
}

// Calculate the squeared difference between expected offspring and actual offspring and save it to
// 'output'
__global__ void resample_error_kernel(
    float * expected_offspring,
    unsigned long long * offspring,
    float * output,
    size_t count
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    float fcount = __uint2float_rn(count);

    for (size_t i = idx; i < count; i += stride) {
        float os = __uint2float_rn(offspring[i]);
        float error = os - expected_offspring[i];

        output[i] = error * error;
    }
}

// Creates a list of the number of offspring for each ancestor where the i-th value in 'offspring'
// is the number of children of the i-th particle 
template<size_t BLOCKS, size_t THREADS>
void ancestors_to_offspring(
    size_t * ancestors,
    unsigned long long int * offspring,
    size_t count
) {
    count_offspring<<<BLOCKS, THREADS>>>(
        ancestors,
        offspring,
        count
    );

    gpuErrchk( cudaDeviceSynchronize() );
}

// Computes the sum of the resample errors
template<size_t BLOCKS, size_t THREADS>
float resample_error(
    float * expected_offspring,
    unsigned long long * offspring,
    float * output,
    size_t count
) {
    resample_error_kernel<<<BLOCKS, THREADS>>>(
        expected_offspring,
        offspring,
        output,
        count
    );

    gpuErrchk( cudaDeviceSynchronize() );

    return thrust::reduce(thrust::device, output, output + count);
}

struct MeanStddev {
    float mean;
    float stddev;

    MeanStddev() {}
    MeanStddev(float mean, float stddev) : mean(mean), stddev(stddev) {}
    
    // Computes the mean and stddev of values
    MeanStddev(std::vector<float> &values) {
        float sum = 0.0;

        for (auto value : values) {
            sum += value;
        }

        mean = sum / (float) values.size();
        sum = 0.0;

        for (auto value : values) {
            float diff = value - mean;
            sum += diff * diff;
        }

        stddev = std::sqrt(sum / (float) values.size());
    }
};

