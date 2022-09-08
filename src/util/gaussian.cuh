#pragma once

#include "consts.h"
#include <random>

class Gaussian {
    float mean;
    float stddev;
    float norm;
    std::normal_distribution<float> distribution;
public:
    Gaussian() {}
    Gaussian(float mean, float stddev) : 
        mean(mean), stddev(stddev), norm(sqrt(consts::TWO_PI) * stddev), distribution(std::normal_distribution<float>(mean, stddev)) 
    {}

    __host__ __device__ float get_mean() const {
        return mean;
    }

    __host__ __device__ float get_stddev() const {
        return stddev;
    }

    __host__ __device__ float get_norm() const {
        return norm;
    }

    __host__ __device__ float density(float x) const {
        return exp(-pow((x - mean), 2) / (2.0 * stddev * stddev)) / norm;
    }

    template<class URNG>
    float generate(URNG& g) {
        return distribution(g);
    }
    
};