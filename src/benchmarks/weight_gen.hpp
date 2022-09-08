#pragma once

#include "../util/channel.hpp"
#include "../util/parse_json.hpp"
#include "../resampling/random.cuh"
#include <random>
#include <functional>
#include <math.h>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

enum class WeightMethod {
    Gamma, 
    RelativeVar
};

// Struct to load weight generation parameters from json file
struct WeightParams {
    WeightMethod method;
    float a;
    float b;

    WeightParams() {}
    WeightParams(WeightMethod method, float a, float b) : method(method), a(a), b(b) {}

    const char * method_name() {
        if (method == WeightMethod::Gamma) {
            return "Gamma";
        }
        else {
            return "RelativeVar";
        }
    }

    bool is_gamma() {
        return method == WeightMethod::Gamma;
    }
};

void parse_value(const rapidjson::Value * dom, WeightParams &into) {
    std::string m;

    parse_field(dom, "method", m);

    if (m == std::string("Gamma")) {
        // Load shape parameter into 'a'
        parse_field(dom, "shape", into.a);
        into.method = WeightMethod::Gamma;

        // Optionally load scale parameter into 'b'
        if (!dom->HasMember("scale")) {
            into.b = 1.0;
        } else {
            parse_field(dom, "scale", into.b);
        }
    }
    else if (m == std::string("RelativeVar")) {
        // Load y parameter into 'a'
        parse_field(dom, "y", into.a);
        into.method = WeightMethod::RelativeVar;
    }
    else throw std::invalid_argument("Invalid method type");
}

// Outputs a vector of unsigned integers representing the number of values generated that 
// are within a certain bin. Bins are calculated from the min and max value in 'samples' where
// the i-th bin is the range [min + (max - min) / 'bins' * i, min + (max - min) / 'bins' * (i + 1))
std::vector<size_t> weight_bins(std::vector<float>& samples, size_t bins) {
    std::vector<size_t> output(bins);
    float min = samples[0];
    float max = samples[0];

    for (auto& value : samples) {
        if (value < min) {
            min = value;
        }

        if (value > max) {
            max = value; 
        }
    }

    float bin_size = (max - min) / (float) bins; 

    for (auto& value : samples) {
        float bin_offset = (value - min) / bin_size;
        // printf("Offset: %f\n", bin_offset);
        size_t bin = (size_t) floor(bin_offset);
        if (bin >= output.size()) {
            continue; 
        }


        output.at(bin)++;
    }

    return output;
}

// Generate a set of samples from the gamma distribution using the xorshift rng
float gamma_samples(
    std::vector<float>& samples, 
    float alpha, 
    float scale,
    size_t count, 
    xorshift &rng
) {
    std::gamma_distribution<float> gamma(alpha, scale);
    samples.clear(); 
    samples.reserve(count);
    float sum = 0.0;

    for (size_t i = 0; i < count; i++) {
        float gamma_sample = gamma(rng);
        sum += gamma_sample;

        samples.push_back(gamma_sample);
    }

    return sum;
}

// Generate a set of samples from the relative variance distribution using the xorshift rng
float relative_var_samples(
    std::vector<float>& samples, 
    float y, 
    size_t count, 
    xorshift &rng
) {
    std::normal_distribution<float> dist(0.0, 1.0);
    samples.clear(); 
    samples.reserve(count);
    float sum = 0.0;
    float inv_sqrt_2pi = 1.0 / sqrt(M_PI * 2.0);

    for (size_t i = 0; i < count; i++) {
        float x = dist(rng);
        float sample = inv_sqrt_2pi *  exp(-0.5 * (x - y) * (x - y));

        samples.push_back(sample);
        sum += sample;
    }

    return sum;
}

// Normalized gamma samples
void dirichlet_samples(std::vector<float>& samples, float alpha, size_t count, xorshift &rng) {
    float sum = gamma_samples(samples, alpha, 1.0, count, rng);
    float norm = 1.0 / sum;

    for (auto& value : samples) {
        value *= norm;
    }
}

// Send 'seqs' number of weight sequences of size 'count' to 'sender'. 
void generate_weight_seqs(
    Channel<std::vector<float>> sender, 
    WeightParams method, 
    size_t count, 
    size_t seqs, 
    xorshift rng
) {
    for (size_t i = 0; i < seqs; i++) {
        std::vector<float> samples;

        if (method.method == WeightMethod::Gamma) {
            // dirichlet_samples(samples, method.a, count, rng);
            gamma_samples(samples, method.a, method.b, count, rng);
        }
        else {
            relative_var_samples(samples, method.a, count, rng);
        }

        sender.send(std::move(samples));
    }
}

// Spawn a new thread that generates weight sequences
std::thread gen_weights(
    Channel<std::vector<float>> sender, 
    WeightParams method, 
    size_t count, 
    size_t seqs
) {
    // unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::random_device rd;
    xorshift rng(rd);

    return std::thread(
        generate_weight_seqs, 
        sender,
        method,
        count,
        seqs,
        rng
    );
}

// Split generating 'seqs' number of weight sequences across 'worker' number of threads
void split_gen_weights(
    std::vector<std::thread>& worker_threads,
    Channel<std::vector<float>> sender, 
    WeightParams method, 
    size_t count, 
    size_t seqs,
    size_t workers
) { 
    std::vector<size_t> worker_seqs;
    worker_seqs.reserve(workers);

    for (size_t i = 0; i < workers; i++) {
        worker_seqs.push_back(0);
    }

    size_t current_worker = 0;

    while (seqs > 0) {
        seqs -= 1; 
        worker_seqs[current_worker] += 1;
        current_worker += 1;
        current_worker = current_worker % workers;
    }

    for (size_t i = 0; i < workers; i++) {
        std::thread t = gen_weights(
            sender,
            method, 
            count,
            worker_seqs[i]
        );

        worker_threads.push_back(std::move(t));
    }
}

// Function to display the bins of a distribution to the console
void display_distribution(
    float * weights, 
    size_t amount, 
    size_t bins,
    size_t scale
) {
    float max = 0.0;

    for (size_t i = 0; i < amount; i++) {
        float weight = weights[i];
        if (weight > max) {
            max = weight;
        }
    }

    float bin_size = max / float(bins);
    std::vector<size_t> counts;
    counts.reserve(bins);

    for (size_t i = 0; i < bins; i++) {
        counts.push_back(0);
    }

    for (size_t i = 0; i < amount; i++) {
        float weight = weights[i];

        float l_bin = 0.0;
        float u_bin = bin_size;

        for (size_t j = 0; j < bins; j++) {
            if (weight > l_bin && weight < u_bin) {
                counts[j]++;
                break;
            }

            l_bin = u_bin;
            u_bin += bin_size;
        }
    }

    float l_bin = 0.0;
    float u_bin = bin_size;

    for (auto& count : counts) {
        printf("%.2f-%.2f:", l_bin, u_bin);
        l_bin = u_bin;
        u_bin += bin_size;

        for (size_t i = 0; i < count / scale; i++) {
            printf("*");
        }
        printf("\n");
    }
    printf("Max: %f\n", max);
}