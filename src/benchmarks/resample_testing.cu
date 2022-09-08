#include "../resampling/helper.cuh"
#include "../util/parse_json.hpp"
#include "../util/progress_bar.hpp"
#include "../resampling/random.cuh"
#include "../resampling/resampling.cuh"
#include "../resampling/metropolis_c1.cuh"
#include "../resampling/segmented_megopolis.cuh"
#include "../resampling/metropolis_c2.cuh"
#include "../resampling/megopolis_aligned.cuh"
#include "error.cuh"
#include "weight_gen.hpp"
#include "resample_config.hpp"

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/adjacent_difference.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <random>
#include <array>
#include <iostream>
#include <cstdint>

const size_t THREADS = 256;
const size_t BLOCKS = 512;

// const size_t THREADS = 2;
// const size_t BLOCKS = 1;

// Container for all host and device buffers used for testing the resampling methods
struct ResampleBuffers
{
    // Allow for different input and output sizes
    size_t size;
    size_t output_size;
    size_t offspring_iters;
    size_t waiting_for;
    Channel<std::vector<float>> weight_gen_channel;
    std::vector<std::vector<float>> weight_seqs;
    std::vector<float> weights;
    unsigned long long *offspring;
    size_t *ancestors;
    float *expected_offspring;
    float *output;

    // Particles are just floats to demonstrate the runtime cost of copying particles from input to
    // output
    float *dev_input;
    float *dev_weights;
    float *dev_output;
    float *dev_error;
    size_t *dev_ancestors;
    unsigned long long *dev_offspring;
    float *dev_expected_offspring;
    float *dev_offspring_average;

    ResampleBuffers(const ResampleBuffers &) = delete;
    ResampleBuffers(ResampleBuffers &&) = delete;

    ResampleBuffers(
        size_t size,
        size_t output_size) : size(size),
                              waiting_for(0),
                              weights(size, 1.0),
                              output_size(output_size)
    {
        offspring = new unsigned long long[size];
        expected_offspring = new float[size];
        output = new float[output_size];
        ancestors = new size_t[output_size];

        gpuErrchk(cudaMalloc(&dev_output, output_size * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_error, size * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_input, size * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_weights, size * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_ancestors, output_size * sizeof(size_t)));
        gpuErrchk(cudaMalloc(&dev_offspring, size * sizeof(unsigned long long)));
        gpuErrchk(cudaMalloc(&dev_expected_offspring, size * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_offspring_average, size * sizeof(float)));

        thrust::sequence(thrust::device, dev_input, dev_input + size, 0);
    }

    ~ResampleBuffers()
    {
        delete[] offspring;
        delete[] expected_offspring;
        delete[] output;
        delete[] ancestors;
        gpuErrchk(cudaFree(dev_input));
        gpuErrchk(cudaFree(dev_output));
        gpuErrchk(cudaFree(dev_error));
        gpuErrchk(cudaFree(dev_weights));
        gpuErrchk(cudaFree(dev_ancestors));
        gpuErrchk(cudaFree(dev_offspring));
        gpuErrchk(cudaFree(dev_expected_offspring));
        gpuErrchk(cudaFree(dev_offspring_average));
    }

    void reset()
    {
        // Only need to reset weights and offspring buffers
        gpuErrchk(cudaMemcpy(dev_weights, weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        thrust::fill_n(thrust::device, dev_offspring, size, 0);
    }

    // Function to sort inputs by weights
    void sort_by_weight()
    {
        thrust::sort_by_key(
            thrust::device,
            dev_weights,
            dev_weights + size,
            dev_input);
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Calculates the number of offspring each particle had using the ancestor buffer
    void calculate_offspring()
    {
        ancestors_to_offspring<BLOCKS, THREADS>(
            dev_ancestors,
            dev_offspring,
            output_size);
    }

    // dev_weights[i] = dev_weights[i] - dev_weights[i - 1]
    void weights_adjacent_diff()
    {
        thrust::adjacent_difference(
            thrust::device,
            dev_weights,
            dev_weights + size,
            dev_weights);
    }

    void reset_offspring_average()
    {
        offspring_iters = 0;
        thrust::fill_n(thrust::device, dev_offspring_average, size, 0.0);
    }

    // Generates a new set of weights
    void change_weights(float alpha, xorshift &rng)
    {
        dirichlet_samples(weights, alpha, size, rng);
    }

    // Generates a multiple sequences of weights in parallel
    void gen_sequences(
        std::vector<std::thread> &threads,
        WeightParams params,
        size_t seqs,
        size_t workers)
    {
        waiting_for += seqs;

        return split_gen_weights(
            threads,
            this->weight_gen_channel,
            params,
            size,
            seqs,
            workers);
    }

    // Blocks the current thread, waiting to receive the next weight sequence.
    // If there are no more sequences to receive then it returns false
    bool next_weight_seq()
    {
        if (waiting_for == 0 && weight_seqs.size() == 0)
        {
            return false;
        }

        if (weight_seqs.size() > 0)
        {
            weights = weight_seqs.back();
            weight_seqs.pop_back();
        }
        else
        {
            weight_gen_channel.receive(weight_seqs);
            waiting_for -= weight_seqs.size();
            weights = weight_seqs.back();
            weight_seqs.pop_back();
        }

        return true;
    }

    // Copies host weights to device weight buffer
    void reset_weights()
    {
        gpuErrchk(cudaMemcpy(dev_weights, weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }

    unsigned long long sum_offspring()
    {
        return thrust::reduce(thrust::device, dev_offspring, dev_offspring + size);
    }

    // Calculates the expected number of offspring from the given device weight sequence
    void calculate_expected_offspring()
    {
        thrust::device_ptr<float> start(dev_weights);
        thrust::device_ptr<float> end = start + size;
        float sum = thrust::reduce(thrust::device, start, end, 0.0);

        // Normalize weight and multiply by output size
        Multiply<float> multi((float)output_size / sum);
        transform_into(
            dev_weights,
            dev_expected_offspring,
            size,
            multi);
    }

    // Add the current offspring counts to the offspring average buffer
    void add_offspring_to_average()
    {
        FloatAddAssignULL func;
        offspring_iters++;

        device_binop(
            dev_offspring_average,
            dev_offspring,
            dev_offspring_average,
            size,
            func);
    }

    float calculate_offspring_bias()
    {
        device_binop(
            dev_offspring,
            dev_expected_offspring,
            dev_offspring,
            size,
            DiffSqrd());

        float bias_sqrd = thrust::reduce(
            thrust::device,
            dev_offspring,
            dev_offspring + size);

        return bias_sqrd;
    }

    // Compute the bias as sum((expected_offspring - avg_offspring)^2)
    float calculate_bias()
    {
        Multiply<float> multi(1.0 / (float)offspring_iters);
        transform_into(
            dev_offspring_average,
            dev_offspring_average,
            size,
            multi);

        device_binop(
            dev_offspring_average,
            dev_expected_offspring,
            dev_offspring_average,
            size,
            DiffSqrd());

        float bias_sqrd = thrust::reduce(
            thrust::device,
            dev_offspring_average,
            dev_offspring_average + size);

        return bias_sqrd;
    }

    // Compute the resample MSE
    float resample_mse()
    {
        return resample_error<BLOCKS, THREADS>(
            dev_expected_offspring,
            dev_offspring,
            dev_error,
            size);
    }

    // Debugging function to see the number of offspring produced by a resampling algorithm
    void display_offspring_sum()
    {
        gpuErrchk(cudaMemcpy(expected_offspring, dev_offspring_average, size * sizeof(float), cudaMemcpyDeviceToHost));

        printf("Offspring Sum\n");

        for (size_t i = 0; i < size; i++)
        {
            printf("%.2f\n", expected_offspring[i]);
        }
        printf("\n");
    }

    // Debugging function to display resampling results
    void display()
    {
        gpuErrchk(cudaMemcpy(offspring, dev_offspring, size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(expected_offspring, dev_expected_offspring, size * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(output, dev_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(ancestors, dev_ancestors, output_size * sizeof(size_t), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; i++)
        {
            printf("[%.2f, %.2f, %d, %d]\n", weights[i], expected_offspring[i], (int)offspring[i], (int)ancestors[i]);
        }

        this->display_offspring_sum();

        printf("\n");
    }
};

// Struct for dynamically choosing a resampling method to use
struct Resamplers
{
    size_t precompute_iters;
    resampling::SystematicResampler<BLOCKS, THREADS, float> sys;
    resampling::MegopolisResampler<BLOCKS, THREADS, float> mego;
    resampling::MegopolisAlignedResampler<BLOCKS, THREADS, float> mego_a;
    resampling::MultinomialResampler<BLOCKS, THREADS, float> multi;
    resampling::MetropolisC1Resampler<BLOCKS, THREADS, float> seg_metro;
    resampling::MetropolisC2Resampler<BLOCKS, THREADS, float> metro_c2;
    resampling::SegMegopolisResampler<BLOCKS, THREADS, float> seg_mego;
    resampling::MetropolisResampler<BLOCKS, THREADS, float> metro;
    resampling::NicelySystematicResampler<BLOCKS, THREADS, float> naive_sys;
    resampling::NaiveStratifiedResampler<BLOCKS, THREADS, float> naive_strat;

    Resamplers(int seed) : seg_metro(seed), seg_mego(seed), metro_c2(seed) {}

    // Cache the number of iteration for a metropolis based algorithm into precompute_iters
    void compute_iters(ResampleBuffers &buffers, float ratio, float e)
    {
        precompute_iters = mego.determine_iters(
            buffers.dev_weights,
            buffers.size,
            buffers.output_size,
            ratio,
            e);
    }

    // Get the current choice for iterations for metropolis based algorithms based on selected
    // iter mode
    size_t get_iters(ResampleBuffers &buffers, IterMode mode, size_t def, float ratio, float e)
    {
        if (mode == IterMode::Dynamic)
        {
            return mego.determine_iters(
                buffers.dev_weights,
                buffers.size,
                buffers.output_size,
                ratio,
                e);
        }
        else if (mode == IterMode::PreCompute)
        {
            return precompute_iters;
        }

        return def;
    }
};

// Execute one resample based on the selected resampling method.
// Returns [MSE, execution_time]
std::array<float, 2> run_resample_error(
    ResampleMethod method,
    Resamplers &res,
    ResampleBuffers &buffers,
    bool display,
    float epsilon,
    resampling::Box<curandState> &rand,
    xorshift &rng)
{
    bool calc_offspring = true;
    auto start = std::chrono::steady_clock::now();
    if (method.method == ResampleType::Megopolis)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        // printf("iters: %i\n", iters);
        res.mego.set_iters(iters);
        res.mego.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::MegopolisAligned)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        res.mego_a.set_iters(iters);
        res.mego_a.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::Metropolis)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        res.metro.set_iters(iters);
        res.metro.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::Multinomial)
    {
        res.multi.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::Systematic)
    {
        res.sys.resample_save_offspring(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_offspring,
            buffers.output_size,
            buffers.size,
            rng);
        calc_offspring = false;
    }
    else if (method.method == ResampleType::NicelySystematic)
    {
        res.naive_sys.resample(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::SortSystematic)
    {
        buffers.sort_by_weight();
        res.sys.resample_save_offspring(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_offspring,
            buffers.output_size,
            buffers.size,
            rng);
        calc_offspring = false;
    }
    else if (method.method == ResampleType::Stratified)
    {
        res.naive_strat.resample(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::MetropolisC1)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        res.seg_metro.set_iters(iters);
        res.seg_metro.set_segment_size(method.segment_size);
        res.seg_metro.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::SegMegopolis)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        res.seg_mego.set_iters(iters);
        res.seg_mego.set_segment_size(method.segment_size);
        res.seg_mego.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }
    else if (method.method == ResampleType::MetropolisC2)
    {
        size_t iters = res.get_iters(buffers, method.mode, method.iters, method.iter_ratio, epsilon);
        res.metro_c2.set_iters(iters);
        res.metro_c2.set_segment_size(method.segment_size);
        res.metro_c2.resample_save_ancestors(
            buffers.dev_output,
            buffers.dev_input,
            buffers.dev_weights,
            buffers.dev_ancestors,
            buffers.output_size,
            buffers.size,
            resampling::to_raw(rand.get()),
            rng);
    }

    auto difference = std::chrono::steady_clock::now() - start;
    auto ns = difference.count();
    float time = static_cast<float>(ns) * 1e-9;

    if (calc_offspring)
    {
        buffers.calculate_offspring();
    }

    buffers.add_offspring_to_average();
    float error = buffers.resample_mse();

    if (display)
    {
        buffers.display();
        printf("\n");
    }
    return {error, time};
}

std::ofstream create_file(const char *path, bool &result)
{
    std::ofstream file;
    file.open(path, std::ofstream::trunc | std::ofstream::out);
    file.close();
    file.open(path, std::ios::in | std::ios::out);

    result = file.is_open();
    return file;
}

int main(int argc, char *argv[])
{
    rapidjson::Document dom = load_config(get_config_file(argc, argv));
    size_t seed = parse_uint(&dom, "seed");
    bool display, save;
    size_t iters, weight_sequences;
    float epsilon;
    std::vector<WeightParams> weight_params;
    std::vector<ResampleParticles> particle_values;
    std::vector<ResampleMethod> methods;
    std::string output_prefix;
    std::string output_suffix;
    Option<size_t> cache_pref;

    // Load parameters
    parse_field(&dom, "resample_methods", methods);
    parse_field(&dom, "weight_params", weight_params);
    parse_field(&dom, "iters", iters);
    parse_field(&dom, "weight_sequences", weight_sequences);
    parse_field(&dom, "display", display, false);
    parse_field(&dom, "particle_values", particle_values);
    parse_field(&dom, "output_prefix", output_prefix);
    parse_field(&dom, "output_suffix", output_suffix);
    parse_field(&dom, "save", save, false);
    parse_field(&dom, "epsilon", epsilon, 0.01f);
    parse_field(&dom, "cache_preference", cache_pref);

    if (cache_pref.is_some())
    {
        size_t pref = cache_pref.unwrap();
        if (pref < 4)
        {
            printf("Setting cache-preference %u\n", (unsigned)pref);
            gpuErrchk(cudaDeviceSetCacheConfig(((cudaFuncCache)pref)));
        }
        else
        {
            printf("Invalid cache-preference %u\n", (unsigned)pref);
        }
    }

    // unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::random_device rd;
    xorshift rng(rd);
    Resamplers resamplers(seed);
    resampling::Box<curandState> rand = resampling::generate_curand_states<BLOCKS, THREADS>(seed);
    std::vector<float> seq_biases;
    std::vector<float> biases;
    std::vector<float> new_biases;
    std::vector<float> bias_stddevs;
    std::vector<float> mse;
    std::vector<float> time;

    std::uint64_t particle_count = 0;
    for (auto &value : particle_values)
    {
        particle_count += value.input_size;
    }

    std::uint64_t iterations = weight_params.size() * iters * weight_sequences * methods.size();
    float count_div = particle_count * iterations;
    std::uint64_t count = 0;

    // For each resampling method execute the experiments
    for (auto &method : methods)
    {
        // Create an output file for the this resampling method
        std::string output_file = output_prefix;
        std::string method_name = method.to_string();
        output_file.append(method_name);
        output_file.append(output_suffix);

        bool result = true;
        auto logger = create_file(output_file.c_str(), result);
        logger.setf(std::ios::scientific, std::ios::floatfield);
        logger << "particles,wmethod,a,b,mse,mse_stddev,bias,avg_bias_stddev,bias_stddev,bias_stddev2,time,time_stddev\n";
        printf("----%s----\n", method_name.c_str());

        // For each particle count execute the experiments
        for (auto &particles : particle_values)
        {
            // Create the resampling buffers to be the correct size
            ResampleBuffers buffers(particles.input_size, particles.output_size);
            printf("----Particles: [%i, %i]----\n", (int)particles.input_size, (int)particles.input_size);

            // For each set of weight distribution parameters execute experiments
            for (auto &weight_param : weight_params)
            {
                biases.clear();
                bias_stddevs.clear();
                mse.clear();
                time.clear();
                std::vector<std::thread> threads;

                // Generate sequences in parallel
                buffers.gen_sequences(threads, weight_param, weight_sequences, 4);

                // Run resampling tests on all the generated weight sequences
                while (buffers.next_weight_seq())
                {
                    // Setup weight buffers
                    buffers.reset_weights();
                    buffers.reset_offspring_average();
                    buffers.calculate_expected_offspring();

                    seq_biases.clear();

                    // If iter method is precompute then we compute the number of metropolis
                    // iterations to use for this given weight sequence
                    if (method.mode == IterMode::PreCompute)
                    {
                        resamplers.compute_iters(buffers, method.iter_ratio, epsilon);
                    }

                    float error = 0.0;
                    float runtime = 0.0;
                    float avg_norm = 1.0 / (float)iters;
                    float particle_norm = 1.0 / (float)buffers.size;

                    // Run resampling on the given weight sequence 'iter' times
                    for (size_t j = 0; j < iters; j++)
                    {
                        count += particles.input_size;

                        // Simple progress bar to track execution state
                        clear_progress(70, "");
                        print_progress((float)count / count_div, 70);

                        buffers.reset();
                        auto stats = run_resample_error(
                            method,
                            resamplers,
                            buffers,
                            display,
                            epsilon,
                            rand,
                            rng);
                        error += stats[0];
                        runtime += stats[1];
                        float seq_bias = buffers.calculate_offspring_bias();
                        seq_bias *= particle_norm;
                        seq_biases.push_back(seq_bias);
                    }

                    // Calculate stats
                    MeanStddev bias_stats(seq_biases);
                    float bias = buffers.calculate_bias() * particle_norm;

                    error = error * avg_norm * particle_norm;
                    runtime = runtime * avg_norm;

                    clear_progress(70, "");

                    mse.push_back(error);
                    new_biases.push_back(bias_stats.mean);
                    biases.push_back(bias);
                    bias_stddevs.push_back(bias_stats.stddev);
                    time.push_back(runtime);
                }

                // Ensure all threads close properly
                for (size_t t = 0; t < threads.size(); t++)
                {
                    threads[t].join();
                }

                // Calculate mean and stddev of results
                MeanStddev bias_stat(biases);
                MeanStddev bias_stddev_stat(bias_stddevs);
                MeanStddev mse_stat(mse);
                MeanStddev time_stat(time);

                // Provide some runtime info
                printf("Error: %f, bias: %f, bias_stddev: %f, runtime: %f\n", mse_stat.mean, bias_stat.mean, bias_stddev_stat.mean, time_stat.mean);

                // Output stats to file
                logger << particles.input_size << ',';
                logger << weight_param.method_name() << ',';
                logger << weight_param.a << ',';
                logger << weight_param.b << ',';
                logger << mse_stat.mean << ',' << mse_stat.stddev << ',';
                logger << bias_stat.mean << ',' << bias_stat.stddev << ',';
                logger << bias_stddev_stat.mean << ',' << bias_stddev_stat.stddev << ',';
                logger << time_stat.mean << ',' << time_stat.stddev;
                logger << '\n';
            }
            clear_progress(70, "Done\n");
        }

        // Finish writing to logging file
        logger.flush();
        logger.close();
    }

    return 0;
}