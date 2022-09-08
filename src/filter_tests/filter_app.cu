// #include "particles.cuh"
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <chrono>

#include "particles.cuh"
#include "../resampling/resampling.cuh"
#include "../resampling/random.cuh"
#include "../benchmarks/resample_config.hpp"
#include "../util/utility.h"
#include "../logging/logger.hpp"
#include "../util/progress_bar.hpp"

const size_t THREADS = 256;
const size_t BLOCKS = 512;

struct Results
{
    double update_time;
    double estimate_time;
    double resample_time;
    double avg_iters;
    std::vector<std::vector<float>> rsmes;

    Results() : update_time(0.0), estimate_time(0.0), resample_time(0.0), avg_iters(0) {}

    float calculate_rsme(size_t runs)
    {
        float rsmes_avg = 0.0;

        for (auto &rsme : rsmes)
        {
            float norm = 1.0 / (float)runs;
            float sum = 0.0;

            for (size_t i = 1; i < rsme.size(); i++)
            {
                // Use norm to get average error at time i for this trajectory
                sum += sqrt(rsme[i] * norm);
            }

            // Add the average
            rsmes_avg += sum / (float)(rsme.size() - 1);
        }

        return rsmes_avg / (float)rsmes.size();
    }

    double calculate_avg_iters(size_t runs)
    {
        double resamples = (double)(runs * this->rsmes.size());

        return avg_iters / resamples;
    }

    void to_csv(std::ofstream &file, size_t runs)
    {
        double total = update_time + estimate_time + resample_time;

        file << update_time << ',' << update_time / total << ',';
        file << resample_time << ',' << resample_time / total << ',';
        file << estimate_time << ',' << estimate_time / total << ',';
        file << total << ',';

        file << this->calculate_rsme(runs) << ',';

        file << this->calculate_avg_iters(runs);
    }
};

struct Resamplers
{
    size_t precompute_iters;
    resampling::MegopolisResampler<BLOCKS, THREADS, float, std::mt19937> mego;
    resampling::MetropolisC1Resampler<BLOCKS, THREADS, float, std::mt19937> metro_c1;
    resampling::MetropolisC2Resampler<BLOCKS, THREADS, float, std::mt19937> metro_c2;
    resampling::MetropolisResampler<BLOCKS, THREADS, float, std::mt19937> metro;
    resampling::NicelySystematicResampler<BLOCKS, THREADS, float, std::mt19937> nicely_sys;

    Resamplers(int seed) : metro_c1(seed), metro_c2(seed) {}

    void set_iters(size_t iters)
    {
        mego.set_iters(iters);
        metro_c1.set_iters(iters);
        metro_c2.set_iters(iters);
        metro.set_iters(iters);
    }

    void set_segment_size(size_t segment_size)
    {
        metro_c1.set_segment_size(segment_size);
        metro_c2.set_segment_size(segment_size);
    }

    size_t compute_iters(float *weights, size_t size, float e)
    {
        return mego.determine_iters(
            weights,
            size,
            size,
            1.0,
            e);
        // printf("%i\n", (int) precompute_iters);
    }

    resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *get_resampler(ResampleType resample_type)
    {
        if (resample_type == ResampleType::Megopolis)
        {
            return (resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *)&mego;
        }
        else if (resample_type == ResampleType::Metropolis)
        {
            return (resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *)&metro;
        }
        else if (resample_type == ResampleType::MetropolisC1)
        {
            return (resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *)&metro_c1;
        }
        else if (resample_type == ResampleType::MetropolisC2)
        {
            return (resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *)&metro_c2;
        }
        else if (resample_type == ResampleType::NicelySystematic)
        {
            return (resampling::Resampler<std::mt19937, BLOCKS, THREADS, float> *)&nicely_sys;
        }
        else
        {
            return nullptr;
        }
    }

    size_t get_iters(float *weights, size_t size, IterMode mode, size_t def, float e)
    {
        if (mode == IterMode::Dynamic)
        {
            return mego.determine_iters(
                weights,
                size,
                size,
                1.0,
                e);
        }
        else if (mode == IterMode::PreCompute)
        {
            return def;
        }

        return def;
    }
};

struct Progress
{
    size_t run_count;
    size_t progress;
    float run_norm;
    float finish_norm;
    size_t size;

    Progress(size_t runs, size_t finish_count, size_t size) : run_count(0),
                                                              progress(0),
                                                              run_norm(1.0 / (float)runs),
                                                              finish_norm(1.0 / (float)finish_count),
                                                              size(size)
    {
    }

    void reprint()
    {
        printf("\033[A");
        clear_progress(this->size, "");
        print_progress((float)this->run_count * this->run_norm, this->size);
        printf("\n");
        clear_progress(this->size, "");
        print_progress((float)this->progress * this->finish_norm, this->size);
    }

    void finish_run()
    {
        this->run_count += 1;
        this->reprint();
    }

    void reset_runs()
    {
        this->run_count = 0;
    }

    void finish_trajectory()
    {
        this->progress += 1;
        this->reprint();
    }
};

std::vector<std::vector<float>> gen_trajectories(
    size_t count,
    size_t time_steps,
    std::mt19937 &rng,
    Gaussian gen_noise,
    Gaussian process_noise)
{
    std::vector<std::vector<float>> output;
    output.reserve(count);

    // Gaussian dist(0.0, 2.0);
    // Gaussian propagate_noise(0.0, sqrt(10.0));

    for (size_t i = 0; i < count; i++)
    {
        std::vector<float> trajectory;
        trajectory.reserve(time_steps);
        float truth = gen_noise.generate(rng);
        // float truth = 0.0;

        for (size_t j = 0; j < time_steps; j++)
        {
            trajectory.push_back(truth);
            truth = filter_ex::propagate_particle(j, truth, process_noise.generate(rng));
        }

        output.push_back(trajectory);
    }

    return output;
}

// Returns RSME
void run_trajectory(
    std::vector<float> &trajectory,
    Gaussian measurement_noise,
    std::mt19937 &rng,
    filter_ex::Filter<BLOCKS, THREADS> &filter,
    Resamplers &resamplers,
    ResampleMethod method,
    float resample_epsilon,
    Results &results,
    std::vector<float> &rsme)
{
    // Gaussian measurement_noise(0.0, 1.0);
    filter.initialize_particles(2.0);

    float update_time = 0.0;
    float resample_time = 0.0;
    float estimate_time = 0.0;
    size_t iter_count = 0;
    auto resampler = resamplers.get_resampler(method.method);
    resamplers.set_segment_size(method.segment_size);

    // Start at one due to filter/simulation timestep missmatch
    for (size_t i = 1; i < trajectory.size(); i++)
    {
        float current = trajectory[i];

        float measurement = current * current / 20.0 + measurement_noise.generate(rng);
        auto start = std::chrono::steady_clock::now();
        filter.propagate(measurement);
        auto difference = std::chrono::steady_clock::now() - start;
        auto ns = difference.count();
        double time = static_cast<double>(ns) * 1e-9;

        update_time += time;
        start = std::chrono::steady_clock::now();

        size_t iters = resamplers.get_iters(filter.get_weights(), filter.get_size(), method.mode, method.iters, resample_epsilon);
        iters = min(iters, (size_t)50000);
        resamplers.set_iters(iters);
        filter.resample(resampler, rng);

        difference = std::chrono::steady_clock::now() - start;
        ns = difference.count();
        time = static_cast<double>(ns) * 1e-9;

        resample_time += time;
        iter_count += iters;

        start = std::chrono::steady_clock::now();
        float estimate = filter.estimate_uniform_weights();
        difference = std::chrono::steady_clock::now() - start;
        ns = difference.count();
        time = static_cast<double>(ns) * 1e-9;
        estimate_time += time;

        // error = || estimate - true || ^ 2
        float error = estimate - current;
        error = error * error;

        if (isnan(error))
        {
            printf("----NAN-----current: %f, estimate: %f\n\n\n", current, estimate);
        }
        rsme[i] += error;
    }

    results.update_time += update_time;
    results.resample_time += resample_time;
    results.estimate_time += estimate_time;
    double avg_iters = (double)iter_count / (double)trajectory.size();
    results.avg_iters += avg_iters;
}

void run_particles(
    size_t particles,
    float measure_noise,
    float process_noise,
    Resamplers &resamplers,
    ResampleMethod method,
    std::string &method_name,
    float resample_epsilon,
    std::vector<std::vector<float>> &paths,
    size_t runs,
    Progress &progress,
    std::mt19937 &rng,
    std::ofstream &logger)
{
    Gaussian m_noise(0.0, measure_noise);
    filter_ex::Filter<BLOCKS, THREADS> filter(particles, measure_noise, process_noise);
    Results results;
    int time_steps = paths[0].size();
    progress.reset_runs();

    printf("----Method: %s, Particles: %i, Noise: %f----\n\n", method_name.c_str(), (int)particles, measure_noise);

    for (auto &trajectory : paths)
    {
        std::vector<float> rsme(trajectory.size(), 0.0);

        for (size_t i = 0; i < runs; i++)
        {
            progress.reprint();

            run_trajectory(
                trajectory,
                m_noise,
                rng,
                filter,
                resamplers,
                method,
                resample_epsilon,
                results,
                rsme);

            progress.finish_run();
        }
        results.rsmes.push_back(rsme);
        progress.finish_trajectory();
    }

    clear_progress(70, "");
    printf("Outputting results\n");
    logger << particles << ',' << paths.size() << ',' << time_steps << ',' << runs << ',' << measure_noise << ',';
    results.to_csv(logger, runs);
    logger << std::endl;
}

std::vector<std::vector<float>> get_trajectories_from_args(
    int argc,
    char *argv[],
    size_t count,
    size_t time_steps,
    std::mt19937 &rng,
    Gaussian gen_noise,
    Gaussian process_noise)
{
    std::vector<std::vector<float>> paths;

    if (argc <= 3)
    {
        paths = gen_trajectories(
            count,
            time_steps,
            rng,
            gen_noise,
            process_noise);

        return paths;
    }

    if (strcmp(argv[2], "-load") == 0)
    {
        if (argc < 4)
        {
            printf("Missing load file, generating...");
            paths = gen_trajectories(
                count,
                time_steps,
                rng,
                gen_noise,
                process_noise);

            return paths;
        }

        rapidjson::Document dom = load_config(argv[3]);
        parse_field(&dom, "paths", paths);

        return paths;
    }

    paths = gen_trajectories(
        count,
        time_steps,
        rng,
        gen_noise,
        process_noise);

    if (strcmp(argv[2], "-save") == 0)
    {
        std::string save = "paths.json";
        if (argc < 4)
        {
            printf("Missing save file, saving to paths.json");
        }
        else
        {
            save = std::string(argv[3]);
        }

        bool result = true;
        auto logger = logging::create_file(save.c_str(), result);

        if (!result)
        {
            printf("Failed to save to %s\n", save.c_str());
            return paths;
        }

        json_se::open_brace(logger);
        json_se::serialize_field_name(logger, "paths");
        json_se::open_bracket(logger);

        bool first = true;

        for (auto &path : paths)
        {
            if (!first)
            {
                json_se::comma(logger);
            }

            first = false;
            json_se::serialize_array(logger, path.data(), path.size());
        }

        json_se::close_bracket(logger);
        json_se::close_brace(logger);
    }

    return paths;
}

int main(int argc, char *argv[])
{
    rapidjson::Document dom = load_config(get_config_file(argc, argv));
    size_t trajectories;
    size_t runs;
    size_t time_steps;
    float epsilon;
    float gen_noise;
    float process_noise;
    std::vector<float> measure_noises;
    std::vector<size_t> particle_values;
    std::vector<ResampleMethod> methods;
    std::string prefix;
    std::string suffix;

    parse_field(&dom, "trajectories", trajectories);
    parse_field(&dom, "runs", runs);
    parse_field(&dom, "time_steps", time_steps);
    parse_field(&dom, "particle_values", particle_values);
    parse_field(&dom, "resample_methods", methods);
    parse_field(&dom, "output_prefix", prefix);
    parse_field(&dom, "output_suffix", suffix);
    parse_field(&dom, "epsilon", epsilon);
    parse_field(&dom, "measure_noises", measure_noises);
    parse_field(&dom, "gen_noises", gen_noise, 2.0f);
    parse_field(&dom, "process_noise", process_noise, 3.16f);

    Gaussian p_noise(0.0, process_noise);
    Gaussian g_noise(0.0, gen_noise);

    printf("Time steps: %i, runs: %i, trajectories: %i\n", (int)time_steps, (int)runs, (int)trajectories);

    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed1);
    // std::random_device rd;
    // std::mt19937 rng(rd);
    Resamplers resamplers(seed1);

    printf("Generating trajectories\n");
    std::vector<std::vector<float>> paths = get_trajectories_from_args(
        argc,
        argv,
        trajectories,
        time_steps,
        rng,
        g_noise,
        p_noise);

    // size_t progress_count = 0;
    // float finish_count = (float)1.0 / (trajectories * methods.size() * particle_values.size());
    // float run_count = (float)1.0 / (trajectories * runs);
    size_t combinations = trajectories * methods.size() * particle_values.size() * measure_noises.size();
    size_t method_runs = trajectories * runs;
    Progress progress(method_runs, combinations, 70);

    for (auto method : methods)
    {
        std::string output_file = prefix;
        std::string method_name = method.to_string();
        output_file.append(method_name);
        output_file.append(suffix);

        bool result = false;
        auto logger = logging::create_file(output_file.c_str(), result);

        if (result)
        {
            printf("Logging to %s\n", output_file.c_str());
        }
        else
        {
            printf("Not logging to %s\n", output_file.c_str());
        }

        logger << "particles,trajectories,steps,runs,noise,update,update_ratio,resample,resample_ratio,estimate,estimate_ratio,avg_time,rsme,avg_iters\n";

        for (auto noise : measure_noises)
        {
            for (auto particles : particle_values)
            {
                run_particles(
                    particles,
                    noise,
                    process_noise,
                    resamplers,
                    method,
                    method_name,
                    epsilon,
                    paths,
                    runs,
                    progress,
                    rng,
                    logger);
            }
        }
    }

    return 0;
}