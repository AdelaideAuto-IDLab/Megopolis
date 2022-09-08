#pragma once

#include <thrust/for_each.h>

#include "buffer.cuh"
#include "../util/utility.h"
#include "../resampling/shared_ptr.cuh"
#include "../resampling/random.cuh"
#include "../resampling/megopolis.cuh"
#include <cstddef>
#include <memory>
#include <vector>
#include <utility>
#include <stdio.h>
#include <random>
#include <functional>

template <typename S, typename M>
struct SensorMeasurement
{
    S sensor;
    M measurement;

    SensorMeasurement() {}
    SensorMeasurement(S sensor, M measurement) : sensor(sensor), measurement(measurement) {}
};

template <typename T>
struct Estimate
{
    T state;
    float uncertainty;

    Estimate() {}
    Estimate(T state, float uncertainty) : state(state), uncertainty(uncertainty) {}
};

// Virtual class for defining a model to use to update particles
template <size_t BLOCKS, size_t THREADS, typename P, typename S, typename M>
class BernoulliModel
{
public:
    float ps;
    float pb;
    float lambda;
    float pdf_c;

    BernoulliModel() {}

    BernoulliModel(
        float ps,
        float pb,
        float lambda,
        float pdf_c) : ps(ps),
                       pb(pb),
                       lambda(lambda),
                       pdf_c(pdf_c)
    {
    }

    virtual std::vector<P> initial(size_t count, size_t seed) = 0;
    virtual P propagate_singular(float dt, P particle, std::mt19937 rng) = 0;

    virtual void propagate(
        float dt,
        float weight,
        P *buffer,
        float *weights,
        size_t count,
        thrust::device_ptr<curandState> rand) = 0;

    virtual void noise(float dt, ParticleBuffer<P> &buffer, thrust::device_ptr<curandState> rand) = 0;
    virtual void birth_particles(size_t count, float weighting, ParticleBuffer<P> &buffer, thrust::device_ptr<curandState> rand) = 0;
    virtual void likelihood(ParticleBuffer<P> &buffer, S &sensor, M measurement, float lambda, float pdf_c) = 0;
    virtual void likelihood_no_measurement(ParticleBuffer<P> &buffer, S &sensor, float lambda, float pdf_c) = 0;
    virtual Estimate<P> estimate(ParticleBuffer<P> &buffer) = 0;

    // Updates a collection of filters with the same kernel
    // Particles should be of size pcount
    // weights should be of size mcount * pcount
    virtual void multi_likelihood(
        SensorMeasurement<S, M> *measurements,
        size_t mcount,
        P *particles,
        float *weights,
        size_t pcount) = 0;

    // Updates particles states without affecting weights
    virtual void multi_propagate(
        float dt,
        P *particles,
        size_t count,
        thrust::device_ptr<curandState> rand) = 0;
};

// BLOCKS = Number of blocks to execute on gpu should be chosen to maximise occupancy on gpu
// THREADS = Number of threads per block
// Templated to improve compile time optimisations for gpu execution
//
// P = Type of particles
// S = Type of sensor
// M = Type of measurements from sensor
template <
    size_t BLOCKS,
    size_t THREADS,
    typename P,
    typename S,
    typename M,
    typename RESAMPLER = resampling::MegopolisResampler<BLOCKS, THREADS, P>>
class BernoulliFilter
{
    std::shared_ptr<BernoulliModel<BLOCKS, THREADS, P, S, M>> model;
    RESAMPLER resampler;
    ParticleBuffer<P> particles;
    float existence;
    size_t filter_particles;
    size_t birth_particles;

    resampling::Box<curandState> rand;

public:
    BernoulliFilter(int seed) : rand(resampling::generate_curand_states<BLOCKS, THREADS>(seed))
    {
    }

    BernoulliFilter(
        std::shared_ptr<BernoulliModel<BLOCKS, THREADS, P, S, M>> bernoulli_model,
        float existence,
        size_t count,
        size_t birth_particles,
        size_t seed) : model(std::move(bernoulli_model)),
                       particles(count + birth_particles),
                       existence(existence),
                       filter_particles(count),
                       birth_particles(birth_particles)
    {
        std::vector<P> initial = model.get()->initial(count, seed);
        particles.append_host_data(initial.data(), count);

        float weight = 1.0 / float(count);
        particles.set_weights(weight);

        rand = resampling::generate_curand_states<BLOCKS, THREADS>(seed);
    }

    // Implemented copy constructors because of Box<curandState>
    BernoulliFilter(const BernoulliFilter<BLOCKS, THREADS, P, S, M> &other) : model(other.model),
                                                                              particles(other.particles),
                                                                              existence(other.existence),
                                                                              filter_particles(other.filter_particles),
                                                                              birth_particles(other.birth_particles)
    {
        curandState *states;
        gpuErrchk(cudaMalloc(&states, BLOCKS * THREADS * sizeof(curandState)));
        gpuErrchk(cudaMemcpy(states, resampling::to_raw(other.rand.get()), BLOCKS * THREADS * sizeof(curandState), cudaMemcpyDeviceToDevice));

        rand = resampling::Box<curandState>(states);
    }

    // Does not copy curand states
    BernoulliFilter &operator=(const BernoulliFilter<BLOCKS, THREADS, P, S, M> &other)
    {
        model = other.model;
        particles = other.particles;
        existence = other.existence;
        filter_particles = other.filter_particles;
        birth_particles = other.birth_particles;
    }

    BernoulliFilter(BernoulliFilter<BLOCKS, THREADS, P, S, M> &other, size_t amount) : model(other.model),
                                                                                       existence(other.existence),
                                                                                       filter_particles(amount),
                                                                                       birth_particles(amount / 8),
                                                                                       particles(amount)
    {
        assert(amount <= other.particles.len());

        curandState *states;
        gpuErrchk(cudaMalloc(&states, BLOCKS * THREADS * sizeof(curandState)));
        gpuErrchk(cudaMemcpy(states, resampling::to_raw(other.rand.get()), BLOCKS * THREADS * sizeof(P), cudaMemcpyDeviceToDevice));

        rand = resampling::Box<curandState>(states);

        particles.append_device_data(other.particles.particles_begin(), amount);
    }

    ParticleBuffer<P> &get_particles()
    {
        return particles;
    }

    void reset_weights()
    {
        float weight = 1.0 / float(particles.len());

        particles.set_weights(weight);
    }

    void birth_new_particles()
    {
        size_t len = particles.len();
        float weight = model.get()->pb * (1.0 - existence) / float(birth_particles);

        model.get()->birth_particles(birth_particles, weight, particles, rand.get());
        particles.set_length(particles.len() + birth_particles);
    }

    float propagate_state(float dt)
    {
        float ps = model.get()->ps;
        float pb = model.get()->pb;

        float r_predict = pb * (1.0 - existence) + ps * existence;
        model.get()->propagate(
            dt,
            ps * existence,
            resampling::to_raw(particles.particles_begin()),
            resampling::to_raw(particles.weights_begin()),
            particles.len(),
            rand.get());

        if (r_predict < 0.0)
        {
            r_predict = 0.0;
        }
        else if (r_predict > 1.0)
        {
            r_predict = 1.0;
        }

        return r_predict;
    }

    float predict(float dt)
    {
        float r_predict = this->propagate_state(dt);
        gpuErrchk(cudaDeviceSynchronize());

        this->birth_new_particles();
        gpuErrchk(cudaDeviceSynchronize());
        return r_predict;
    }

    void update(float existence_pred, S &sensor, Option<M> measurement)
    {
        float prev_sum = particles.sum_weights();
        float lambda = model.get()->lambda;
        float pdf_c = model.get()->pdf_c;

        if (measurement.is_some())
        {
            model.get()->likelihood(particles, sensor, measurement.unwrap(), lambda, pdf_c);
        }
        else
        {
            model.get()->likelihood_no_measurement(particles, sensor, lambda, pdf_c);
        }

        // TODO: Remove Sychronize from here
        gpuErrchk(cudaDeviceSynchronize());
        float multi = particles.sum_weights() / prev_sum;
        existence = (existence_pred * multi) / ((lambda * pdf_c) * (1.0 - existence_pred) + existence_pred * multi);

        if (existence < 0.0)
        {
            existence = 0.0;
        }
        else if (existence > 1.0)
        {
            existence = 1.0;
        }
    }

    template <typename RNG>
    void resample(RNG &rng)
    {
        particles.swap();
        thrust::device_ptr<P> current = particles.particles_begin();
        thrust::device_ptr<P> previous = particles.previous_begin();
        thrust::device_ptr<float> weights = particles.weights_begin();

        // TODO: This does not check if the kernel excecutes properly only that it launches properly
        resampler.resample(
            resampling::to_raw(current),
            resampling::to_raw(previous),
            resampling::to_raw(weights),
            filter_particles,
            particles.len(),
            resampling::to_raw(rand.get()),
            rng);

        particles.set_length(filter_particles);
    }

    template <typename RNG>
    void step(float dt, S &sensor, Option<M> measurement, RNG &rng)
    {
        float existence_pred = this->predict(dt);
        this->update(existence_pred, sensor, measurement);
        this->resample(rng);
        float weights = 1.0 / float(particles.len());
        particles.set_weights(weights);
        model.get()->noise(dt, particles, rand.get());
    }

    Estimate<P> get_estimate()
    {
        BernoulliModel<BLOCKS, THREADS, P, S, M> *model_ptr = model.get();
        return model_ptr->estimate(particles);
    }

    float get_existence() const
    {
        return existence;
    }

    void set_existence(float e)
    {
        existence = e;
    }

    void copy_particles_to_host(P *host_ptr, size_t count) const
    {
        particles.copy_to_host(host_ptr, count, 0);
    }

    void copy_sample(size_t count, size_t offset, P *ptr, resampling::Device device) const
    {
        if (device == resampling::Device::Gpu)
        {
            thrust::device_ptr<P> dev_ptr(ptr);
            particles.copy_to_device(dev_ptr, count, offset);
        }
        else
        {
            particles.copy_to_host(ptr, count, offset);
        }
    }

    template <typename RNG>
    void copy_random_sample(size_t count, P *ptr, resampling::Device device, RNG &rng) const
    {
        size_t max_offset = particles.calculate_maximum_offset(count);
        std::uniform_int_distribution<size_t> dist(0, max_offset);
        size_t offset = dist(rng);

        this->copy_sample(count, offset, ptr, device);
    }

    void copy_random_sample_gpu(
        size_t count,
        thrust::device_ptr<P> ptr,
        curandState *rand) const
    {
        particles.template rand_copy_to_device<BLOCKS, THREADS>(ptr, count, rand);
    }

    template <typename RNG>
    size_t random_particle_offset(size_t count, RNG &rng) const
    {
        size_t max_offset = particles.calculate_maximum_offset(count);
        std::uniform_int_distribution<size_t> dist(0, max_offset);
        return dist(rng);
    }

    std::shared_ptr<BernoulliModel<BLOCKS, THREADS, P, S, M>> &get_model()
    {
        return model;
    }
};