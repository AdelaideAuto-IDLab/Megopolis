#pragma once

#include "helper.cuh"

#include <stdio.h>
#include <atomic>
#include <cstdint>

namespace resampling {
    template <typename T>
    class Box
    {
        thrust::device_ptr<T> ptr;

    public:
        Box(const Box<T> &) = delete;
        Box &operator=(const Box<T> &) = delete;

        Box() {}
        Box(T *ptr) : ptr(ptr) {}
        ~Box()
        {
            gpuErrchk(cudaFree(to_raw(ptr)));
        }

        Box(Box<T> &&other) : ptr(other.ptr)
        {
            T *null = nullptr;
            other.ptr = thrust::device_ptr<T>(null);
        }

        Box &operator=(Box<T> &&other)
        {
            gpuErrchk(cudaFree(to_raw(ptr)));
            ptr = other.ptr;
            T *null = nullptr;
            other.ptr = thrust::device_ptr<T>(null);

            return *this;
        }

        thrust::device_ptr<T> get() const
        {
            return ptr;
        }

        T *get_raw()
        {
            return to_raw(ptr);
        }
    };

    template <Device D, typename T>
    class Rc
    {
        std::atomic<std::uint64_t> *count;
        T *ptr;

        void free_data()
        {
            delete count;

            if (D == Device::Gpu)
            {
                gpuErrchk(cudaFree(ptr));
            }
            else
            {
                delete[] ptr;
            }
        }

    public:
        Rc() : count(nullptr), ptr(nullptr) {}

        // ptr must be to an array type pointer
        Rc(T *ptr) : ptr(ptr)
        {
            count = new std::atomic<std::uint64_t>(1);
        }

        ~Rc()
        {
            if (count == nullptr)
            {
                return;
            }

            std::uint64_t prev_count = count->fetch_sub(1);

            if (prev_count == 1)
            {
                this->free_data();
            }
        }

        Rc(Rc &&other) : count(other.count), ptr(other.ptr)
        {
            other.count = nullptr;
            other.ptr = nullptr;
        }

        Rc &operator=(Rc &&other)
        {
            if (count != nullptr)
            {
                std::uint64_t prev_count = count->fetch_sub(1);

                if (prev_count == 1)
                {
                    this->free_data();
                }
            }

            count = other.count;
            ptr = other.ptr;

            other.count = nullptr;
            other.ptr = nullptr;

            return *this;
        }

        Rc(const Rc &other) : count(other.count), ptr(other.ptr)
        {
            count->fetch_add(1);
        }

        Rc &operator=(const Rc &other)
        {
            if (count != nullptr)
            {
                std::uint64_t prev_count = count->fetch_sub(1);

                if (*count == 1)
                {
                    this->free_data();
                }
            }

            count = other.count;
            ptr = other.ptr;

            count->fetch_add(1);
            return *this;
        }

        // Copies array_size amount of T into a new Rc
        template <Device ND>
        Rc<ND, T> clone_to_device(size_t array_size)
        {
            T *new_ptr;
            size_t size = array_size * sizeof(T);

            if (ND == Device::Gpu)
            {
                gpuErrchk(cudaMalloc(&new_ptr, size));

                if (D == Device::Cpu)
                {
                    gpuErrchk(cudaMemcpy(new_ptr, ptr, size, cudaMemcpyHostToDevice));
                }
                else
                {
                    gpuErrchk(cudaMemcpy(new_ptr, ptr, size, cudaMemcpyDeviceToDevice));
                }
            }
            else
            {
                new_ptr = new T[array_size];

                if (D == Device::Gpu)
                {
                    gpuErrchk(cudaMemcpy(new_ptr, ptr, size, cudaMemcpyDeviceToHost));
                }
                else
                {
                    memcpy(new_ptr, ptr, size);
                }
            }

            return Rc<ND, T>(new_ptr);
        }

        __host__ __device__ T *raw()
        {
            return ptr;
        }

        // Returns non const ptr for passing to gpu in const function
        __host__ __device__ T *const_raw() const
        {
            return ptr;
        }
    };
}