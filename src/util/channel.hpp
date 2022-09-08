#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <utility>
#include <thread>


template<typename T>
struct MutexQueue {
    std::vector<T> values;
    bool written;
    std::mutex guard;
    std::condition_variable cv;

    MutexQueue() : written(false) {} 
};

template<typename T>
class Channel {
    std::shared_ptr<MutexQueue<T>> queue;

public:
    Channel() : queue(new MutexQueue<T>()) {}

    void send(T value) {
        MutexQueue<T> *guarded = queue.get();

        {
            std::lock_guard<std::mutex> lk(guarded->guard);
            guarded->values.emplace_back(std::move(value));
            guarded->written = true;
        }

        guarded->cv.notify_one();
    }

    void send_multiple(T * values, size_t amount) {
        MutexQueue<T> *guarded = queue.get();
        
        {
            std::lock_guard<std::mutex> lk(guarded->guard);

            for (size_t i = 0; i < amount; i++) {
                guarded->values.push_back(values[i]);
            }
            guarded->written = true;
        }

        guarded->cv.notify_one();
    }

    // Swaps the underlying 'values' vector with 'into'. Blocking operation
    void receive(std::vector<T>& into) {
        MutexQueue<T> *guarded = queue.get();
        std::unique_lock<std::mutex> lk(guarded->guard);

        guarded->cv.wait(lk, [&] { return guarded->written; });
        guarded->written = false;
        guarded->values.swap(into);
        guarded->values.clear();

        lk.unlock();
    }

    void try_receive(std::vector<T>& into) {
        MutexQueue<T> *guarded = queue.get();
        std::unique_lock<std::mutex> lk(guarded->guard);

        if (!guarded->written) {
            lk.unlock();
            return;
        }

        guarded->written = false;
        guarded->values.swap(into);
        guarded->values.clear();

        lk.unlock();
    }
};

template<typename T>
void one_to_many_channels(Channel<T> receiver, std::vector<Channel<T>> channels) {
    std::vector<T> buffer;

    while(true) {
        receiver.receive(buffer);

        for (auto& channel : channels) {
            channel.send(buffer.data(), buffer.size());
        }
    }
}

template<typename T>
std::pair<std::thread, Channel<T>> combine_sending_channels(std::vector<Channel<T>> channels) {
    Channel<T> receiver;

    std::thread thread(one_to_many_channels<T>, receiver, std::move(channels));

    return make_pair(thread, receiver, std::move(channels));
}