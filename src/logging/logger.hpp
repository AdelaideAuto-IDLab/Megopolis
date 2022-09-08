#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <utility>
#include <stdio.h>
#include "../util/channel.hpp"
#include "messages.hpp"

namespace logging
{
    std::ofstream create_file(const char * path, bool &result) {
        std::ofstream file;
        file.open(path, std::ofstream::trunc | std::ofstream::out );
        file.close();
        file.open(path, std::ios::in | std::ios::out );

        result = file.is_open();
        return file;
    }

    class Logger {
        std::chrono::steady_clock::time_point start;
        Channel<LogMessage> channel;
        std::ofstream file;
        float time_multi;
        bool written_first;

    public: 
        Logger(float time_multi, const char * path, bool& result) :
            start(std::chrono::steady_clock::now()), 
            time_multi(time_multi),
            written_first(false) 
        {
            file.open(path, std::ofstream::trunc | std::ofstream::out );
            file.close();
            file.open(path, std::ios::in | std::ios::out );

            result = file.is_open();

            if (result) {
                file << "[\n\n]";
            }
        }

        Channel<LogMessage> get_channel() {
            return channel;
        }

        bool write(LogMessage &message, float time) {
            if (!file.is_open()) {
                return false;
            }

            if (written_first) {
                file << ",\n";
            }
            else {
                written_first = true;
            }

            file << '\t';
            file << '[' << time * time_multi << ',';            
            message.serialize(file);
            file << ']';

            return true;
        }

        void start_logging() {
            std::vector<LogMessage> messages;

            while (true) {
                channel.receive(messages);
                file.seekp(-2, std::ios_base::end);

                auto difference = std::chrono::steady_clock::now() - start;
                auto ns = difference.count();
                double time = static_cast<double>(ns) * 1e-9;
                for (auto& message : messages) {
                    this->write(message, time);
                }

                file << "\n]";
                file.flush();
                messages.clear();
            }
        }

    };

    std::pair<Channel<LogMessage>, std::thread> start_logger(float time_multi, const char * file, bool &result) {
        Logger logger(time_multi, file, result);
        Channel<LogMessage> channel = logger.get_channel();
        std::thread handle(&Logger::start_logging, std::move(logger));

        return std::make_pair(channel, std::move(handle));
    }
} // namespace Logging
