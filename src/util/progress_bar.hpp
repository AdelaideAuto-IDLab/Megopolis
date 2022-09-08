#pragma once

#include <iostream>
#include <iomanip>
#include <math.h>

// Simple progress bar
void print_progress(float percent, size_t bar_len) {
    int filled = ceil(percent * (float) bar_len);
    filled = std::min((int) bar_len, std::max(0, filled));
    std::cout << std::setprecision(0);
    int out_percent = ceil(100.0 * percent);
    std::cout << " %" << out_percent << " [";

    for (auto i = 0; i < filled; i++) {
        std::cout << "=";
    } 

    for (size_t i = filled + 1; i < bar_len; i++) {
        std::cout << " ";
    }

    std::cout << "]\r";
    std::cout << std::flush;
}

// Must be called to clear the previously output progress bar.
// Works for outputing to standard out
void clear_progress(size_t bar_len, const char * message) {
    for (size_t i = 0; i < bar_len + 8; i++) {
        std::cout << " ";
    }

    std::cout << "\r";
    std::cout << message;
}