#pragma once

#include <mutex>
#include <utility>
#include <memory>

// @Polish: Improve command line argument passing
const char* get_config_file(int argc, char *argv[]) {
    if (argc < 2) {
        // No argument passed so default to config.json
        return "config.json";
    }
    return argv[1];
}

// Optional value wrapper
template<typename T>
struct Option {
   T value;
   bool valid;

public:
   Option() : valid(false) {}

   Option(T value) : value(value), valid(true) {}

   T unwrap() {
      assert(valid);
      return value;
   }

   T unwrap_or(T other) {
      if (valid) {
         return value;
      }
      else {
         return other;
      }
   }
    
   T& unwrap_ref() {
       return value;
   }

   bool is_some() {
      return valid;
   }

   bool is_none() {
      return !valid;
   }
};

// Wrapper class for holding a guarded value
template <typename T>
class MutexValue {
    std::mutex guard; 
    T value;

public:
    MutexValue(T value) : value(std::move(value)) {}

    // Lock must be called before get_ref
    std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(guard);
    }

    T& get_ref() {
        return value;
    }
};

// Wrapper class for having a shared reference to a guarded value
template <typename T>
class ArcMutex {
    std::shared_ptr<MutexValue<T>> ptr; 

public:
    ArcMutex(T value) : ptr(new MutexValue<T>(std::move(value))) {}

    // Lock must be called before get_ref
    std::unique_lock<std::mutex> lock() {
        return ptr.get()->lock();
    }

    T& get_ref() {
        return ptr.get()->get_ref();
    }
};