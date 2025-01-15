#include "../inc/utils.h"

Timer::Timer(bool print) : print_(print){};

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

scalar_t Timer::time() {
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    scalar_t dur_ms =  duration.count()*1e-6;
    if (print_)
        std::cout << dur_ms << " ms" << std::endl;
    start_time = end_time;
    return dur_ms;
}

scalar_t Timer::time(std::string info) {
    if (print_)
        std::cout << info;
    return time();
}
