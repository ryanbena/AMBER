#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>

Timer::Timer(bool print) : print_(print){}

void Timer::start() {
    start_time = std::chrono::steady_clock::now();
}

float Timer::time() {
    end_time = std::chrono::steady_clock::now();
    duration = end_time - start_time;
    float dur_ms =  duration.count()*1.0e3f;
    if (print_)
        std::cout << dur_ms << " ms" << std::endl;
    start_time = end_time;
    return dur_ms;
}

float Timer::time(std::string info) {
    if (print_)
        std::cout << info;
    return time();
}

std::string getCurrentDateTime(void){
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert to tm struct
    std::tm now_tm;
    localtime_r(&now_time_t, &now_tm); // Linux/macOS

    // Format date/time string
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();

}