#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include "Types.h"

#define PRINT_TIMING false

class Timer
{
public: 
    Timer(bool print);
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::chrono::duration<scalar_t, std::nano> duration;

    void start();
    scalar_t time();
    scalar_t time(std::string info);

    bool print_;
};