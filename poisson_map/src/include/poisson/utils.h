#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#define PRINT_TIMING false

class Timer{
public: 
    Timer(bool print);
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::chrono::duration<float> duration;

    void start();
    float time();
    float time(std::string info);

    bool print_;
};

std::string getCurrentDateTime(void);