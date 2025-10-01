#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include <chrono>

#define PRINT_TIMING false

class Timer{
public: 
    Timer(bool print);
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::chrono::duration<float, std::nano> duration;

    void start();
    float time();
    float time(std::string info);

    bool print_;
};

float x_to_j(const float x, const float xc);
float y_to_i(const float y, const float yc);
float q_to_yaw(const int q, const float yawc);
float yaw_to_q(const float yaw, const float yawc);
float ang_diff(const float a1, const float a2);
float q_wrap(float qi);
float trilinear_interpolation(const float *grid, const float i, const float j, const float k);
float bilinear_interpolation(const float *grid, const float i, const float j);
std::string getCurrentDateTime(void);