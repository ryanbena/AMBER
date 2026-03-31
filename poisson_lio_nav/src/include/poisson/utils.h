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

float x_to_j(const float x, const float xc);
float y_to_i(const float y, const float yc);
float ang_diff(const float a1, const float a2);
void low_pass(std::vector<float>& v_filter, const std::vector<float>& v_new, const float wc, const float dt);
float bilinear_interpolation(const float *grid, const float i, const float j, const int buffer);
int8_t bilinear_interpolation(const int8_t *grid, const float i, const float j, const int buffer);
std::string getCurrentDateTime(void);