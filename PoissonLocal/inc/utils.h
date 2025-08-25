#pragma once
#include <chrono>
#include <iostream>
#include <fstream>

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

float j_to_x(const int j, const float xc);
float i_to_y(const int i, const float yc);
float q_to_yaw(const int q, const float yawc);
float x_to_j(const float x, const float xc);
float y_to_i(const float y, const float yc);
float yaw_to_q(const float yaw, const float yawc);
float ang_diff(const float a1, const float a2);
float softMin(const float x0, const float xmin, const float alpha);
float softMax(const float x0, const float xmax, const float alpha);
float q_wrap(float qi);
float trilinear_interpolation(const float *grid, const float i, const float j, const float k);
float bilinear_interpolation(const float *grid, const float i, const float j);
bool writeDataToFile(const bool flag, const float *data_ptr, const int data_length, const std::string& filename);