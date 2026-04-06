#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

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
float bilinear_interpolation(const float *grid, const float i, const float j);
int8_t bilinear_interpolation(const int8_t *grid, const float i, const float j);

float extract_grid_value(const float *grid, const float x, const float y, const float xc, const float yc);
Eigen::Vector3f extract_grid_gradient(const float *grid, const float x, const float y, const float xc, const float yc, const float x_eps = 0.2f, const float y_eps = 0.2f);
Eigen::Matrix3f extract_grid_hessian(const float *grid, const float x, const float y, const float xc, const float yc, const float x_eps = 0.2f, const float y_eps = 0.2f);