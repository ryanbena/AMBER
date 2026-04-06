#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>

#include "utils.h"
#include "definitions.h"

#include <Eigen/Sparse>
#include <Eigen/Geometry>

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

/* Convert x Coordinate to j Index*/
float x_to_j(const float x, const float xc){

    return (x-xc) / DS;

}
  
/* Convert y Coordinate to i Index */
float y_to_i(const float y, const float yc){

    return (y-yc) / DS;

}

/* Compute difference between two angles wrapped between [-pi, pi] */
float ang_diff(const float a1, const float a2){
    
    float a3 = a1 - a2;
    while(a3 <= -M_PI){
        a3 += 2.0f*M_PI;
    }
    while(a3 > M_PI){
        a3 -= 2.0f*M_PI;
    }
    return a3;

}

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float ic = std::clamp(i, 0.0f, (float)(IMAX-2));
    const float jc = std::clamp(j, 0.0f, (float)(JMAX-2));

    const float i0 = std::floor(ic);
    const float j0 = std::floor(jc);
    const float i1 = i0 + 1.0f;
    const float j1 = j0 + 1.0f;

    const float f00 = grid[(int)i0*JMAX+(int)j0];
    const float f10 = grid[(int)i1*JMAX+(int)j0];
    const float f01 = grid[(int)i0*JMAX+(int)j1];
    const float f11 = grid[(int)i1*JMAX+(int)j1];
    
    const float wi = i - i0;
    const float wj = j - j0;

    const float f0 = (1.0f - wi) * f00 + wi * f10;
    const float f1 = (1.0f - wi) * f01 + wi * f11;

    const float f = (1.0f - wj) * f0 + wj * f1;

    return f;

};


int8_t bilinear_interpolation(const int8_t *grid, const float i, const float j){

    const float ic = std::clamp(i, 0.0f, (float)(IMAX-2));
    const float jc = std::clamp(j, 0.0f, (float)(JMAX-2));
    
    const float i0 = std::floor(ic);
    const float j0 = std::floor(jc);
    const float i1 = i0 + 1.0f;
    const float j1 = j0 + 1.0f;

    const float f00 = (float)grid[(int)i0*JMAX+(int)j0];
    const float f10 = (float)grid[(int)i1*JMAX+(int)j0];
    const float f01 = (float)grid[(int)i0*JMAX+(int)j1];
    const float f11 = (float)grid[(int)i1*JMAX+(int)j1];

    const float wi = i - i0;
    const float wj = j - j0;

    const float f0 = (1.0f - wi) * f00 + wi * f10;
    const float f1 = (1.0f - wi) * f01 + wi * f11;

    const float f = (1.0f - wj) * f0 + wj * f1;

    return (int8_t)std::round(f);

};

/* Extract the Value from a Numerical Grid using Bilinear Interpolation */
float extract_grid_value(const float *grid, const float x, const float y, const float xc, const float yc){

    const float ic = y_to_i(y, yc);
    const float jc = x_to_j(x, xc);            
    
    const float value = bilinear_interpolation(grid, ic, jc);
    
    return value;

};

/* Extract the Gradient from a Numerical Grid using a Central Difference Scheme */
Eigen::Vector3f extract_grid_gradient(const float *grid, const float x, const float y, const float xc, const float yc, const float x_eps, const float y_eps){

    const float xp = x + x_eps;
    const float xm = x - x_eps;
    const float yp = y + y_eps;
    const float ym = y - y_eps;

    const float hxp = extract_grid_value(grid, xp, y, xc, yc);
    const float hxm = extract_grid_value(grid, xm, y, xc, yc);
    const float hyp = extract_grid_value(grid, x, yp, xc, yc);
    const float hym = extract_grid_value(grid, x, ym, xc, yc);

    Eigen::Vector3f gradient{(hxp - hxm) / (xp - xm), (hyp - hym) / (yp - ym), 0.0f};

    return gradient;

};

/* Extract the Hessian from a Numerical Grid using a Central Difference Scheme */
Eigen::Matrix3f extract_grid_hessian(const float *grid, const float x, const float y, const float xc, const float yc, const float x_eps, const float y_eps){
    
    const float xp = x + x_eps;
    const float xm = x - x_eps;
    const float yp = y + y_eps;
    const float ym = y - y_eps;
    
    const float h = extract_grid_value(grid, x, y, xc, yc);
    const float hxp = extract_grid_value(grid, xp, y, xc, yc);
    const float hxm = extract_grid_value(grid, xm, y, xc, yc);
    const float hyp = extract_grid_value(grid, x, yp, xc, yc);
    const float hym = extract_grid_value(grid, x, ym, xc, yc);
    const float hpp = extract_grid_value(grid, xp, yp, xc, yc);
    const float hpm = extract_grid_value(grid, xp, ym, xc, yc);
    const float hmp = extract_grid_value(grid, xm, yp, xc, yc);
    const float hmm = extract_grid_value(grid, xm, ym, xc, yc);
    
    const float d2hdx2 = (hxp + hxm - 2.0f*h) / (x_eps * x_eps);
    const float d2hdy2 = (hyp + hym - 2.0f*h) / (y_eps * y_eps);
    const float d2hdxdy = (hpp + hmm - hpm - hmp) / (4.0f * x_eps * y_eps);
    
    Eigen::Matrix3f hessian;
    hessian.row(0) << d2hdx2, d2hdxdy, 0.0f;
    hessian.row(1) << d2hdxdy, d2hdy2, 0.0f;
    hessian.row(2) << 0.0f, 0.0f, 0.0f;

    return hessian;

};