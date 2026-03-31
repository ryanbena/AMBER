#include "utils.h"
#include "poisson.h"
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

/* Convert x Coordinate to j Index*/
float x_to_j(const float x, const float xc){

    return (x-xc) / DS;

}
  
/* Convert y Coordinate to i Index */
float y_to_i(const float y, const float yc){

    return (y-yc) / DS;

}

/* Low Pass Filter */
void low_pass(std::vector<float>& v_filter, const std::vector<float>& v_new, const float wc, const float dt){

    const float kc = 1.0f - std::exp(-wc*dt);
    for(char i=0; i<v_filter.size(); i++){
        v_filter[i] *= (1.0f - kc);
        v_filter[i] += kc * v_new[i];
    }

}

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j, const int buffer){

    const float ic = std::clamp(i, (float)buffer, (float)(IMAX-buffer-2)); // Saturated Because of Finite Grid Size
    const float jc = std::clamp(j, (float)buffer, (float)(JMAX-buffer-2)); // Numerical Derivatives Shrink Effective Grid Size  

    const float i1f = std::floor(ic);
    const float j1f = std::floor(jc);
    const float i2f = i1f + 1.0f;
    const float j2f = j1f + 1.0f;

    const float f1 = (i2f - i) * grid[(int)i1f*JMAX+(int)j1f] + (i - i1f) * grid[(int)i2f*JMAX+(int)j1f];
    const float f2 = (i2f - i) * grid[(int)i1f*JMAX+(int)j2f] + (i - i1f) * grid[(int)i2f*JMAX+(int)j2f];
    return (j2f - j) * f1 + (j - j1f) * f2;

}

int8_t bilinear_interpolation(const int8_t *grid, const float i, const float j, const int buffer){

    const float ic = std::clamp(i, (float)buffer, (float)(IMAX-buffer-2)); // Saturated Because of Finite Grid Size
    const float jc = std::clamp(j, (float)buffer, (float)(JMAX-buffer-2)); // Numerical Derivatives Shrink Effective Grid Size  

    const float i1f = std::floor(ic);
    const float j1f = std::floor(jc);
    const float i2f = i1f + 1.0f;
    const float j2f = j1f + 1.0f;

    const float f1 = (i2f - i) * (float)grid[(int)i1f*JMAX+(int)j1f] + (i - i1f) * (float)grid[(int)i2f*JMAX+(int)j1f];
    const float f2 = (i2f - i) * (float)grid[(int)i1f*JMAX+(int)j2f] + (i - i1f) * (float)grid[(int)i2f*JMAX+(int)j2f];
    return (int)std::round((j2f - j) * f1 + (j - j1f) * f2);

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