#include "../inc/utils.h"
#include "../inc/poisson.h"
#include <math.h>

Timer::Timer(bool print) : print_(print){};

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

float Timer::time() {
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    float dur_ms =  duration.count()*1e-6f;
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

/* Auxiliary Functions */
float softMin(const float x0, const float xmin, const float alpha){
    float xf = xmin + logf(1.0f+expf(alpha*(x0-xmin))) / alpha;
    return xf;
};

float softMax(const float x0, const float xmax, const float alpha){
    float xf = xmax - logf(1.0f+expf(alpha*(xmax-x0))) / alpha;
    return xf;
};

/* Perform a trilinear interpolation on a 3-D grid */
float trilinear_interpolation(const float *grid, const float i, const float j, const float k){

    const float i1f = floor(i);
    const float j1f = floor(j);
    const float k1f = floor(k);
    const float i2f = ceil(i);
    const float j2f = ceil(j);
    const float k2f = ceil(k);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    int k1 = (int)k1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;
    int k2 = (int)k2f;

    if((i1 != i2) && (j1 != j2) && (k1 != k2)){
        const float f11 = (k2f - k) * grid[k1*imax*jmax+i1*jmax+j1] + (k - k1f) * grid[k2*imax*jmax+i1*jmax+j1];
        const float f12 = (k2f - k) * grid[k1*imax*jmax+i1*jmax+j2] + (k - k1f) * grid[k2*imax*jmax+i1*jmax+j2];
        const float f21 = (k2f - k) * grid[k1*imax*jmax+i2*jmax+j1] + (k - k1f) * grid[k2*imax*jmax+i2*jmax+j1];
        const float f22 = (k2f - k) * grid[k1*imax*jmax+i2*jmax+j2] + (k - k1f) * grid[k2*imax*jmax+i2*jmax+j2];
        const float f1 = (i2f - i) * f11 + (i - i1f) * f21;
        const float f2 = (i2f - i) * f12 + (i - i1f) * f22;
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if((i1 != i2) && (k1 != k2)){
        const float f1 = (i2f - i) * grid[k1*imax*jmax+i1*jmax+(int)j] + (i - i1f) * grid[k1*imax*jmax+i2*jmax+(int)j];
        const float f2 = (i2f - i) * grid[k2*imax*jmax+i1*jmax+(int)j] + (i - i1f) * grid[k2*imax*jmax+i2*jmax+(int)j];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((j1 != j2) && (k1 != k2)){
        const float f1 = (j2f - j) * grid[k1*imax*jmax+(int)i*jmax+j1] + (j - j1f) * grid[k1*imax*jmax+(int)i*jmax+j2];
        const float f2 = (j2f - j) * grid[k2*imax*jmax+(int)i*jmax+j1] + (j - j1f) * grid[k2*imax*jmax+(int)i*jmax+j2];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2 - i) * grid[(int)k*imax*jmax+i1*jmax+j1] + (i - i1) * grid[(int)k*imax*jmax+i2*jmax+j1];
        const float f2 = (i2 - i) * grid[(int)k*imax*jmax+i1*jmax+j2] + (i - i1) * grid[(int)k*imax*jmax+i2*jmax+j2];
        return (j2 - j) * f1 + (j - j1) * f2;
    }
    else if(k1 != k2){
        return (k2 - k) * grid[k1*imax*jmax+(int)i*jmax+(int)j] + (k - k1) * grid[k2*imax*jmax+(int)i*jmax+(int)j];
    }
    else if(i1 != i2){
        return (i2 - i) * grid[(int)k*imax*jmax+i1*jmax+(int)j] + (i - i1) * grid[(int)k*imax*jmax+i2*jmax+(int)j];
    }
    else if(j1 != j2){
        return (j2 - j) * grid[(int)k*imax*jmax+(int)i*jmax+j1] + (j - j1) * grid[(int)k*imax*jmax+(int)i*jmax+j2];
    }
    else{
        return grid[(int)k*imax*jmax+(int)i*jmax+(int)j];
    }

};

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float i1f = floor(i);
    const float j1f = floor(j);
    const float i2f = ceil(i);
    const float j2f = ceil(j);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;

    if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2 - i) * grid[i1*jmax+j1] + (i - i1) * grid[i2*jmax+j1];
        const float f2 = (i2 - i) * grid[i1*jmax+j2] + (i - i1) * grid[i2*jmax+j2];
        return (j2 - j) * f1 + (j - j1) * f2;
    }
    else if(i1 != i2){
        return (i2 - i) * grid[i1*jmax+(int)j] + (i - i1) * grid[i2*jmax+(int)j];
    }
    else if(j1 != j2){
        return (j2 - j) * grid[(int)i*jmax+j1] + (j - j1) * grid[(int)i*jmax+j2];
    }
    else{
        return grid[(int)i*jmax+(int)j];
    }

};
