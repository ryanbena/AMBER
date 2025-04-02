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

};

float softMin(const float x0, const float xmin, const float alpha){
    float xf = xmin + logf(1.0f+expf(alpha*(x0-xmin))) / alpha;
    return xf;
};

float softMax(const float x0, const float xmax, const float alpha){
    float xf = xmax - logf(1.0f+expf(alpha*(xmax-x0))) / alpha;
    return xf;
};

/* Wrap Angle Indices */
float q_wrap(float qi){

    float qf = qi;
    while(qf < 0.0f) qf += (float)QMAX;
    while(qf >= (float)QMAX) qf -= (float)QMAX;
    return qf;

}

/* Perform a trilinear interpolation on a 3-D grid */
float trilinear_interpolation(const float *grid, const float i, const float j, const float k){

    const float i1f = floorf(i);
    const float j1f = floorf(j);
    const float k1f = floorf(k);
    const float i2f = ceilf(i);
    const float j2f = ceilf(j);
    const float k2f = ceilf(k);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int k1 = (int)q_wrap(k1f);
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;
    const int k2 = (int)q_wrap(k2f);

    if((i1 != i2) && (j1 != j2) && (k1 != k2)){
        const float f11 = (k2f - k) * grid[k1*IMAX*JMAX+i1*JMAX+j1] + (k - k1f) * grid[k2*IMAX*JMAX+i1*JMAX+j1];
        const float f12 = (k2f - k) * grid[k1*IMAX*JMAX+i1*JMAX+j2] + (k - k1f) * grid[k2*IMAX*JMAX+i1*JMAX+j2];
        const float f21 = (k2f - k) * grid[k1*IMAX*JMAX+i2*JMAX+j1] + (k - k1f) * grid[k2*IMAX*JMAX+i2*JMAX+j1];
        const float f22 = (k2f - k) * grid[k1*IMAX*JMAX+i2*JMAX+j2] + (k - k1f) * grid[k2*IMAX*JMAX+i2*JMAX+j2];
        const float f1 = (i2f - i) * f11 + (i - i1f) * f21;
        const float f2 = (i2f - i) * f12 + (i - i1f) * f22;
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if((i1 != i2) && (k1 != k2)){
        const float f1 = (i2f - i) * grid[k1*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1f) * grid[k1*IMAX*JMAX+i2*JMAX+(int)j];
        const float f2 = (i2f - i) * grid[k2*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1f) * grid[k2*IMAX*JMAX+i2*JMAX+(int)j];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((j1 != j2) && (k1 != k2)){
        const float f1 = (j2f - j) * grid[k1*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1f) * grid[k1*IMAX*JMAX+(int)i*JMAX+j2];
        const float f2 = (j2f - j) * grid[k2*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1f) * grid[k2*IMAX*JMAX+(int)i*JMAX+j2];
        return (k2f - k) * f1 + (k - k1f) * f2;
    }
    else if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2 - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+j1] + (i - i1) * grid[(int)k*IMAX*JMAX+i2*JMAX+j1];
        const float f2 = (i2 - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+j2] + (i - i1) * grid[(int)k*IMAX*JMAX+i2*JMAX+j2];
        return (j2 - j) * f1 + (j - j1) * f2;
    }
    else if(k1 != k2){
        return (k2 - k) * grid[k1*IMAX*JMAX+(int)i*JMAX+(int)j] + (k - k1) * grid[k2*IMAX*JMAX+(int)i*JMAX+(int)j];
    }
    else if(i1 != i2){
        return (i2 - i) * grid[(int)k*IMAX*JMAX+i1*JMAX+(int)j] + (i - i1) * grid[(int)k*IMAX*JMAX+i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2 - j) * grid[(int)k*IMAX*JMAX+(int)i*JMAX+j1] + (j - j1) * grid[(int)k*IMAX*JMAX+(int)i*JMAX+j2];
    }
    else{
        return grid[(int)k*IMAX*JMAX+(int)i*JMAX+(int)j];
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
        const float f1 = (i2 - i) * grid[i1*JMAX+j1] + (i - i1) * grid[i2*JMAX+j1];
        const float f2 = (i2 - i) * grid[i1*JMAX+j2] + (i - i1) * grid[i2*JMAX+j2];
        return (j2 - j) * f1 + (j - j1) * f2;
    }
    else if(i1 != i2){
        return (i2 - i) * grid[i1*JMAX+(int)j] + (i - i1) * grid[i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2 - j) * grid[(int)i*JMAX+j1] + (j - j1) * grid[(int)i*JMAX+j2];
    }
    else{
        return grid[(int)i*JMAX+(int)j];
    }

};

bool writeDataToFile(const bool flag, const float *data_ptr, const int data_length, const std::string& filename){

    if(!flag){
        std::ofstream outFile(filename);
        if(outFile.is_open()){
            for(int n = 0; n < data_length; n++){
                outFile << data_ptr[n] << std::endl;
            }
            outFile.close();
        } 
        else{
            std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        }
    }
    return true;

};