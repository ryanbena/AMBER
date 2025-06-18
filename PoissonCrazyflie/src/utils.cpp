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

void quat_rotate(float *vb, const float *vi, const float *q){

	float dp1 = q[1]*vi[0] + q[2]*vi[1] + q[3]*vi[2];
	float dp2 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    vb[0] = q[0]*q[0]*vi[0] + 2.0f*q[0]*(vi[1]*q[3] - vi[2]*q[2]) + 2.0f*dp1*q[1] - dp2*vi[0];
    vb[1] = q[0]*q[0]*vi[1] + 2.0f*q[0]*(vi[2]*q[1] - vi[0]*q[3]) + 2.0f*dp1*q[2] - dp2*vi[1];
    vb[2] = q[0]*q[0]*vi[2] + 2.0f*q[0]*(vi[0]*q[2] - vi[1]*q[1]) + 2.0f*dp1*q[3] - dp2*vi[2];

}

float softMin(const float x0, const float xmin, const float alpha){
    float xf = xmin + logf(1.0f+expf(alpha*(x0-xmin))) / alpha;
    return xf;
};

float softMax(const float x0, const float xmax, const float alpha){
    float xf = xmax - logf(1.0f+expf(alpha*(xmax-x0))) / alpha;
    return xf;
};

/* Perform a bilinear interpolation on a 2-D grid */
float bilinear_interpolation(const float *grid, const float i, const float j){

    const float i1f = floorf(i);
    const float j1f = floorf(j);
    const float i2f = ceilf(i);
    const float j2f = ceilf(j);

    const int i1 = (int)i1f;
    const int j1 = (int)j1f;
    const int i2 = (int)i2f;
    const int j2 = (int)j2f;

    if((i1 != i2) && (j1 != j2)){
        const float f1 = (i2f - i) * grid[i1*JMAX+j1] + (i - i1f) * grid[i2*JMAX+j1];
        const float f2 = (i2f - i) * grid[i1*JMAX+j2] + (i - i1f) * grid[i2*JMAX+j2];
        return (j2f - j) * f1 + (j - j1f) * f2;
    }
    else if(i1 != i2){
        return (i2f - i) * grid[i1*JMAX+(int)j] + (i - i1f) * grid[i2*JMAX+(int)j];
    }
    else if(j1 != j2){
        return (j2f - j) * grid[(int)i*JMAX+j1] + (j - j1f) * grid[(int)i*JMAX+j2];
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