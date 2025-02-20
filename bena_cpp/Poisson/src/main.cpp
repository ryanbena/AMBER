#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

#include "../inc/kernel.hpp"
#include "../inc/Types.h"
#include "../inc/utils.h"

#include <Eigen/Core>

void writeMatrixToFile(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << matrix.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
        outFile.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
    }
}

double softMin(const double x0, const double xmin, const double alpha){
    double xf = xmin + log(1.0+exp(alpha*(x0-xmin))) / alpha;
    return xf;
}

double softMax(const double x0, const double xmax, const double alpha){
    double xf = xmax - log(1.0+exp(alpha*(xmax-x0))) / alpha;
    return xf;
}

const double xmax = 4.0; // grid x dimension
const double ymax = 4.0; // grid y dimension
const double ds = 0.01; // grid resolution

const int grid_imax = round(xmax/ds);
const int grid_jmax = round(ymax/ds);

const double drone_radius = 0.25;

Eigen::MatrixXd hgrid = Eigen::MatrixXd::Zero(grid_imax, grid_jmax);

// Main
int main(void){

    Timer timer(true);
    timer.start();   

    const int imax = grid_imax;
    const int jmax = grid_jmax;
    const double dA = ds * ds; //grid cell area

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(imax, jmax);
    Eigen::MatrixXd sdf = Eigen::MatrixXd::Zero(imax, jmax);
    Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(imax, jmax);
    Eigen::MatrixXd vy = Eigen::MatrixXd::Zero(imax, jmax);
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(imax, jmax);

    const double h0 = 0.0; // Set boundary level set value
    const double dh0 = 1.0; // Set dh Value

    hgrid.setConstant(h0);
    b.setConstant(1.0);
    vx.setConstant(0.0);
    vy.setConstant(0.0);
    f.setConstant(0.0);

    // Set Border    
    b.block(0, 0, imax, 1).setConstant(0.0);
    b.block(0, 0, 1, jmax).setConstant(0.0);
    b.block(0, jmax-1, imax, 1).setConstant(0.0);
    b.block(imax-1, 0, 1, jmax).setConstant(0.0);

    // Set Box Corners
    Eigen::MatrixXd xy_box(4,4); // xmin, ymin, width, height
    xy_box.row(0) << 0.60, 0.88, 0.80, 0.32;
    xy_box.row(1) << 0.60, 1.20, 0.40, 0.16;
    xy_box.row(2) << 1.24, 1.56, 1.36, 0.40;
    xy_box.row(3) << 0.84, 2.76, 0.28, 0.28;
    
    // Find Indices
    Eigen::MatrixXd ij_box(4,4); // xmin, ymin, width, height
    ij_box.row(0) << round(xy_box(0,0)/ds), round(xy_box(0,1)/ds), round(xy_box(0,2)/ds), round(xy_box(0,3)/ds);
    ij_box.row(1) << round(xy_box(1,0)/ds), round(xy_box(1,1)/ds), round(xy_box(1,2)/ds), round(xy_box(1,3)/ds);
    ij_box.row(2) << round(xy_box(2,0)/ds), round(xy_box(2,1)/ds), round(xy_box(2,2)/ds), round(xy_box(2,3)/ds);
    ij_box.row(3) << round(xy_box(3,0)/ds), round(xy_box(3,1)/ds), round(xy_box(3,2)/ds), round(xy_box(3,3)/ds);
    
    // Assign Occupancy
    for(int k = 0; k < ij_box.rows(); k++){
        b.block(ij_box(k,0), ij_box(k,1), ij_box(k,2), ij_box(k,3)).setConstant(-1.0); // Fill Cells with -1.0
    }

    // Find Boundaries (Any Occupied Point that Borders an Unoccupied Point)
    Eigen::MatrixXd b0(imax, jmax);
    memcpy(b0.data(), b.data(), sizeof(double)*imax*jmax);
    for(int i = 1; i < imax-1; i++){
        for(int j = 1; j < jmax-1; j++){
            if(b0(i,j)==1.0){
                const double neighbors = b0(i+1,j) + b0(i-1,j) + b0(i,j+1) + b0(i,j-1) + b0(i+1,j+1) + b0(i-1,j+1) + b0(i-1,j-1) + b0(i+1,j-1);
                if(neighbors<8.0){
                    b(i,j) = 0.0;
                    hgrid(i,j) = h0;
                }
            }
        }
    }

    // Solve SDF to Inflate Occupancy Grid
    sdf = hgrid;
    for(int i = 0; i < imax; i++){
        for(int j = 0; j < jmax; j++){
            double min_dist_sq = 1.0e10;
            int x = 0;
            double distx_sq = 0.0;
            while(distx_sq < min_dist_sq){
                int y = 0;
                double dist_sq = distx_sq;
                while(dist_sq < min_dist_sq){
                    // Check if we hit a boundary
                    bool free_space = true;
                    try{free_space &= (bool)b(i+x,j+y);}
                    catch(...){};
                    try{free_space &= (bool)b(i+x,j-y);}
                    catch(...){};
                    try{free_space &= (bool)b(i-x,j+y);}
                    catch(...){};
                    try{free_space &= (bool)b(i-x,j-y);}
                    catch(...){};
                    // If we did, update the minimum distance
                    if(!free_space) min_dist_sq = dist_sq;
                    y++;
                    dist_sq = distx_sq + dA*(double)(y*y);
                }
                x++;
                distx_sq = dA * (double)(x*x);
            }
            sdf(i,j) = sqrt(min_dist_sq) * b(i,j)- ceil(drone_radius/ds) * ds;
        
        }
    }

    // Recompute Inflated Occupancy Grid Using SDF
    b.setConstant(1.0);
    for(int i = 0; i < imax; i++){
        for(int j = 0; j < jmax; j++){
            if(sdf(i,j) < 0.0) b(i,j) = -1.0;
        }
    }
    memcpy(b0.data(), b.data(), sizeof(double)*imax*jmax);
    for(int i = 1; i < imax-1; i++){
        for(int j = 1; j < jmax-1; j++){
            if(b0(i,j)==1.0){
                const double neighbors = b0(i+1,j) + b0(i-1,j) + b0(i,j+1) + b0(i,j-1) + b0(i+1,j+1) + b0(i-1,j+1) + b0(i-1,j-1) + b0(i+1,j-1);
                if(neighbors<8.0){
                    b(i,j) = 0.0;
                    hgrid(i,j) = h0;
                }
            }
        }
    }
    // Reset Border    
    b.block(0, 0, imax, 1).setConstant(0.0);
    b.block(0, 0, 1, jmax).setConstant(0.0);
    b.block(0, jmax-1, imax, 1).setConstant(0.0);
    b.block(imax-1, 0, 1, jmax).setConstant(0.0);

    // Compute Boundary Gradients from Occupancy Grid
    const int blend = 1; // How Many Pixels Will Be Used to Blend Gradients (>= 1, <= buffer)
    for(int i = blend; i < imax-blend; i++){
        for(int j = blend; j < jmax-blend; j++){
            if(!b(i,j)){
                vx(i,j) = 0.0;
                vy(i,j) = 0.0;
                for(int p = -blend; p <= blend; p++){
                    for(int q = -blend; q <= blend; q++){
                        if(q > 0){
                            vx(i,j) += b(i+q,j+p);
                            vy(i,j) += b(i+p,j+q);
                        }
                        else if (q < 0){
                            vx(i,j) -= b(i+q,j+p);
                            vy(i,j) -= b(i+p,j+q);
                        }
                    }
                }
                const double V = sqrt(vx(i,j)*vx(i,j) + vy(i,j)*vy(i,j));
                vx(i,j) *= dh0 / V;
                vy(i,j) *= dh0 / V;
            }
        }
    }

    // Solve Interpolation Problem to Warm Start vx
    int ileft, iright;
    for(int j = 1; j < (jmax-1); j++){
        for(int i1 = 0; i1 < (imax-1); i1++){
            if(!b(i1,j)){
                ileft = i1;
                for(int i2 = (ileft+1); i2 < imax; i2++){
                    if(!b(i2,j)){
                        iright = i2;
                        break;
                    }
                }
                const int di = iright - ileft;
                if(di > 1){
                    for(int i3 = ileft; i3 < (iright+1); i3++){
                        const double k = (double)(i3-ileft) / (double)di;
                        vx(i3,j) = (1.0-k) * vx(ileft,j) + k * vx(iright,j);
                    }
                }
            }
        }
    }

    // Solve Interpolation Problem to Warm Start vy
    int jbottom, jtop;
    for(int i = 1; i < (imax-1); i++){
        for(int j1 = 0; j1 < (jmax-1); j1++){
            if(!b(i,j1)){
                jbottom = j1;
                for(int j2 = (jbottom+1); j2 < jmax; j2++){
                    if(!b(i,j2)){
                        jtop = j2;
                        break;
                    }
                }
                const int dj = jtop - jbottom;
                if(dj > 1){
                    for(int j3 = jbottom; j3 < (jtop+1); j3++){
                        const double k = (double)(j3-jbottom) / (double)dj;
                        vy(i,j3) = (1.0-k) * vy(i,jbottom) + k * vy(i,jtop);
                    }
                }
            }
        }
    }

    // Solve Laplace's Equation for vx and vy
    int v_iters = 0;
    double v_rss = 1.0e0; 
    const double v_RelTol = pow(ds/4.0, 2.0);
    while(v_rss > v_RelTol){
        double vx_rss = 0.0;
        double vy_rss = 0.0;
        for(int i = 1; i < imax-1; i++){
            for(int j = 1; j < jmax-1; j++){
                if(b(i,j)){
                    const double vx_old = vx(i,j);
                    const double vy_old = vy(i,j);
                    vx(i,j) = (vx(i+1,j) + vx(i-1,j) + vx(i,j+1) + vx(i,j-1)) / 4.0;
                    vy(i,j) = (vy(i+1,j) + vy(i-1,j) + vy(i,j+1) + vy(i,j-1)) / 4.0;
                    vx_rss += (vx(i,j) - vx_old) * (vx(i,j) - vx_old);
                    vy_rss += (vy(i,j) - vy_old) * (vy(i,j) - vy_old);
                }
            }
        }
        v_rss = fmax(vx_rss, vy_rss);
        v_iters++;
    }

    // Compute Forcing Function
    const double alpha = 2.0;
    for(int i = 1; i < (imax-1); i++){
        for(int j = 1; j < (jmax-1); j++){
            f(i,j) = (vx(i+1,j) - vx(i-1,j)) / (2.0*ds) + (vy(i,j+1) - vy(i,j-1)) / (2.0*ds);
            if(b(i,j) > 0.0){
                f(i,j) = softMin(f(i,j), -4.0, alpha);
                f(i,j) = softMax(f(i,j), 0.0, alpha);
            }
            else if(b(i,j) < 0.0){
                f(i,j) = softMax(f(i,j), 4.0, alpha);
                f(i,j) = softMin(f(i,j), 0.0, alpha);
            }
            else{
                f(i,j) = 0.0;
            }
            f(i,j) *= dA;
        }
    }

    // Solve Poisson's Equation
    hgrid = sdf;
    int h_iters = 0;
    double h_rss = 1.0e0;
    const double h_RelTol = pow(ds/4.0, 2.0);
    while(h_rss > h_RelTol){
        h_rss = 0.0;
        for(int i = 1; i < imax-1; i++){
            for(int j = 1; j < jmax-1; j++){
                if(b(i,j)){
                    const double h_old = hgrid(i,j);
                    hgrid(i,j) = hgrid(i+1,j) + hgrid(i-1,j) + hgrid(i,j+1) + hgrid(i,j-1);
                    hgrid(i,j) -= f(i,j);
                    hgrid(i,j) /= 4.0;
                    h_rss += (hgrid(i,j) - h_old) * (hgrid(i,j) - h_old);
                }
            }
        }
        h_iters++;
    }
    timer.time("CPU Solve Time: ");

    timer.start();
    int epochs = 0;
    hgrid = sdf;
    Eigen::MatrixXd h_old = sdf;
    h_rss = 1.0e0;
    while(h_rss > h_RelTol){
    //    // CUDA!!!!!
        int iter_per_epoch = 500;
        Kernel::Poisson(hgrid, f, b, iter_per_epoch);
        Eigen::MatrixXd dh = hgrid - h_old;
        h_rss = dh.norm() / iter_per_epoch;
        h_old = hgrid;
        epochs++;
        printf("Epoch: %u, (Error: %lf) \n", epochs, h_rss);
    }
    timer.time("GPU Solve Time: ");
    
    printf("Total Laplace Iterations: %u \n", v_iters);
    printf("Total Poisson Iterations: %u \n", h_iters);

    writeMatrixToFile(sdf, "sdf_safe_set.csv");
    writeMatrixToFile(hgrid, "poisson_safe_set.csv");
    writeMatrixToFile(vx, "guidance_field_x.csv");
    writeMatrixToFile(vy, "guidance_field_y.csv");
    writeMatrixToFile(b, "occupancy_map.csv");
    writeMatrixToFile(f, "forcing_function.csv");
	return 0;
    
}