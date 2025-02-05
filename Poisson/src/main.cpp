#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include "../inc/kernel.hpp"
#include "../inc/Types.h"
#include "../inc/utils.h"

#include <Eigen/Core>

#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

const int imax = 120;
const int jmax = 120;

const double ds = 0.0254; // grid resolution
const double dA = ds * ds; //grid cell area

double yaw = 0.0;

const double h0 = 0.0; // Set boundary level set value
const double dh0 = 1.0; // Set dh Value

const double v_RelTol = pow(ds/4.0, 2.0);
const double h_RelTol = pow(ds/4.0, 2.0);

const double go2_radius = 0.30;
const double g1_radius = 0.25;

bool save_flag = false;

Eigen::MatrixXd hgrid = Eigen::MatrixXd::Zero(imax, jmax);
Eigen::MatrixXd b = Eigen::MatrixXd::Zero(imax, jmax);
Eigen::MatrixXd sdf = Eigen::MatrixXd::Zero(imax, jmax);
Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(imax, jmax);
Eigen::MatrixXd vy = Eigen::MatrixXd::Zero(imax, jmax);
Eigen::MatrixXd f = Eigen::MatrixXd::Zero(imax, jmax);

void poisson_init(void){

    hgrid.setConstant(h0);
    b.setConstant(1.0);

}

/* Compute the Signed Distance Function *//*
void signed_distance_function(void){

    double rounded_buffer = ceil(go2_radius/ds) * ds;
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
                    if(!free_space){
                        min_dist_sq = dist_sq;
                    }
                    y++;
                    dist_sq = distx_sq + dA*(double)(y*y);
                }
                x++;
                distx_sq = dA * (double)(x*x);
            }
            sdf(i,j) = sqrt(min_dist_sq) * b(i,j); // Signed Distance to Centroid
            sdf(i,j) -= rounded_buffer; // Buffered
        }
    }

}
*/

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_boundary(void){
    
    // Set Border
    b.block(0, 0, imax, 1).setConstant(0.0);
    b.block(0, 0, 1, jmax).setConstant(0.0);
    b.block(0, jmax-1, imax, 1).setConstant(0.0);
    b.block(imax-1, 0, 1, jmax).setConstant(0.0);

    Eigen::MatrixXd b0(imax, jmax);
    memcpy(b0.data(), b.data(), sizeof(double)*imax*jmax);
    for(int i = 1; i < imax-1; i++){
        for(int j = 1; j < jmax-1; j++){
            if(b0(i,j)==1.0){
                const double neighbors = b0(i+1,j) + b0(i-1,j) + b0(i,j+1) + b0(i,j-1) + b0(i+1,j+1) + b0(i-1,j+1) + b0(i-1,j-1) + b0(i+1,j-1);
                if(neighbors < 8.0){
                    b(i,j) = 0.0;
                    hgrid(i,j) = h0;
                }
            }
        }
    }

}

/* Buffer Occupancy Grid with 2-D Robot Shape */
void inflate_occupancy_grid(void){

    //yaw += 0.05;

    const double D = 0.70; // Max Robot Dimension to Define Template Size
    int dim = ceil((ceil(D / ds) + 1.0) / 2.0) * 2.0 - 1.0;
    Eigen::MatrixXd robot_grid = Eigen::MatrixXd::Zero(dim, dim);

    const double ar = 0.35;
    const double br = 0.35;
    
    for(int i = 0; i < dim; i++){
        const double xi = (double)i * ds - D/2.0;
        for(int j = 0; j < dim; j++){
            const double yi = (double)j * ds - D/2.0;
            
            const double xb = cos(yaw)*xi + sin(yaw)*yi;
            const double yb = -sin(yaw)*xi + cos(yaw)*yi;

            const double dist = xb*xb/(ar*ar) + yb*yb/(br*br);
            if(dist <= 1.0){
                robot_grid(i,j) = -1.0;
            }
        }
    }
 
    Eigen::MatrixXd b0(imax, jmax);
    memcpy(b0.data(), b.data(), sizeof(double)*imax*jmax);

    int lim = (dim - 1)/2;
    for(int i = 1; i < imax-1; i++){
        int ilow = std::max(i - lim, 0);
        int itop = std::min(i + lim, imax);
        for(int j = 1; j < jmax-1; j++){
            int jlow = std::max(j - lim, 0);
            int jtop = std::min(j + lim, jmax);
            if(!b0(i,j)){
                for(int p = ilow; p < itop; p++){
                    for(int q = jlow; q < jtop; q++){
                        if(b(p,q) != -1.0){
                            b(p,q) += robot_grid(p-i+lim, q-j+lim);
                        }
                    }
                }
            }
        }
    }

}

/* Using Occupancy Grid, Find Desired Boundary Gradients */
void compute_boundary_gradients(void){

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
                if(V != 0.0){
                    vx(i,j) *= dh0 / V;
                    vy(i,j) *= dh0 / V;
                }
            }
        }
    }

}

/* Solve Laplace's Equation for Guidance Field */
int laplace_cpu(void){
    
    int v_iters = 0;
    double v_rss = 1.0e0; 
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
        v_rss = vx_rss + vy_rss;
        v_iters++;
    }

    return v_iters;

}

double softMin(const double x0, const double xmin, const double alpha){
    double xf = xmin + log(1.0+exp(alpha*(x0-xmin))) / alpha;
    return xf;
}

double softMax(const double x0, const double xmax, const double alpha){
    double xf = xmax - log(1.0+exp(alpha*(xmax-x0))) / alpha;
    return xf;
}

/* Compute Forcing Function from Guidance Field */
void compute_forcing_function(void){

    const double max_div = 4.0;
    const double alpha = 2.0;

    for(int i = 1; i < (imax-1); i++){
        for(int j = 1; j < (jmax-1); j++){
            f(i,j) = (vx(i+1,j) - vx(i-1,j)) / (2.0*ds) + (vy(i,j+1) - vy(i,j-1)) / (2.0*ds);
            if(b(i,j) > 0.0){
                f(i,j) = softMin(f(i,j), -max_div, alpha);
                f(i,j) = softMax(f(i,j), 0.0, alpha);
            }
            else if(b(i,j) < 0.0){
                f(i,j) = softMax(f(i,j), max_div, alpha);
                f(i,j) = softMin(f(i,j), 0.0, alpha);
            }
            else{
                f(i,j) = 0.0;
            }
            f(i,j) *= dA;
        }
    }

}

/* Solve Poisson's Equation */
int poisson_cpu(void){

    int h_iters = 0;
    double h_rss = 1.0e0;
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

    return h_iters;

}

int laplace_gpu(void){
    
    int v_iters = 0;
    int epochs = 0;

    int iter_per_epoch = 100;
    
    Eigen::MatrixXd vx_old(imax, jmax);
    Eigen::MatrixXd vy_old(imax, jmax);
    memcpy(vx_old.data(), vx.data(), sizeof(double)*imax*jmax);
    memcpy(vy_old.data(), vy.data(), sizeof(double)*imax*jmax);

    Eigen::MatrixXd f0 = Eigen::MatrixXd::Zero(imax, jmax);

    double v_rss = 1.0e0;
    while(v_rss > v_RelTol){

        /* CUDA!!!!! */       
        Kernel::Poisson(vx, f0, b, iter_per_epoch);
        Kernel::Poisson(vy, f0, b, iter_per_epoch);
        
        Eigen::MatrixXd dvx = vx - vx_old;
        Eigen::MatrixXd dvy = vy - vy_old;

        double vx_rss = dvx.norm() * dvx.norm() / iter_per_epoch;
        double vy_rss = dvy.norm() * dvy.norm() / iter_per_epoch;
        v_rss = vx_rss + vy_rss;

        memcpy(vx_old.data(), vx.data(), sizeof(double)*imax*jmax);
        memcpy(vy_old.data(), vy.data(), sizeof(double)*imax*jmax);
        
        epochs++;
        //printf("Epoch: %u, (Error: %lf) \n", epochs, h_rss);

    }

    v_iters = epochs * iter_per_epoch;
    return v_iters;

}

int poisson_gpu(void){
    
    int h_iters = 0;
    int epochs = 0;

    int iter_per_epoch = 100;
    
    Eigen::MatrixXd h_old(imax, jmax);
    memcpy(h_old.data(), hgrid.data(), sizeof(double)*imax*jmax);

    double h_rss = 1.0e0;
    while(h_rss > h_RelTol){

        /* CUDA!!!!! */
        Kernel::Poisson(hgrid, f, b, iter_per_epoch);
        Eigen::MatrixXd dh = hgrid - h_old;
        h_rss = dh.norm() * dh.norm() / iter_per_epoch;
        memcpy(h_old.data(), hgrid.data(), sizeof(double)*imax*jmax);
        epochs++;

    }
    
    h_iters = epochs * iter_per_epoch;
    return h_iters;

}

bool writeDataToFile(bool flag){

    if(!flag){
        const std::string& filename = "poisson_safety_grid.csv";
        std::ofstream outFile(filename);
        if(outFile.is_open()){
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    outFile << hgrid(i,j) << std::endl;
                }
            }
            outFile.close();
        } 
        else{
            std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        }
    }
    return true;

}

class OccupancyGridSubscriber : public rclcpp::Node{

    public:
        
        OccupancyGridSubscriber() : Node("occupancy_grid_subscriber"){
            
            message.data.resize(imax*jmax);
            auto topic_callback = [this](std_msgs::msg::UInt8MultiArray::UniquePtr msg) -> void {

                Timer timer(true);
                timer.start();

                // Assign Occupancy
                b.setConstant(1.0);
                for(int i = 0; i < imax; i++){
                    for(int j = 0; j < jmax; j++){
                        if(msg->data[jmax*i+j]){
                            b(i,j) = -1.0; // Fill Cells with -1.0
                        }
                    }
                }

                find_boundary();
                inflate_occupancy_grid();
                find_boundary();
                compute_boundary_gradients();
                
                int v_iters = laplace_cpu();
                compute_forcing_function();
                int h_iters = poisson_cpu();

                for(int i = 0; i < imax; i++){
                    for(int j = 0; j < jmax; j++){
                        message.data[jmax*i+j] = hgrid(i,j);
                    }
                }
                this->publisher_->publish(message);

                timer.time("Solve Time: ");
                printf("Laplace Iterations: %u \n", v_iters);
                printf("Poisson Iterations: %u \n", h_iters);

                save_flag = writeDataToFile(save_flag);
            
            };
            
            publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("safety_grid_topic", 1);
            subscription_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>("occ_grid_topic", 1, topic_callback);
        
        }

    private:

        std_msgs::msg::Float64MultiArray message;

        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr subscription_;
    

};

int main(int argc, char * argv[]){

    poisson_init();

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OccupancyGridSubscriber>());
    rclcpp::shutdown();

  return 0;

}