#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "../inc/kernel.hpp"
#include "../inc/poisson.h"
#include "../inc/utils.h"

#include <Eigen/Core>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

using std::placeholders::_1;

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"){
            
            tgrid = std::chrono::high_resolution_clock::now();
            sleep(1);
            dtgrid0 = std::chrono::high_resolution_clock::now() - tgrid;

            hgrid.setConstant(h0);
            
            hgrid_message.data.resize(imax*jmax);
            u_message.data.resize(10);
                        
            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_grid_topic", 1);
            u_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_command_topic", 10);
            occ_grid_suber_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>("occ_grid_topic", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1));
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/MacLane/pose", 1, std::bind(&PoissonControllerNode::optitrack_state_callback, this, std::placeholders::_1));

        }

    private:

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

        void nominal_controller(const float rx, const float ry, const float yaw){

            // Define Reference Trajectory
            float rxd = 1.75f;
            float ryd = 1.75f;
            float yawd = 0.0f;

            // Compute Errors
            float rxe = rxd - rx;
            float rye = ryd - ry;
            float yawe = ang_diff(yawd,yaw);

            const float kp = 0.5f;
            vx = kp * rxe;
            vy = kp * rye;
            vyaw = kp * yawe;

        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(Eigen::MatrixXf &bound){
            
            // Set Border
            bound.block(0, 0, imax, 1).setConstant(0.0f);
            bound.block(0, 0, 1, jmax).setConstant(0.0f);
            bound.block(0, jmax-1, imax, 1).setConstant(0.0f);
            bound.block(imax-1, 0, 1, jmax).setConstant(0.0f);

            Eigen::MatrixXf b0(imax, jmax);
            memcpy(b0.data(), bound.data(), sizeof(float)*imax*jmax);
            for(int i = 1; i < imax-1; i++){
                for(int j = 1; j < jmax-1; j++){
                    if(b0(i,j)==1.0f){
                        const float neighbors = b0(i+1,j) + b0(i-1,j) + b0(i,j+1) + b0(i,j-1) + b0(i+1,j+1) + b0(i-1,j+1) + b0(i-1,j-1) + b0(i+1,j-1);
                        if(neighbors < 8.0f){
                            bound(i,j) = 0.0f;
                        }
                    }
                }
            }

        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void zero_boundary(Eigen::MatrixXf &grid, Eigen::MatrixXf &bound){
            
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(!bound(i,j)) grid(i,j) = h0;
                }
            }

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(Eigen::MatrixXf &bound, const float yaw){

            /* Step 1: Create Robot Kernel */
            const float go2_length = 0.80f;
            const float go2_width = 0.40f;
            //const float g1_radius = 0.25f;
            
            const float D = sqrt(go2_length*go2_length + go2_width*go2_width); // Max Robot Dimension to Define Template Size
            int dim = ceil((ceil(D / ds) + 1.0f) / 2.0f) * 2.0f - 1.0f;
            Eigen::MatrixXf robot_grid = Eigen::MatrixXf::Zero(dim, dim);

            const float MOS = 1.0f;
            const float ar = MOS * go2_length / 2.0f;
            const float br = MOS * go2_width / 2.0f;
            
            for(int i = 0; i < dim; i++){
                const float xi = (float)i * ds - D/2.0f;
                for(int j = 0; j < dim; j++){
                    const float yi = (float)j * ds - D/2.0f;
                    
                    const float xb = cosf(yaw)*xi + sinf(yaw)*yi;
                    const float yb = -sinf(yaw)*xi + cosf(yaw)*yi;

                    const float dist = pow(xb/br, 4.0f) + pow(yb/ar, 4.0f);
                    if(dist <= 1.0f){
                        robot_grid(i,j) = -1.0f;
                    }
                }
            }
        
            /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
            Eigen::MatrixXf b0(imax, jmax);
            memcpy(b0.data(), bound.data(), sizeof(float)*imax*jmax);

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
                                bound(p,q) += robot_grid(p-i+lim, q-j+lim);
                            }
                        }
                    }
                }
            }
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(bound(i,j) < -1.0f) bound(i,j) = -1.0f;
                }
            }

        };

        /* Using Occupancy Grid, Find Desired Boundary Gradients */
        void compute_boundary_gradients(Eigen::MatrixXf &guidance_x, Eigen::MatrixXf &guidance_y, Eigen::MatrixXf &bound){

            const int blend = 1; // How Many Pixels Will Be Used to Blend Gradients (>= 1, <= buffer)
            for(int i = blend; i < imax-blend; i++){
                for(int j = blend; j < jmax-blend; j++){
                    if(!bound(i,j)){
                        guidance_x(i,j) = 0.0f;
                        guidance_y(i,j) = 0.0f;
                        for(int p = -blend; p <= blend; p++){
                            for(int q = -blend; q <= blend; q++){
                                if(q > 0){
                                    guidance_x(i,j) += bound(i+q,j+p);
                                    guidance_y(i,j) += bound(i+p,j+q);
                                }
                                else if (q < 0){
                                    guidance_x(i,j) -= bound(i+q,j+p);
                                    guidance_y(i,j) -= bound(i+p,j+q);
                                }
                            }
                        }
                        const float V = sqrt(guidance_x(i,j)*guidance_x(i,j) + guidance_y(i,j)*guidance_y(i,j));
                        if(V != 0.0f){
                            guidance_x(i,j) *= dh0 / V;
                            guidance_y(i,j) *= dh0 / V;
                        }
                    }
                }
            }

        };

        /* Compute Forcing Function from Guidance Field */
        void compute_forcing_function(Eigen::MatrixXf &force, Eigen::MatrixXf &guidance_x, Eigen::MatrixXf &guidance_y, Eigen::MatrixXf &bound){

            const float max_div = 4.0f;
            const float alpha = 2.0f;

            for(int i = 1; i < (imax-1); i++){
                for(int j = 1; j < (jmax-1); j++){
                    force(i,j) = (guidance_x(i+1,j) - guidance_x(i-1,j)) / (2.0f*ds) + (guidance_y(i,j+1) - guidance_y(i,j-1)) / (2.0f*ds);
                    if(bound(i,j) > 0.0f){
                        force(i,j) = softMin(force(i,j), -max_div, alpha);
                        force(i,j) = softMax(force(i,j), 0.0f, alpha);
                    }
                    else if(bound(i,j) < 0.0f){
                        force(i,j) = softMax(force(i,j), max_div, alpha);
                        force(i,j) = softMin(force(i,j), 0.0f, alpha);
                    }
                    else{
                        force(i,j) = 0.0f;
                    }
                }
            }

        };

        /* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
        int poisson(Eigen::MatrixXf &grid, Eigen::MatrixXf &force, Eigen::MatrixXf &bound, const float relTol = 1.0e-4f, const float N = 70.0f, const bool gpu_flag = false){
            
            if(!gpu_flag){
                const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5

                int iters = 0;
                const int max_iters = 10000;
                float rss;
                for(int k = 0; k < max_iters; k++){

                    // Checkerboard Pass
                    rss = 0.0f;
                    for(int i = 1; i < imax-1; i++){
                        for(int j = 1; j < jmax-1; j++){
                            bool red = (i%2)==(j%2);
                            if(bound(i,j) && red){
                                float dg = grid(i+1,j) + grid(i-1,j) + grid(i,j+1) + grid(i,j-1);
                                dg -= force(i,j) * ds * ds;
                                dg /= 4.0f;
                                dg -= grid(i,j);
                                grid(i,j) += w_SOR * dg;
                                rss += dg * dg;
                            }
                        }
                    }
                    for(int i = 1; i < imax-1; i++){
                        for(int j = 1; j < jmax-1; j++){
                            bool black = (i%2)!=(j%2);
                            if(bound(i,j) && black){
                                float dg = grid(i+1,j) + grid(i-1,j) + grid(i,j+1) + grid(i,j-1);
                                dg -= force(i,j) * ds * ds;
                                dg /= 4.0f;
                                dg -= grid(i,j);
                                grid(i,j) += w_SOR * dg;
                                rss += dg * dg;
                            }
                        }
                    }
                    
                    rss = sqrt(rss) * ds;
                    iters++;
                    if(rss < relTol) break;
                
                }

                return iters;
            }
            else{

                return Kernel::Poisson(grid.data(), force.data(), bound.data(), relTol); // CUDA!

            }

        };

        /* Compute the Poisson Safety Function */
        void solve_poisson_safety_function(Eigen::MatrixXf &grid, Eigen::MatrixXf &guidance_x, Eigen::MatrixXf &guidance_y, Eigen::MatrixXf &occ, const float yaw, const bool gpu_flag){
            
            Eigen::MatrixXf bound(imax, jmax);
            memcpy(bound.data(), occ.data(), sizeof(float)*imax*jmax);
            
            find_boundary(bound);
            inflate_occupancy_grid(bound, yaw);
            find_boundary(bound);
            zero_boundary(grid, bound);
            compute_boundary_gradients(guidance_x, guidance_y, bound);
            
            const float v_RelTol = 1.0e-4f;
            vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 45.0f, gpu_flag);
            vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 45.0f, gpu_flag);
            
            Eigen::MatrixXf force(imax, jmax);
            compute_forcing_function(force, guidance_x, guidance_y, bound);

            const float h_RelTol = 1.0e-4f;
            h_iters = poisson(grid, force, bound, h_RelTol, 70.0f, gpu_flag);

        };

        float softMin(const float x0, const float xmin, const float alpha){
            float xf = xmin + logf(1.0f+expf(alpha*(x0-xmin))) / alpha;
            return xf;
        };

        float softMax(const float x0, const float xmax, const float alpha){
            float xf = xmax - logf(1.0f+expf(alpha*(xmax-x0))) / alpha;
            return xf;
        };

        /* Perform a bilinear interpolation on a 2-D grid */
        float bilinear_interpolation(Eigen::MatrixXf &grid, const float i, const float j){

            float f, f1, f2, f11, f12, f21, f22;
            
            const float i1 = floor(i);
            const float j1 = floor(j);
            const float i2 = ceil(i);
            const float j2 = ceil(j);

            if((i1 != i2) && (j1 != j2)){
                f11 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j1);
                f12 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j2);
                f21 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j1);
                f22 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j2);
                f1 = (j2 - j) / (j2 - j1) * (f11 + f21);
                f2 = (j - j1) / (j2 - j1) * (f12 + f22);
                f = f1 + f2;
            }
            else if(i1 != i2){
                f1 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j);
                f2 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j);
                f = f1 + f2;
            }
            else if(j1 != j2){
                f1 = (j2 - j) / (j2 - j1) * grid((int)i,(int)j1);
                f2 = (j - j1) / (j2 - j1) * grid((int)i,(int)j2);
                f = f1 + f2;
            }
            else{
                f = grid((int)i,(int)j);
            }

            return f;

        };

        void safety_filter(const float rx, const float ry){

            // Fractional Index Corresponding to Current Position
            float ir = (float)imax - ry / ds;
            float jr = rx / ds;

            const float x_eps = 1.0e-3f; // Small Perturbation for Numerical Gradients (meters)
            float i_eps = x_eps / ds;

            float ic = fminf(fmaxf(i_eps, ir), (float)(imax-1)-i_eps); // Saturated Because Numerical Derivatives Shrink Effective Grid Size
            float jc = fminf(fmaxf(i_eps, jr), (float)(jmax-1)-i_eps);

            // Get Safety Function Value
            h = bilinear_interpolation(hgrid, ic, jc);
            float ht = bilinear_interpolation(hgridt, ic, jc);
            //float hy = bilinear_interpolation(hgridy, ic, jc);

            // Compute Time Derivative
            float dhdt_raw = (h - ht) / dtgrid;
            const float wv = 100.0f; // Low Pass Filter Cutoff
            float kv = 1.0f - expf(-wv*dtgrid);
            dhdt *= 1.0f - kv;
            dhdt += kv * dhdt_raw;
            h += dhdt * grid_age;

            // Compute Angular Derivative
            //dhdyaw = (hy - h) / yaw_eps;

            // Compute Gradient
            float ip = ic - i_eps;
            float im = ic + i_eps;
            float jp = jc + i_eps;
            float jm = jc - i_eps;
            gradhx = (bilinear_interpolation(hgrid, ic, jp) - bilinear_interpolation(hgrid, ic, jm)) / (2.0f * x_eps);
            gradhy = (bilinear_interpolation(hgrid, ip, jc) - bilinear_interpolation(hgrid, im, jc)) / (2.0f * x_eps);
            
            
            // Single Integrator Safety Filter
            const float alpha = 0.6f;
            const float issf = 4.0f;

            float b = gradhx*gradhx + gradhy*gradhy; // + dhdyaw*dhdyaw;
            float a = gradhx*vx + gradhy*vy;
            //a += dhdyaw*vyaw;
            a += alpha*h;
            a += dhdt;
            a -= 1.0f/issf * b;

            vxs = -a * gradhx / b;
            vys = -a * gradhy / b;
            //vyaws = -a * dhdyaw / b;

            if(a<=0.0f){
                vx += vxs;
                vy += vys;
                //vyaw += vyaws;
            }

        };

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

        };

        void occ_grid_callback(std_msgs::msg::UInt8MultiArray::UniquePtr msg){
            
            const bool gpu_flag = true;

            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Compute Initial Grid Age
            const float camera_latency = 0.016f;
            const float segmenter_latency = 0.035f;
            const float poisson_latency = 0.03f;
            grid_age = camera_latency + segmenter_latency + poisson_latency;
            
            // Compute Loop Time
            dtgrid0 = std::chrono::high_resolution_clock::now() - tgrid;
            tgrid = std::chrono::high_resolution_clock::now();
            dtgrid = dtgrid0.count() * 1.0e-9f;

            // Solve Poisson Safety Function (Old Occupancy, New Orientation)
            memcpy(hgridt.data(), hgrid.data(), sizeof(float)*imax*jmax);
            //memcpy(vxgridt.data(), vxgrid.data(), sizeof(float)*imax*jmax);
            //memcpy(vygridt.data(), vygrid.data(), sizeof(float)*imax*jmax);
            //solve_poisson_safety_function(hgridt, vxgridt, vygridt, occ, yaw, gpu_flag);
            //printf("Laplace Old Iterations: %u \n", vx_iters + vy_iters);
            //printf("Poisson Old Iterations: %u \n", h_iters);
            
            // Update Occupancy
            occ = Eigen::MatrixXf::Ones(imax, jmax);
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(msg->data[jmax*i+j]) occ(i,j) = -1.0f;
                }
            }

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            //memcpy(hgrid.data(), hgridt.data(), sizeof(float)*imax*jmax);
            //memcpy(vxgrid.data(), vxgridt.data(), sizeof(float)*imax*jmax);
            //memcpy(vygrid.data(), vygridt.data(), sizeof(float)*imax*jmax);
            solve_poisson_safety_function(hgrid, vxgrid, vygrid, occ, yaw, gpu_flag);
            //printf("Laplace Iterations: %u \n", vx_iters + vy_iters);
            //printf("Poisson Iterations: %u \n", h_iters);

            // Solve Poisson Safety Function (New Occupancy, Epsilon Perturbed Orientation)
            //memcpy(hgridy.data(), hgrid.data(), sizeof(float)*imax*jmax);
            //memcpy(vxgridy.data(), vxgrid.data(), sizeof(float)*imax*jmax);
            //memcpy(vygridy.data(), vygrid.data(), sizeof(float)*imax*jmax);
            //solve_poisson_safety_function(hgridy, vxgridy, vygridy, occ, yaw + yaw_eps, gpu_flag);
            //printf("Laplace Eps Iterations: %u \n", vx_iters + vy_iters);
            //printf("Poisson Eps Iterations: %u \n", h_iters);

            h_flag = true;

            // Publish Poisson Safety Function Grid
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    hgrid_message.data[jmax*i+j] = hgrid(i,j);
                }
            }
            this->hgrid_publisher_->publish(hgrid_message);

            solve_timer.time("Solve Time: ");
            std::cout << "Loop Time: " << dtgrid*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vx << "," << vy << ">" << std::endl;
            //save_flag = writeDataToFile(save_flag);

        };

        void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){

            // Interpret State
            const float rc[2] = {0.0f, 0.3f}; // Location of OptiTrack Origin in Grid Frame
            rx = data->pose.position.x + rc[0];
            ry = data->pose.position.y + rc[1];
            float sin_yaw = 2.0f * (data->pose.orientation.w * data->pose.orientation.z); 
            float cos_yaw = 1.0f - 2.0f * data->pose.orientation.z * data->pose.orientation.z;
            yaw = atan2f(sin_yaw, cos_yaw);

            // Apply Nominal Control
            nominal_controller(rx, ry, yaw);

            // Increment Age of Latest Grid
            dtstate0 = std::chrono::high_resolution_clock::now() - tstate;
            tstate = std::chrono::high_resolution_clock::now();
            dtstate = dtstate0.count() * 1.0e-9f;
            //t += dtstate;
            grid_age += dtstate;
            
            // Apply Safety Filter
            if(h_flag) safety_filter(rx, ry);

            // Publish Control Action
            u_message.data[0] = rx;
            u_message.data[1] = ry;
            u_message.data[2] = yaw;
            u_message.data[3] = vx;
            u_message.data[4] = vy;
            u_message.data[5] = vyaw;
            u_message.data[6] = vxs;
            u_message.data[7] = vys;
            u_message.data[8] = vyaws;
            u_message.data[9] = h;
            this->u_publisher_->publish(u_message);

        };

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        //float yaw_eps = 5.0f * M_PI / 180.0;

        bool save_flag = false;
        bool h_flag = false;

        int vx_iters, vy_iters, h_iters;
        
        float rx = 1.75f;
        float ry = 1.75f;
        float yaw = 0.0f;

        //float t = -5.0f;
        std::chrono::high_resolution_clock::time_point tgrid, tstate;
        std::chrono::duration<float, std::nano> dtgrid0, dtstate0;
        float dtgrid = 1.0e10f;
        float dtstate = 1.0e10f;
        float grid_age = 0.0f;

        float vx, vy, vyaw;
        float vxs, vys, vyaws;
        float h, dhdt, dhdyaw, gradhx, gradhy;

        Eigen::MatrixXf f0 = Eigen::MatrixXf::Zero(imax, jmax);
        Eigen::MatrixXf occ = Eigen::MatrixXf::Ones(imax, jmax);

        Eigen::MatrixXf hgrid = Eigen::MatrixXf::Zero(imax, jmax);
        Eigen::MatrixXf vxgrid = Eigen::MatrixXf::Zero(imax, jmax);
        Eigen::MatrixXf vygrid = Eigen::MatrixXf::Zero(imax, jmax);

        Eigen::MatrixXf hgridt = Eigen::MatrixXf::Zero(imax, jmax);
        //Eigen::MatrixXf vxgridt = Eigen::MatrixXf::Zero(imax, jmax);
        //Eigen::MatrixXf vygridt = Eigen::MatrixXf::Zero(imax, jmax);

        //Eigen::MatrixXf hgridy = Eigen::MatrixXf::Zero(imax, jmax);
        //Eigen::MatrixXf vxgridy = Eigen::MatrixXf::Zero(imax, jmax);
        //Eigen::MatrixXf vygridy = Eigen::MatrixXf::Zero(imax, jmax);
        
        std_msgs::msg::Float32MultiArray hgrid_message;
        std_msgs::msg::Float32MultiArray u_message;

        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr u_publisher_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;
    
};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoissonControllerNode>());
    rclcpp::shutdown();

  return 0;

}