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
#include "../inc/mpc_cbf.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

using std::placeholders::_1;

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"){
            
            t_grid = std::chrono::high_resolution_clock::now();
            t_state = std::chrono::high_resolution_clock::now();
            sleep(1);
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;

            f0 = (float *)malloc(imax*jmax*tmax*sizeof(float));
            hgrid = (float *)malloc(imax*jmax*tmax*sizeof(float));
            vxgrid = (float *)malloc(imax*jmax*tmax*sizeof(float));
            vygrid = (float *)malloc(imax*jmax*tmax*sizeof(float));
            h0grid = (float *)malloc(imax*jmax*tmax*sizeof(float));

            for(int n = 0; n < imax*jmax; n++){
                occ[n] = 1.0f;
                occ_vi[n] = 0.0f;
                occ_vj[n] = 0.0f;
            }
            
            for(int n = 0; n < imax*jmax*tmax; n++){
                f0[n] = 0.0f;
                hgrid[n] = h0;
            }
            
            mpc_controller.setup_QP();
            mpc_controller.solve();

            hgrid_message.data.resize(imax*jmax);
            u_message.data.resize(10);
            mpc_message.data.resize(mpc_controller.nZ);
                        
            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_grid_topic", 1);
            u_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_command_topic", 10);
            mpc_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("mpc_solution_topic", 1);
            occ_grid_suber_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>("occ_grid_topic", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1));
            optflow_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("optical_flow_topic", 1, std::bind(&PoissonControllerNode::optical_flow_callback, this, std::placeholders::_1));
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/drone/pose", 1, std::bind(&PoissonControllerNode::optitrack_state_callback, this, std::placeholders::_1));
            
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

        /* Nominal Single Integrator Tracker */
        void nominal_controller(const float rx, const float ry, const float yaw){

            // Define Reference Trajectory
            rxd = 1.75f;
            ryd = 1.75f;

            // Compute Errors
            float rxe = rxd - rx;
            float rye = ryd - ry;
            
            float err = sqrtf(rxe*rxe+rye*rye);
            yawd = 0.0f;
            //if(err > 0.5f) yawd = atan2f(rye, rxe);

            float yawe = ang_diff(yawd,yaw);

            const float kp = 0.5f;
            vx = kp * rxe;
            vy = kp * rye;
            vyaw = kp * yawe;

        };

        void propogate_occupancy_grid(float *bound, const float *occ, const float *occ_vi, const float *occ_vj, const float t){
            
            for(int n = 0; n < imax*jmax; n++){
                bound[n] = 1.0f;
            }
            
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(occ[i*jmax+j]==-1.0f){
                        const float vi = occ_vi[i*jmax+j];
                        const float vj = occ_vj[i*jmax+j];
                        const float if_new = (float)i + vi * t;
                        const float jf_new = (float)j + vj * t;
                        const int i_new = std::min(std::max((int)roundf(if_new), 0), imax-1);
                        const int j_new = std::min(std::max((int)roundf(jf_new), 0), jmax-1);
                        bound[i_new*jmax+j_new] = -1.0f;
                    }
                }
            }
        
        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(float *bound){
            
            // Set Border
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(i==0 || i==(imax-1) || j==0 || j==(jmax-1)) bound[i*jmax+j] = 0.0f;
                }
            }

            float b0[imax*jmax];
            memcpy(b0, bound, imax*jmax*sizeof(float));
            for(int i = 1; i < imax-1; i++){
                for(int j = 1; j < jmax-1; j++){
                    if(b0[i*jmax+j]==1.0f){
                        const float neighbors = b0[(i+1)*jmax+j] + b0[(i-1)*jmax+j] + b0[i*jmax+(j+1)] + b0[i*jmax+(j-1)] + b0[(i+1)*jmax+(j+1)] + b0[(i-1)*jmax+(j+1)] + b0[(i-1)*jmax+(j-1)] + b0[(i+1)*jmax+(j-1)];
                        if(neighbors < 8.0f){
                            bound[i*jmax+j] = 0.0f;
                        }
                    }
                }
            }

        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void zero_boundary(float *grid, const float *bound){
            
            for(int n = 0; n < imax*jmax; n++){
                if(!bound[n]) grid[n] = h0;
            }

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, const float yawk){

            /* Step 1: Create Robot Kernel */
            const float length = 0.80f; // Go2
            const float width = 0.40f;
            //const float length = 0.50f; // G1
            //const float width = 0.50f;
            //const float length = 0.40f; // Drone
            //const float width = 0.40f;
            
            const float D = sqrtf(length*length + width*width); // Max Robot Dimension to Define Template Size
            const int dim = ceil((ceil(D / ds) + 1.0f) / 2.0f) * 2.0f - 1.0f;
            float robot_grid[dim*dim];

            const float MOS = 1.0f;
            const float ar = MOS * length / 2.0f;
            const float br = MOS * width / 2.0f;
            
            for(int i = 0; i < dim; i++){
                const float xi = (float)i * ds - D/2.0f;
                for(int j = 0; j < dim; j++){
                    robot_grid[i*dim+j] = 0.0;
                    const float yi = (float)j * ds - D/2.0f;
                    const float xb = cosf(yawk)*xi + sinf(yawk)*yi;
                    const float yb = -sinf(yawk)*xi + cosf(yawk)*yi;
                    const float dist = powf(xb/br, 4.0f) + powf(yb/ar, 4.0f);
                    if(dist <= 1.0f) robot_grid[i*dim+j] = -1.0f;
                }
            }
        
            /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
            float b0[imax*jmax];
            memcpy(b0, bound, imax*jmax*sizeof(float));

            int lim = (dim - 1)/2;
            for(int i = 1; i < imax-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, imax);
                for(int j = 1; j < jmax-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, jmax);
                    if(!b0[i*jmax+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*jmax+q] += robot_grid[(p-i+lim)*dim+(q-j+lim)];
                            }
                        }
                    }
                }
            }
            for(int n = 0; n < imax*jmax; n++){
                if(bound[n] < -1.0f) bound[n] = -1.0f;
            }

        };

        /* Using Occupancy Grid, Find Desired Boundary Gradients */
        void compute_boundary_gradients(float *guidance_x, float *guidance_y, const float *bound){

            const int blend = 1; // How Many Pixels Will Be Used to Blend Gradients (>= 1, <= buffer)
            for(int i = blend; i < imax-blend; i++){
                for(int j = blend; j < jmax-blend; j++){
                    if(!bound[i*jmax+j]){
                        guidance_x[i*jmax+j] = 0.0f;
                        guidance_y[i*jmax+j] = 0.0f;
                        for(int p = -blend; p <= blend; p++){
                            for(int q = -blend; q <= blend; q++){
                                if(q > 0){
                                    guidance_x[i*jmax+j] += bound[(i+q)*jmax+(j+p)];
                                    guidance_y[i*jmax+j] += bound[(i+p)*jmax+(j+q)];
                                }
                                else if (q < 0){
                                    guidance_x[i*jmax+j] -= bound[(i+q)*jmax+(j+p)];
                                    guidance_y[i*jmax+j] -= bound[(i+p)*jmax+(j+q)];
                                }
                            }
                        }
                        const float V = sqrtf(guidance_x[i*jmax+j]*guidance_x[i*jmax+j] + guidance_y[i*jmax+j]*guidance_y[i*jmax+j]);
                        if(V != 0.0f){
                            guidance_x[i*jmax+j] *= dh0 / V;
                            guidance_y[i*jmax+j] *= dh0 / V;
                        }
                    }
                }
            }

        };

        /* Compute Forcing Function from Guidance Field */
        void compute_forcing_function(float *force, const float *guidance_x, const float *guidance_y, const float *bound){

            const float max_div = 4.0f;
            const float alpha = 2.0f;

            for(int k = 0; k < tmax; k++){
                for(int i = 1; i < (imax-1); i++){
                    for(int j = 1; j < (jmax-1); j++){
                        force[k*imax*jmax+i*jmax+j] = (guidance_x[k*imax*jmax+(i+1)*jmax+j] - guidance_x[k*imax*jmax+(i-1)*jmax+j]) / (2.0f*ds) + (guidance_y[k*imax*jmax+i*jmax+(j+1)] - guidance_y[k*imax*jmax+i*jmax+(j-1)]) / (2.0f*ds);
                        if(bound[k*imax*jmax+i*jmax+j] > 0.0f){
                            force[k*imax*jmax+i*jmax+j] = softMin(force[k*imax*jmax+i*jmax+j], -max_div, alpha);
                            force[k*imax*jmax+i*jmax+j] = softMax(force[k*imax*jmax+i*jmax+j], 0.0f, alpha);
                        }
                        else if(bound[k*imax*jmax+i*jmax+j] < 0.0f){
                            force[k*imax*jmax+i*jmax+j] = softMax(force[k*imax*jmax+i*jmax+j], max_div, alpha);
                            force[k*imax*jmax+i*jmax+j] = softMin(force[k*imax*jmax+i*jmax+j], 0.0f, alpha);
                        }
                        else{
                            force[k*imax*jmax+i*jmax+j] = 0.0f;
                        }
                    }
                }
            }

        };

        /* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
        int poisson(float *grid, const float *force, const float *bound, const float relTol = 1.0e-4f, const float N = 70.0f, const bool gpu_flag = false){
            
            if(!gpu_flag){

                const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5

                int iters = 0;
                const int max_iters = 10000;
                float rss;
                for(int n = 0; n < max_iters; n++){

                    // Checkerboard Pass
                    rss = 0.0f;
                    for(int k = 0; k < tmax; k++){
                        
                        // Red Pass
                        for(int i = 1; i < imax-1; i++){
                            for(int j = 1; j < jmax-1; j++){
                                const bool red = (((i%2)+(j%2))%2) == 0;
                                if(bound[k*imax*jmax+i*jmax+j] && red){
                                    float dg = 0.0f;
                                    dg += (grid[k*imax*jmax+(i+1)*jmax+j] + grid[k*imax*jmax+(i-1)*jmax+j]);
                                    dg += (grid[k*imax*jmax+i*jmax+(j+1)] + grid[k*imax*jmax+i*jmax+(j-1)]);
                                    dg -= force[k*imax*jmax+i*jmax+j] * ds*ds;
                                    dg /= 4.0f;
                                    dg -= grid[k*imax*jmax+i*jmax+j];
                                    grid[k*imax*jmax+i*jmax+j] += w_SOR * dg;
                                    rss += dg * dg;
                                }
                            }
                        }
                        // Black Pass
                        for(int i = 1; i < imax-1; i++){
                            for(int j = 1; j < jmax-1; j++){
                                const bool black = (((i%2)+(j%2))%2) == 1;
                                if(bound[k*imax*jmax+i*jmax+j] && black){
                                    float dg = 0.0f;
                                    dg += (grid[k*imax*jmax+(i+1)*jmax+j] + grid[k*imax*jmax+(i-1)*jmax+j]);
                                    dg += (grid[k*imax*jmax+i*jmax+(j+1)] + grid[k*imax*jmax+i*jmax+(j-1)]);
                                    dg -= force[k*imax*jmax+i*jmax+j] * ds*ds;
                                    dg /= 4.0f;
                                    dg -= grid[k*imax*jmax+i*jmax+j];
                                    grid[k*imax*jmax+i*jmax+j] += w_SOR * dg;
                                    rss += dg * dg;
                                }
                            }
                        }

                    }
                    rss = sqrtf(rss) * ds / (float)tmax;
                    iters++;
                    if(rss < relTol) break;
                
                }

                return iters;

            }
            else{

                return Kernel::Poisson(grid, force, bound, relTol); // CUDA!

            }

        };

        /* Compute the Poisson Safety Function */
        void solve_poisson_safety_function(float *grid, float *guidance_x, float *guidance_y, const float *occ, const float *occ_vi, const float *occ_vj){
            
            const bool gpu_flag = true;

            float *bound, *force;
            bound = (float *)malloc(imax*jmax*tmax*sizeof(float));
            force = (float *)malloc(imax*jmax*tmax*sizeof(float));
            
            for(int k = 0; k < tmax; k++){

                const float t_k = (float)k * DT + grid_age;
                const float yaw_k = (float)k * DT * vyaw + yaw;
                
                float *bound_slice = bound + k*imax*jmax;
                float *grid_slice = grid + k*imax*jmax;
                float *guidance_x_slice = guidance_x + k*imax*jmax;
                float *guidance_y_slice = guidance_y + k*imax*jmax;
                memcpy(bound_slice, occ, imax*jmax*sizeof(float));
                
                propogate_occupancy_grid(bound_slice, occ, occ_vi, occ_vj, t_k);
                find_boundary(bound_slice);
                inflate_occupancy_grid(bound_slice, yaw_k);
                find_boundary(bound_slice);
                zero_boundary(grid_slice, bound_slice);
                compute_boundary_gradients(guidance_x_slice, guidance_y_slice, bound_slice);

            }
            
            const float v_RelTol = 1.0e-4f;
            vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 45.0f, gpu_flag);
            vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 45.0f, gpu_flag);
            
            compute_forcing_function(force, guidance_x, guidance_y, bound);

            const float h_RelTol = 1.0e-4f;
            h_iters = poisson(grid, force, bound, h_RelTol, 70.0f, gpu_flag);

            free(bound);
            free(force);
            
        };

        void solve_mpc(const float rx, const float ry, const float yaw){
            
            const float xd[2] = {rxd, ryd};
            const float x[2] = {rx, ry};
            mpc_controller.update_cost(xd, x);

            // Perform SQP Iterations
            for(int i=0; i<20; i++){
                mpc_controller.update_constraints(x, hgrid);
                mpc_controller.solve();
            }
            float z[mpc_controller.nZ];
            mpc_controller.extract_solution(z);
            vx = z[mpc_controller.nX+0];
            vy = z[mpc_controller.nX+1];

            for(int i = 0; i < mpc_controller.nZ; i++){
                mpc_message.data[i] = z[i];
            }
            this->mpc_publisher_->publish(mpc_message);

        }

        void safety_filter(const float rx, const float ry, const float yaw){

            // Fractional Index Corresponding to Current Position
            const float ir = (float)imax - ry / ds;
            const float jr = rx / ds;

            const float x_eps = 1.0e-3f; // Small Perturbation for Numerical Gradients (meters)
            const float y_eps = 1.0e-3f;

            const float i_eps = x_eps / ds;
            const float j_eps = y_eps / ds;

            const float ic = fminf(fmaxf(i_eps, ir), (float)(imax-1)-i_eps); // Saturated Because of Finite Grid Size
            const float jc = fminf(fmaxf(i_eps, jr), (float)(jmax-1)-i_eps); // Numerical Derivatives Shrink Effective Grid Size

            // Get Safety Function Value
            const float kc = grid_age / DT;
            h = (1.0f - kc) * bilinear_interpolation(hgrid, ic, jc) + kc * bilinear_interpolation(hgrid+imax*jmax, ic, jc);
            const float h0 = (1.0f - kc) * bilinear_interpolation(h0grid, ic, jc) + kc * bilinear_interpolation(h0grid+imax*jmax, ic, jc);
            dhdt = (h - h0) / dt_grid;

            // Compute Gradients
            const float ip = ic - i_eps;
            const float im = ic + i_eps;
            const float jp = jc + j_eps;
            const float jm = jc - j_eps;
            dhdx = (bilinear_interpolation(hgrid, ic, jp) - bilinear_interpolation(hgrid, ic, jm)) / (2.0f * x_eps);
            dhdy = (bilinear_interpolation(hgrid, ip, jc) - bilinear_interpolation(hgrid, im, jc)) / (2.0f * y_eps);
            
            // Single Integrator Safety Filter
            const float alpha = 0.6f;
            const float issf = 100.0f;

            const float b = dhdx*dhdx + dhdy*dhdy;
            float a = dhdx*vx + dhdy*vy;
            a += dhdt;
            a += alpha*h;
            a -= 1.0f/issf * b;

            vxs = -a * dhdx / b;
            vys = -a * dhdy / b;

            if(a<=0.0f){
                vx += vxs;
                vy += vys;
            }

        };

        bool writeDataToFile(bool flag){

            if(!flag){
                const std::string& filename = "poisson_safety_grid.csv";
                std::ofstream outFile(filename);
                if(outFile.is_open()){
                    for(int i = 0; i < imax; i++){
                        for(int j = 0; j < jmax; j++){
                            outFile << hgrid[i*jmax+j] << std::endl;
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
            
            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Compute Initial Grid Age
            const float camera_latency = 0.016f;
            const float segmenter_latency = 0.035f;
            const float poisson_latency = 0.030f;
            grid_age = camera_latency + segmenter_latency + poisson_latency;

            // Update Occupancy
            for(int n = 0; n < imax*jmax; n++){
                occ[n] = 1.0f;
                if(msg->data[n]) occ[n] = -1.0f;
            }
            
            // Store Old Solution
            memcpy(h0grid, hgrid, imax*jmax*tmax*sizeof(float));

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            solve_poisson_safety_function(hgrid, vxgrid, vygrid, occ, occ_vi, occ_vj);
            printf("Laplace Iterations: %u \n", vx_iters + vy_iters);
            printf("Poisson Iterations: %u \n", h_iters);
            h_flag = true;
            
            // Compute Grid of dh/dt Values
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;
            t_grid = std::chrono::high_resolution_clock::now();
            dt_grid = dt_grid0.count() * 1.0e-9f;
            
            // Publish Poisson Safety Function Grid
            int k = tmax-1;
            for(int n = 0; n < imax*jmax; n++){
                hgrid_message.data[n] = hgrid[k*imax*jmax+n];
            }
            this->hgrid_publisher_->publish(hgrid_message);
            
            solve_timer.time("Solve Time: ");
            std::cout << "Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vx << "," << vy << "," << vyaw << ">" << std::endl;
            //save_flag = writeDataToFile(save_flag);
        
        };

        void optical_flow_callback(std_msgs::msg::Float32MultiArray::UniquePtr msg){
            
            // Store Optical Flow Data
            for(int n = 0; n < imax*jmax; n++){
                occ_vi[n] = msg->data[0*imax*jmax+n];
                occ_vj[n] = msg->data[1*imax*jmax+n];
            }
        
        };

        void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){
            
            // Interpret State
            const float rc[2] = {0.0f, 0.3f}; // Location of OptiTrack Origin in Grid Frame
            rx = data->pose.position.x + rc[0];
            ry = data->pose.position.y + rc[1];
            float sin_yaw = 2.0f * (data->pose.orientation.w * data->pose.orientation.z); 
            float cos_yaw = 1.0f - 2.0f * data->pose.orientation.z * data->pose.orientation.z;
            yaw = atan2f(sin_yaw, cos_yaw);
            if(yaw<0.0f) yaw += 2.0f*M_PI;

            // Increment Age of Latest Grid
            dt_state0 = std::chrono::high_resolution_clock::now() - t_state;
            t_state = std::chrono::high_resolution_clock::now();
            dt_state = dt_state0.count() * 1.0e-9f;
            //t += dt_state;
            grid_age += dt_state;
            
            // Apply Nominal Control & Safety Filter
            nominal_controller(rx, ry, yaw);
            if(h_flag) safety_filter(rx, ry, yaw);
            if(h_flag) solve_mpc(rx, ry, yaw);

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

        MPC mpc_controller;
        
        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        bool save_flag = false;
        bool h_flag = false;

        int vx_iters, vy_iters, h_iters;
        
        float rx = 1.75f;
        float ry = 1.75f;
        float yaw = 0.0f;

        //float t = -5.0f;
        std::chrono::high_resolution_clock::time_point t_grid, t_state;
        std::chrono::duration<float, std::nano> dt_grid0, dt_state0;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;
        float grid_age = 0.0f;

        float rxd, ryd, yawd;
        float vx, vy, vyaw;
        float vxs, vys, vyaws;
        float h, dhdt, dhdyaw, dhdx, dhdy;

        
        float occ[imax*jmax];
        float occ_vi[imax*jmax];
        float occ_vj[imax*jmax];
        float *f0, *hgrid, *vxgrid, *vygrid, *h0grid;
        
        std_msgs::msg::Float32MultiArray hgrid_message;
        std_msgs::msg::Float32MultiArray u_message;
        std_msgs::msg::Float32MultiArray mpc_message;

        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr u_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mpc_publisher_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr optflow_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;
    
};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoissonControllerNode>());
    rclcpp::shutdown();

  return 0;

}