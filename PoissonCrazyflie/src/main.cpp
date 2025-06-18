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
#include "nav_msgs/msg/occupancy_grid.hpp"

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>

#define KEY_UP 65
#define KEY_DOWN 66
#define KEY_RIGHT 67
#define KEY_LEFT 68

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"){
            
            t_grid = std::chrono::high_resolution_clock::now();
            t_state = std::chrono::high_resolution_clock::now();

            hgrid = (float *)malloc(IMAX*JMAX*TMAX*sizeof(float));
            
            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                occ_vi[n] = 0.0f;
                occ_vj[n] = 0.0f;
            }
            
            for(int n = 0; n < IMAX*JMAX*TMAX; n++){
                hgrid[n] = h0;
            }

            const float xd[6] = {rxd, ryd, rzd, 0.0f, 0.0f, 0.0f};
            const float x[6] = {rx, ry, rz, vx, vy, vz};
            mpc_controller.setup_QP();
            mpc_controller.initialize_solution(xd, x, 0.0f);
            mpc_controller.solve();

            hgrid_message.data.resize(IMAX*JMAX);
            mpc_message.data.resize(mpc_controller.nZ+3);
            control_message.data.resize(3);
            
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            rclcpp::SubscriptionOptions options3;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options3.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_grid_topic", 1);
            mpc_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("mpc_solution_topic", 1);
            control_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("control_topic", 1);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occ_grid_topic", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            optflow_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("optical_flow_topic", 1, std::bind(&PoissonControllerNode::optical_flow_callback, this, std::placeholders::_1), options2);
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/Crazyflie/pose", 1, std::bind(&PoissonControllerNode::optitrack_state_callback, this, std::placeholders::_1), options3);
            
            // Create Timer for Reference
            const float ref_period = 0.01f;
            auto ref_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(ref_period));
            ref_timer_ = this->create_wall_timer(ref_timer_period, std::bind(&PoissonControllerNode::reference_callback, this));
            
            // Create Timer for MPC
            const float mpc_period = 0.01f;
            auto mpc_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(mpc_period));
            mpc_timer_ = this->create_wall_timer(mpc_timer_period, std::bind(&PoissonControllerNode::mpc_callback, this));
        
        }

    private:
        
        void reference_callback(void){
            
            char ch;
            if (read(STDIN_FILENO, &ch, 1) > 0) {

                // Define Reference Trajectory
                if(ch == KEY_UP) rxd -= 0.01f;
                if(ch == KEY_DOWN) rxd += 0.01f;
                if(ch == KEY_LEFT) ryd -= 0.01f;
                if(ch == KEY_RIGHT) ryd += 0.01f;
            
            }

        };   
    
        /* Nominal Single Integrator Tracker */
        void single_int_position_controller(void){

            const float kp = 0.5f;
            vxd = kp * (rxd - rx);
            vyd = kp * (ryd - ry);
            vzd = kp * (rzd - rz);

        };

        /* Nominal Single Integrator Tracker */
        void single_int_velocity_controller(void){
            
            const float kd = 5.5f;
            mux = kd * (vxd - vx);
            muy = kd * (vyd - vy);
            muz = kd * (vzd - vz);

        };

        /* Propogate Forward in Time for MPC */
        void propogate_occupancy_grid(float *bound, const float *occ, const float *occ_vi, const float *occ_vj, const int k){
            
            for(int n = 0; n < IMAX*JMAX; n++){
                bound[n] = 1.0f;
            }
            
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(occ[i*JMAX+j]==-1.0f){
                        const float if_new = (float)i + occ_vi[i*JMAX+j] * (float)k * DT;
                        const float jf_new = (float)j + occ_vj[i*JMAX+j] * (float)k * DT;
                        const int i_new = std::min(std::max((int)roundf(if_new), 0), IMAX-1);
                        const int j_new = std::min(std::max((int)roundf(jf_new), 0), JMAX-1);
                        bound[i_new*JMAX+j_new] = -1.0f;
                    }
                }
            }
        
        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(float *bound){
            
            // Set Border
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
                }
            }

            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++){
                if(b0[n]==1.0f){
                    if(b0[n+1]==-1.0f || 
                       b0[n-1]==-1.0f || 
                       b0[n+JMAX]==-1.0f || 
                       b0[n-JMAX]==-1.0f || 
                       b0[n+JMAX+1]==-1.0f || 
                       b0[n-JMAX+1]==-1.0f || 
                       b0[n+JMAX-1]==-1.0f || 
                       b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
                }
            }

        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_and_fix_boundary(float *grid, float *bound){
            
            // Set Border
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
                }
            }

            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++){
                if(b0[n]==1.0f){
                    if(b0[n+1]==-1.0f || 
                       b0[n-1]==-1.0f || 
                       b0[n+JMAX]==-1.0f || 
                       b0[n-JMAX]==-1.0f || 
                       b0[n+JMAX+1]==-1.0f || 
                       b0[n-JMAX+1]==-1.0f || 
                       b0[n+JMAX-1]==-1.0f || 
                       b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
                }
                if(!bound[n]) grid[n] = h0;
            }

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound){

            const float MoS = 1.2f; // Margin of Safety
            const float diameter = 0.13f; // Crazyflie Diameter
            const float D = MoS * diameter;
            const float R = D / 2.0f;
            const float R2 = R * R;
            const int dim = ceilf((ceilf(D / DS) + 1.0f) / 2.0f) * 2.0f - 1.0f;
            const int lim = (dim - 1)/2;
            float robot_grid[dim*dim];

             /* Step 1: Create Robot Kernel */
            for(int i = 0; i < dim; i++){
                const float yi = (float)(i-lim) * DS;
                for(int j = 0; j < dim; j++){
                    robot_grid[i*dim+j] = 0.0;
                    const float xi = (float)(j-lim) * DS;
                    const float r2 = xi*xi + yi*yi;
                    if(r2 <= R2) robot_grid[i*dim+j] = -1.0f;
                }
            }
        
            /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int i = 1; i < IMAX-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 1; j < JMAX-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, JMAX);
                    if(!b0[i*JMAX+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*JMAX+q] += robot_grid[(p-i+lim)*dim+(q-j+lim)];
                            }
                        }
                    }
                }
            }
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] < -1.0f) bound[n] = -1.0f;
            }

        };

        /* Compute Forcing Function for Average Flux */
        void compute_fast_forcing_function(float *force, const float *bound){

            float perimeter_c = 0.0f;
            float area_c = 0.0f;
            
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
                    else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
                }
            }
            
            float perimeter_o = 2.0f*(float)IMAX*DS + 2.0f*(float)JMAX*DS + perimeter_c;
            float area_o = (float)IMAX*(float)JMAX*DS*DS - area_c;
            float force_o = -dh0 * perimeter_o / area_o * DS*DS;
            float force_c = 0.0f;
            if(area_c != 0.0f) force_c = dh0 * perimeter_c / area_c * DS*DS;
            
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] > 0.0f){
                    force[n] = force_o;
                }
                else if(bound[n] < 0.0f){
                    force[n] = force_c;
                }
                else{
                    force[n] = 0.0f;
                }
            }
        
        };

        /* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
        int poisson(float *grid, const float *force, const float *bound, const float relTol = 1.0e-4f, const float N = 25.0f, const bool gpu_flag = false){
            
            const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            
            if(!gpu_flag){

                int iters = 0;
                const int max_iters = 10000;
                float rss;
                for(int n = 0; n < max_iters; n++){

                    // Checkerboard Pass
                    rss = 0.0f;
                    
                    for(int k = 0; k < TMAX; k++){
                        
                        float *grid_slice = grid + k*IMAX*JMAX;
                        const float *force_slice = force + k*IMAX*JMAX;
                        const float *bound_slice = bound + k*IMAX*JMAX;
                        
                        // Red Pass
                        for(int i = 1; i < IMAX-1; i++){
                            for(int j = 1; j < JMAX-1; j++){
                                const bool red = (((i%2)+(j%2))%2) == 0;
                                if(bound_slice[i*JMAX+j] && red){
                                    float dg = 0.0f;
                                    dg += (grid_slice[(i+1)*JMAX+j] + grid_slice[(i-1)*JMAX+j]);
                                    dg += (grid_slice[i*JMAX+(j+1)] + grid_slice[i*JMAX+(j-1)]);
                                    dg -= force_slice[i*JMAX+j];
                                    dg /= 4.0f;
                                    dg -= grid_slice[i*JMAX+j];
                                    grid_slice[i*JMAX+j] += w_SOR * dg;
                                    rss += dg * dg;
                                }
                            }
                        }
                        // Black Pass
                        for(int i = 1; i < IMAX-1; i++){
                            for(int j = 1; j < JMAX-1; j++){
                                const bool black = (((i%2)+(j%2))%2) == 1;
                                if(bound_slice[i*JMAX+j] && black){
                                    float dg = 0.0f;
                                    dg += (grid_slice[(i+1)*JMAX+j] + grid_slice[(i-1)*JMAX+j]);
                                    dg += (grid_slice[i*JMAX+(j+1)] + grid_slice[i*JMAX+(j-1)]);
                                    dg -= force_slice[i*JMAX+j];
                                    dg /= 4.0f;
                                    dg -= grid_slice[i*JMAX+j];
                                    grid_slice[i*JMAX+j] += w_SOR * dg;
                                    rss += dg * dg;
                                }
                            }
                        }

                    }

                    rss = sqrtf(rss) * DS / (float)TMAX;
                    iters++;
                    if(rss < relTol) break;

                
                }

                return iters;

            }
            else{

                return Kernel::Poisson(grid, force, bound, relTol, w_SOR); // CUDA!

            }

        };

        /* Compute the Poisson Safety Function */
        void solve_poisson_safety_function(float *grid, const float *occ, const float *occ_vi, const float *occ_vj){
            
            const bool gpu_flag = true;

            float *bound, *force;
            bound = (float *)malloc(IMAX*JMAX*TMAX*sizeof(float));
            force = (float *)malloc(IMAX*JMAX*TMAX*sizeof(float));

            #pragma omp parallel for
            for(int k=0; k<TMAX; k++){

                float *force_slice = force + k*IMAX*JMAX;
                float *bound_slice = bound + k*IMAX*JMAX;
                float *grid_slice = grid + k*IMAX*JMAX;

                memcpy(bound_slice, occ, IMAX*JMAX*sizeof(float));
                propogate_occupancy_grid(bound_slice, occ, occ_vi, occ_vj, k);
                find_boundary(bound_slice);
                inflate_occupancy_grid(bound_slice);
                find_and_fix_boundary(grid_slice, bound_slice);
                compute_fast_forcing_function(force_slice, bound_slice);

            }

            const float h_RelTol = 1.0e-4f;
            h_iters = poisson(grid, force, bound, h_RelTol, 25.0f, gpu_flag);

            free(bound);
            free(force);
            
        };

        void safety_filter(const float rx, const float ry, const float rz, const float dt, const bool filter_flag){

            // Fractional Index Corresponding to Current Position
            const float ir = ry / DS;
            const float jr = rx / DS;

            // Saturated Because of Finite Grid Size
            const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
            const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size

            // Get Safety Function Rate
            const float *hgrid0 = hgrid+0*IMAX*JMAX;
            const float *hgrid1 = hgrid+1*IMAX*JMAX;
            const float h0 = bilinear_interpolation(hgrid0, ic, jc);
            const float h1 = bilinear_interpolation(hgrid1, ic, jc);
            dhdt = (h1 - h0) / DT;
            
            // Get Safety Function Value
            h = h0;
            h += dhdt * dt;

            // Compute Gradients
            const float i_eps = 5.0f;
            const float j_eps = 5.0f;
            const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
            const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
            const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
            const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));

            const float dhdx0 = (bilinear_interpolation(hgrid0, ic, jp) - bilinear_interpolation(hgrid0, ic, jm)) / ((jp-jm)*DS);
            const float dhdy0 = (bilinear_interpolation(hgrid0, ip, jc) - bilinear_interpolation(hgrid0, im, jc)) / ((ip-im)*DS);
            const float dhdx1 = (bilinear_interpolation(hgrid1, ic, jp) - bilinear_interpolation(hgrid1, ic, jm)) / ((jp-jm)*DS);
            const float dhdy1 = (bilinear_interpolation(hgrid1, ip, jc) - bilinear_interpolation(hgrid1, im, jc)) / ((ip-im)*DS);
            const float kgrad = dt/DT;
            dhdx = dhdx0*(1.0f-kgrad) + dhdx1* kgrad;
            dhdy = dhdy0*(1.0f-kgrad) + dhdy1* kgrad;

            // Single Integrator Safety Filter
            const float issf = 100.0f;

            const float b = dhdx*dhdx + dhdy*dhdy;
            float a = dhdx*vxd + dhdy*vyd;
            a += dhdt;
            a += WN*h;
            a -= 1.0f/issf * b;

            if(fabsf(b) < 1.0e-3f){
                vxs = 0.0f;
                vys = 0.0f;
                vzs = 0.0f;
            }
            else{
                vxs = -a * dhdx / b;
                vys = -a * dhdy / b;
                vzs = 0.0f;
            }

            if(a<=0.0f & filter_flag){
                vxd += vxs;
                vyd += vys;
                vzd += vzs;
            }

        };

        void predictive_safety_filter(void){
            
            if(h_flag){

                const float xd[6] = {rxd, ryd, rzd, 0.0f, 0.0f, 0.0f};
                const float x[6] = {rx, ry, rz, vx, vy, vz};
                
                // Perform SQP Iterations
                const int sqp_iters = 1;
                const float sqp_weight = 1.0f;
                mpc_controller.initialize_solution(xd, x, sqp_weight);
                for(int i=0; i<sqp_iters; i++){
                    mpc_controller.update_cost_and_constraints(hgrid);
                    mpc_fail_flag = mpc_controller.solve();
                }
                mpc_age = 0.0f;

            }

        };

        void mpc_callback(void){
            
            //predictive_safety_filter();
            
            // Publish ROS Message
            float z[mpc_controller.nZ];
            mpc_controller.extract_solution(z);
            for(int i = 0; i < mpc_controller.nZ; i++){
                mpc_message.data[i] = z[i];
            }
            mpc_message.data[mpc_controller.nZ+0] = rxd;
            mpc_message.data[mpc_controller.nZ+1] = ryd;
            mpc_message.data[mpc_controller.nZ+2] = rzd;
            this->mpc_publisher_->publish(mpc_message);

        };

        void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){
            
            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Compute Grid Timing
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;
            t_grid = std::chrono::high_resolution_clock::now();
            dt_grid = dt_grid0.count() * 1.0e-9f;
            grid_age = dt_grid;

            // Update Occupancy
            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                if(msg->data[n]) occ[n] = -1.0f;
            }

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            solve_poisson_safety_function(hgrid, occ, occ_vi, occ_vj);
            printf("Poisson Iterations: %u \n", h_iters);
            h_flag = true;
            
            // Publish Poisson Safety Function Grid (Interpolated)
            int k = 0;
            for(int n = 0; n < IMAX*JMAX; n++){
                hgrid_message.data[n] = hgrid[k*IMAX*JMAX+n];
            }
            this->hgrid_publisher_->publish(hgrid_message);

            solve_timer.time("Grid Solve Time: ");
            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << mux << "," << muy << "," << muz << ">" << std::endl;
        
        };

        /* Store Optical Flow Data */
        void optical_flow_callback(const std_msgs::msg::Float32MultiArray::UniquePtr msg){
          
          for(int n = 0; n < IMAX*JMAX; n++){
              occ_vi[n] = msg->data[0*IMAX*JMAX+n];
              occ_vj[n] = msg->data[1*IMAX*JMAX+n];
          }
      
        };

        void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){
            
            // Increment Age of Latest Grid
            dt_state0 = std::chrono::high_resolution_clock::now() - t_state;
            t_state = std::chrono::high_resolution_clock::now();
            dt_state = dt_state0.count() * 1.0e-9f;
            
            // Interpret State
            const float rc[3] = {1.75f, 1.75f, 0.00f}; // Location of OptiTrack Origin in Grid Frame
            vx = (data->pose.position.x + rc[0] - rx) / dt_state;
            vy = (data->pose.position.y + rc[1] - ry) / dt_state;
            vz = (data->pose.position.z + rc[2] - rz) / dt_state;
            rx = data->pose.position.x + rc[0];
            ry = data->pose.position.y + rc[1];
            rz = data->pose.position.z + rc[2];

            grid_age += dt_state;
            mpc_age += dt_state;

            // Apply Nominal Control & Safety Filter
            single_int_position_controller();
            single_int_velocity_controller();

            
            if(h_flag){
                predictive_safety_filter();
                if(!mpc_fail_flag){
                    mpc_controller.set_ad(&mux, &muy, &muz, mpc_age);
                }
                else{
                    safety_filter(rx, ry, rz, grid_age, mpc_fail_flag);
                    single_int_velocity_controller();
                }
            }

            control_message.data[0] = mux;
            control_message.data[1] = muy;
            control_message.data[2] = fminf(muz, 7.0f);
            this->control_publisher_->publish(control_message);

        };
        
        // MPC Controller Object
        MPC mpc_controller;

        // Poisson Variables
        int h_iters;
        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        // Flags
        bool h_flag = false;
        bool mpc_fail_flag = false;

        // State Variables
        float rx = 1.75f;
        float ry = 1.75f;
        float rz = 0.00f;
        float vx = 0.00f;
        float vy = 0.00f;
        float vz = 0.00f;

        // Reference Trajectory Variables
        float rxd = 1.75f;
        float ryd = 1.75f;
        float rzd = 1.00f;
        float vxd = 0.00f;
        float vyd = 0.00f;
        float vzd = 0.00f;

        // Time Variables
        std::chrono::high_resolution_clock::time_point t_grid, t_state;
        std::chrono::duration<float, std::nano> dt_grid0, dt_state0;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;
        float grid_age = 0.0f;
        float mpc_age = 0.0f;

        // Other Controller Variables
        float mux, muy, muz;
        float vxs, vys, vzs;
        float h, dhdt, dhdx, dhdy, dhdz;
        
        // Poisson Arrays
        float occ[IMAX*JMAX];
        float occ_vi[IMAX*JMAX];
        float occ_vj[IMAX*JMAX];
        float *hgrid;
        
        // Publisher Messages
        std_msgs::msg::Float32MultiArray hgrid_message;
        std_msgs::msg::Float32MultiArray mpc_message;
        std_msgs::msg::Float32MultiArray control_message;

        // Timers, Publishers, & Subscribers
        rclcpp::TimerBase::SharedPtr mpc_timer_;
        rclcpp::TimerBase::SharedPtr ref_timer_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mpc_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr control_publisher_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr optflow_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;
    
};

struct termios oldt;
struct termios newt;

void setNonBlocking(bool enable) {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if(enable) fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    else fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
}

void setRawMode(bool enable) {

    if(enable){
        tcgetattr(STDIN_FILENO, &oldt);  // save old settings
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO); // disable buffering and echo
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    } 
    else{
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // restore old settings
    }
}

int main(int argc, char * argv[]){

    setRawMode(true);
    setNonBlocking(true);
    
    rclcpp::init(argc, argv);   
    rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::Node::SharedPtr poissonNode = std::make_shared<PoissonControllerNode>();
    executor.add_node(poissonNode);

    try{
        executor.spin();
        throw("Terminated");
    }
    catch(const char* msg){
        rclcpp::shutdown();
        setRawMode(false);
        setNonBlocking(false);
        std::cout << msg << std::endl;
    }

  return 0;

}