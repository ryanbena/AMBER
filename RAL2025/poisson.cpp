#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>

#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_2d.h"
#include "mpc_cbf_3d.h"
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>

#include <time.h>
#include <cmath>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

#define KEY_UP 65
#define KEY_DOWN 66
#define KEY_RIGHT 67
#define KEY_LEFT 68
#define SPACEBAR 32
#define COMMA 44
#define PERIOD 46

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"), sport_req(this){
            
            // Create CSV for Saving
            std::string baseFileName = "experiment_data";
            std::string dateTime = getCurrentDateTime();
            std::string fileName = baseFileName + "_" + dateTime + ".csv";
            outFile.open(fileName);

            // Initialize Clocks
            t_grid = std::chrono::high_resolution_clock::now();
            t_state = std::chrono::high_resolution_clock::now();

            // Initialize Occupancy Grids & Flow Fields
            for(int n = 0; n < IMAX*JMAX; n++){
                conf0[n] = 0;
                conf1[n] = 0;
            }

            // Initialize Poisson Grids
            cudaMallocHost((void**)&hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&hgrid0, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&bound, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&force, IMAX*JMAX*QMAX*sizeof(float));
            dhdt_grid = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            for(int n=0; n < IMAX*JMAX*QMAX; n++){
                hgrid1[n] = h0;
                hgrid0[n] = h0;
                dhdt_grid[n] = 0.0f;
            }
            Kernel::poissonInit();
            robot_kernel_dim = initialize_robot_kernel(robot_kernel);

            // Initialize QP for MPC Problem
            #if MPC3D_FLAG
                mpc3d_controller.setup_QP();
                mpc3d_controller.solve();
            #else
                mpc2d_controller.setup_QP();
                mpc2d_controller.solve();
            #endif

            // Create Publishers & Subscribers
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            pose_suber_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1,std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1), options2);
            
            // Create Timer for Reference & Logging
            const float timer_period = 0.01f;
            auto log_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(timer_period));
            timer_ = this->create_wall_timer(log_timer_period, std::bind(&PoissonControllerNode::timer_callback, this));
            
            // Start Up the Unitree Go2
            sport_req.StandUp(req);
            sleep(1);
            sport_req.BalanceStand(req);
            sleep(1);
            sport_req.EconomicGait(req);
            sleep(1);
            //sport_req.ClassicWalk(req, true);
            //sleep(1);
            sport_req.SpeedLevel(req, 1);
            sleep(1);
        
        }

    private:
  
        void timer_callback(void){

            if(!start_flag){
                rxd = rx;
                ryd = ry;
                yawd = yaw;
            }
            else{
                float save_data[7] = {rx, ry, yaw, vx, vy, vyaw, h};
                for(int n = 0; n < 7; n++){
                    outFile << save_data[n];
                    outFile << ",";
                }
                outFile << std::endl;
            }
            
            // Update Reference Trajectory
            char ch;
            if (read(STDIN_FILENO, &ch, 1) > 0) {
                float drxdb = 0.0f;
                float drydb = 0.0f;
                float dyawd = 0.0f; 
                if(ch == KEY_UP) drxdb = 0.02f;
                if(ch == KEY_DOWN) drxdb = -0.02f;
                if(ch == KEY_LEFT) drydb = 0.02f;
                if(ch == KEY_RIGHT) drydb = -0.02f;
                if(ch == COMMA) dyawd = 0.02f;
                if(ch == PERIOD) dyawd = -0.02f;
                rxd += cosf(yaw)*drxdb - sinf(yaw)*drydb;
                ryd += sinf(yaw)*drxdb + cosf(yaw)*drydb;
                yawd += dyawd;
                if(ch == SPACEBAR){
                    if(start_flag) stop_flag = true;
                    start_flag = true;
                }
            }

            //Timer control_timer(true);
            //control_timer.start();

            // Run MPC with SQP
            if(h_flag){
                const float xd[3] = {rxd, ryd, yawd};
                const float x[3] = {rx, ry, yaw};
                const float ud[3] = {vxn, vyn, vyawn};
                const float xc[3] = {rxc, ryc, yawc};
                #if MPC3D_FLAG
                    std::lock_guard<std::mutex> lock(mpc_mutex);
                    for(int i=0; i<SQP_ITERS; i++){
                        mpc3d_controller.update_cost(xd, x, ud);
                        mpc3d_controller.update_constraints(hgrid1, dhdt_grid, xc);
                        mpc3d_controller.solve();
                    }
                    mpc3d_controller.set_input(&vxd, &vyd, &vyawd);
                #else
                    std::lock_guard<std::mutex> lock(mpc_mutex);
                    for(int i=0; i<SQP_ITERS; i++){
                        mpc2d_controller.update_cost_and_constraints(hgrid1, xc, xd, x);
                        mpc2d_controller.solve();
                    }
                    mpc2d_controller.set_input(&vxd, &vyd);
                #endif
            }

            //control_timer.time("MPC Solve Time: ");

        };   
    
        /* Nominal Single Integrator Tracker */
        void nominal_controller(const float rx, const float ry, const float yaw){

            // Compute Errors
            float rxe = rxd - rx;
            float rye = ryd - ry;
            float yawe = ang_diff(yawd,yaw);

            const float kp = 0.5f;
            vxn = kp * rxe;
            vyn = kp * rye;
            vyawn = kp * yawe;

        };

        /* Threshold Occupancy Map with Hysterisis */
        void build_occ_map(float *occ_map, const int8_t *conf1, const int8_t *conf0){
            
            const int8_t T_hi = 105;
            const int8_t T_lo = 64;
            for(int n=0; n<IMAX*JMAX; n++){
                occ_map[n] = 1.0f;
                bool strong = conf1[n] >= T_hi;
                bool weak = conf1[n] >= T_lo && conf0[n] >= T_lo;
                if(strong || weak) occ_map[n] = -1.0f;
            }
 
        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(float *grid, float *bound, const bool fix_flag){
            
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
                if(fix_flag && !bound[n]) grid[n] = h0;
            }

        };
        
        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        int initialize_robot_kernel(float*& kernel){
            
            /* Create Robot Kernel */
            robot_length = 0.74f; // Go2
            robot_width = 0.37f;
            robot_MOS = 1.35f;
            
            const float ar = robot_MOS * robot_length / 2.0f;
            const float br = robot_MOS * robot_width / 2.0f;
            const float D = 2.0f * sqrtf(ar*ar + br*br); // Max Robot Dimension to Define Kernel Size
            const int dim = ceilf(ceilf(D / DS) / 2.0f) * 2.0f; //Make Sure Kernel Dimension is Even

            #if MPC3D_FLAG
                kernel = (float *)malloc(dim*dim*QMAX*sizeof(float));
                for(int q=0; q<QMAX; q++){
                    float *kernel_slice = kernel + q*dim*dim;
                    const float yawq = q_to_yaw(q, yawc);
                    fill_elliptical_robot_kernel(kernel_slice, yawq, dim, 2.0f);
                }
            #else
                kernel = (float *)malloc(dim*dim*sizeof(float));
                fill_elliptical_robot_kernel(kernel, yaw, dim, 2.0f);
            #endif

            return dim;

        };

        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        void fill_elliptical_robot_kernel(float *kernel, const float yawq, const int dim, const float expo){
            
            const float ar = robot_MOS * robot_length / 2.0f;
            const float br = robot_MOS * robot_width / 2.0f;
            for(int i = 0; i < dim; i++){
                const float yi = (float)(i-dim/2)*DS;
                for(int j = 0; j < dim; j++){
                    kernel[i*dim+j] = 0.0f;
                    const float xi = (float)(j-dim/2)*DS;
                    const float xb = cosf(yawq)*xi + sinf(yawq)*yi;
                    const float yb = -sinf(yawq)*xi + cosf(yawq)*yi;
                    const float dist = powf(fabsf(xb/ar), expo) + powf(fabsf(yb/br), expo);
                    if(dist <= 1.0f) kernel[i*dim+j] = -1.0f;
                }
            }

        };


        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, const float *kernel){
            
            // Convolve Robot Kernel with Occupancy Grid, Along the Boundary
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));

            int lim = (robot_kernel_dim - 1)/2;
            for(int i = 1; i < IMAX-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 1; j < JMAX-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, JMAX);
                    if(!b0[i*JMAX+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*JMAX+q] += kernel[(p-i+lim)*robot_kernel_dim+(q-j+lim)];
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


        /* Compute the Poisson Safety Function */
        bool solve_poisson_safety_function(void){

            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Initialize Temporary Grids
            float occ[IMAX*JMAX];
            float *hgrid_temp = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            memcpy(hgrid_temp, hgrid1, IMAX*JMAX*QMAX*sizeof(float));

            // Execute Poisson Pre-Processing
            build_occ_map(occ, conf1, conf0);
            find_boundary(hgrid_temp, occ, false);
            #if (!MPC3D_FLAG) 
                fill_elliptical_robot_kernel(robot_kernel, yaw, robot_kernel_dim, 2.0f);
            #endif

            #pragma omp parallel for num_threads(6)
            for(int q=0; q<QMAX; q++){
                float *force_slice = force + q*IMAX*JMAX;
                float *bound_slice = bound + q*IMAX*JMAX;
                float *hgrid_slice = hgrid_temp + q*IMAX*JMAX;
                float *kernel_slice = robot_kernel + q*robot_kernel_dim*robot_kernel_dim;
                memcpy(bound_slice, occ, IMAX*JMAX*sizeof(float));
                inflate_occupancy_grid(bound_slice, kernel_slice);
                find_boundary(hgrid_slice, bound_slice, true);
                compute_fast_forcing_function(force_slice, bound_slice);
            }

            // Solve Poisson's Equation
            const float relTol = 1.0e-4f;
            const int N = IMAX/5;
            const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(float)(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            int iters = Kernel::poissonSolve(hgrid_temp, force, bound, relTol, w_SOR); // CUDA!
            
            // Transfer Solutions into Necessary Locations
            memcpy(conf0, conf1, IMAX*JMAX*sizeof(int8_t));
            memcpy(hgrid0, hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            memcpy(hgrid1, hgrid_temp, IMAX*JMAX*QMAX*sizeof(float));
            free(hgrid_temp);
            if(h_flag) dhdt_flag = true;
            
            printf("Poisson Iterations: %u \n", iters);
            solve_timer.time("Grid Solve Time: ");

            return true;

        };

        /* Display Poisson Safety Function Grid (Interpolated) */
        void display_poisson_safety_function(void){

            Timer display_timer(true);
            display_timer.start();

            float grid[IMAX*JMAX];
            #if MPC3D_FLAG
                const float qr = yaw_to_q(yaw,yawc);
                const float q1f = floorf(qr);
                const float q2f = ceilf(qr);
                const int q1 = (int)q_wrap(q1f);
                const int q2 = (int)q_wrap(q2f);
                for(int n = 0; n < IMAX*JMAX; n++){
                    if(q1f!=q2f) grid[n] = (q2f - qr) * hgrid1[q1*IMAX*JMAX+n] + (qr - q1f) * hgrid1[q2*IMAX*JMAX+n];
                    else grid[n] = hgrid1[q1*IMAX*JMAX+n];
                }
            #else
                memcpy(grid, hgrid1, IMAX*JMAX*sizeof(float));
            #endif

            // Populate Float Grayscale Poisson Image with Chosen q & k Values
            cv::Mat poisson_img = cv::Mat::zeros(IMAX, JMAX, CV_32FC1);
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    poisson_img.at<float>(i,j) = grid[i*JMAX+j];
                }
            }
            
            // Convert to 8-bit Grayscale
            cv::Mat gray_img;
            cv::normalize(poisson_img, gray_img, 0, 255, cv::NORM_MINMAX);
            gray_img.convertTo(gray_img, CV_8U);
            
            // Convert to Colormap
            cv::Mat color_img;
            cv::applyColorMap(gray_img, color_img, cv::COLORMAP_HOT);
            
            // Resize for Display
            cv::Mat resized_img;
            const int upscale = 6;
            cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);
            
            // Add Current Location & Goal Location
            cv::Point curr_pt = cv::Point(upscale*x_to_j(rx,rxc),upscale*y_to_i(ry,ryc));
            cv::Point goal_pt = cv::Point(upscale*x_to_j(rxd,rxc),upscale*y_to_i(ryd,ryc));
            cv::circle(resized_img, curr_pt, upscale, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::circle(resized_img, goal_pt, upscale, cv::Scalar(0, 127, 0), cv::FILLED);

            // Add MPC Trajectory
            #if MPC3D_FLAG
                float z[TMAX*LINEAR_STATE_LENGTH_3D+(TMAX-1)*LINEAR_INPUT_LENGTH_3D];
                mpc3d_controller.extract_solution(z);
                for(int n = 1; n < TMAX; n++){
                    const int j_traj = x_to_j(z[LINEAR_STATE_LENGTH_3D*n+0], rxc);
                    const int i_traj = y_to_i(z[LINEAR_STATE_LENGTH_3D*n+1], ryc);
                    cv::Point traj_pt = cv::Point(upscale*j_traj, upscale*i_traj);
                    cv::circle(resized_img, traj_pt, upscale/2, cv::Scalar(255, 0, 0), cv::FILLED);
                }
            #else
                float z[TMAX*LINEAR_STATE_LENGTH_2D+(TMAX-1)*LINEAR_INPUT_LENGTH_2D];
                mpc2d_controller.extract_solution(z);
                for(int n = 1; n < TMAX; n++){
                    const int j_traj = x_to_j(z[LINEAR_STATE_LENGTH_2D*n+0], rxc);
                    const int i_traj = y_to_i(z[LINEAR_STATE_LENGTH_2D*n+1], ryc);
                    cv::Point traj_pt = cv::Point(upscale*j_traj, upscale*i_traj);
                    cv::circle(resized_img, traj_pt, upscale/2, cv::Scalar(255, 0, 0), cv::FILLED);
                }
            #endif

            // Vertical Flip Image for Display 
            cv::Mat flipped_img;
            cv::flip(resized_img, flipped_img, 0);
            
            // Display Final Image
            cv::imshow("Poisson Solution", flipped_img);
            cv::waitKey(1);

            display_timer.time("Display Time: ");

        }

        void update_dhdt_grid(void){

            const float wc = 10.0f;
            const float kc = 1.0f - expf(-wc*dt_grid);

            for(int i = 0; i<IMAX; i++){
                for(int j = 0; j<JMAX; j++){
                    for(int q = 0; q<QMAX; q++){
                        const float i0 = i + dry / DS;
                        const float j0 = j + drx / DS;
                        const bool in_grid = (i0 >= 0.0f) && (i0 <= (IMAX-1)) && (j0 >= 0.0f) && (j0 <= (JMAX-1));
                        float dhdt_ij = 0.0f;
                        if(in_grid){
                            const float h0 = trilinear_interpolation(hgrid0, i0, j0, q);
                            const float h1 = trilinear_interpolation(hgrid1, i, j, q);
                            dhdt_ij = (h1 - h0) / dt_grid;
                        }
                        dhdt_grid[q*IMAX*JMAX+i*JMAX+j] *= 1.0f - kc;
                        dhdt_grid[q*IMAX*JMAX+i*JMAX+j] += kc * dhdt_ij;
                    }
                }
            }

        }


        void safety_filter(const bool filter_flag){

            // Fractional Indices Corresponding to Current State
            const float ic = y_to_i(ry, ryc);
            const float jc = x_to_j(rx, rxc);
            const float qc = yaw_to_q(yaw,yawc);

            // Get Safety Function Value & Rate
            h = trilinear_interpolation(hgrid1, ic, jc, qc);
            dhdt = trilinear_interpolation(dhdt_grid, ic, jc, qc);
            
            // Forward Propogate h to Remove Latency
            const float h_pred = h + dhdt * grid_age;
    
            // Compute Gradients
            const float x_eps = 5.0f;
            const float hxp = trilinear_interpolation(hgrid1, ic, jc+x_eps, qc);
            const float hxm = trilinear_interpolation(hgrid1, ic, jc-x_eps, qc);
            const float hyp = trilinear_interpolation(hgrid1, ic+x_eps, jc, qc);
            const float hym = trilinear_interpolation(hgrid1, ic-x_eps, jc, qc);
            dhdx = (hxp-hxm) / (2.0f*x_eps*DS);
            dhdy = (hyp-hym) / (2.0f*x_eps*DS);
            dhdq = 0.0f;
            
            // Single Integrator Safety Filter
            const float issf1 = 5.0f;
            const float issf2 = 20.0f;
            const float b = dhdx*dhdx + dhdy*dhdy + dhdq*dhdq;
            const float Lgh_norm = sqrtf(b);
            float a = WN*h_pred;
            a += dhdt; // Dynamic Environment
            a += dhdx*vxd + dhdy*vyd + dhdq*vyawd; // Min Norm Controller
            a -= Lgh_norm / issf1 + Lgh_norm*Lgh_norm / issf2; // Robustness

            vx = vxd;
            vy = vyd;
            vyaw = vyawd;
            
            if((a<0.0f) && (b>1.0e-2f) && filter_flag){
                vx += -a * dhdx / b;
                vy += -a * dhdy / b;
                vyaw += -a * dhdq / b;
            }

        };


        void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){

            // Compute Grid Timing
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;
            t_grid = std::chrono::high_resolution_clock::now();
            dt_grid = dt_grid0.count() * 1.0e-9f;
            grid_age = dt_grid;
            
            // Read Message Data
            drx = msg->info.origin.position.x - rxc;
            dry = msg->info.origin.position.y - ryc;
            rxc = msg->info.origin.position.x;
            ryc = msg->info.origin.position.y;
            for(int n = 0; n < IMAX*JMAX; n++) conf1[n] = msg->data[n];

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            h_flag = solve_poisson_safety_function();

            // Update Grid of dh/dt Values
            if(dhdt_flag) update_dhdt_grid();

            // Display Results
            display_poisson_safety_function();
            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vxb << "," << vyb << "," << vyaw << ">" << std::endl;
        
        };

        void state_update_callback(const unitree_go::msg::SportModeState::SharedPtr data){
            
            // Increment Age of Latest Grid
            dt_state0 = std::chrono::high_resolution_clock::now() - t_state;
            t_state = std::chrono::high_resolution_clock::now();
            dt_state = dt_state0.count() * 1.0e-9f;
            grid_age += dt_state;

            // Interpret State
            rx = data->position[0];
            ry = data->position[1];
            float sin_yaw = 2.0f * (data->imu_state.quaternion[0] * data->imu_state.quaternion[3]); 
            float cos_yaw = 1.0f - 2.0f * data->imu_state.quaternion[3] * data->imu_state.quaternion[3];
            yaw = atan2f(sin_yaw, cos_yaw);

            // Apply Nominal Control
            nominal_controller(rx, ry, yaw);

            // Apply Safety Filter
            safety_filter(h_flag);

            // Transform to Body-Fixed
            vxb = cosf(yaw)*vx+ sinf(yaw)*vy;
            vyb = -sinf(yaw)*vx + cosf(yaw)*vy;
            if(fabsf(vxb)>100.0f || fabsf(vyb)>100.0f || fabsf(vyaw)>100.0f) stop_flag = true; // Check for Valid Control Action
            
            //Publish Control Action
            if(start_flag && !stop_flag){
                vxb = fminf(fmaxf(vxb, -4.0f), 4.0f);
                vyb = fminf(fmaxf(vyb, -4.0f), 4.0f);
                vyaw = fminf(fmaxf(vyaw, -4.0f), 4.0f);
                sport_req.Move(req, vxb, vyb, vyaw); // Send Command
            }
            if(stop_flag){
                sport_req.StopMove(req);
                rclcpp::shutdown();
            }

        };

        std::mutex mpc_mutex;
        MPC2D mpc2d_controller;
        MPC3D mpc3d_controller;

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        bool h_flag = false;
        bool dhdt_flag = false;
        bool start_flag = false;
        bool stop_flag = false;
        
        // Define State
        float drx = 0.0f;
        float dry = 0.0f;
        float rx = 0.0f;
        float ry = 0.0f;
        float yaw = 0.0f;

        // Define Reference Trajectory
        float rxd = 0.0f;
        float ryd = 0.0f;
        float yawd = 0.0f;

        // Define Occupancy Grid Origin
        float rxc = -2.00f;
        float ryc = -2.00f;
        float yawc = 0.0f;
        float rxc0 = -2.00f;
        float ryc0 = -2.00f;
        float yawc0 = 0.0f;

        std::chrono::high_resolution_clock::time_point t_grid, t_state;
        std::chrono::duration<float, std::nano> dt_grid0, dt_state0;
        float grid_age = 0.0f;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;

        float vxn, vyn, vyawn;
        float vxd, vyd, vyawd; 
        float vx, vy, vyaw;
        float vxb, vyb;
        float h, dhdt, dhdx, dhdy, dhdq;
        
        int8_t conf1[IMAX*JMAX];
        int8_t conf0[IMAX*JMAX];
        float *hgrid1, *hgrid0, *bound, *force, *robot_kernel, *dhdt_grid;

        float robot_length, robot_width, robot_MOS;
        int robot_kernel_dim;
        
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr pose_suber_;
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber_;
        unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
        SportClient sport_req;

        std::ofstream outFile;

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