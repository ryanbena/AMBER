#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <random>
#include <cmath>

#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "cloud_merger.h"
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int32.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <time.h>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"), sport_req(this){
            
            // Initialize Clocks
            t_grid = std::chrono::steady_clock::now();
            t_state = std::chrono::steady_clock::now();

            // Initialize Occupancy Grids
            for(int n = 0; n < IMAX*JMAX; n++){
                occ1[n] = 1.0f;
                occ0[n] = 1.0f; 
                conf[n] = 0;
            }

            // Initialize Poisson Grids
            cudaMallocHost((void**)&hgrid1, IMAX*JMAX*sizeof(float));
            cudaMallocHost((void**)&hgrid0, IMAX*JMAX*sizeof(float));
            cudaMallocHost((void**)&bound, IMAX*JMAX*sizeof(float));
            cudaMallocHost((void**)&force, IMAX*JMAX*sizeof(float));
            dhdt_grid = (float *)malloc(IMAX*JMAX*sizeof(float));
            for(int n=0; n < IMAX*JMAX; n++){
                hgrid1[n] = h0;
                hgrid0[n] = h0;
                dhdt_grid[n] = 0.0f;
            }
            Kernel::poissonInit();
            robot_kernel_dim = initialize_robot_kernel(robot_kernel);
            buff = robot_kernel_dim / 2;

            // Initialize QP for MPC Problem
            xd = x;
            xd_traj.resize(3*TMAX);
            mpc3d_controller.reset_QP();
            mpc3d_controller.solve();

            // TF Listener
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            // Subscribers
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            twist_suber_ = this->create_subscription<geometry_msgs::msg::Twist>("u_des", 1, std::bind(&PoissonControllerNode::teleop_callback, this, std::placeholders::_1), options2);
            key_suber_ = this->create_subscription<std_msgs::msg::Int32>("key_press", 1, std::bind(&PoissonControllerNode::keyboard_callback, this, std::placeholders::_1), options2);

            // MPC Loop Timer
            mpc_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
            mpc_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&PoissonControllerNode::mpc_callback, this), mpc_callback_group_);

            // CBF-QP Loop Timer
            cbf_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            cbf_timer_ = this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&PoissonControllerNode::cbf_callback, this), cbf_callback_group_);            

            // Start Up the Unitree Go2
            sport_req.RecoveryStand(req);
            sleep(1);
            sport_req.SpeedLevel(req, 1);
            sleep(1);

        }

    private:

        void teleop_callback(geometry_msgs::msg::Twist::UniquePtr msg){
                        
            const std::vector<float> vtb = {(float)msg->linear.x, (float)msg->linear.y, (float)msg->angular.z};

            // Teleop Velocity Command
            vt = {std::cos(x[2])*vtb[0] - std::sin(x[2])*vtb[1],
                  std::sin(x[2])*vtb[0] + std::cos(x[2])*vtb[1],
                  vtb[2]}; 

            // Re-initialize If Not Started
            if(!start_flag) vt = {0.0f, 0.0f, 0.0f};

        };

        void keyboard_callback(std_msgs::msg::Int32::UniquePtr msg){
                                    
            // Check for Flags
            int ch = msg->data;
            switch(ch){
                case ' ':
                    space_counter++;
                    if(space_counter>=3) start_flag = true;
                    if(space_counter>=6) stop_flag = true;
                    break;
                case 'r':
                    realtime_sf_flag = !realtime_sf_flag;
                    break;
                case 'p':
                    predictive_sf_flag = !predictive_sf_flag;
                    break;
                default:
                    break;
            }

            switch(ch){
                case '1':
                    wn = 0.5f;
                    break;
                case '2':
                    wn = 1.0f;
                    break;
                case '3':
                    wn = 1.5f;
                    break;
                case '4':
                    wn = 2.0f;
                    break;
                case '5':
                    wn = 4.0f;
                    break;
                case '6':
                    wn = 8.0f;
                    break;
                default:
                    break;
            }

        };
        
        void mpc_callback(void){

            //Timer mpc_timer(true);
            //mpc_timer.start();

            // Define Waypoints
            // const float waypts[4][3] = {{ 0.0f,  0.0f,  0.0f},
            //                             { 3.0f,  0.0f,  0.0f},
            //                             { 0.0f,  2.0f,  3.14f},
            //                             {-4.0f,  2.0f,  3.14f}};

            const float waypts[11][3] = {{-31.738f, -87.877f,  0.037f},
                                         {-20.467f, -87.328f,  0.030f},
                                         { -9.969f, -87.626f, -0.058f},
                                         {-10.177f, -90.996f, -1.983f},
                                         {-13.458f, -97.497f, -2.146f},
                                         {-15.349f, -101.03f, -2.194f},
                                         {-18.394f, -102.28f,  3.129f},
                                         {-28.529f, -102.12f, -3.104f},
                                         {-35.350f, -101.87f,  2.959f},
                                         {-35.460f, -100.02f,  1.395f},
                                         {-33.027f, -90.675f,  1.217f}};

            // Identify Current Waypoint (If Waypoint Reached, Update the Counter)
            if(start_flag){
                const float ex = xd[0] - x[0];
                const float ey = xd[1] - x[1];
                const float e = std::sqrt(ex*ex+ey*ey);
                if(e<0.4f){
                    if(waypt_counter<10) waypt_counter++;
                    else waypt_counter=0;
                }
            }

            // Define Current Waypoint in SE(3)
            Eigen::Vector3f r_waypt;
            Eigen::Matrix3f R_waypt;
            r_waypt << waypts[waypt_counter][0], waypts[waypt_counter][1], 0.0f;
            R_waypt = Eigen::AngleAxisf(waypts[waypt_counter][2], Eigen::Vector3f::UnitZ());

            // Initialize Odom Frame Waypoint
            Eigen::Vector3f r_waypt_odom = r_waypt;
            Eigen::Matrix3f R_waypt_odom = R_waypt;

            // Transform to Odom Frame
            try{
                
                transform_ = tf_buffer_->lookupTransform("odom", "map", rclcpp::Time(0));
                
                Eigen::Vector3f r_map;
                r_map(0) = transform_.transform.translation.x;
                r_map(1) = transform_.transform.translation.y;
                r_map(2) = transform_.transform.translation.z;
                
                Eigen::Quaternionf q_map2odom;
                q_map2odom.w() = transform_.transform.rotation.w;
                q_map2odom.x() = transform_.transform.rotation.x;
                q_map2odom.y() = transform_.transform.rotation.y;
                q_map2odom.z() = transform_.transform.rotation.z;
                Eigen::Matrix3f R_map2odom = q_map2odom.toRotationMatrix();

                r_waypt_odom = R_map2odom * r_waypt + r_map;
                R_waypt_odom = R_map2odom * R_waypt;

            }
            catch(tf2::TransformException &ex){

                std::cout << "Global Map Transform Not Available Yet" << std::endl;
            
            }

            // Flatten Back to SE(2)
            xd[0] = r_waypt_odom(0);
            xd[1] = r_waypt_odom(1);
            xd[2] = std::atan2(R_waypt_odom(1,0), R_waypt_odom(0,0));
            xd[2] = std::fmod(xd[2]-x[2]+M_PI, 2.0f*M_PI) + x[2] - M_PI;

            // Rollout Desired Trajectory
            for(int k=0; k<TMAX; k++){
                xd_traj[3*k+0] = xd[0];
                xd_traj[3*k+1] = xd[1];
                xd_traj[3*k+2] = xd[2];
            }

            // Run MPC with SQP Loops until Cost is Stable
            if(predictive_sf_flag && h_flag && mpc_mutex.try_lock()){
                std::lock_guard<std::mutex> lock(mpc_mutex, std::adopt_lock);
                mpc3d_controller.reset_xbar_and_ubar(x, vn);
                for(int n=0; n<MAX_SQP_ITERS; n++){
                    mpc3d_controller.line_search(hgrid1, xc, wn, buff);
                    mpc3d_controller.update_cost(xd_traj, vn, n);
                    mpc3d_controller.update_constraints(hgrid1, x, xc, wn, issf, buff);
                    mpc3d_controller.solve();
                    if(mpc3d_controller.update_residual()<1.0f) break;
                }
                mpc3d_controller.set_input(vd);
            }

            //mpc_timer.time("MPC Solve Time: ");

        };
    
        /* Nominal Single Integrator Proportional Tracker */
        void nominal_controller(void){

            const float kp = 0.5f;
            vn[0] += std::clamp(kp*(xd[0]-x[0]), -1.0f, 1.0f);
            vn[1] += std::clamp(kp*(xd[1]-x[1]), -1.0f, 1.0f);
            vn[2] += std::clamp(kp*ang_diff(xd[2],x[2]), -1.0f, 1.0f);

        };

        /* Threshold Occupancy Map with Hysterisis */
        void build_occ_map(float *occ_map, const float *occ_map_old, const int8_t *conf){
            
            const int8_t T_hi = 85;
            const int8_t T_lo = 64;
                        
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    const int i0 = i + (int)std::round(dx[1] / DS);
                    const int j0 = j + (int)std::round(dx[0] / DS);
                    const bool in_grid = (i0 >= 0) && (i0 < IMAX) && (j0 >= 0) && (j0 < JMAX);
                    const bool strong = conf[i*JMAX+j] >= T_hi;
                    const bool weak = conf[i*JMAX+j] >= T_lo;
                    if(strong) occ_map[i*JMAX+j] = -1.0f;
                    else if(weak && in_grid){
                        if(occ_map_old[i0*JMAX+j0]==-1.0f) occ_map[i*JMAX+j] = -1.0f;
                    }
                    else occ_map[i*JMAX+j] = 1.0f;
                }
            }
 
        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void add_boundary(float *bound){
            
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
        void find_and_lock_boundary(float *grid, float *bound){
            
            // Set Border
            const int imin = buff;
            const int jmin = buff;
            const int imax = IMAX-buff;
            const int jmax = JMAX-buff;
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i<=imin||i>=(imax-1)||j<=jmin||j>=(jmax-1)){
                        bound[i*JMAX+j] = 0.0f;
                        grid[i*JMAX+j] = h0;
                    }
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
        
        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        int initialize_robot_kernel(float*& kernel){
            
            /* Create Robot Kernel */
            const float robot_radius = 1.0f * 0.37f; // Go2
            const int dim = 2 * (int)std::ceil(std::ceil(2.0f * robot_radius / DS) / 2.0f); //Make Sure Kernel Dimension is Even

            kernel = (float *)malloc(dim*dim*sizeof(float));
            for(int i = 0; i < dim; i++){
                const float y = (float)(i-dim/2)*DS;
                for(int j = 0; j < dim; j++){
                    kernel[i*dim+j] = 0.0f;
                    const float x = (float)(j-dim/2)*DS;
                    const float dist = x*x + y*y;
                    if(dist <= robot_radius*robot_radius) kernel[i*dim+j] = -1.0f;
                }
            }

            return dim;

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, const float *kernel){
            
            // Convolve Robot Kernel with Occupancy Grid, Along the Boundary
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));

            int lim = (robot_kernel_dim - 1)/2;
            for(int i = 0; i < IMAX; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 0; j < JMAX; j++){
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
            const int imin = buff;
            const int jmin = buff;
            const int imax = IMAX-buff;
            const int jmax = JMAX-buff;
            for(int i = imin+1; i < imax-1; i++){
                for(int j = jmin+1; j < jmax-1; j++){
                    if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
                    else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
                }
            }
            
            float perimeter_o = (float)(2*(IMAX-2*buff)+2*(JMAX-2*buff))*DS + perimeter_c;
            float area_o = (float)((IMAX-2*buff)*(JMAX-2*buff))*DS*DS - area_c;
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

            // Execute Poisson Pre-Processing
            build_occ_map(occ1, occ0, conf);
            add_boundary(occ1);

            // Initialize Temporary Grids
            float *hgrid_temp = (float *)malloc(IMAX*JMAX*sizeof(float));
            memcpy(hgrid_temp, hgrid1, IMAX*JMAX*sizeof(float));
            memcpy(bound, occ1, IMAX*JMAX*sizeof(float));
            inflate_occupancy_grid(bound, robot_kernel);
            find_and_lock_boundary(hgrid_temp, bound);
            compute_fast_forcing_function(force, bound);

            // Solve Poisson's Equation
            const float relTol = 1.0e-4f;
            const int N = IMAX/5;
            const float w_SOR = 2.0f/(1.0f+std::sin(M_PI/(float)(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            int iters = Kernel::poissonSolve(hgrid_temp, force, bound, relTol, w_SOR); // CUDA!
            
            // Transfer Solutions into Necessary Locations
            memcpy(occ0, occ1, IMAX*JMAX*sizeof(float));
            memcpy(hgrid0, hgrid1, IMAX*JMAX*sizeof(float));
            memcpy(hgrid1, hgrid_temp, IMAX*JMAX*sizeof(float));
            free(hgrid_temp);
            if(h_flag) dhdt_flag = true;
            
            solve_timer.time("Poisson Solve Time: ");
            printf("Poisson Iterations: %u \n", iters);

            return true;

        };

        /* Display Poisson Safety Function Grid (Interpolated) */
        void display_poisson_safety_function(void){

            Timer display_timer(true);
            display_timer.start();

            // Populate Float Grayscale Poisson Image with Chosen q & k Values
            cv::Mat poisson_img = cv::Mat::zeros(IMAX-2*buff, JMAX-2*buff, CV_32FC1);
            const int imin = buff;
            const int jmin = buff;
            const int imax = IMAX-buff;
            const int jmax = JMAX-buff;
            for (int i = imin; i < imax; i++){
                for (int j = jmin; j < jmax; j++){
                    poisson_img.at<float>(i-imin,j-imin) = hgrid1[i*JMAX+j];
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
            const int upscale = 1;
            if(upscale == 1) resized_img = color_img;
            else cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);
            
            // Add Current Location & Goal Location
            cv::Point curr_pt = cv::Point(upscale*(x_to_j(x[0],xc[0])-jmin),upscale*(y_to_i(x[1],xc[1])-imin));
            cv::Point goal_pt = cv::Point(upscale*(x_to_j(xd[0],xc[0])-jmin),upscale*(y_to_i(xd[1],xc[1])-imin));
            cv::circle(resized_img, curr_pt, upscale, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::circle(resized_img, goal_pt, 2*upscale, cv::Scalar(0, 127, 0), cv::FILLED);

            // Add MPC Trajectory
            for(int n = 1; n < TMAX; n++){
                const int j_traj = x_to_j(mpc3d_controller.sol(STATES*n+0), xc[0]) - jmin;
                const int i_traj = y_to_i(mpc3d_controller.sol(STATES*n+1), xc[1]) - imin;
                cv::Point traj_pt = cv::Point(upscale*j_traj, upscale*i_traj);
                cv::circle(resized_img, traj_pt, upscale/2, cv::Scalar(255, 0, 0), cv::FILLED);
            }

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
            const float kc = 1.0f - std::exp(-wc*dt_grid);
            const int imin = buff;
            const int jmin = buff;
            const int imax = IMAX-buff;
            const int jmax = JMAX-buff;
            for(int i = imin; i<imax; i++){
                for(int j = jmin; j<jmax; j++){
                    const float i0 = (float)i + dx[1]/DS;
                    const float j0 = (float)j + dx[0]/DS;
                    const bool in_grid = (i0 >= (float)imin) && (j0 >= (float)jmin) && (i0 <= (float)(imax-1)) && (j0 <= (float)(jmax-1));
                    float dhdt_ij = 0.0f;
                    if(in_grid){
                        const float h0 = bilinear_interpolation(hgrid0, i0, j0, buff);
                        const float h1 = bilinear_interpolation(hgrid1, (float)i, (float)j, buff);
                        dhdt_ij = (h1 - h0) / dt_grid;
                    }
                    dhdt_grid[i*JMAX+j] *= 1.0f - kc;
                    dhdt_grid[i*JMAX+j] += kc * dhdt_ij;
                }
            }

        }


        void safety_filter(const std::vector<float> vd){

            // Fractional Indices Corresponding to Current State
            const float ic = y_to_i(x[1], xc[1]);
            const float jc = x_to_j(x[0], xc[0]);            

            // Get Safety Function Value & Rate
            h = bilinear_interpolation(hgrid1, ic, jc, buff);
            dhdt = bilinear_interpolation(dhdt_grid, ic, jc, buff);
        
            // Compute Current Gradient
            const float eps = 2.0f;
            float hxp = bilinear_interpolation(hgrid1, ic, jc + eps, buff);
            float hxm = bilinear_interpolation(hgrid1, ic, jc - eps, buff);
            float hyp = bilinear_interpolation(hgrid1, ic + eps, jc, buff);
            float hym = bilinear_interpolation(hgrid1, ic - eps, jc, buff);
            dhdx = (hxp-hxm) / (2.0f*eps*DS);
            dhdy = (hyp-hym) / (2.0f*eps*DS);
            Eigen::Vector3f gradh{dhdx, dhdy, 0.0f};

            // Single Integrator Safety Filter
            const float b = gradh.dot(gradh);
            const float Lgh_norm = std::sqrt(b);
            // float ISSf = Lgh_norm/issf + Lgh_norm*Lgh_norm/issf;
            float ISSf = std::pow(Lgh_norm + 0.5f, 2.0f) / issf;
            float a = wn * h;
            a += dhdt; // Dynamic Environment
            a += dhdx * vd[0] + dhdy * vd[1]; // Min Norm Controller
            a -= ISSf; // Input-to-State Safety (Robustness)
            
            // Analytical Safety Filter
            const float sigma_sontag = 0.1f;
            float lambda = 0.0f;
            if(b>1.0e-4f) lambda = 1.0f * (-a + std::sqrt(a*a+sigma_sontag*b*b)) / (2.0f*b); // Half Sontag
            Eigen::Vector3f vs = lambda * gradh;

            v = vd;
            if(realtime_sf_flag){
                v[0] += vs(0);
                v[1] += vs(1);
                v[2] += vs(2);
            }

        };


        void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){

            // Compute Grid Timing
            dt_grid = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_grid).count();
            t_grid = std::chrono::steady_clock::now();

            // Read Message Data
            dx[0] = msg->info.origin.position.x - xc[0];
            dx[1] = msg->info.origin.position.y - xc[1];
            xc[0] = msg->info.origin.position.x;
            xc[1] = msg->info.origin.position.y;
            for(int n = 0; n < IMAX*JMAX; n++) conf[n] = msg->data[n];

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            h_flag = solve_poisson_safety_function();

            // Update Grid of dh/dt Values
            if(start_flag && dhdt_flag) update_dhdt_grid();

            // Display Results
            if(start_flag) display_poisson_safety_function();
            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "State: <" << x[0] << "," << x[1] << "," << x[2] << ">" << std::endl;
            std::cout << "Command: <" << vb[0] << "," << vb[1] << "," << vb[2] << ">" << std::endl;

        };

        void cbf_callback(void){
        
            // Query Current Body -> Odom Transform
            Eigen::Vector3f r{x[0], x[1], 0.0f};
            Eigen::Quaternionf q;
            Eigen::Matrix3f R;

            try{
                
                transform_ = tf_buffer_->lookupTransform("odom", "body", rclcpp::Time(0));
                
                r(0) = transform_.transform.translation.x;
                r(1) = transform_.transform.translation.y;
                r(2) = transform_.transform.translation.z;
                
                q.w() = transform_.transform.rotation.w;
                q.x() = transform_.transform.rotation.x;
                q.y() = transform_.transform.rotation.y;
                q.z() = transform_.transform.rotation.z;
                R = q.toRotationMatrix();

                dt_state = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_state).count();
                t_state = std::chrono::steady_clock::now();

                // Interpret State
                x[0] = r(0);
                x[1] = r(1);
                x[2] = std::atan2(R(1,0), R(0,0));

            }
            catch(tf2::TransformException &ex){

                std::cout << "Odom Not Available" << std::endl;
            
            }

            // Feedforward + Feedback Tracking Control
            vn = vt;
            //nominal_controller();
            
            // Safety Filter
            if(predictive_sf_flag){
                if(h_flag) safety_filter(vd); // Apply Safety Filter
            }
            else{
                if(h_flag) safety_filter(vn); // Apply Safety Filter
            }
            
            // Transform to Body-Fixed
            const std::vector<float> vb_new = {std::cos(x[2])*v[0] + std::sin(x[2])*v[1],
                                              -std::sin(x[2])*v[0] + std::cos(x[2])*v[1],
                                               v[2]};

            // Low Pass Filter
            // low_pass(vb, vb_new, 10.0f, dt_state);
            low_pass(vb, vb_new, 5.0f, dt_state); // Better for human interaction

            // Check for Failures
            if(std::abs(vb[0])>10.0f || std::abs(vb[1])>10.0f || std::abs(vb[2])>10.0f) sit_flag = true; // Check for Valid Control Action

            // Saturate
            vb[0] = std::clamp(vb[0], -2.5f, 3.8f);
            vb[1] = std::clamp(vb[1], -1.0f, 1.0f);
            vb[2] = std::clamp(vb[2], -4.0f, 4.0f);

            //Publish Control Action
            if(stop_flag){
                sport_req.StopMove(req);
                sleep(2);
                sport_req.StandDown(req);
                rclcpp::shutdown();
            }
            else if(sit_flag){
                sport_req.StopMove(req);
                sleep(2);
                sport_req.StandDown(req);
            }
            else if(start_flag){
                sport_req.Move(req, vb[0], vb[1], vb[2]); // Send Command
            }

        };

        std::mutex mpc_mutex;
        MPC3D mpc3d_controller;

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        float wn = 1.0f;
        float issf = 5.0f;

        bool h_flag = false;
        bool dhdt_flag = false;

        bool start_flag = false;
        bool sit_flag = false;
        bool stop_flag = false;
        bool predictive_sf_flag = false;
        bool realtime_sf_flag = false;
        int space_counter = 0;
        int waypt_counter = 0;
        
        // Define State
        std::vector<float> x = {0.0f, 0.0f, 0.0f};
        std::vector<float> xd = {0.0f, 0.0f, 0.0f};
        std::vector<float> xd_traj;
        std::vector<float> xc = {-2.0f, -2.0f, 0.0f};
        std::vector<float> xc0 = {-2.0f, -2.0f, 0.0f};
        std::vector<float> dx = {0.0f, 0.0f, 0.0f};

        std::chrono::steady_clock::time_point t_grid, t_state;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;

        std::vector<float> vt = {0.0f, 0.0f, 0.0f};
        std::vector<float> vn = {0.0f, 0.0f, 0.0f};
        std::vector<float> vd = {0.0f, 0.0f, 0.0f};
        std::vector<float> v = {0.0f, 0.0f, 0.0f};
        std::vector<float> vb = {0.0f, 0.0f, 0.0f};
        float h, dhdt, dhdx, dhdy;
        
        float occ1[IMAX*JMAX];
        float occ0[IMAX*JMAX];
        int8_t conf[IMAX*JMAX];
        float *hgrid1, *hgrid0, *bound, *force, *robot_kernel, *dhdt_grid;
        int robot_kernel_dim, buff;
        
        rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
        rclcpp::CallbackGroup::SharedPtr cbf_callback_group_;
        rclcpp::TimerBase::SharedPtr mpc_timer_;
        rclcpp::TimerBase::SharedPtr cbf_timer_;
        
        rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr key_suber_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;

        geometry_msgs::msg::TransformStamped transform_;
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber_;
        unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
        SportClient sport_req;

};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto mappingNode = std::make_shared<CloudMergerNode>();
    auto poissonNode = std::make_shared<PoissonControllerNode>();
    executor.add_node(mappingNode);
    executor.add_node(poissonNode);
    
    try{
        executor.spin();
        throw("Terminated");
    }
    catch(const char* msg){
        rclcpp::shutdown();
        std::cout << msg << std::endl;
    }

  return 0;

}