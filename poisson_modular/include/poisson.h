#pragma once

#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>

#include "kernel.hpp"
#include "definitions.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <poisson_msgs/msg/poisson_grid.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PoissonNode : public rclcpp::Node{

    public:
        
        PoissonNode() : Node("poisson_solver"){

            // Start Time
            t = std::chrono::steady_clock::now();
            
            // Initialize Map Message
            map_msg_.data.resize(IMAX*JMAX);
            map_msg_.header.frame_id = "map";
            map_msg_.header.stamp = this->now();
            map_msg_.info.width  = JMAX;
            map_msg_.info.height = IMAX;
            map_msg_.info.resolution = DS;
            map_msg_.info.origin.position.x = rc(0);
            map_msg_.info.origin.position.y = rc(1);
            map_msg_.info.origin.position.z = 0.0f;
            map_msg_.info.origin.orientation.w = 1.0;
            map_msg_.info.origin.orientation.x = 0.0f;
            map_msg_.info.origin.orientation.y = 0.0f;
            map_msg_.info.origin.orientation.z = 0.0f;

            // Initialize Poisson Message
            poisson_msg_.data.resize(IMAX*JMAX);
            poisson_msg_.header.frame_id = "map";
            poisson_msg_.header.stamp = this->now();
            poisson_msg_.width = JMAX;
            poisson_msg_.height = IMAX;
            poisson_msg_.resolution = DS;
            poisson_msg_.origin.position.x = rc(0);
            poisson_msg_.origin.position.y = rc(1);
            poisson_msg_.origin.position.z = 0.0f;
            poisson_msg_.origin.orientation.w = 1.0;
            poisson_msg_.origin.orientation.x = 0.0f;
            poisson_msg_.origin.orientation.y = 0.0f;
            poisson_msg_.origin.orientation.z = 0.0f;

            // Define Useful Rotation Matrices
            R_ned2enu.row(0) << 0.0f, 1.0f, 0.0f;
            R_ned2enu.row(1) << 1.0f, 0.0f, 0.0f;
            R_ned2enu.row(2) << 0.0f, 0.0f, -1.0f;
            R_nwu2enu.row(0) << 0.0f, -1.0f, 0.0f;
            R_nwu2enu.row(1) << 1.0f, 0.0f, 0.0f;
            R_nwu2enu.row(2) << 0.0f, 0.0f, 1.0f;
            
            // Initialize Point Clouds
            raw_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            masked_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            
            // Initialize Occupancy Grids
            for(int n = 0; n < IMAX*JMAX; n++){
                occ1_[n] = 1.0f;
                occ0_[n] = 1.0f; 
                conf_[n] = 0;
            }

            // Initialize Poisson Grids
            cudaMallocHost((void**)&hgrid_, IMAX*JMAX*sizeof(float));
            cudaMallocHost((void**)&bound_, IMAX*JMAX*sizeof(float));
            cudaMallocHost((void**)&force_, IMAX*JMAX*sizeof(float));
            for(int n=0; n < IMAX*JMAX; n++) hgrid_[n] = h0;
            
            // Initialize Robot Kernel
            Kernel::poissonInit();
            robot_kernel_dim = initialize_robot_kernel(robot_kernel_);

            // Create Subscribers & Publishers
            rclcpp::QoS lidar_qos(1);
            rclcpp::QoS odom_qos(1);
            lidar_qos.best_effort();
            odom_qos.best_effort();
            lidar_suber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/world/caltech/model/x500_livox_mid360_0/link/link/sensor/livox_mid360/scan/points", lidar_qos, std::bind(&PoissonNode::lidar_callback, this, std::placeholders::_1));
            odom_suber_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>("/fmu/out/vehicle_odometry", odom_qos, std::bind(&PoissonNode::pose_callback, this, std::placeholders::_1));
            map_puber_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);
            cbf_puber_ = this->create_publisher<poisson_msgs::msg::PoissonGrid>("poisson_cbf", 1);

       }

    private:

        /* Register the Incoming Point Cloud & Mask Unwanted Points */
        void register_and_mask_cloud(void){
        
            // Transform Scan from LiDAR (IMU) Frame to Body Frame
            Eigen::Affine3f imu2body = Eigen::Affine3f::Identity();
            imu2body.linear() = R_nwu2enu; //rotation
            imu2body.translation() << 0.12f, 0.0f, 0.15f; //translation
            pcl::transformPointCloud(*raw_cloud_, *raw_cloud_, imu2body);
        
            // Crop Point Cloud in Body Frame
            for(const auto& pt : raw_cloud_->points){
                const float dist = std::sqrt(pt.x*pt.x+pt.y*pt.y);
                if(dist>minR && dist<maxR) masked_cloud_->points.push_back(pt);
            }

            raw_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

            // Transform Scan from Body Frame to Odom Frame
            Eigen::Affine3f body2inertial = Eigen::Affine3f::Identity();
            body2inertial.linear() = R; //rotation
            body2inertial.translation() << r(0), r(1), r(2); //translation
            pcl::transformPointCloud(*masked_cloud_, *masked_cloud_, body2inertial);

        }

        /* Increment A Cell for Every Point Cloud Point */
        void populate_raw_occupancy_map(void){

            // Create Occupancy Grid object
            raw_map_.setTo(0.0f);
            for(const auto& pt : masked_cloud_->points){
                const float ic = y_to_i(pt.y, rc(1));
                const float jc = x_to_j(pt.x, rc(0));
                const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX-1)) && (jc > 0.0f) && (jc < (float)(JMAX-1));
                if(!in_grid) continue;
                const bool above_ground = pt.z > 0.05f;
                if(!above_ground) continue;
                raw_map_.at<float>((int)std::round(ic),(int)std::round(jc)) += 1.0f;             
            }

            masked_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            
        }

        /* Create Gaussian Kernel to Smooth Occupancy Map */
        cv::Mat gaussian_kernel(int kernel_size, float sigma){
            // Create kernel_sizexkernel_size array of floats
            cv::Mat kernel(kernel_size, kernel_size, CV_32F); 

            int half = kernel_size/2;
            // Iterate through each cell
            for(int i=-half; i<=half; i++){
                for (int j=-half; j<=half; j++){
                    float val = std::exp(-(float)(i*i+j*j)/(2.0f*sigma*sigma));
                    kernel.at<float>(i+half, j+half) = val;
                }
            }

            return kernel;
        }

        /* Update Occupancy Map Confidence Values */
        void update_confidence_values(void){

            // Parameters
            const float thresh = 1.0f;
            const float beta_up = 1.0f;
            const float beta_dn = 4.0f;
            
            // Apply Gaussian Decay Kernel to Raw Occupancy Data
            cv::Mat smooth_occ;
            cv::filter2D(raw_map_, smooth_occ, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // Update Confidence
            int8_t conf_temp[IMAX*JMAX];
            for(int i = 0; i < IMAX; i++) {
                for(int j = 0; j < JMAX; j++){

                    // Initialize Confidence Value (Accounting for Egomotion)
                    float conf_ij = 0.0f;
                    float i0 = (float)i + dr(1) / DS;
                    float j0 = (float)j + dr(0) / DS;
                    const bool in_grid = (i0 >= 0.0f) && (i0 <= (float)(IMAX-1)) && (j0 >= 0.0f) && (j0 <= (float)(JMAX-1));
                    if(in_grid) conf_ij = (float)bilinear_interpolation(conf_, i0, j0) / 127.0f;
                    
                    // Determine Update Dynamics from Raw Smoothed Occupancy
                    const float occ_ij = smooth_occ.at<float>(i,j);
                    float sig, C;
                    if(occ_ij > thresh){
                        sig = 1.0f - std::exp(-beta_up*occ_ij*dt);
                        C = 1.0f;
                    }
                    else{
                        sig = 1.0f - std::exp(-beta_dn*dt);
                        C = 0.0f;
                    }
                    
                    // Update Confidence Value
                    conf_ij *= 1.0f - sig;
                    conf_ij += sig * C;
                    conf_temp[i*JMAX+j] = (int8_t)std::round(127.0f*conf_ij);

                }
            }

            // Copy Tempory Grid Back to Global Grid
            memcpy(conf_, conf_temp, IMAX*JMAX*sizeof(int8_t));

        }

        /* Threshold Occupancy Map with Hysterisis */
        void build_occ_map(float *occ_map, const float *occ_map_old, const int8_t *conf){
            
            const int8_t T_hi = 85;
            const int8_t T_lo = 64;
                        
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    const int i0 = i + (int)std::round(dr(1) / DS);
                    const int j0 = j + (int)std::round(dr(0) / DS);
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
            const int imin = 0;
            const int jmin = 0;
            const int imax = IMAX-1;
            const int jmax = JMAX-1;
            for(int i = 0; i < IMAX; i++) bound[i*JMAX+jmin] = 0.0f;
            for(int i = 0; i < IMAX; i++) bound[i*JMAX+jmax] = 0.0f;
            for(int j = 0; j < JMAX; j++) bound[imin*JMAX+j] = 0.0f;
            for(int j = 0; j < JMAX; j++) bound[imax*JMAX+j] = 0.0f;

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
        
        /* Construct n x n Kernel Using Circle */
        int initialize_robot_kernel(float*& kernel){
            
            /* Create Robot Kernel */
            const float robot_radius = 0.5f; // X500
            const float D = 2.0f * robot_radius; // Max Robot Dimension to Define Kernel Size
            const int dim = 2 * (int)std::ceil(std::ceil(D / DS) / 2.0f); //Make Sure Kernel Dimension is Even

            kernel = (float *)malloc(dim*dim*sizeof(float));
            for(int i = 0; i < dim; i++){
                const float yi = (float)(i-dim/2)*DS;
                for(int j = 0; j < dim; j++){
                    kernel[i*dim+j] = 0.0f;
                    const float xi = (float)(j-dim/2)*DS;
                    const float dist = (xi*xi + yi*yi) / (robot_radius*robot_radius);
                    if(dist <= 1.0f) kernel[i*dim+j] = -1.0f;
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
            
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
                    else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
                }
            }
            
            float perimeter_o = (float)(2*IMAX+2*JMAX)*DS + perimeter_c;
            float area_o = (float)(IMAX*JMAX)*DS*DS - area_c;
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
        void solve_poisson_safety_function(void){

            // Execute Poisson Pre-Processing
            build_occ_map(occ1_, occ0_, conf_);
            add_boundary(occ1_);

            // Initialize Temporary Grids
            memcpy(bound_, occ1_, IMAX*JMAX*sizeof(float));
            inflate_occupancy_grid(bound_, robot_kernel_);
            find_and_lock_boundary(hgrid_, bound_);
            compute_fast_forcing_function(force_, bound_);

            // Solve Poisson's Equation
            const float relTol = 1.0e-4f;
            const int N = IMAX/5;
            const float w_SOR = 2.0f/(1.0f+std::sin(M_PI/(float)(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            int iters = Kernel::poissonSolve(hgrid_, force_, bound_, relTol, w_SOR); // CUDA!
            
            // // Transfer Solutions into Necessary Locations
            memcpy(occ0_, occ1_, IMAX*JMAX*sizeof(float));
            
            // std::cout << "Poisson Iterations: " << iters << std::endl;

        };

        /* Publish Occupancy Map */
        void publish_occupancy_map(void){

            map_msg_.header.frame_id = "map";
            map_msg_.header.stamp = this->now();
            map_msg_.info.origin.position.x = rc(0);
            map_msg_.info.origin.position.y = rc(1);
            for(int n=0; n<IMAX*JMAX; n++) map_msg_.data[n] = conf_[n];
            map_puber_->publish(map_msg_); 

        };

        /* Publish Poisson Safety Function Grid */
        void publish_poisson_safety_function(void){

            poisson_msg_.header.frame_id = "map";
            poisson_msg_.header.stamp = this->now();
            poisson_msg_.origin.position.x = rc(0);
            poisson_msg_.origin.position.y = rc(1);
            for(int n=0; n<IMAX*JMAX; n++) poisson_msg_.data[n] = hgrid_[n];
            this->cbf_puber_->publish(poisson_msg_);

        };

        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Start timer
            Timer poisson_timer(true);
            poisson_timer.start();

            // Compute Deltas
            dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - t).count();
            t = std::chrono::steady_clock::now();
            dr = r - r_map;
            r_map << r(0), r(1), r(2);
            rc << r(0) - maxX, r(1) - maxY, 0.0f;
            
            // Populate Point Cloud with LiDAR Points
            pcl::fromROSMsg(*msg, *raw_cloud_);
            
            // Solve for Map & Poisson
            register_and_mask_cloud();
            populate_raw_occupancy_map();
            update_confidence_values();
            solve_poisson_safety_function();
            
            // Publish
            publish_occupancy_map();
            publish_poisson_safety_function();

            poisson_timer.time("Poisson Solve Time: ");

        };

        void pose_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg){

            Eigen::Vector3f r_ned{msg->position[0], msg->position[1], msg->position[2]};
            r = R_ned2enu * r_ned;
            
            Eigen::Quaternionf q;
            q.w() = msg->q[0];
            q.x() = msg->q[1];
            q.y() = msg->q[2];
            q.z() = msg->q[3];

            Eigen::Matrix3f R_ned(q);
            R = R_ned2enu.transpose() * R_ned * R_ned2enu;

        }
        
        // Time
        std::chrono::steady_clock::time_point t;
        float dt = 1.0e10f;

        // Point Clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr masked_cloud_;

        // CV Occupancy Grid
        cv::Mat raw_map_ = cv::Mat::zeros(IMAX, JMAX, CV_32F);

        // LiDAR Cropping Parameters
        const float minR = 0.5f;
        const float maxX = (float)(JMAX/2) * DS;
        const float maxY = (float)(IMAX/2) * DS;
        const float maxR = std::sqrt(maxX*maxX+maxY*maxY);

        // Robot States
        Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f r_map{0.0f, 0.0f, 0.0f};
        Eigen::Matrix3f R, R_ned2enu, R_nwu2enu;

        // Grid States
        Eigen::Vector3f rc{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f dr{0.0f, 0.0f, 0.0f};

        // Grids
        float occ1_[IMAX*JMAX];
        float occ0_[IMAX*JMAX];
        int8_t conf_[IMAX*JMAX];
        float *hgrid_, *bound_, *force_;
        float *robot_kernel_;

        // Poisson Parameters
        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        // Gaussian Kernel for Convolution
        const cv::Mat gauss_kernel = gaussian_kernel(3, 2.0);

        // Robot Kernel
        int robot_kernel_dim;

        // Messages
        nav_msgs::msg::OccupancyGrid map_msg_;
        poisson_msgs::msg::PoissonGrid poisson_msg_;

        // Subscribers
        rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_suber_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_suber_;

        // Publishers
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_puber_;
        rclcpp::Publisher<poisson_msgs::msg::PoissonGrid>::SharedPtr cbf_puber_;
        
};