#pragma once

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "unitree_go/msg/sport_mode_state.hpp"
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <Eigen/Dense>

#include "utils.h"
#include "poisson.h"

const float MOS = 1.35f;
const float minX = 0.370f * MOS; // Must be >= 0.370
const float maxX = (float)(JMAX/2) * DS;
const float minY = 0.185f * MOS; // Must be >= 0.185
const float maxY = (float)(IMAX/2) * DS;
const float minZ = 0.05f;
const float maxZ = 0.80f;

class CloudMergerNode : public rclcpp::Node{
    
    public:
        
        CloudMergerNode() : Node("cloud_merger"){

            // Initialize Cloud Message
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "odom";
            
            // Initialize Map Message
            map_msg.data.resize(IMAX*JMAX);
            map_msg.header.stamp = this->now();
            map_msg.header.frame_id = "odom";
            map_msg.info.width  = IMAX;
            map_msg.info.height = JMAX;
            map_msg.info.resolution = DS;
            map_msg.info.origin.position.x = -maxX + r(0);
            map_msg.info.origin.position.y = -maxY + r(1);
            map_msg.info.origin.position.z = 0.0f;
            map_msg.info.origin.orientation.w = 1.0;
            map_msg.info.origin.orientation.x = 0.0f;
            map_msg.info.origin.orientation.y = 0.0f;
            map_msg.info.origin.orientation.z = 0.0f;

            // Construct Initial Grids
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    const float x = (float)(j-JMAX/2) * DS;
                    const float y = (float)(i-IMAX/2) * DS;
                    polar_coordinates_r2[i*JMAX+j] = x*x+y*y;
                    polar_coordinates_th[i*JMAX+j] = std::atan2(y,x);
                    old_conf[i*JMAX+j] = 0;
                }
            }

            // Start Time
            t = std::chrono::steady_clock::now();

            // Create Subscribers & Publishers
            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/livox/lidar", 1, std::bind(&CloudMergerNode::lidar_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/utlidar/cloud_deskewed", 1, std::bind(&CloudMergerNode::combined_callback, this, std::placeholders::_1));
            robot_pose_sub_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1, std::bind(&CloudMergerNode::pose_callback, this, std::placeholders::_1));
            cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("poisson_cloud", 1);
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

            combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

        }

    private:
        
        void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            //Start timer
            Timer map_timer(true);
            map_timer.start();

            dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - t).count();
            t = std::chrono::steady_clock::now();
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr unitree_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *unitree_cloud);

            // Mask Robot Body with Circle
            pcl::PointCloud<pcl::PointXYZ>::Ptr masked_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& pt : unitree_cloud->points){
                const float dist = std::sqrt((pt.x-r(0))*(pt.x-r(0)) + (pt.y-r(1))*(pt.y-r(1)));
                if((dist>minX) && (dist>minY)) masked_cloud->points.push_back(pt);
            }
            masked_cloud->width = masked_cloud->points.size();
            masked_cloud->height = 1;

            *masked_cloud += *combined_cloud_;
            combined_cloud_->clear();

            // Create Occupancy Grid object
            cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
            for(const auto& pt : masked_cloud->points){
                const bool in_plane = (pt.z > minZ) && (pt.z < maxZ);
                if(!in_plane) continue;
                const float ic = (pt.y - r(1)) / DS + (float)(IMAX/2);
                const float jc = (pt.x - r(0)) / DS + (float)(JMAX/2);
                const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX-1)) && (jc > 0.0f) && (jc < (float)(JMAX-1));
                if(!in_grid) continue;
                raw_map.at<float>((int)std::round(ic),(int)std::round(jc)) = 1.0f;             
            }

            // BUILD MAP HERE
            for(int n=0; n<IMAX*JMAX; n++) confidence_values[n] = 0;
            Filtered_Occupancy_Convolution(confidence_values, raw_map, old_conf);
            memcpy(old_conf, confidence_values, IMAX*JMAX*sizeof(int8_t));

            // Publish Filtered Point Cloud
            pcl::toROSMsg(*masked_cloud, cloud_msg);
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "odom";
            cloud_pub_->publish(cloud_msg);

            // Publish Confidence Map
            for(int n=0; n<IMAX*JMAX; n++) map_msg.data[n] = confidence_values[n];
            map_msg.header.stamp = this->now();
            map_msg.info.origin.position.x = -maxX + r(0);
            map_msg.info.origin.position.y = -maxY + r(1);
            map_pub_->publish(map_msg);
            
            map_timer.time("Occ Map Solve Time: ");

        }

        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Populate Point Cloud with LiDAR Points
            pcl::PointCloud<pcl::PointXYZ>::Ptr livox_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *livox_cloud);

            // Define Transform from LiDAR Frame to Body Frame
            Eigen::Matrix3f R_lidar, R_y_trim, R_p_trim, R_r_trim;
            R_lidar << -1.0f,  0.0f,  0.0f, //180 deg pitch
                        0.0f,  1.0f,  0.0f,
                        0.0f,  0.0f, -1.0f; 
            R_r_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitX());
            R_p_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitY());
            R_y_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitZ());
            Eigen::Matrix3f R_lidar2body = R_y_trim * R_p_trim * R_r_trim * R_lidar;
            Eigen::Vector3f r_lidar{-0.05f, 0.0f, 0.18f}; // Location of the LiDAR in Body Frame

            // Perform Affine Transform
            Eigen::Affine3f lidar2body = Eigen::Affine3f::Identity();
            lidar2body.linear() = R_lidar2body;  //rotation to final frame
            lidar2body.translation() = r_lidar; // translation in final frame
            pcl::transformPointCloud(*livox_cloud, *livox_cloud, lidar2body);

            // Mask Robot Body with Hyper-Ellipse in Body Frame
            pcl::PointCloud<pcl::PointXYZ>::Ptr masked_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& pt : livox_cloud->points){
                float ellipse_norm = std::pow(pt.x/minX,8.0f) + std::pow(pt.y/minY,8.0f);
                if(ellipse_norm > 1.0f) masked_cloud->points.push_back(pt);
            }
            masked_cloud->width = masked_cloud->points.size();
            masked_cloud->height = 1;

            // Transform from Body Frame to Inertial Frame
            Eigen::Affine3f body2inertial = Eigen::Affine3f::Identity();
            body2inertial.linear() = R;
            body2inertial.translation() = r;
            pcl::transformPointCloud(*masked_cloud, *masked_cloud, body2inertial);

            // Add Points into Combined Cloud
            *combined_cloud_ += *masked_cloud;

        } 

        void pose_callback(const unitree_go::msg::SportModeState::SharedPtr msg){

            r(0) = msg->position[0];
            r(1) = msg->position[1];
            r(2) = msg->position[2];

            q.w() = msg->imu_state.quaternion[0];
            q.x() = msg->imu_state.quaternion[1];
            q.y() = msg->imu_state.quaternion[2];
            q.z() = msg->imu_state.quaternion[3];
            R = q.toRotationMatrix();
            
            rpy(0) = msg->imu_state.rpy[0];
            rpy(1) = msg->imu_state.rpy[1];
            rpy(2) = msg->imu_state.rpy[2];

        }

        //  CREATE GAUSSIAN KERNEL
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

        //  BUFFERED CONVOLUTION
        void Filtered_Occupancy_Convolution(int8_t *confidence_values, const cv::Mat& occupancy_data, const int8_t *old_conf_map){

            // Shift Confidence Values Based on Egomotion
            const float di = (r(1) - r_map(1)) / DS;
            const float dj = (r(0) - r_map(0)) / DS;
            r_map = r;

            for(int i = 0; i < IMAX; i++) {
                for(int j = 0; j < JMAX; j++){
                    float i0 = (float)i + di;
                    float j0 = (float)j + dj;
                    const bool in_grid = (i0 >= 0.0f) && (i0 <= (float)(IMAX-1)) && (j0 >= 0.0f) && (j0 <= (float)(JMAX-1));
                    if(in_grid) confidence_values[i*JMAX+j] = bilinear_interpolation_int8(old_conf_map, i0, j0);
                }
            }

            // Apply Gaussian decay kernel to occupancy_data
            cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // set parameters
            float sig, C, beta_up, beta_dn;
            const float thresh = 1.0f;
            // const float thresh = 4.0f; // Better for human interaction
            bool front_flag = true;
            
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    
                    const float r2 = polar_coordinates_r2[i*JMAX+j];
                    const float th = polar_coordinates_th[i*JMAX+j];
                    const bool range_flag = r2 > 1.44f;
                    const bool angle_flag = std::abs(ang_diff(rpy(2), th)) > 0.6f;
                    if(range_flag || angle_flag) front_flag = false;
                    else front_flag = true;
                    
                    float val_binary = buffered_binary.at<float>(i,j);
                    float conf = (float)confidence_values[i*JMAX+j] / 127.0f;
                    if(val_binary > thresh){
                        if(front_flag) beta_up = 4.0f; //Go2 Front LiDAR only
                        else beta_up = 1.0f; // Livox Mid360
                        sig = 1.0f - std::exp(-beta_up*val_binary*dt);
                        C = 1.0f;
                    }
                    else{
                        if(front_flag) beta_dn = 4.0f;
                        else beta_dn = 4.0f;
                        sig = 1.0f - std::exp(-beta_dn*dt);
                        C = 0.0f;
                    }
                    conf *= 1.0f - sig;
                    conf += sig * C;
                    confidence_values[i*JMAX+j] = (int8_t)std::round(127.0f*conf);

                }
            }

        }
    
            sensor_msgs::msg::PointCloud2 cloud_msg;
            nav_msgs::msg::OccupancyGrid map_msg;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr robot_pose_sub_;
            
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
            rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

            pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud_;
            
            Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f r_map{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f rpy{0.0f, 0.0f, 0.0f};
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            Eigen::Quaternionf q = Eigen::Quaternionf::Identity();

            std::chrono::steady_clock::time_point t;
            float dt = 1.0e10f;

            // Generate gaussian kernel for convolution later
            const cv::Mat gauss_kernel = gaussian_kernel(9, 2.0);
            
            int8_t confidence_values[IMAX*JMAX];
            int8_t old_conf[IMAX*JMAX];
            float polar_coordinates_r2[IMAX*JMAX];
            float polar_coordinates_th[IMAX*JMAX];
            cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);

}; 
