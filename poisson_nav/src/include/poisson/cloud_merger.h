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
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>

#include <cmath>
#include <Eigen/Dense>

#include "utils.h"
#include "poisson.h"

bool initialized = false;

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

            t_start = std::chrono::steady_clock::now();

            // Initialize Odom Message
            odom_msg.header.stamp = this->now();
            odom_msg.header.frame_id = "odom";

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

            combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            global_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::io::loadPCDFile("map.pcd", *global_cloud_);

            // Start Time
            t = std::chrono::steady_clock::now();

            // Create Subscribers
            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/livox/lidar", 1, std::bind(&CloudMergerNode::livox_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/utlidar/cloud_deskewed", 1, std::bind(&CloudMergerNode::utlidar_callback, this, std::placeholders::_1));
            robot_pose_sub_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1, std::bind(&CloudMergerNode::pose_callback, this, std::placeholders::_1));
            
            // Create Publishers
            odom_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("aligned_pose", 1);
            cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("poisson_cloud", 1);
            global_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("global_cloud", 1);
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

            // Create Timers
            map_timer_ = this->create_wall_timer(std::chrono::milliseconds(200), std::bind(&CloudMergerNode::occ_map_callback, this));

        }

    private:

        void register_lidar_scan(pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud){
            
            Timer registration_timer(true);
            registration_timer.start();

            // Initialize Point Clouds
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr registered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            const float res[2] = {4.0e-1f, 2.0e-1f};
            //const float eps[2] = {1.0e-4f, 1.0e-6f};
            for(int i=0; i<2; i++){

                // Initialize GICP
                //gicp.setMaxCorrespondenceDistance(4.0f*res[i]);
                //gicp.setMaximumIterations(100);
                //gicp.setTransformationEpsilon(eps[i]);
                pcl::VoxelGrid<pcl::PointXYZ> voxel;
                voxel.setLeafSize(res[i], res[i], res[i]);
                
                // Voxel Filter Map and Scan
                filtered_global_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
                voxel.setInputCloud(global_cloud_);
                voxel.filter(*filtered_global_cloud);
                //*filtered_scan_cloud = *scan_cloud;
                filtered_scan_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
                voxel.setInputCloud(scan_cloud);
                voxel.filter(*filtered_scan_cloud);

                // Register Scan
                gicp.setInputSource(filtered_scan_cloud);
                gicp.setInputTarget(filtered_global_cloud);
                gicp.align(*registered_cloud, initial_guess);
                
                Eigen::Matrix4f transform = gicp.getFinalTransformation();
                R_cloud = transform.block<3,3>(0,0);
                r_cloud = transform.block<3,1>(0,3);
                
                initial_guess = transform;

            } 

            registration_timer.time("Point Cloud Registration Time: ");
        
            pcl::toROSMsg(*global_cloud_, global_cloud_msg);
            global_cloud_msg.header.stamp = this->now();
            global_cloud_msg.header.frame_id = "odom";
            global_cloud_pub_->publish(global_cloud_msg);

            pcl::toROSMsg(*registered_cloud, cloud_msg);
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "odom";
            cloud_pub_->publish(cloud_msg);

            Eigen::Quaternionf q_cloud(R_cloud);
            std::cout << "Estimated Position Error: <" << 100.0f * r_cloud(0) << " , " << 100.0f * r_cloud(1) << " , " << 100.0f * r_cloud(2) << "> cm" << std::endl;
            std::cout << "Estimated Attitude Error: " << 180.0f/M_PI * 2.0f*std::asin(std::sqrt(q_cloud.vec().dot(q_cloud.vec()))) << " deg" << std::endl;

        }

        void occ_map_callback(void){

            //Start timer
            Timer map_timer(true);
            map_timer.start();

            scan_counter++;
            if(scan_counter >= 5 && registration_counter < 10){
                scan_counter = 0;
                register_lidar_scan(combined_cloud_);
                registration_counter++;
                if(registration_counter == 10){
                    R_init = R_cloud * R_init;
                    r_init = R_cloud * r_init + r_cloud;
                }
            }

            // Create Occupancy Grid object
            cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
            for(const auto& pt : combined_cloud_->points){
                const bool in_plane = (pt.z > minZ) && (pt.z < maxZ);
                if(!in_plane) continue;
                const float ic = (pt.y - r(1)) / DS + (float)(IMAX/2);
                const float jc = (pt.x - r(0)) / DS + (float)(JMAX/2);
                const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX-1)) && (jc > 0.0f) && (jc < (float)(JMAX-1));
                if(!in_grid) continue;
                raw_map.at<float>((int)std::round(ic),(int)std::round(jc)) += 1.0f;             
            }

            // Publish Filtered Point Cloud
            // pcl::toROSMsg(*combined_cloud_, cloud_msg);
            // cloud_msg.header.stamp = this->now();
            // cloud_msg.header.frame_id = "odom";
            // cloud_pub_->publish(cloud_msg);
            
            combined_cloud_->clear();

            // BUILD MAP HERE
            dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - t).count();
            t = std::chrono::steady_clock::now();
            for(int n=0; n<IMAX*JMAX; n++) confidence_values[n] = 0;
            Filtered_Occupancy_Convolution(confidence_values, raw_map, old_conf);
            memcpy(old_conf, confidence_values, IMAX*JMAX*sizeof(int8_t));

            // Publish Confidence Map
            for(int n=0; n<IMAX*JMAX; n++) map_msg.data[n] = confidence_values[n];
            map_msg.header.stamp = this->now();
            map_msg.header.frame_id = "odom";
            map_msg.info.origin.position.x = -maxX + r(0);
            map_msg.info.origin.position.y = -maxY + r(1);
            map_pub_->publish(map_msg);
            
            map_timer.time("Occ Map Solve Time: ");

        }
        
        void utlidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr utlidar_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *utlidar_cloud);

            // Mask Robot Body with Circle
            pcl::PointCloud<pcl::PointXYZ>::Ptr masked_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for(const auto& pt : utlidar_cloud->points){
                const float dist = std::sqrt((pt.x-r(0))*(pt.x-r(0)) + (pt.y-r(1))*(pt.y-r(1)));
                if((dist>minX) && (dist>minY)) masked_cloud->points.push_back(pt);
            }
            masked_cloud->width = masked_cloud->points.size();
            masked_cloud->height = 1;

            // Transform from Odom Frame to Inertial Frame
            Eigen::Affine3f body2inertial = Eigen::Affine3f::Identity();
            body2inertial.linear() = R_init;
            body2inertial.translation() = r_init;
            pcl::transformPointCloud(*masked_cloud, *masked_cloud, body2inertial);

            // Add Points into Combined Cloud
            *combined_cloud_ += *masked_cloud;

        }

        void livox_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Populate Point Cloud with LiDAR Points
            pcl::PointCloud<pcl::PointXYZ>::Ptr livox_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *livox_cloud);

            // Define Transform from LiDAR Frame to Body Frame
            Eigen::Matrix3f R_lidar, R_y_trim, R_p_trim, R_r_trim, R_lidar2body;
            R_lidar << -1.0f, 0.0f, 0.0f, //z-axis flip & 180 degree yaw
                0.0f,  1.0f, 0.0f,
                0.0f,  0.0f, -1.0f; 
            R_r_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitX());
            R_p_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitY());
            R_y_trim = Eigen::AngleAxisf(0.0f * M_PI/180.0f, Eigen::Vector3f::UnitZ());
            R_lidar2body = R_y_trim * R_p_trim * R_r_trim * R_lidar;
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

            Eigen::Vector3f r_odom;
            r_odom << msg->position[0], msg->position[1], msg->position[2];

            Eigen::Quaternionf q_odom;
            q_odom.w() = msg->imu_state.quaternion[0];
            q_odom.x() = msg->imu_state.quaternion[1];
            q_odom.y() = msg->imu_state.quaternion[2];
            q_odom.z() = msg->imu_state.quaternion[3];
            Eigen::Matrix3f R_odom = q_odom.toRotationMatrix();
            
            if(!init_flag){
                R_init = R_odom.transpose();
                r_init = -R_init * r_odom;
                init_flag = true;
            }

            r = R_init * r_odom + r_init;
            R = R_init * R_odom;
            Eigen::Quaternionf q(R);
            yaw = std::atan2(R(1,0), R(0,0));
            odom_msg.pose.position.x = r(0);
            odom_msg.pose.position.y = r(1);
            odom_msg.pose.position.z = r(2);
            odom_msg.pose.orientation.w = q.w();
            odom_msg.pose.orientation.x = q.x();
            odom_msg.pose.orientation.y = q.y();
            odom_msg.pose.orientation.z = q.z();
            odom_msg.header.stamp = this->now();
            odom_msg.header.frame_id = "odom";
            odom_pub_->publish(odom_msg);

        }

        //  CREATE GAUSSIAN KERNEL
        cv::Mat gaussian_kernel(int kernel_size, float sigma){
            // Create kernel_sizexkernel_size array of floats
            cv::Mat kernel(kernel_size, kernel_size, CV_32F); 

            int half = kernel_size/2;
            // Iterate through each cell
            for(int i=-half; i<=half; i++){
                for(int j=-half; j<=half; j++){
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
                    const bool in_grid = (i0 >= 0.0f) && (i0 <= (float)(IMAX-2)) && (j0 >= 0.0f) && (j0 <= (float)(JMAX-2));
                    if(in_grid) confidence_values[i*JMAX+j] = bilinear_interpolation(old_conf_map, i0, j0, 0);
                }
            }

            // Apply Gaussian decay kernel to occupancy_data
            cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // set parameters
            float sig, C, beta_up, beta_dn;
            const float thresh = 4.0f;
            bool front_flag = true;
            
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    
                    const float r2 = polar_coordinates_r2[i*JMAX+j];
                    const float th = polar_coordinates_th[i*JMAX+j];
                    const bool range_flag = r2 > 1.44f;
                    const bool angle_flag = std::abs(ang_diff(yaw, th)) > 0.6f;
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
    
            bool init_flag = false;
            int scan_counter = 0;
            int registration_counter = 0;
        
            geometry_msgs::msg::PoseStamped odom_msg;
            sensor_msgs::msg::PointCloud2 cloud_msg;
            sensor_msgs::msg::PointCloud2 global_cloud_msg;
            nav_msgs::msg::OccupancyGrid map_msg;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr robot_pose_sub_;
            
            rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr odom_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_cloud_pub_;
            rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

            rclcpp::TimerBase::SharedPtr map_timer_;

            pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud_;
            pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud_;
            
            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

            Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f R_init = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f R_cloud = Eigen::Matrix3f::Identity();
            Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f r_map{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f r_init{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f r_cloud{0.0f, 0.0f, 0.0f};
            float yaw = 0.0f;

            std::chrono::steady_clock::time_point t, t_start;
            float dt = 1.0e10f;

            // Generate gaussian kernel for convolution later
            const cv::Mat gauss_kernel = gaussian_kernel(5, 2.0);
            
            int8_t confidence_values[IMAX*JMAX];
            int8_t old_conf[IMAX*JMAX];
            float polar_coordinates_r2[IMAX*JMAX];
            float polar_coordinates_th[IMAX*JMAX];
            cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);

}; 
