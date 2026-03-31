#pragma once

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
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
            
            // Initialize Map Message
            map_msg.data.resize(IMAX*JMAX);
            map_msg.header.stamp = this->now();
            map_msg.header.frame_id = "odom";
            map_msg.info.width  = IMAX;
            map_msg.info.height = JMAX;
            map_msg.info.resolution = DS;
            map_msg.info.origin.position.x = r(0) - maxX;
            map_msg.info.origin.position.y = r(1) - maxY;
            map_msg.info.origin.position.z = 0.0f;
            map_msg.info.origin.orientation.w = 1.0f;
            map_msg.info.origin.orientation.x = 0.0f;
            map_msg.info.origin.orientation.y = 0.0f;
            map_msg.info.origin.orientation.z = 0.0f;

            // Initialize Grids
            for(int n=0; n<IMAX*JMAX; n++) old_conf[n] = 0;

            // Start Time
            t = std::chrono::steady_clock::now();

            // Create TF Listener
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            // Create Subscribers & Publishers
            lio_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1, std::bind(&CloudMergerNode::lio_callback, this, std::placeholders::_1));
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

        }

    private:

        void lio_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){

            //Start timer
            Timer map_timer(true);
            map_timer.start();

            // Populate Point Cloud with LiDAR Points
            pcl::PointCloud<pcl::PointXYZ>::Ptr lio_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *lio_cloud);

            // Query Current Body -> Odom Transform
            try{

                transform_ = tf_buffer_->lookupTransform("odom", "body", rclcpp::Time(0));
                r(0) = transform_.transform.translation.x;
                r(1) = transform_.transform.translation.y;
                r(2) = transform_.transform.translation.z;
            
                Eigen::Quaternionf q;
                q.w() = transform_.transform.rotation.w;
                q.x() = transform_.transform.rotation.x;
                q.y() = transform_.transform.rotation.y;
                q.z() = transform_.transform.rotation.z;
                R = q.toRotationMatrix();

            }
            catch(tf2::TransformException &ex){
            
                std::cout << "Odom Not Available" << std::endl;
            
            }

            // Create Local Occupancy Grid object
            cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
            Eigen::Vector3f pt_odom, pt_body;
            const float z_body = 0.27f;
            for(const auto& pt : lio_cloud->points){
                pt_odom << pt.x, pt.y, pt.z;
                pt_body = R.transpose() * (pt_odom - r);
                const float z = pt_body(2) + z_body;
                const bool in_plane = (z > minZ) && (z < maxZ);
                if(!in_plane) continue;
                const float ic = (pt.y - r(1)) / DS + (float)(IMAX/2);
                const float jc = (pt.x - r(0)) / DS + (float)(JMAX/2);
                const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX-1)) && (jc > 0.0f) && (jc < (float)(JMAX-1));
                if(!in_grid) continue;
                raw_map.at<float>((int)std::round(ic),(int)std::round(jc)) += 1.0f;             
            }
            
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
            map_msg.info.origin.position.x = r(0) - maxX;
            map_msg.info.origin.position.y = r(1) - maxY;
            map_pub_->publish(map_msg);
            
            map_timer.time("Occ Map Solve Time: ");

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
            
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    
                    float val_binary = buffered_binary.at<float>(i,j);
                    float conf = (float)confidence_values[i*JMAX+j] / 127.0f;
                    if(val_binary > thresh){
                        beta_up = 1.0f;
                        sig = 1.0f - std::exp(-beta_up*val_binary*dt);
                        C = 1.0f;
                    }
                    else{
                        beta_dn = 4.0f;
                        sig = 1.0f - std::exp(-beta_dn*dt);
                        C = 0.0f;
                    }
                    conf *= 1.0f - sig;
                    conf += sig * C;
                    confidence_values[i*JMAX+j] = (int8_t)std::round(127.0f*conf);

                }
            }

        }
        
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lio_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
        nav_msgs::msg::OccupancyGrid map_msg;

        geometry_msgs::msg::TransformStamped transform_;
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        
        Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f r_map{0.0f, 0.0f, 0.0f};
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); 

        std::chrono::steady_clock::time_point t;
        float dt = 1.0e10f;

        // Generate gaussian kernel for convolution later
        const cv::Mat gauss_kernel = gaussian_kernel(5, 2.0);
        
        int8_t confidence_values[IMAX*JMAX];
        int8_t old_conf[IMAX*JMAX];
        cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);

}; 
