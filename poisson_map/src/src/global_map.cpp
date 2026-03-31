#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include "unitree_go/msg/sport_mode_state.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include <cmath>
#include <Eigen/Dense>

#include "utils.h"

const float MOS = 1.35f;
const float minX = 0.370f * MOS; // Must be >= 0.370
const float minY = 0.185f * MOS; // Must be >= 0.185

class CloudMergerNode : public rclcpp::Node{
    
    public:
        
        CloudMergerNode() : Node("cloud_merger"){

            // Delay Startup
            const int delay_sec = 10;
            int delay_count = 0;
            while(delay_count<delay_sec){
                std::cout << "Start Mapping in " << delay_sec-delay_count << " seconds" << std::endl;
                delay_count++;
                sleep(1);
            }

            global_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

            // Create Subscribers
            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/livox/lidar", 1, std::bind(&CloudMergerNode::livox_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/utlidar/cloud_deskewed", 1, std::bind(&CloudMergerNode::utlidar_callback, this, std::placeholders::_1));
            robot_pose_sub_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1, std::bind(&CloudMergerNode::pose_callback, this, std::placeholders::_1));
            
            // Create Publishers
            global_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("global_cloud", 1);
            
            // Create Timers
            map_timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&CloudMergerNode::map_callback, this));

        }

    private:

        void map_callback(void){
      
            // Apply Voxel Filter
            if(!map_save_flag){
                pcl::VoxelGrid<pcl::PointXYZ> voxel;
                voxel.setLeafSize(0.1f, 0.1f, 0.1f);
                voxel.setInputCloud(global_cloud_);
                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                voxel.filter(*filtered_cloud);
                global_cloud_ = filtered_cloud;
                std::cout << "Map Size: " << global_cloud_->size() << " points" << std::endl;
            }

            sensor_msgs::msg::PointCloud2 global_cloud_msg;
            pcl::toROSMsg(*global_cloud_, global_cloud_msg);
            global_cloud_msg.header.stamp = this->now();
            global_cloud_msg.header.frame_id = "odom";
            global_cloud_pub_->publish(global_cloud_msg);

            // Save Once Filtered Map is Big Enough
            if(global_cloud_->size() > 50000 && !map_save_flag){
                map_save_flag = true;
                pcl::io::savePCDFileBinary("map.pcd", *global_cloud_);
                std::cout << "==========MAP SAVED==========" << std::endl;
            }

        }
       
        void utlidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
               
            if(init_flag){

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
                *global_cloud_ += *masked_cloud;
            
            }
            
        }

        void livox_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            if(init_flag){

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
                *global_cloud_ += *masked_cloud;

            }

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
            
        }

            bool init_flag = false;
            bool map_save_flag = false;
        
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr robot_pose_sub_;

            pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_cloud_pub_;

            rclcpp::TimerBase::SharedPtr map_timer_;
            
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

            Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f R_init = Eigen::Matrix3f::Identity();
            Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
            Eigen::Vector3f r_init{0.0f, 0.0f, 0.0f};

            std::chrono::steady_clock::time_point t_start;

};


int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto mappingNode = std::make_shared<CloudMergerNode>();
    executor.add_node(mappingNode);
    
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
