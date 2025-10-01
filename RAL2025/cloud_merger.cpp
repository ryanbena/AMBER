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

// Ground detection
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <cmath>
#include <Eigen/Dense>

#include "utils.h"

bool initialized = false;

const float minx = 0.50f;
const float miny = 0.25f;
const float minZ = 0.08f;
const float maxZ = 0.80f;
const float maxXY = 2.0f;
const int grid_length = 120;
const float resolution = (2.0 * maxXY) / (float)grid_length;

//Cropping variables
const float range_crop = maxXY-resolution;

class CloudMerger : public rclcpp::Node
{
    public:
        CloudMerger()
        : Node("cloud_merger"){

            binary_map.data.resize(grid_length*grid_length);

            for(int i=0; i<grid_length; i++){
                for(int j=0; j<grid_length; j++){
                    const float x = (float)(j-grid_length/2)*resolution;
                    const float y = (float)(i-grid_length/2)*resolution;
                    polar_coordinates_r2[i*grid_length+j] = x*x+y*y;
                    polar_coordinates_th[i*grid_length+j] = atan2f(y,x);
                    old_conf[i*grid_length+j] = 0;
                }
            }

            t = std::chrono::steady_clock::now();

            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 1, std::bind(&CloudMerger::lidar_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/utlidar/cloud_deskewed", 1, std::bind(&CloudMerger::combined_callback, this, std::placeholders::_1));
            robot_pose_sub_ = this->create_subscription<unitree_go::msg::SportModeState>(
                "sportmodestate", 1,std::bind(&CloudMerger::pose_callback, this, std::placeholders::_1));
            livox_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("livox_comb", 1);
            unmasked_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("unmasked_combined", 1);
            binary_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

            combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

            // --- Static transform broadcaster ---
            static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp = this->now();
            tf.header.frame_id = "odom";       // parent frame
            tf.child_frame_id = "livox_frame"; // sensor frame
            tf.transform.translation.x = 0.0;  // LiDAR offset in robot coordinates
            tf.transform.translation.y = 0.0;
            tf.transform.translation.z = 0.0;
            tf.transform.rotation.x = 0.0;
            tf.transform.rotation.y = 0.0;
            tf.transform.rotation.z = 0.0;
            tf.transform.rotation.w =1.0;
            static_broadcaster_->sendTransform(tf);

        }
    private:
        void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            //Start timer
            auto start = std::chrono::steady_clock::now();
            dt = start-t;
            t = std::chrono::steady_clock::now();

            //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZI>);
            //removeGroundPlane(combined_cloud_, cloud_no_ground);

            pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *odom_cloud);

            //*odom_cloud += *cloud_no_ground;
            *odom_cloud += *combined_cloud_;

            // Get pose values
            float rx = latest_pose_.position[0];
            float ry = latest_pose_.position[1];

            // Create Occupancy Grid object
            cv::Mat raw_map = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            
            for (const auto& pt : odom_cloud->points){
                // center points
                float xi = pt.x - rx;
                float yi = pt.y - ry;
                float zi = pt.z;

                if(zi>minZ && zi<maxZ && std::abs(xi) <= range_crop && std::abs(yi) <= range_crop){
                    int i = (int)std::round((pt.y-ry+maxXY)/resolution);
                    int j = (int)std::round((pt.x-rx+maxXY)/resolution);
                    if (i>=0 && i<grid_length && j>=0 && j<grid_length) raw_map.at<float>(i,j) = 1.0f;
                }               
            }

            // BUILD MAP HERE
            for(int n=0; n<grid_length*grid_length; n++) confidence_values[n] = 0;
            Filtered_Occupancy_Convolution(confidence_values, raw_map, old_conf);
            memcpy(old_conf, confidence_values, grid_length*grid_length*sizeof(int8_t));

            //Publish
            sensor_msgs::msg::PointCloud2 unmasked_out;
            pcl::toROSMsg(*odom_cloud, unmasked_out);
            unmasked_out.header = msg->header;
            unmasked_pub_ -> publish(unmasked_out);

            // Publish binary map
            for(int n=0; n<grid_length*grid_length; n++) binary_map.data[n] = confidence_values[n];
            binary_map.header = msg->header;
            binary_map.info.width  = grid_length;
            binary_map.info.height = grid_length;
            binary_map.info.resolution = resolution;
            binary_map.info.origin.position.x = -maxXY + rx;
            binary_map.info.origin.position.y = -maxXY + ry;
            binary_map.info.origin.position.z = 0.0;
            binary_map.info.origin.orientation.w = 1.0;

            binary_grid_pub_->publish(binary_map);

             // End timer right after publishing
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << std::endl;
            
            combined_cloud_->clear();
        }

        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Populate Point Cloud with LiDAR Points
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *cloud);

            // Translate from LiDAR Origin to Body Origin (In Livox Upside-down Coordinates)
            const float x_trim = 0.05f;
            const float y_trim = 0.0f;
            const float z_trim = -0.18f;
            Eigen::Affine3f lidar2body = Eigen::Affine3f::Identity();
            lidar2body.translation() << x_trim, y_trim, z_trim;
            pcl::transformPointCloud(*cloud, *cloud, lidar2body);

            // Mask Robot Body with Hyper-Ellipse
            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& pt : cloud->points){
                float ellipse_norm = std::pow(pt.x/minx,8.0f) + std::pow(pt.y/miny,8.0f);
                if(ellipse_norm > 1.0f) filtered->points.push_back(pt);
            }
            filtered->width = filtered->points.size();
            filtered->height = 1;

            // Rotate from LiDAR Frame to Unrotated Frame
            Eigen::Matrix3f R_flip, R_yaw, R_pitch, R_roll, R_body;
            R_flip << -1.0f, 0.0f, 0.0f, //z-axis flip & 180 degree yaw
                0.0f,  1.0f, 0.0f,
                0.0f,  0.0f, -1.0f; 
            const float roll_trim = 0.0f*M_PI/180.0f;
            const float pitch_trim = 0.0f*M_PI/180.0f;
            const float roll_angle = latest_pose_.imu_state.rpy[0] + roll_trim;   
            const float pitch_angle = latest_pose_.imu_state.rpy[1] + pitch_trim;   
            const float yaw_angle = latest_pose_.imu_state.rpy[2];
            R_roll = Eigen::AngleAxisf(roll_angle, Eigen::Vector3f::UnitX());
            R_pitch = Eigen::AngleAxisf(pitch_angle, Eigen::Vector3f::UnitY());
            R_yaw = Eigen::AngleAxisf(yaw_angle, Eigen::Vector3f::UnitZ());
            R_body = R_yaw*R_pitch*R_roll;
            Eigen::Affine3f rotated2fixed = Eigen::Affine3f::Identity();
            rotated2fixed.linear() = R_body * R_flip;  //rotation
            pcl::transformPointCloud(*filtered, *filtered, rotated2fixed);

            // Move from Body-Centered Frame to Inertial Frame
            const float x_pos = latest_pose_.position[0];
            const float y_pos = latest_pose_.position[1];
            const float z_pos = latest_pose_.position[2];
            Eigen::Affine3f body2inertial = Eigen::Affine3f::Identity();
            body2inertial.translation() << x_pos, y_pos, z_pos;
            pcl::transformPointCloud(*filtered, *filtered, body2inertial);

            // Add Points into Combined Cloud
            *combined_cloud_ += *filtered;

        } 

        void pose_callback(const unitree_go::msg::SportModeState::SharedPtr msg){

            latest_pose_ = *msg;

        }

        //  CREATE GAUSSIAN KERNEL
        cv::Mat gaussian_kernel(int kernel_size, float sigma){
            // Create kernel_sizexkernel_size array of floats
            cv::Mat kernel(kernel_size, kernel_size, CV_32F); 

            int half = kernel_size/2;
            // Iterate through each cell
            for(int i=-half; i<=half; i++){
                for (int j=-half; j<=half; j++){
                    float val = std::exp(-(i*i+j*j)/(2.0*sigma*sigma));
                    kernel.at<float>(i+half, j+half) = val;
                }
            }

            return kernel;
        }

        //  BUFFERED CONVOLUTION
        void Filtered_Occupancy_Convolution(int8_t *confidence_values, const cv::Mat& occupancy_data, const int8_t *old_conf_map){

            float rx = latest_pose_.position[0];
            float ry = latest_pose_.position[1];
            // Shift Confidence Values Based on Egomotion
            const float drx = rx-rx_map;
            const float dry = ry-ry_map;
            rx_map = rx;
            ry_map = ry;
            int di = (int)std::round(dry/resolution);
            int dj = (int)std::round(drx/resolution);
            // cv::Mat tf_mat = (cv::Mat_<float>(2,3) << 1,0,-dj, 0,1, -di);
            // cv::warpAffine(old_conf_map, confidence_values, tf_mat,cv::Size(grid_length, grid_length), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0.0f);

            //cv::Mat confidence_values = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            for(int i = 0; i < grid_length; i++) {
                int inew = i - di;
                if (inew > -1 && inew < grid_length) {
                    for (int j = 0; j < grid_length; j++) {
                        int jnew = j - dj;
                        if (jnew > -1 && jnew < grid_length) {
                            confidence_values[inew*grid_length+jnew] = old_conf_map[i*grid_length+j];
                            //confidence_values.at<float>(inew, jnew) = old_conf_map.at<float>(i, j);
                        }
                    }
                }
            }

            // Apply Gaussian decay kernel to occupancy_data
            cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // set parameters
            float sig, C, beta_up, beta_dn;
            const float Th = 1.0f;
            bool front_flag = true;
            float dt_sec = dt.count();
            
            
            for(int i=0; i<grid_length; i++){
                for(int j=0; j<grid_length; j++){
                    
                    const float r2 = polar_coordinates_r2[i*grid_length+j];
                    const float th = polar_coordinates_th[i*grid_length+j];
                    const bool range_flag = r2 > 1.44f;
                    const bool angle_flag = fabsf(ang_diff(latest_pose_.imu_state.rpy[2], th)) > 0.6f;
                    if(range_flag || angle_flag) front_flag = false;
                    else front_flag = true;
                    
                    float val_binary = buffered_binary.at<float>(i,j);
                    float conf = (float)confidence_values[i*grid_length+j] / 127.0f;
                    if(val_binary > Th){
                        if(front_flag) beta_up = 4.0f; //Go2 Front LiDAR only
                        else beta_up = 1.0f; // Livox Mid360
                        sig = 1.0f - std::exp(-beta_up*val_binary*dt_sec);
                        C = 1.0f;
                    }
                    else{
                        if(front_flag) beta_dn = 4.0f;
                        else beta_dn = 2.0f;
                        sig = 1.0f - std::exp(-beta_dn*dt_sec);
                        C = 0.0f;
                    }
                    conf *= 1.0f - sig;
                    conf += sig * C;
                    confidence_values[i*grid_length+j] = (int8_t)(127.0f * conf);

                }
            }

        }

        void removeGroundPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_removed_cloud)  {
            pcl::PointIndices::Ptr ground_candidate_indices(new pcl::PointIndices);
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_candidates(new pcl::PointCloud<pcl::PointXYZI>);
            
            for (size_t i = 0; i < input_cloud->points.size(); ++i) {
                const auto& pt = input_cloud->points[i];
                if (pt.z < minZ){// &&  std::abs(xi) <= range_crop+0.5 && std::abs(yi) <= range_crop+0.5 ) {
                    ground_candidates->points.push_back(pt);
                    ground_candidate_indices->indices.push_back(i);
                }
            }
            ground_candidates->width = ground_candidates->points.size();
            ground_candidates->height = 1;

            if (ground_candidates->empty()) {
                std::cout << "No ground candidates found under Z threshold." << std::endl;
                *ground_removed_cloud = *input_cloud;  // Return unmodified cloud
                return;
            }

            sensor_msgs::msg::PointCloud2 livox_raw;
            pcl::toROSMsg(*ground_candidates, livox_raw);

            livox_raw.header.stamp = this->now();
            livox_raw.header.frame_id = "odom"; 
            livox_pub_ -> publish(livox_raw);

            // Create the segmentation object
            pcl::SACSegmentation<pcl::PointXYZI> seg;            
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
            seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));        // Prefer planes perpendicular to Z (i.e. horizontal)
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(minZ);  // Adjust this threshold based on sensor noise
            seg.setInputCloud(ground_candidates);
            seg.segment(*inliers, *coefficients);


            pcl::PointIndices::Ptr full_cloud_inliers(new pcl::PointIndices);
            for (int idx : inliers->indices) {
                full_cloud_inliers->indices.push_back(ground_candidate_indices->indices[idx]);
            }

            // Extract non-ground (outlier) points
            pcl::ExtractIndices<pcl::PointXYZI> extract;
            extract.setInputCloud(input_cloud);
            extract.setIndices(full_cloud_inliers);
            extract.setNegative(true);  // True = remove inliers (i.e., remove the plane)
            extract.filter(*ground_removed_cloud);

        }
    
            nav_msgs::msg::OccupancyGrid binary_map;

            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr robot_pose_sub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr unmasked_pub_;
            rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr binary_grid_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr livox_pub_;

            std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

            pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud_;
            unitree_go::msg::SportModeState latest_pose_;
            float rx_map = 0.0;
            float ry_map = 0.0;

            std::chrono::steady_clock::time_point t;
            std::chrono::duration<double> dt;
            // Generate gaussian kernel for convolution later
            const cv::Mat gauss_kernel  = gaussian_kernel(9, 2.0);
            
            int8_t confidence_values[grid_length*grid_length];
            int8_t old_conf[grid_length*grid_length];
            float polar_coordinates_r2[grid_length*grid_length];
            float polar_coordinates_th[grid_length*grid_length];
            cv::Mat buffered_binary = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            double roll_ground = 0.0;
            double pitch_ground = 0.0;
};

int main(int argc, char *argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CloudMerger>());
    rclcpp::shutdown();
    return 0;
}             
