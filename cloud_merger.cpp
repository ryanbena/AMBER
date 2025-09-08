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

const double trimX = 0.0;
const double trimY = 0.0;
const double trimZ = 0.225;
bool initialized = false;

const float minx = 0.75;
const float miny = 0.18;
const float minZ = 0.1;
const float maxZ = 1.5;
const float maxXY = 2.0;
const float grid_length = 60;
const float resolution = (2.0 * maxXY) / grid_length;

//Cropping variables
const float range_crop = maxXY-resolution;

class CloudMerger : public rclcpp::Node
{
    public:
        CloudMerger()
        : Node("cloud_merger"){
            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10, std::bind(&CloudMerger::lidar_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/utlidar/cloud_deskewed", 10, std::bind(&CloudMerger::combined_callback, this, std::placeholders::_1));
            robot_pose_sub_ = this->create_subscription<unitree_go::msg::SportModeState>(
                "sportmodestate", 10,std::bind(&CloudMerger::pose_callback, this, std::placeholders::_1));
            livox_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("livox_comb", 10);
            unmasked_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("unmasked_combined2", 10);
            raw_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map_convert2", 10);
            binary_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid2", 10);

            combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

            // --- Static transform broadcaster ---
            static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

            geometry_msgs::msg::TransformStamped t;
            t.header.stamp = this->now();
            t.header.frame_id = "odom";       // parent frame
            t.child_frame_id = "livox_frame"; // sensor frame
            t.transform.translation.x = 0.0;  // LiDAR offset in robot coordinates
            t.transform.translation.y = 0.0;
            t.transform.translation.z = 0.0;
            t.transform.rotation.x = 0.0;
            t.transform.rotation.y = 0.0;
            t.transform.rotation.z = 0.0;
            t.transform.rotation.w =1.0;

            static_broadcaster_->sendTransform(t);

        }
    private:
        void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            //Start timer
            auto start = std::chrono::steady_clock::now();
            dt = start-t;
            t=start;

            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZI>);
            removeGroundPlane(combined_cloud_, cloud_no_ground);

            pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *odom_cloud);

            *odom_cloud += *cloud_no_ground;

            // Get pose values
            float rx = latest_pose_.position[0];
            float ry = latest_pose_.position[1];

            // Create Occupancy Grid object
            nav_msgs::msg::OccupancyGrid raw_grid;
            raw_grid.data.assign(grid_length*grid_length, 0); //Initialize blank raw grid map
            cv::Mat raw_map = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            
            for (const auto& pt : odom_cloud->points)
            {
                // center points
                float xi = pt.x - rx;
                float yi = pt.y - ry;
                float zi = pt.z;

                if(zi>minZ && zi<maxZ &&
                    std::abs(xi) <= range_crop && std::abs(yi) <= range_crop){
                        int i = static_cast<int>(std::round((pt.y -ry+maxXY)/resolution));
                        int j = static_cast<int>(std::round((pt.x -rx+maxXY)/resolution));
                        if (i>=0 && i<grid_length && j>=0 && j<grid_length){
                            int idx = i*grid_length + j; 
                            raw_grid.data[idx] = 127.0;
                            raw_map.at<float>(i,j) = 1.0f;
                        }
                }               
            }

            // BUILD MAP HERE
            cv::Mat Confidence_values_conv = Filtered_Occupancy_Convolution(raw_map, old_conf);
            nav_msgs::msg::OccupancyGrid binary_map = thresholding_std(Confidence_values_conv, old_conf);
            old_conf = Confidence_values_conv;

            //Publish
            sensor_msgs::msg::PointCloud2 unmasked_out;
            pcl::toROSMsg(*odom_cloud, unmasked_out);
            unmasked_out.header = msg->header;
            unmasked_pub_ -> publish(unmasked_out);

            //Publish Occup Map
            raw_grid.header = msg->header;
            raw_grid.info.width  = grid_length;
            raw_grid.info.height = grid_length;
            raw_grid.info.resolution = resolution;
            raw_grid.info.origin.position.x = -maxXY + rx;
            raw_grid.info.origin.position.y = -maxXY + ry;
            raw_grid.info.origin.position.z = 0.0;
            raw_grid.info.origin.orientation.w = 1.0;

            raw_grid_pub_->publish(raw_grid);

            // Publish binary map
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
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *cloud);

            // Angles are negative of actual desired direction
            const float pitch = 0.0*M_PI/180.0;
            const float roll = 0.0*M_PI/180.0;

            // Get curr IMU reading
            double x_pos = latest_pose_.position[0]+trimX;
            double y_pos = latest_pose_.position[1]+trimY;
            double z_pos = latest_pose_.position[2]+trimZ;
            float roll_angle = latest_pose_.imu_state.rpy[0]+roll;   
            float pitch_angle = latest_pose_.imu_state.rpy[1]+pitch;   
            float yaw_angle = latest_pose_.imu_state.rpy[2];    

            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZI>);

            //ELipse masking
            for (const auto& pt : cloud->points)
            {
                double ellipse_norm = std::pow(pt.x / minx, 2.0) + std::pow(pt.y / miny, 2.0);
                if (ellipse_norm > 1.0){
                    filtered->points.push_back(pt);}
            }
            filtered->width = filtered->points.size();
            filtered->height = 1;

            // // incoming lidar moves by current x,y, yaw
            Eigen::Matrix3f R_yaw;
            Eigen::Matrix3f R_pitch;
            Eigen::Matrix3f R_roll;

            Eigen::Matrix3f R_flip; //z-axis flip & 180 yaw
            R_flip << -1.0f, 0.0f, 0.0f,
                0.0f,  1.0f, 0.0f,
                0.0f,  0.0f, -1.0f;  

            R_roll = Eigen::AngleAxisf(roll_angle, Eigen::Vector3f::UnitX());
            R_pitch = Eigen::AngleAxisf(pitch_angle, Eigen::Vector3f::UnitY());
            R_yaw = Eigen::AngleAxisf(yaw_angle, Eigen::Vector3f::UnitZ());

            // Create affine transformation
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.linear() = R_roll*R_pitch*R_yaw*R_flip;  //rotation
            transform.translation() << x_pos, y_pos, z_pos;
            pcl::transformPointCloud(*filtered, *filtered, transform);

            *combined_cloud_ += *filtered;
        } 

        void pose_callback(const unitree_go::msg::SportModeState::SharedPtr msg){
            latest_pose_ = *msg;
        }

        //  CREATE GAUSSIAN KERNEL
        cv::Mat gaussian_kernel(int kernel_size, float sigma, bool normalize){
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
            // Ignoring normalize step until necessary
            return kernel;
        }

        //  BUFFERED CONVOLUTION
        cv::Mat Filtered_Occupancy_Convolution(
            const cv::Mat& occupancy_data,
            const cv::Mat& old_conf_map){

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

            cv::Mat confidence_values = cv::Mat::zeros(grid_length, grid_length, CV_32F);

            for (int i = 0; i < grid_length; i++) {
                int inew = i - di;
                if (inew > -1 && inew < grid_length) {
                    for (int j = 0; j < grid_length; j++) {
                        int jnew = j - dj;
                        if (jnew > -1 && jnew < grid_length) {
                            confidence_values.at<float>(inew, jnew) = old_conf_map.at<float>(i, j);
                        }
                    }
                }
            }

            // Apply Gaussian decay kernel to occupancy_data
            cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            // set parameters
            const float C_plus = 1.0;
            const float C_minus = 0.0;
            const float Th = 1.0;
            const float beta1 = 4.0;
            const float beta2 = 1.0;
            float dt_sec = dt.count();  
            const float sig2 = 1.0-std::exp(-beta2*dt_sec);

            for (int r=0; r<grid_length; r++){
                for(int c=0; c<grid_length; c++){
                    float val_binary = buffered_binary.at<float>(r,c);
                    float conf = confidence_values.at<float>(r,c);
                    if( val_binary > Th){

                        float sig1 = 1.0 - std::exp(-beta1*val_binary*dt_sec);
                        confidence_values.at<float>(r,c) = (1.0-sig1)*conf + sig1*C_plus;
                    }
                    else{
                        confidence_values.at<float>(r,c) = (1.0-sig2)*conf + sig2*C_minus;
                    }
                }
            }
            return confidence_values;
        }

        // HYSTERISIS THRESHOLDING
        nav_msgs::msg::OccupancyGrid thresholding_std(
            const cv::Mat conf_map,
            const cv::Mat old_conf,
            const float T_hi = 0.9, const float T_lo = 0.5) {

            std::vector<int8_t> binary_map(grid_length*grid_length, 0);

            for(int y=0; y<grid_length; y++){
                for(int x=0; x<grid_length; x++){
                    int idx = y*grid_length+x;
                    double val = conf_map.at<float>(y,x);
                    double old_val = conf_map.at<float>(y,x);

                    bool strong = val>=T_hi;
                    bool weak = val >= T_lo && val <T_hi;
                    bool grow_mask = old_val >= T_lo && weak;

                    binary_map[idx] = (strong||grow_mask) ? 100:0;
                }
            }
 
            nav_msgs::msg::OccupancyGrid grid_msg;
            grid_msg.data = binary_map;
            return grid_msg;
        }
        void removeGroundPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_removed_cloud)  {
            pcl::PointIndices::Ptr ground_candidate_indices(new pcl::PointIndices);
            pcl::PointCloud<pcl::PointXYZI>::Ptr ground_candidates(new pcl::PointCloud<pcl::PointXYZI>);
            
            // Get pose values
            float rx = latest_pose_.position[0];
            float ry = latest_pose_.position[1];

            for (size_t i = 0; i < input_cloud->points.size(); ++i) {
                const auto& pt = input_cloud->points[i];
                // center points
                float xi = pt.x - rx;
                float yi = pt.y - ry;
                float zi = pt.z;

                if (pt.z < 0.05){// &&  std::abs(xi) <= range_crop+0.5 && std::abs(yi) <= range_crop+0.5 ) {
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
            seg.setDistanceThreshold(0.05);  // Adjust this threshold based on sensor noise
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
    
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr robot_pose_sub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr unmasked_pub_;
            rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr raw_grid_pub_;
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
            const cv::Mat gauss_kernel  = gaussian_kernel(9, 2.0,false);
            
            cv::Mat old_conf = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            cv::Mat buffered_binary = cv::Mat::zeros(grid_length, grid_length, CV_32F);
            double roll_ground = 0.0;
            double pitch_ground = 0.0;
};
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CloudMerger>());
    rclcpp::shutdown();
    return 0;
}             
