#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include <unistd.h>
#include <cmath>

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

// Create a sport_request class for sport command request
class sport_request : public rclcpp::Node {

    public:

        sport_request() : Node("req_sender"){

            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            
            poisson_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("safety_command_topic", 1, std::bind(&sport_request::poisson_topic_callback, this, std::placeholders::_1), options1);
            state_subscription_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1, std::bind(&sport_request::state_topic_callback, this, std::placeholders::_1), options2);

            req_puber = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 1); // the req_puber is set to subscribe "/api/sport/request" topic with dt   
            pose_puber = this->create_publisher<geometry_msgs::msg::PoseStamped>("MacLane_pose_internal", 1);
            
            sport_req.SwitchGait(req, 0);
            req_puber->publish(req);
            sleep(1);
            sport_req.StandUp(req);
            req_puber->publish(req);
            sleep(1);
            sport_req.BalanceStand(req);
            req_puber->publish(req);
            sleep(1);
            //sport_req.ContinuousGait(req, false);
            //req_puber->publish(req);
            //sleep(1);
        };

    private:

        void state_topic_callback(unitree_go::msg::SportModeState::SharedPtr msg){

            geometry_msgs::msg::PoseStamped internal_pose;
            internal_pose.pose.position.x = msg->position[0];
            internal_pose.pose.position.y = msg->position[1];
            internal_pose.pose.position.z = msg->position[2];
            internal_pose.pose.orientation.w = msg->imu_state.quaternion[0];
            internal_pose.pose.orientation.x = msg->imu_state.quaternion[1];
            internal_pose.pose.orientation.y = msg->imu_state.quaternion[2];
            internal_pose.pose.orientation.z = msg->imu_state.quaternion[3];
            pose_puber->publish(internal_pose);

        };

        void poisson_topic_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg){
                
            rx = msg->data[0];
            ry = msg->data[1];
            yaw = msg->data[2];
            vx = msg->data[3];
            vy = msg->data[4];
            vyaw = msg->data[5];
            vxs = msg->data[6];
            vys = msg->data[7];
            vyaws = msg->data[8];
            h = msg->data[9];

            if(!stop_flag){

                t += dt;

                if(t > 0.0f){

                    // Convert Velocity Commands to Body Frame
                    float vxb = cosf(yaw)*vx+ sinf(yaw)*vy;
                    float vyb = -sinf(yaw)*vx + cosf(yaw)*vy;

                    // Send Command
                    vxb = fminf(fmaxf(vxb, -3.0f), 3.0f);
                    vyb = fminf(fmaxf(vyb, -2.0f), 2.0f);
                    vyaw = fminf(fmaxf(vyaw, -2.0f), 2.0f);
                    //if((vxb*vxb+vyb*vyb > 0.01f) || (vyaw*vyaw > 0.1f)){
                        sport_req.Move(req, vxb, vyb, vyaw);
                        req_puber->publish(req);
                    //}
                    //logData();

                    if(t > t_stop){

                        sport_req.StopMove(req);
                        req_puber->publish(req);
                        sleep(1);

                        sport_req.StandDown(req);
                        req_puber->publish(req);

                        writeDataToFile();
                        stop_flag = true;
                        //rclcpp::shutdown();

                    }
                }
            }

        };

        void logData(){

            t_buffer[iter] = t;
            rx_buffer[iter] = rx;
            ry_buffer[iter] = ry;
            yaw_buffer[iter] = yaw;
            vx_buffer[iter] = vx;
            vy_buffer[iter] = vy;
            vyaw_buffer[iter] = vyaw;
            vxs_buffer[iter] = vxs;
            vys_buffer[iter] = vys;
            vyaws_buffer[iter] = vyaws;
            h_buffer[iter] = h;
            iter++;

            if(h < 0.0f) std::cout << "Safety Violation!: " << h << std::endl;
        
        };

        void writeDataToFile(void){
            
            // Log Data Perminantly
            time_t rawtime;
            struct tm * timeinfo;
            char filename[80];
            
            time(&rawtime);
            timeinfo = localtime(&rawtime);
            sprintf(filename, "data_log/poisson_go2_data_%04d%02d%02d_%02d%02d.csv", timeinfo->tm_year+1900, timeinfo->tm_mon+1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min);

            std::ofstream outFile(filename);
            if(outFile.is_open()){
                for(int i=0; i<BUFF; i++){
                    outFile << t_buffer[i] << ", " << rx_buffer[i] << ", " << ry_buffer[i] << ", " << yaw_buffer[i] << ", " << vx_buffer[i] << ", " << vy_buffer[i] << ", " << vyaw_buffer[i] << ", " << vxs_buffer[i] << ", " << vys_buffer[i] << ", " << vyaws_buffer[i] << ", " << h_buffer[i] << std::endl;
                }
                outFile.close();
                std::cout << "Data logged at: " << filename << std::endl;
            } 
            else{
                std::cerr << "Error: Could not open file " << filename << " for writing.\n";
            }

            // Log Data Temporarily (For Plotting Script)
            char filename_temp[80];
            sprintf(filename_temp, "data_log/poisson_data_latest.csv");
            std::ofstream outFile_temp(filename_temp);
            if(outFile_temp.is_open()){
                for(int i=0; i<BUFF; i++){
                    outFile_temp << t_buffer[i] << ", " << rx_buffer[i] << ", " << ry_buffer[i] << ", " << yaw_buffer[i] << ", " << vx_buffer[i] << ", " << vy_buffer[i] << ", " << vyaw_buffer[i] << ", " << vxs_buffer[i] << ", " << vys_buffer[i] << ", " << vyaws_buffer[i] << ", " << h_buffer[i] << std::endl;
                }
                outFile_temp.close();
            } 
            else{
                std::cerr << "Error: Could not open file " << filename_temp << " for writing.\n";
            }

        };

        rclcpp::TimerBase::SharedPtr timer_; // ROS2 timer
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber;
        unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
        SportClient sport_req;

        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_puber;
        rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_subscription_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr poisson_subscription_;
        
        // Initialize Timing Variables
        bool stop_flag = false;
        float t = -5.0f; // running time count
        float dt = 0.002f; //control time step
        float t_stop = 300.0f; // experiment duration
        static const int BUFF = 120000; //must be >= t_stop/dt

        int iter = 0; // running iteration count

        // Initialize Message Variables
        float rx = 0.0f;
        float ry = 0.0f;
        float yaw = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;
        float vyaw = 0.0f;
        float vxs = 0.0f;
        float vys = 0.0f;
        float vyaws = 0.0f;
        float h = 0.0f;

        float t_buffer[BUFF];
        float rx_buffer[BUFF];
        float ry_buffer[BUFF];
        float yaw_buffer[BUFF];
        float vx_buffer[BUFF];
        float vy_buffer[BUFF];
        float vyaw_buffer[BUFF];
        float vxs_buffer[BUFF];
        float vys_buffer[BUFF];
        float vyaws_buffer[BUFF];
        float h_buffer[BUFF];

};

int main(int argc, char *argv[]){

    rclcpp::init(argc, argv); // Initialize rclcpp
    rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::Node::SharedPtr relayNode = std::make_shared<sport_request>();
    executor.add_node(relayNode);
    
    try{
        executor.spin(); //Run ROS2 node
        throw("Terminated");
    }
    catch(const char* msg){
        std::cout << msg << std::endl;
        rclcpp::shutdown();
    }

    return 0;

}