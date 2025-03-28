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

#include <unistd.h>
#include <cmath>

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

// Create a soprt_request class for soprt commond request
class soprt_request : public rclcpp::Node {

    public:

        soprt_request() : Node("req_sender"){

            req_puber = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 1); // the req_puber is set to subscribe "/api/sport/request" topic with dt
            timer_ = this->create_wall_timer(std::chrono::milliseconds(int(dt * 1000)), std::bind(&soprt_request::timer_callback, this));
            
            auto poisson_topic_callback = [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) -> void {
                
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
 
            };
            poisson_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("safety_command_topic", 1, poisson_topic_callback);

            sleep(1);
            sport_req.SwitchGait(req, 0);
            req_puber->publish(req);
            sleep(1);
            sport_req.StandUp(req);
            req_puber->publish(req);
            sleep(1);
            sport_req.BalanceStand(req);
            req_puber->publish(req);
            sleep(1);

        };

    private:

        void timer_callback(){

            t += dt;

            if(t > 0.0f){

                // Convert Velocity Commands to Body Frame
                float vxb = cosf(yaw)*vx+ sinf(yaw)*vy;
                float vyb = -sinf(yaw)*vx + cosf(yaw)*vy;

                // Send Command
                vxb = fminf(fmaxf(vxb, -3.0f), 3.0f);
                vyb = fminf(fmaxf(vyb, -2.0f), 2.0f);
                vyaw = fminf(fmaxf(vyaw, -2.0f), 2.0f);
                sport_req.Move(req, vxb, vyb, vyaw);
                req_puber->publish(req);
                
                logData();

                if(t > t_stop){

                    sport_req.StopMove(req);
                    req_puber->publish(req);
                    sleep(1);

                    sport_req.StandDown(req);
                    req_puber->publish(req);

                    writeDataToFile();

                    rclcpp::shutdown();

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

        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr poisson_subscription_;
        
        // Initialize Timing Variables
        float t = -5.0f; // running time count
        float dt = 0.001f; //control time step
        float t_stop = 60.0f; // experiment duration
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
    rclcpp::TimerBase::SharedPtr timer_; // Create a timer callback object to send sport request in time intervals
    rclcpp::spin(std::make_shared<soprt_request>()); //Run ROS2 node
    rclcpp::shutdown();
    return 0;

}