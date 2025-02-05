#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <arpa/inet.h>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <unistd.h>
#include <cmath>
#include <cstring>

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

#define BUF 120000 // Maxout after 120 s @ 1000 Hz

using std::placeholders::_1;

// Create a soprt_request class for soprt commond request
class soprt_request : public rclcpp::Node {

public:

    soprt_request() : Node("req_sender"){

        complete = false;
        h_flag = false;
        t = -5; // Runing time count

        pose_state_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/G1/pose", 10, std::bind(&soprt_request::optitrack_state_callback, this, _1));
        timer_ = this->create_wall_timer(std::chrono::milliseconds(int(dt * 1000)), std::bind(&soprt_request::timer_callback, this));

        tgrid = t;
        hgrid = std_msgs::msg::Float64MultiArray();
        h0grid = std_msgs::msg::Float64MultiArray();
        
        auto h_topic_callback = [this](std_msgs::msg::Float64MultiArray::UniquePtr msg) -> void {
            
            dtgrid = t - tgrid;
            tgrid = t;

            h0grid.data = hgrid.data;
            hgrid.data = msg->data;
            
            if(!h_flag){
                h_flag = true;
                std::cout << "First Grid Received!" << std::endl;
            }

        };
        
        h_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("safety_grid_topic", 1, h_topic_callback);
        openG1Socket("192.168.1.223", 12345);

    };

private:

    void timer_callback(){

        t += dt;

        if ((t > 0.0) && (!complete)){

            // Define Refecrence Trajectory
            const double amp = 0.0;
            const double freq = 1.0 / 40.0;
            double w = 2.0 * M_PI * freq;
            double rxd = amp*sin(w*t) + 1.5;
            double ryd = amp*sin(w*t) + 1.5;
            double vxd = amp*w*cos(w*t);
            double vyd = amp*w*cos(w*t);

            // Define Reference Yaw
            const double yawd = M_PI/4.0;
            const double yawd_dot = 0.0;
            
            // Compute Errors
            double rxe = rxd - rx;
            double rye = ryd - ry;
            double yawe = yawd - yaw;
            
            // Proportional Control + Tracking
            const double kp = 0.8;
            double vx = kp * rxe + vxd;
            double vy = kp * rye + vyd;
            double vyaw = kp * yawe + yawd_dot;

            if(h_flag){

                // Fractional Index Corresponding to Current Position
                const double rc[2] = {0.0, 0.0}; // Location of OptiTrack Origin in Grid Frame
                double ir = (double)imax - (ry+rc[1]) / ds;
                double jr = (rx+rc[0]) / ds;

                const double x_eps = 1.0e-4; // Small Perturbation for Numerical Gradients (meters)
                double i_eps = x_eps / ds;

                double ic = fmin(fmax(i_eps, ir), (double)(imax-1)-i_eps); // Saturated Because Numerical Derivatives Shrink Effective Grid Size
                double jc = fmin(fmax(i_eps, jr), (double)(jmax-1)-i_eps);

                // Get Safety Function Value
                h = bilinear_interpolation(&hgrid, ic, jc);
                h0 = bilinear_interpolation(&h0grid, ic, jc);
                if(h < 0.0) std::cout << "Safety Violation!: " << h << std::endl;

                // Compute Time Derivative
                double dhdt_raw = (h - h0) / dtgrid;
                const double wv = 10.0; // Low Pass Filter Cutoff
                double kv = 1.0 - exp(-wv*dtgrid);
                dhdt *= 1.0 - kv;
                dhdt += kv * dhdt_raw;

                // Compute Gradient
                double ip = ic - i_eps;
                double im = ic + i_eps;
                double jp = jc + i_eps;
                double jm = jc - i_eps;
                gradhx = (bilinear_interpolation(&hgrid, ic, jp) - bilinear_interpolation(&hgrid, ic, jm)) / (2.0 * x_eps);
                gradhy = (bilinear_interpolation(&hgrid, ip, jc) - bilinear_interpolation(&hgrid, im, jc)) / (2.0 * x_eps);
                
                // Single Integrator Safety Filter
                const double alpha = 1.0;
                const double issf = 3.0;

                double b = gradhx*gradhx + gradhy*gradhy;
                double a = gradhx*vx + gradhy*vy;
                a += alpha*h;
                a += dhdt;
                a -= 1.0/issf * b;

                const double beta = 3.0; // Smoothing Parameter
                vx -= exp(-beta*fmax(a, 0.0)) * a * gradhx / b;
                vy -= exp(-beta*fmax(a, 0.0)) * a * gradhy / b;
            
            }

            double vxb = cos(yaw)*vx+ sin(yaw)*vy;
            double vyb = -sin(yaw)*vx + cos(yaw)*vy;

            vxb = fmin(fmax(vxb, -0.6), 0.6);
            vyb = fmin(fmax(vyb, -0.4), 0.4);

            moveG1((float)vxb, (float)vyb, (float)vyaw);
            std::cout << "Moving!" << t << std::endl;

            logData();

            if(t >= 120.0) complete = true;
            if(complete){
                moveG1(0.0f, 0.0f, 0.0f);
                close(g1_socket);
                writeDataToFile();
            }

        }
    };

    void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){

        rx = data->pose.position.x;
        ry = data->pose.position.y;

        double sin_yaw = 2.0 * (data->pose.orientation.w * data->pose.orientation.z); 
        double cos_yaw = 1.0 - 2.0 * data->pose.orientation.z * data->pose.orientation.z;
        yaw = atan2(sin_yaw, cos_yaw);
        //double yaw_deg = yaw * 180.0/M_PI;
        //std::cout << yaw_deg << std::endl;
    
    };

    // Perform a bilinear interpolation on a 2-D grid
    double bilinear_interpolation(std_msgs::msg::Float64MultiArray *grid_ptr, const double i, const double j){

        double f, f1, f2, f11, f12, f21, f22;
        
        const double i1 = floor(i);
        const double j1 = floor(j);
        const double i2 = ceil(i);
        const double j2 = ceil(j);

        if((i1 != i2) && (j1 != j2)){
            f11 = (i2 - i) / (i2 - i1) * grid_ptr->data[jmax*(int)i1+(int)j1];
            f12 = (i2 - i) / (i2 - i1) * grid_ptr->data[jmax*(int)i1+(int)j2];
            f21 = (i - i1) / (i2 - i1) * grid_ptr->data[jmax*(int)i2+(int)j1];
            f22 = (i - i1) / (i2 - i1) * grid_ptr->data[jmax*(int)i2+(int)j2];
            f1 = (j2 - j) / (j2 - j1) * (f11 + f21);
            f2 = (j - j1) / (j2 - j1) * (f12 + f22);
            f = f1 + f2;
        }
        else if(i1 != i2){
            f1 = (i2 - i) / (i2 - i1) * grid_ptr->data[jmax*(int)i1+(int)j];
            f2 = (i - i1) / (i2 - i1) * grid_ptr->data[jmax*(int)i2+(int)j];
            f = f1 + f2;
        }
        else if(j1 != j2){
            f1 = (j2 - j) / (j2 - j1) * grid_ptr->data[jmax*(int)i+(int)j1];
            f2 = (j - j1) / (j2 - j1) * grid_ptr->data[jmax*(int)i+(int)j2];
            f = f1 + f2;
        }
        else{
            f = grid_ptr->data[jmax*(int)i+(int)j];
        }

        return f;

    }

    void logData(void){
        tbuffer[iter] = t;
        xbuffer[iter] = rx;
        ybuffer[iter] = ry;
        hbuffer[iter] = h;
        iter++;
    }

    void writeDataToFile(void){
        const std::string& filename = "poisson_safety_values.csv";
        std::ofstream outFile(filename);
        if(outFile.is_open()){
            for(int i=0; i<BUF; i++){
                outFile << tbuffer[i] << ", " << xbuffer[i] << ", " << ybuffer[i] << ", " << hbuffer[i] << std::endl;
            }
            outFile.close();
        } 
        else{
            std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        }
    }

    void openG1Socket(const std::string& ip_address, int port){

        g1_socket = socket(AF_INET, SOCK_DGRAM, 0); // Create socket
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(port);
        inet_pton(AF_INET, ip_address.c_str(), &dest_addr.sin_addr);

    }
    
    void moveG1(float vx, float vy, float vyaw){

        std::vector<uint8_t> buffer(12); // Create buffer to hold packed data (3 floats = 12 bytes)
        memcpy(&buffer[0], &vx, 4);
        memcpy(&buffer[4], &vy, 4);
        memcpy(&buffer[8], &vyaw, 4);
        
        sendto(g1_socket, buffer.data(), buffer.size(), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr)); // Send the packed data
        std::cout << "vx: " << vx << ", " << "vy: " << vy << ", " "vyaw: " << vyaw << std::endl;

    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_state_suber;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr h_subscription_;

    rclcpp::TimerBase::SharedPtr timer_; // ROS2 timer
    int g1_socket;
    struct sockaddr_in dest_addr;

    bool complete;
    bool h_flag;

    double t; // runing time count
    double dt = 0.005; //control time step

    double tgrid;
    double dtgrid = 1.0e10;
    double dhdt = 0.0;

    double rx, ry, yaw;
    double h, h0, gradhx, gradhy;
    
    double tbuffer[BUF];
    double xbuffer[BUF];
    double ybuffer[BUF];
    double hbuffer[BUF];
    int iter = 0;

    const int imax = 120;
    const int jmax = 120;
    const double ds = 0.0254;
    std_msgs::msg::Float64MultiArray hgrid;
    std_msgs::msg::Float64MultiArray h0grid;

};

int main(int argc, char *argv[]){

    rclcpp::init(argc, argv); // Initialize rclcpp
    rclcpp::TimerBase::SharedPtr timer_; // Create a timer callback object to send sport request in time intervals
    rclcpp::spin(std::make_shared<soprt_request>()); //Run ROS2 node
    rclcpp::shutdown();
    return 0;

}