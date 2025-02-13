#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include <unistd.h>
#include <cmath>

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

#define BUF 24000

using std::placeholders::_1;

// Create a soprt_request class for soprt commond request
class soprt_request : public rclcpp::Node {

public:

    soprt_request() : Node("req_sender"){

        h_flag = false;
        hy_flag = false;
        t = -5; // Runing time count

        pose_state_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/MacLane/pose", 1, std::bind(&soprt_request::optitrack_state_callback, this, _1));
        req_puber = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10); // the req_puber is set to subscribe "/api/sport/request" topic with dt
        timer_ = this->create_wall_timer(std::chrono::milliseconds(int(dt * 1000)), std::bind(&soprt_request::timer_callback, this));

        tgrid = t;
        hgrid.data.resize(imax*jmax);
        //hgridy.data.resize(imax*jmax);
        h0grid.data.resize(imax*jmax);
        
        auto hgrid_topic_callback = [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) -> void {
            
            dtgrid = t - tgrid;
            tgrid = t;
            memcpy(&h0grid.data[0], &hgrid.data[0], sizeof(double)*imax*jmax);
            memcpy(&hgrid.data[0], &msg->data[0], sizeof(double)*imax*jmax);
            if(!h_flag){
                h_flag = true;
                std::cout << "First Grid Received!" << std::endl;
            }
            
        };
        hgrid_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("safety_grid_topic", 1, hgrid_topic_callback);

/*
        auto hgridy_topic_callback = [this](std_msgs::msg::Float64MultiArray::SharedPtr msgy) -> void {
        
            memcpy(&hgridy.data[0], &msgy->data[0], sizeof(double)*imax*jmax);
            if(!hy_flag){
                hy_flag = true;
                std::cout << "First Epsilon Grid Received!" << std::endl;
            }
            
        };
        hgridy_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("eps_safety_grid_topic", 1, hgridy_topic_callback);

*/
/*
        auto hdata_topic_callback = [this](std_msgs::msg::Float64MultiArray::UniquePtr msg) -> void {

            //hdata.data = msg->data;
            h = msg->data[0];
            dhdt = msg->data[1];
            dhdyaw = msg->data[2];
            gradhx = msg->data[3];
            gradhy = msg->data[4];

            if(!h_flag){
                h_flag = true;
                std::cout << "First Data Received!" << std::endl;
            }
            
        };
        hdata_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("safety_data_topic", 1, hdata_topic_callback);
*/
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

        if(t > 0.0){

            const double t_idle = 0.0;
            const double t_walk = 120.0;

            // Define Reference Trajectory
            const double amp = 0.0;
            const double freq = 1.0 / 20.0;
            double w = 2.0 * M_PI * freq;
            rxd = amp*sin(w*t) + 1.5;
            ryd = amp*sin(w*t) + 1.5;
            double vxd = amp*w*cos(w*t);
            double vyd = amp*w*cos(w*t);

            /*
            // Initial Position
            double rx0, ry0;
            const int test_case = 3;
            switch (test_case){
                case 1:
                    rx0 = 2.0;
                    ry0 = 0.5;
                    break;
                case 2:
                    rx0 = 2.5;
                    ry0 = 1.3;
                    break;
                case 3:
                    rx0 = 1.5;
                    ry0 = 2.0;
                    break;
            }

            // Final Position
            double rxf = 1.5;
            double ryf = 0.5;

            // Reference
            if(t < t_idle){
                rxd = rx0;
                ryd = ry0;
                vxd = 0.0;
                vyd = 0.0;
            }
            else{
                double gamma = (t-t_idle)/t_walk;
                rxd = rxf*gamma + rx0*(1.0-gamma);
                ryd = ryf*gamma + ry0*(1.0-gamma);
                vxd = (rxf-rx0) / t_walk;
                vyd = (ryf-ry0) / t_walk;
            }
            */

            // Compute Errors
            double rxe = rxd - rx;
            double rye = ryd - ry;
            
            // Define Reference Yaw
            //double rxe_yaw = rxf - rx;
            //double rye_yaw = ryf - ry;
            //double err = sqrt(rxe_yaw*rxe_yaw+rye_yaw*rye_yaw);
            //if(err > 0.35){
            //    yawd = atan2(rye_yaw,rxe_yaw);
            //}
            //double yawe = ang_diff(yawd, yaw);
            
            // Proportional Control + Tracking
            //const double kpy = 1.0;
            //vyaw = kpy * yawe + vyawd;

            const double kpv = 0.5;
            vx = kpv * rxe + vxd;
            vy = kpv * rye + vyd;

            vx = fmin(fmax(vx, -1.0), 1.0);
            vy = fmin(fmax(vy, -1.0), 1.0);

            if(h_flag){

                // Fractional Index Corresponding to Current Position
                const double rc[2] = {0.0, 0.0}; // Location of OptiTrack Origin in Grid Frame
                double ir = (double)imax - (ry+rc[1]) / ds;
                double jr = (rx+rc[0]) / ds;

                const double x_eps = 1.0e-3; // Small Perturbation for Numerical Gradients (meters)
                double i_eps = x_eps / ds;

                double ic = fmin(fmax(i_eps, ir), (double)(imax-1)-i_eps); // Saturated Because Numerical Derivatives Shrink Effective Grid Size
                double jc = fmin(fmax(i_eps, jr), (double)(jmax-1)-i_eps);

                // Get Safety Function Value
                h = bilinear_interpolation(&hgrid, ic, jc);
                h0 = bilinear_interpolation(&h0grid, ic, jc);

                if(h < 0.0){
                    std::cout << "Safety Violation!: " << h << std::endl;
                } 

                // Compute Time Derivative
                double dhdt_raw = (h - h0) / dtgrid;
                const double wv = 30.0; // Low Pass Filter Cutoff
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
                const double alpha = 0.8;
                const double issf = 4.0;

                double b = gradhx*gradhx + gradhy*gradhy;
                double a = gradhx*vx + gradhy*vy;
                a += alpha*h;
                a += dhdt;
                a -= 1.0/issf * b;

                vxs = -a * gradhx / b;
                vys = -a * gradhy / b;

                if(a<=0.0){
                    vx += vxs;
                    vy += vys;
                }

            }
            
            /*
            vyaw = 0.0;
            if(h_flag && hy_flag){
                double iry = (double)imax - (ry+rc[1]) / ds;
                double jry = (rx+rc[0]) / ds;
                double icy = fmin(fmax(i_eps, iry), (double)(imax-1)-i_eps); // Saturated Because Numerical Derivatives Shrink Effective Grid Size
                double jcy = fmin(fmax(i_eps, jry), (double)(jmax-1)-i_eps);
                hy = bilinear_interpolation(&hgridy, icy, jcy);
                const double kpy = 5.0;
                double yaw_eps = 1.0 * M_PI / 180.0;
                dhdyaw = (hy - h) / yaw_eps;
                vyaw = kpy*dhdyaw;
            }
            */
            
            const double yawd = 0.0 * M_PI / 180.0;
            double yawe = ang_diff(yawd,yaw);
            const double kpy = 0.5;
            vyaw = kpy * yawe;

            double vxb = cos(yaw)*vx+ sin(yaw)*vy;
            double vyb = -sin(yaw)*vx + cos(yaw)*vy;

            vxb = fmin(fmax(vxb, -1.0), 1.0);
            vyb = fmin(fmax(vyb, -1.0), 1.0);

            // Send Command
            sport_req.Move(req, (float)vxb, (float)vyb, (float)vyaw);
            req_puber->publish(req);
            //std::cout << "Command Sent!" << vxb << "," << vyb << std::endl;
            logData();

            if(t >= (t_idle+t_walk)){

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

    // Compute difference between two angles wrapped between [-pi, pi]
    double ang_diff(const double a1, const double a2){
        double a3 = a1 - a2;
        while(a3 <= -M_PI){
            a3 += 2.0*M_PI;
        }
        while(a3 > M_PI){
            a3 -= 2.0*M_PI;
        }
        return a3;
    }

    void logData(void){
        tbuffer[iter] = t;
        xbuffer[iter] = rx;
        ybuffer[iter] = ry;
        vxbuffer[iter] = vx;
        vybuffer[iter] = vy;
        vxsbuffer[iter] = vxs;
        vysbuffer[iter] = vys;
        hbuffer[iter] = h;
        iter++;
    }

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
            for(int i=0; i<BUF; i++){
                outFile << tbuffer[i] << ", " << xbuffer[i] << ", " << ybuffer[i] << ", " << vxbuffer[i] << ", " << vybuffer[i] << ", " << vxsbuffer[i] << ", " << vysbuffer[i] << ", " << hbuffer[i] << std::endl;
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
            for(int i=0; i<BUF; i++){
                outFile_temp << tbuffer[i] << ", " << xbuffer[i] << ", " << ybuffer[i] << ", " << vxbuffer[i] << ", " << vybuffer[i] << ", " << vxsbuffer[i] << ", " << vysbuffer[i] << ", " << hbuffer[i] << std::endl;
            }
            outFile_temp.close();
        } 
        else{
            std::cerr << "Error: Could not open file " << filename_temp << " for writing.\n";
        }

    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_state_suber;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr hgrid_subscription_;
    //rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr hgridy_subscription_;
    //rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr hdata_subscription_;

    rclcpp::TimerBase::SharedPtr timer_; // ROS2 timer
    
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber;

    unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
    SportClient sport_req;

    bool h_flag = false;
    bool hy_flag = false;

    double t; // runing time count
    double dt = 0.005; //control time step

    double tgrid;
    double dtgrid = 1.0e10;

    double rx, ry, yaw, rxd, ryd;
    double yawd = 0.0;
    double vx, vy, vyaw, vxd, vyd;
    double vyawd = 0.0;
    double vxs, vys;
    double h, dhdt, dhdyaw, gradhx, gradhy;
    double h0, hy;
    
    double tbuffer[BUF];
    double xbuffer[BUF];
    double ybuffer[BUF];
    double vxbuffer[BUF];
    double vybuffer[BUF];
    double vxsbuffer[BUF];
    double vysbuffer[BUF];
    double hbuffer[BUF];
    int iter = 0;

    //double yaw0 = 0; // initial yaw angle

    const int imax = 120;
    const int jmax = 120;
    const double ds = 0.0254;
    std_msgs::msg::Float64MultiArray hgrid;
    //std_msgs::msg::Float64MultiArray hgridy;
    std_msgs::msg::Float64MultiArray h0grid;
    //std_msgs::msg::Float64MultiArray hdata;

};

int main(int argc, char *argv[]){

    rclcpp::init(argc, argv); // Initialize rclcpp
    rclcpp::TimerBase::SharedPtr timer_; // Create a timer callback object to send sport request in time intervals
    rclcpp::spin(std::make_shared<soprt_request>()); //Run ROS2 node
    rclcpp::shutdown();
    return 0;

}