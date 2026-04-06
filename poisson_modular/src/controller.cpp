#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <mutex>

#include "kernel.hpp"
#include "definitions.h"
#include "utils.h"
#include "mpc_cbf.h"
#include "poisson.h"

#include <Eigen/Sparse>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "geometry_msgs/msg/twist.hpp"
#include "px4_msgs/msg/vehicle_attitude_setpoint.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"
#include "px4_msgs/msg/vehicle_status.hpp"
#include "px4_msgs/srv/vehicle_command.hpp"
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include "poisson_msgs/msg/poisson_grid.hpp"

class ControlNode : public rclcpp::Node{

    public:
        
        ControlNode() : Node("safe_control"){

            // Initialize OSQP for MPC Problem
            mpc3d_controller.reset_QP(r);
            mpc3d_controller.solve();
            
            // Initialize Clocks
            t_start = std::chrono::steady_clock::now();
            t_grid = std::chrono::steady_clock::now();
            t_odom = std::chrono::steady_clock::now();

            // Initialize Poisson Grids
            hgrid0 = (float *)malloc(IMAX*JMAX*sizeof(float));
            hgrid1 = (float *)malloc(IMAX*JMAX*sizeof(float));
            dhdt_grid = (float *)malloc(IMAX*JMAX*sizeof(float));
            for(int n=0; n < IMAX*JMAX; n++){
                hgrid0[n] = h0;
                hgrid1[n] = h0;
                dhdt_grid[n] = 0.0f;
            }

            // Initialize Reference
            rd = r;
            rd_traj.resize(3*(N_HORIZON+1));

            // Useful Transforms
            R_ned2enu.row(0) << 0.0f, 1.0f, 0.0f;
            R_ned2enu.row(1) << 1.0f, 0.0f, 0.0f;
            R_ned2enu.row(2) << 0.0f, 0.0f, -1.0f;
            
            // Subscribers
            rclcpp::QoS odom_qos(1);
            odom_qos.best_effort();
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            rclcpp::SubscriptionOptions options3;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options3.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            cbf_grid_suber_ = this->create_subscription<poisson_msgs::msg::PoissonGrid>("poisson_cbf", 1, std::bind(&ControlNode::cbf_callback, this, std::placeholders::_1), options1);
            odom_suber_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>("/fmu/out/vehicle_odometry", odom_qos, std::bind(&ControlNode::odom_callback, this, std::placeholders::_1), options2);
            status_suber_ = this->create_subscription<px4_msgs::msg::VehicleStatus>("/fmu/out/vehicle_status_v1", odom_qos, std::bind(&ControlNode::status_callback, this, std::placeholders::_1), options3);
            // twist_suber_ = this->create_subscription<geometry_msgs::msg::Twist>("u_des", 1, std::bind(&ControlNode::teleop_callback, this, std::placeholders::_1), options3);
            
            // Publishers
            offboard_control_mode_puber_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 1);
            cmd_puber_ = this->create_publisher<px4_msgs::msg::VehicleAttitudeSetpoint>("/fmu/in/vehicle_attitude_setpoint_v1", 1);

            // Timers
            t_start = std::chrono::steady_clock::now();
            cmd_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&ControlNode::controller_callback, this));
            mpc_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
            mpc_timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&ControlNode::mpc_callback, this), mpc_callback_group_);

       }

    private:

        void status_callback(px4_msgs::msg::VehicleStatus::SharedPtr status){

            if(status->nav_state==14) start_flag = true;
            else start_flag = false;

        }

        void teleop_callback(geometry_msgs::msg::Twist::SharedPtr msg){
                          
            // Teleop Velocity Command
            Eigen::Vector3f vdb{(float)msg->linear.x, (float)msg->linear.y, 0.0f};
            vd = R * vdb;

            // Goal Point
            rd(0) += vd(0) * 0.01f;
            rd(1) += vd(1) * 0.01f;
            rd(2) = 5.0f;
            
            ad << 0.0f, 0.0f, 0.0f;
            
            yawd += (float)msg->angular.z * 0.01f; 

        };

        void position_controller(void){

            // rd(0) = r(0);
            // rd(1) = r(1);
            // Eigen::Vector3f rd_dot{v(0), v(1), 0.0f}; // Use this when rdx = rx and rdy = ry
            Eigen::Vector3f rd_dot = vd;

            const float kr = 2.0f;
            uv = kr * (rd - r) + vd;
            // uv_dot = kr * (rd_dot - v) + ad;

        };

        void velocity_controller(void){

            const float kv = 2.0f;
            ud = kv * (uv_safe - v) + uv_dot;
            
            ud(0) = std::clamp(ud(0), -0.5f*grav, 0.5f*grav);
            ud(1) = std::clamp(ud(1), -0.5f*grav, 0.5f*grav);
            ud(2) = std::clamp(ud(2), -0.4f*grav, 0.4f*grav);

       };

        void update_dhdt_grid(void){

            const float wc = 10.0f;
            const float kc = 1.0f - std::exp(-wc*dt_grid);

            const float di = dr(1) / DS;
            const float dj = dr(0) / DS;

            for(int i = 0; i<IMAX; i++){
                for(int j = 0; j<JMAX; j++){
                    const float i0 = (float)i + di;
                    const float j0 = (float)j + dj;
                    const bool in_grid = (i0 >= 0.0f) && (i0 <= (float)(IMAX-1)) && (j0 >= 0.0f) && (j0 <= (float)(JMAX-1));
                    float dhdt_ij = 0.0f;
                    if(in_grid){
                        const float h0 = bilinear_interpolation(hgrid0, i0, j0);
                        const float h1 = bilinear_interpolation(hgrid1, i, j);
                        dhdt_ij = (h1 - h0) / dt_grid;
                    }
                    dhdt_grid[i*JMAX+j] *= 1.0f - kc;
                    dhdt_grid[i*JMAX+j] += kc * dhdt_ij;
                }
            }

        };

        void safety_filter_rd1(void){

            // Identify Nominal Control Action
            Eigen::Vector3f uv_nom = uv_mpc;
            
            // Get Safety Function Rate, Value & Gradient
            h = extract_grid_value(hgrid1, r(0), r(1), rc(0), rc(1));
            dhdt = extract_grid_value(dhdt_grid, r(0), r(1), rc(0), rc(1));
            dhdr = extract_grid_gradient(hgrid1, r(0), r(1), rc(0), rc(1));
            // if(h<0.0f) std::cout << "Safety Violation: h = " << h << std::endl;

            // Single Integrator Safety Filter
            const float b = dhdr.dot(dhdr);
            const float ISSf1 = issf;
            const float ISSf2 = issf;
            const float Lgh_norm = std::sqrt(b);
            float ISSf = Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2;
            float a = wn*h;
            a += dhdt; // Dynamic Environment
            a += dhdr.dot(uv_nom); // Min Norm Controller
            a -= ISSf; // Input-to-State Safety (Robustness)
            
            // Analytical Safety Filter
            const float k_sontag = 1.0f;
            const float sigma_sontag = 0.0f;
            float lambda = 0.0f;
            if(b>1.0e-4f) lambda = k_sontag * (-a + std::sqrt(a*a+sigma_sontag*b*b)) / (2.0f*b); // Half Sonta

            uv_safe = uv_nom + lambda * dhdr;
            uv_dot << 0.0f, 0.0f, 0.0f;

            // Analytical Safety Filter Rate
            // float lambda_dot =0.0f;
            // if(b>1.0e-4f){
            //     const float dlda = k_sontag * ( -1.0f + a / std::sqrt(a*a+sigma_sontag*b*b) ) / (2.0f*b);
            //     const float dldb = k_sontag * ( sigma_sontag*b*b / std::sqrt(a*a+sigma_sontag*b*b) + a - std::sqrt(a*a+sigma_sontag*b*b) ) / (2.0f*b*b);
            //     const float adot = wn*h_dot + d2hdt2 + dhdx_dot*uv(0) + dhdx_pred*uv_dot(0) + dhdy_dot*uv(1) + dhdy_pred*uv_dot(1) - ISSf_dot;
            //     const float bdot = 2.0f*dhdx_dot*dhdx_pred + 2.0f*dhdy_dot*dhdy_pred;
            //     lambda_dot = dlda * adot + dldb * bdot;
            // }
            
            // uv_dot(0) += lambda_dot * dhdx_pred + lambda * dhdx_dot;
            // uv_dot(1) += lambda_dot * dhdy_pred + lambda * dhdy_dot;
            // uv_dot(2) += 0.0f;

        };

        void safety_filter_rd2(void){
            
            // Get Safety Function Value
            h = extract_grid_value(hgrid1, r(0), r(1), rc(0), rc(1));
            dhdr = extract_grid_gradient(hgrid1, r(0), r(1), rc(0), rc(1));
            hess_h = extract_grid_hessian(hgrid1, r(0), r(1), rc(0), rc(1));

            // Double Integrator Safety Filter
            const float b = dhdr.dot(dhdr);
            const float ISSf1 = issf;
            const float ISSf2 = issf;
            const float Lgh_norm = std::sqrt(b);
            float ISSf = Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2;
            const float zeta = 1.0f;
            float a = wn * wn * h;
            a += 2.0f * zeta * wn * dhdr.dot(v);
            a += v.transpose() * hess_h * v;
            a += dhdr.dot(ud); 
            a -= ISSf; // Input-to-State Safety (Robustness)
            
            // Analytical Safety Filter
            const float sigma_sontag = 0.1f;
            float lambda = 0.0f;
            if(b>1.0e-4f) lambda = 1.0f * (-a + std::sqrt(a*a+sigma_sontag*b*b)) / (2.0f*b); // Half Sontag

            ud(0) += lambda * dhdr(0);
            ud(1) += lambda * dhdr(1);
            ud(2) += lambda * dhdr(2);

            ud(0) = std::clamp(ud(0), -0.8f*grav, 0.8f*grav);
            ud(1) = std::clamp(ud(1), -0.8f*grav, 0.8f*grav);
            ud(2) = std::clamp(ud(2), -0.4f*grav, 0.4f*grav);

        };

        /* Display Poisson Safety Function Grid */
        void display_poisson_safety_function(void){

            // Timer display_timer(true);
            // display_timer.start();

            // Populate Float Grayscale Poisson Image with Chosen q & k Values
            cv::Mat poisson_img = cv::Mat::zeros(IMAX, JMAX, CV_32FC1);
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    poisson_img.at<float>(i,j) = hgrid1[i*JMAX+j];
                }
            }

            // Convert to 8-bit Grayscale
            cv::Mat gray_img;
            cv::normalize(poisson_img, gray_img, 0, 255, cv::NORM_MINMAX);
            gray_img.convertTo(gray_img, CV_8U);

            // Convert to Colormap
            cv::Mat color_img;
            cv::applyColorMap(gray_img, color_img, cv::COLORMAP_HOT);

            // Resize for Display
            cv::Mat resized_img;
            const int upscale = 3;
            cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);

            // Add Current Location & Goal Location
            cv::Point curr_pt = cv::Point(upscale*x_to_j(r(0), rc(0)), upscale*y_to_i(r(1), rc(1)));
            cv::Point goal_pt = cv::Point(upscale*x_to_j(rd(0), rc(0)), upscale*y_to_i(rd(1), rc(1)));
            cv::circle(resized_img, curr_pt, upscale, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::circle(resized_img, goal_pt, upscale, cv::Scalar(0, 127, 0), cv::FILLED);

            // Add MPC Trajectory
            for(int k = 1; k <= N_HORIZON; k++){
                const int j_traj = x_to_j(mpc3d_controller.xbar(STATES*k+0), rc(0));
                const int i_traj = y_to_i(mpc3d_controller.xbar(STATES*k+1), rc(1));
                cv::Point traj_pt = cv::Point(upscale*j_traj, upscale*i_traj);
                cv::circle(resized_img, traj_pt, upscale, cv::Scalar(255, 0, 0), cv::FILLED);
            }

            // Vertical Flip Image for Display 
            cv::Mat flipped_img;
            cv::flip(resized_img, flipped_img, 0);

            // Display Final Image
            cv::imshow("Poisson Solution", flipped_img);
            cv::waitKey(1);

            // display_timer.time("Display Time: ");

        };

        void virtual_force(void){

            const Eigen::Vector3f gravity{0.0f, 0.0f, -grav}; 
            fd = mass * (ud - gravity);

        }

        void attitude_reference(void){

            Eigen::Vector3f yawd_vec{-std::sin(yawd), std::cos(yawd), 0.0f};

            Eigen::Vector3f bzd = fd;
            bzd.normalize();

            Eigen::Vector3f bxd = yawd_vec.cross(bzd);
            bxd.normalize();

            Eigen::Vector3f byd = bzd.cross(bxd);
            byd.normalize();

            Eigen::Matrix3f Rd;
            Rd.col(0) = bxd;
            Rd.col(1) = byd;
            Rd.col(2) = bzd;

            qd = Eigen::Quaternionf(R_ned2enu.transpose() * Rd * R_ned2enu);

        }

        void throttle_percentage(void){

            const Eigen::Vector3f unit_z{0.0f, 0.0f, 1.0f};
            F = fd.dot(R*unit_z);

        };

        void publish_offboard_control_mode(void){
            
            px4_msgs::msg::OffboardControlMode mode_msg{};
            mode_msg.position = false;
            mode_msg.velocity = false;
            mode_msg.acceleration = false;
            mode_msg.attitude = true;
            mode_msg.body_rate = false;
            mode_msg.thrust_and_torque = false;
            mode_msg.direct_actuator = false;
            mode_msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            offboard_control_mode_puber_->publish(mode_msg);

            px4_msgs::msg::VehicleAttitudeSetpoint cmd_msg{};
            cmd_msg.q_d[0] = qd.w();
            cmd_msg.q_d[1] = qd.x();
            cmd_msg.q_d[2] = qd.y();
            cmd_msg.q_d[3] = qd.z();
            cmd_msg.thrust_body[0] = 0.0f;
            cmd_msg.thrust_body[1] = 0.0f;
            const float Fmax = 1.37f * mass * grav; 
            cmd_msg.thrust_body[2] = std::clamp(-F / Fmax, -1.0f, 0.0f);
            cmd_msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
            this->cmd_puber_->publish(cmd_msg);

            //std::cout << "Command: <" << fd(0) << " , " << fd(1) << " , " << fd(2) << ">" << std::endl;
            //std::cout << "Throttle: " << -100 * cmd_msg.thrust_body[2] << "%" << std::endl;
        
        }

        void mpc_callback(void){ 

            // Timer mpc_timer(true);
            // mpc_timer.start();

            
            // Define Waypoints
            const int num_waypts = 4;
            const float waypts[num_waypts][3] = {{ 0.0f,  0.0f,  0.0f},
                                                 {-4.0f,  0.0f,  0.0f},
                                                 {-4.0f, -4.0f,  0.0f},
                                                 { 0.0f, -4.0f,  0.0f}};

            // Identify Current Waypoint (If Waypoint Reached, Update the Counter)
            if(start_flag){
                if(std::sqrt((rd(0)-r(0))*(rd(0)-r(0))+(rd(1)-r(1))*(rd(1)-r(1))) < 0.5f){
                    if(waypt_counter < (num_waypts-1)) waypt_counter++;
                    else waypt_counter = 0;
                }
            }
            else{
                waypt_counter = 0;
                r_start = r;
            }

            // Define Current Waypoint
            rd << r_start(0) + waypts[waypt_counter][0], r_start(1) + waypts[waypt_counter][1], r_start(2) + waypts[waypt_counter][2];

            // Rollout Desired Trajectory
            for(int k=0; k<(N_HORIZON+1); k++){
                rd_traj(3*k+0) = rd(0);
                rd_traj(3*k+1) = rd(1);
                rd_traj(3*k+2) = rd(2);
            }

            // Run MPC with SQP Loops until Cost is Stable
            if(h_flag && mpc_mutex.try_lock()){
                std::lock_guard<std::mutex> lock(mpc_mutex, std::adopt_lock);
                
                mpc3d_controller.shift_linearization(r);
                for(int i=0; i<SQP_MAX_ITERS; i++){
                    mpc3d_controller.update_cost(rd, uv);
                    mpc3d_controller.update_constraints(hgrid1, dhdt_grid, r, rc, wn, issf);
                    mpc3d_controller.solve();
                    mpc3d_controller.line_search();
                    mpc3d_controller.rollout_safety_filter(hgrid1, dhdt_grid, rc, wn, issf);
                }
                
                mpc3d_controller.set_input(uv_mpc);
                // std::cout << "MPC Command: < " << uv_mpc(0) << " , " << uv_mpc(1) << " , " << uv_mpc(2) << " >" << std::endl;
            }

            // mpc_timer.time("MPC Solve Time: ");
            
            // std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            // std::cout << "Control Loop Time: " << dt_odom*1.0e3f << " ms" << std::endl;
            // std::cout << "Position: <" << r(0) << " , " << r(1) << " , " << r(2) << ">" << std::endl;

        };

        void controller_callback(void){

            // position_controller(); // rd, vd -> uv (proportional control)
            uv_safe = uv_mpc;
            // safety_filter_rd1(); // uv -> uv_safe (CBF-QP)
            velocity_controller(); // uv_safe -> ud (proportional control)
            safety_filter_rd2(); // ud -> ud (CBF-QP)
            virtual_force();
            attitude_reference();
            throttle_percentage();
            publish_offboard_control_mode();

        };

        void cbf_callback(poisson_msgs::msg::PoissonGrid::SharedPtr msg){

            // Compute Grid Timing
            dt_grid = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_grid).count();
            t_grid = std::chrono::steady_clock::now();
            grid_age = dt_grid;

            // Read Message Data
            dr(0) = msg->origin.position.x - rc(0);
            dr(1) = msg->origin.position.y - rc(1);
            rc(0) = msg->origin.position.x;
            rc(1) = msg->origin.position.y;
            
            memcpy(hgrid0, hgrid1, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++) hgrid1[n] = msg->data[n];
            if(h_flag) dhdt_flag = true;
            h_flag = true;
            if(dhdt_flag) update_dhdt_grid(); // Update Grid of dh/dt Values

            display_poisson_safety_function();

        };

        void odom_callback(px4_msgs::msg::VehicleOdometry::UniquePtr msg){

            dt_odom = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_odom).count();
            t_odom = std::chrono::steady_clock::now();
            
            q.w() = msg->q[0];
            q.x() = msg->q[1];
            q.y() = msg->q[2];
            q.z() = msg->q[3];

            Eigen::Matrix3f R_ned(q);
            R = R_ned2enu.transpose() * R_ned * R_ned2enu;

            Eigen::Vector3f r_ned{msg->position[0], msg->position[1], msg->position[2]};
            Eigen::Vector3f r_enu = R_ned2enu * r_ned;
            v = (r_enu - r) / dt_odom;
            r = r_enu;
            
        };

        std::mutex mpc_mutex;
        MPC3D mpc3d_controller;

        float wn = 3.0f;
        float issf = 3.0f;

        bool start_flag = false;
        bool h_flag = false;
        bool dhdt_flag = false;
        int waypt_counter = 0;

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        const float mass = 1.80f;
        const float grav = 9.81f; 

        std::chrono::steady_clock::time_point t_start, t_grid, t_odom;
        float grid_age = 0.0f;
        float dt_grid = 1.0e10f;
        float dt_odom = 1.0e10f;

        // Robot States
        Eigen::Vector3f r{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f v{0.0f, 0.0f, 0.0f};
        Eigen::Quaternionf q;
        Eigen::Matrix3f R, R_ned2enu;

        // References
        Eigen::Vector3f r_start{0.0f, 0.0f, 0.0f};
        Eigen::VectorXd rd_traj;
        Eigen::Vector3f rd{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f vd{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f ad{0.0f, 0.0f, 0.0f};
        float yawd = 0.0f;

        // Virtual Velocity Commands
        Eigen::Vector3f uv{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f uv_dot{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f uv_mpc{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f uv_safe{0.0f, 0.0f, 0.0f};

        // Control Signals
        Eigen::Vector3f ud{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f u{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f fd;
        float F;
        Eigen::Quaternionf qd;
        
        // Grid States
        Eigen::Vector3f rc{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f dr{0.0f, 0.0f, 0.0f};

        // Safety Values
        float h, dhdt;
        Eigen::Vector3f dhdr;
        Eigen::Matrix3f hess_h;

        // Grids
        float occ1[IMAX*JMAX];
        float occ0[IMAX*JMAX];
        int8_t conf[IMAX*JMAX];
        float *hgrid1, *hgrid0, *dhdt_grid;

        rclcpp::Publisher<px4_msgs::msg::VehicleAttitudeSetpoint>::SharedPtr cmd_puber_;
        rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_puber_;

        rclcpp::Subscription<poisson_msgs::msg::PoissonGrid>::SharedPtr cbf_grid_suber_;
        rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_suber_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
        rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_suber_;
        
        rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
        rclcpp::TimerBase::SharedPtr cmd_timer_, mpc_timer_;

};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto poissonNode = std::make_shared<PoissonNode>();
    auto controlNode = std::make_shared<ControlNode>();
    executor.add_node(poissonNode);
    executor.add_node(controlNode);
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