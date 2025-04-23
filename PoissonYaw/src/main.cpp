#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "../inc/kernel.hpp"
#include "../inc/poisson.h"
#include "../inc/utils.h"
#include "../inc/mpc_cbf.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>

#define KEY_UP 65
#define KEY_DOWN 66
#define KEY_RIGHT 67
#define KEY_LEFT 68

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"){
            
            t_grid = std::chrono::high_resolution_clock::now();
            t_state = std::chrono::high_resolution_clock::now();
            t_flow = std::chrono::high_resolution_clock::now();

            f0 = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            hgrid = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            vxgrid = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            vygrid = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            h0grid = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            
            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                occ_vi_old[n] = 0.0f;
                occ_vj_old[n] = 0.0f;
                occ_vi[n] = 0.0f;
                occ_vj[n] = 0.0f;
            }
            
            for(int n = 0; n < IMAX*JMAX*QMAX*TMAX; n++){
                f0[n] = 0.0f;
                hgrid[n] = h0;
            }

            mpc_controller.setup_QP();
            mpc_controller.solve();

            hgrid_message.data.resize(IMAX*JMAX);
            u_message.data.resize(4);
            mpc_message.data.resize(mpc_controller.nZ+2);
            
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            rclcpp::SubscriptionOptions options3;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options3.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

            rclcpp::QoS qos_profile(5);
            qos_profile.reliability(rclcpp::ReliabilityPolicy::BestEffort);
            qos_profile.history(rclcpp::HistoryPolicy::KeepLast);

            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_grid_topic", qos_profile);
            u_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_command_topic", qos_profile);
            mpc_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("mpc_solution_topic", qos_profile);
            occ_grid_suber_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>("occ_grid_topic", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            optflow_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("optical_flow_topic", 1, std::bind(&PoissonControllerNode::optical_flow_callback, this, std::placeholders::_1), options2);
            //pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/MacLane/pose", 1, std::bind(&PoissonControllerNode::optitrack_state_callback, this, std::placeholders::_1), options3);
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/Shakira/pose", 1, std::bind(&PoissonControllerNode::optitrack_state_callback, this, std::placeholders::_1), options3);

            // Create Timer for Reference
            const float ref_period = 0.01f;
            auto ref_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(ref_period));
            ref_timer_ = this->create_wall_timer(ref_timer_period, std::bind(&PoissonControllerNode::reference_callback, this));
            
            // Create Timer for MPC
            const float mpc_period = 0.01f;
            auto mpc_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(mpc_period));
            mpc_timer_ = this->create_wall_timer(mpc_timer_period, std::bind(&PoissonControllerNode::mpc_callback, this));
            
            // Define Reference Trajectory
            rxd = 1.75f;
            ryd = 1.75f;
            yawd = 0.0f;

            openG1Socket("169.254.21.27", 5005);
        
        }

    private:

        void openG1Socket(const std::string& ip_address, int port){

            g1_socket = socket(AF_INET, SOCK_DGRAM, 0); // Create socket
            dest_addr.sin_family = AF_INET;
            dest_addr.sin_port = htons(port);
            inet_pton(AF_INET, ip_address.c_str(), &dest_addr.sin_addr);
        
        }
        
        void moveG1(float *yaw, float *vx, float *vy, float *vyaw){
        
            std::vector<uint8_t> buffer(16); // Create buffer to hold packed data (4 floats = 16 bytes)
            memcpy(&buffer[0], yaw, 4);
            memcpy(&buffer[4], vx, 4);
            memcpy(&buffer[8], vy, 4);
            memcpy(&buffer[12], vyaw, 4);
            sendto(g1_socket, buffer.data(), buffer.size(), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr)); // Send the packed data
        
        }
        
        void reference_callback(void){
            
            char ch;
            if (read(STDIN_FILENO, &ch, 1) > 0) {

                // Define Reference Trajectory
                if(ch == KEY_UP) rxd -= 0.01f;
                if(ch == KEY_DOWN) rxd += 0.01f;
                if(ch == KEY_LEFT) ryd -= 0.01f;
                if(ch == KEY_RIGHT) ryd += 0.01f;
            
            }

            // Define Reference Heading
            //const float rxe = rxd - rx;
            //const float rye = ryd - ry;
            //const float err = sqrtf(rxe*rxe+rye*rye);
            //if(err > 0.4f) yawd = atan2f(rye,rxe);

        };   
    
        /* Nominal Single Integrator Tracker */
        void nominal_controller(const float rx, const float ry, const float yaw){

            // Compute Errors
            float rxe = rxd - rx;
            float rye = ryd - ry;
            float yawe = ang_diff(yawd,yaw);

            const float kp = 0.5f;
            vx = kp * rxe;
            vy = kp * rye;
            vyaw = kp * yawe;

        };

        void propogate_occupancy_grid(float *bound, const float *occ, const float *occ_vi, const float *occ_vj, const int k){
            
            for(int n = 0; n < IMAX*JMAX; n++){
                bound[n] = 1.0f;
            }
            
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(occ[i*JMAX+j]==-1.0f){
                        const float if_new = (float)i + occ_vi[i*JMAX+j] * (float)k * DT;
                        const float jf_new = (float)j + occ_vj[i*JMAX+j] * (float)k * DT;
                        const int i_new = std::min(std::max((int)roundf(if_new), 0), IMAX-1);
                        const int j_new = std::min(std::max((int)roundf(jf_new), 0), JMAX-1);
                        bound[i_new*JMAX+j_new] = -1.0f;
                    }
                }
            }
        
        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(float *bound){
            
            // Set Border
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
                }
            }

            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++){
                if(b0[n]==1.0f){
                    if(b0[n+1]==-1.0f || 
                       b0[n-1]==-1.0f || 
                       b0[n+JMAX]==-1.0f || 
                       b0[n-JMAX]==-1.0f || 
                       b0[n+JMAX+1]==-1.0f || 
                       b0[n-JMAX+1]==-1.0f || 
                       b0[n+JMAX-1]==-1.0f || 
                       b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
                }
            }

        };

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_and_fix_boundary(float *grid, float *bound){
            
            // Set Border
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
                }
            }

            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++){
                if(b0[n]==1.0f){
                    if(b0[n+1]==-1.0f || 
                       b0[n-1]==-1.0f || 
                       b0[n+JMAX]==-1.0f || 
                       b0[n-JMAX]==-1.0f || 
                       b0[n+JMAX+1]==-1.0f || 
                       b0[n-JMAX+1]==-1.0f || 
                       b0[n+JMAX-1]==-1.0f || 
                       b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
                }
                if(!bound[n]) grid[n] = h0;
            }

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, const float yawk){

            /* Step 1: Create Robot Kernel */
            //const float length = 0.76f; // Go2
            //const float width = 0.35f;
            const float length = 0.35f; // G1
            const float width = 1.02f;
            //const float length = 0.40f; // Drone
            //const float width = 0.40f;
            
            const float D = sqrtf(length*length + width*width); // Max Robot Dimension to Define Kernel Size
            const int dim = ceilf((ceilf(D / DS) + 1.0f) / 2.0f) * 2.0f - 1.0f;
            float robot_grid[dim*dim];

            const float MOS = 1.4f;
            const float ar = MOS * length / 2.0f;
            const float br = MOS * width / 2.0f;

            const float expo = 4.0f;
            for(int i = 0; i < dim; i++){
                const float yi = (float)(dim-1-i)*DS - D/2.0f;
                for(int j = 0; j < dim; j++){
                    robot_grid[i*dim+j] = 0.0;
                    const float xi = (float)j*DS - D/2.0f;
                    const float xb = cosf(yawk)*xi + sinf(yawk)*yi;
                    const float yb = -sinf(yawk)*xi + cosf(yawk)*yi;
                    const float dist = powf(fabsf(xb/ar), expo) + powf(fabsf(yb/br), expo);
                    if(dist <= 1.0f) robot_grid[i*dim+j] = -1.0f;
                    //if(fabsf(xb/ar) <= 1.0f && fabsf(yb/br) <= 1.0f) robot_grid[i*dim+j] = -1.0f;
                }
            }
        
            /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));

            int lim = (dim - 1)/2;
            for(int i = 1; i < IMAX-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 1; j < JMAX-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, JMAX);
                    if(!b0[i*JMAX+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*JMAX+q] += robot_grid[(p-i+lim)*dim+(q-j+lim)];
                            }
                        }
                    }
                }
            }
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] < -1.0f) bound[n] = -1.0f;
            }

        };

        /* Using Occupancy Grid, Find Desired Boundary Gradients */
        void compute_boundary_gradients(float *guidance_x, float *guidance_y, const float *bound){

            // Set Border Gradients
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0) guidance_x[i*JMAX+j] = dh0;
                    if(j==0) guidance_y[i*JMAX+j] = dh0;
                    if(i==(IMAX-1)) guidance_x[i*JMAX+j] = -dh0;
                    if(j==(JMAX-1)) guidance_y[i*JMAX+j] = -dh0;
                }
            }
            
            // Set Additional Boundary Gradients
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(!bound[i*JMAX+j]){
                        guidance_x[i*JMAX+j] = 0.0f;
                        guidance_y[i*JMAX+j] = 0.0f;
                        for(int p = -1; p <= 1; p++){
                            for(int q = -1; q <= 1; q++){
                                if(q > 0){
                                    guidance_x[i*JMAX+j] += bound[(i+q)*JMAX+(j+p)];
                                    guidance_y[i*JMAX+j] += bound[(i+p)*JMAX+(j+q)];
                                }
                                else if (q < 0){
                                    guidance_x[i*JMAX+j] -= bound[(i+q)*JMAX+(j+p)];
                                    guidance_y[i*JMAX+j] -= bound[(i+p)*JMAX+(j+q)];
                                }
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(!bound[i*JMAX+j]){
                        const float V = sqrtf(guidance_x[i*JMAX+j]*guidance_x[i*JMAX+j] + guidance_y[i*JMAX+j]*guidance_y[i*JMAX+j]);
                        if(V != 0.0f){
                            guidance_x[i*JMAX+j] *= dh0 / V;
                            guidance_y[i*JMAX+j] *= dh0 / V;
                        }
                    }
                }
            }

        };

        /* Compute Forcing Function from Guidance Field */
        void compute_optimal_forcing_function(float *force, const float *guidance_x, const float *guidance_y, const float *bound){

            const float max_div = 10.0f;
            const float alpha = 3.0f;

            for(int i = 1; i < (IMAX-1); i++){
                for(int j = 1; j < (JMAX-1); j++){
                    force[i*JMAX+j] = (guidance_x[(i+1)*JMAX+j] - guidance_x[(i-1)*JMAX+j]) / (2.0f*DS) + (guidance_y[i*JMAX+(j+1)] - guidance_y[i*JMAX+(j-1)]) / (2.0f*DS);
                    if(bound[i*JMAX+j] > 0.0f){
                        force[i*JMAX+j] = softMin(force[i*JMAX+j], -max_div, alpha);
                        force[i*JMAX+j] = softMax(force[i*JMAX+j], 0.0f, alpha);
                    }
                    else if(bound[i*JMAX+j] < 0.0f){
                        force[i*JMAX+j] = softMax(force[i*JMAX+j], max_div, alpha);
                        force[i*JMAX+j] = softMin(force[i*JMAX+j], 0.0f, alpha);
                    }
                    else{
                        force[i*JMAX+j] = 0.0f;
                    }
                }
            }

        };

        /* Compute Forcing Function for Average Flux */
        void compute_fast_forcing_function(float *force, const float *bound){

            float perimeter_c = 0.0f;
            float area_c = 0.0f;
            
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
                    else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
                }
            }
            
            float perimeter_o = 2.0f*(float)IMAX*DS + 2.0f*(float)JMAX*DS + perimeter_c;
            float area_o = (float)IMAX*(float)JMAX*DS*DS - area_c;
            float force_o = -dh0 * perimeter_o / area_o * DS*DS;
            float force_c = 0.0f;
            if(area_c != 0.0f) force_c = dh0 * perimeter_c / area_c * DS*DS;
            
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] > 0.0f){
                    force[n] = force_o;
                }
                else if(bound[n] < 0.0f){
                    force[n] = force_c;
                }
                else{
                    force[n] = 0.0f;
                }
            }
        
        };

        /* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
        int poisson(float *grid, const float *force, const float *bound, const float relTol = 1.0e-4f, const float N = 25.0f, const bool gpu_flag = false){
            
            const float w_SOR = 2.0f/(1.0f+sinf(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            
            if(!gpu_flag){

                int iters = 0;
                const int max_iters = 10000;
                float rss;
                for(int n = 0; n < max_iters; n++){

                    // Checkerboard Pass
                    rss = 0.0f;
                    
                    for(int k = 0; k < TMAX; k++){   
                        for(int q = 0; q < QMAX; q++){
                        
                            float *grid_slice = grid + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                            const float *force_slice = force + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                            const float *bound_slice = bound + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                            
                            // Red Pass
                            for(int i = 1; i < IMAX-1; i++){
                                for(int j = 1; j < JMAX-1; j++){
                                    const bool red = (((i%2)+(j%2))%2) == 0;
                                    if(bound_slice[i*JMAX+j] && red){
                                        float dg = 0.0f;
                                        dg += (grid_slice[(i+1)*JMAX+j] + grid_slice[(i-1)*JMAX+j]);
                                        dg += (grid_slice[i*JMAX+(j+1)] + grid_slice[i*JMAX+(j-1)]);
                                        dg -= force_slice[i*JMAX+j];
                                        dg /= 4.0f;
                                        dg -= grid_slice[i*JMAX+j];
                                        grid_slice[i*JMAX+j] += w_SOR * dg;
                                        rss += dg * dg;
                                    }
                                }
                            }
                            // Black Pass
                            for(int i = 1; i < IMAX-1; i++){
                                for(int j = 1; j < JMAX-1; j++){
                                    const bool black = (((i%2)+(j%2))%2) == 1;
                                    if(bound_slice[i*JMAX+j] && black){
                                        float dg = 0.0f;
                                        dg += (grid_slice[(i+1)*JMAX+j] + grid_slice[(i-1)*JMAX+j]);
                                        dg += (grid_slice[i*JMAX+(j+1)] + grid_slice[i*JMAX+(j-1)]);
                                        dg -= force_slice[i*JMAX+j];
                                        dg /= 4.0f;
                                        dg -= grid_slice[i*JMAX+j];
                                        grid_slice[i*JMAX+j] += w_SOR * dg;
                                        rss += dg * dg;
                                    }
                                }
                            }

                        }
                    }

                    rss = sqrtf(rss) * DS / (float)(QMAX*TMAX);
                    iters++;
                    if(rss < relTol) break;

                
                }

                return iters;

            }
            else{

                return Kernel::Poisson(grid, force, bound, relTol, w_SOR); // CUDA!

            }

        };

        /* Compute the Poisson Safety Function */
        void solve_poisson_safety_function(float *grid, float *guidance_x, float *guidance_y, const float *occ, const float *occ_vi, const float *occ_vj){
            
            const bool gpu_flag = true;

            float *bound, *force;
            bound = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            force = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            

            #pragma omp parallel for
            for(int k=0; k<TMAX; k++){

                float *bound_slice_q0 = (float *)malloc(IMAX*JMAX*sizeof(float));
                memcpy(bound_slice_q0, occ, IMAX*JMAX*sizeof(float));
                propogate_occupancy_grid(bound_slice_q0, occ, occ_vi, occ_vj, k);
                find_boundary(bound_slice_q0);
                
                for(int q=0; q<QMAX; q++){
                
                    const float yaw_k = (float)q * DQ;
                    //const float yaw_k = yaw;
                
                    float *force_slice = force + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *bound_slice = bound + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *grid_slice = grid + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *guidance_x_slice = guidance_x + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *guidance_y_slice = guidance_y + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    
                    memcpy(bound_slice, bound_slice_q0, IMAX*JMAX*sizeof(float));
                    inflate_occupancy_grid(bound_slice, yaw_k);
                    find_and_fix_boundary(grid_slice, bound_slice);
                    //compute_boundary_gradients(guidance_x_slice, guidance_y_slice, bound_slice);
                    compute_fast_forcing_function(force_slice, bound_slice);

                }
                
                free(bound_slice_q0);

            }
            
            //const float v_RelTol = 1.0e-4f;
            //vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 25.0f, gpu_flag);
            //vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 25.0f, gpu_flag);
            //for(int k=0; k<TMAX; k++){
                //for(int q=0; q<QMAX; q++){
                    //float *force_slice = force + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    //float *guidance_x_slice = guidance_x + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    //float *guidance_y_slice = guidance_y + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    //compute_optimal_forcing_function(force_slice, guidance_x_slice, guidance_y_slice, bound_slice);
                    //for(int n=0; n<IMAX*JMAX; n++){
                    //    force_slice[n] *= DS*DS;
                    //}
                //}
            //}

            const float h_RelTol = 1.0e-4f;
            h_iters = poisson(grid, force, bound, h_RelTol, 25.0f, gpu_flag);

            free(bound);
            free(force);
            
        };

        void safety_filter(const float rx, const float ry, const float yaw, const float dt, const bool filter_flag){

            // Fractional Index Corresponding to Current Position
            const float ir = (float)(IMAX-1) - ry / DS;
            const float jr = rx / DS;
            const float qr = yaw / DQ;

            const float x_eps = 1.0f*DS; // Small Perturbation for Numerical Gradients (meters)
            const float y_eps = 1.0f*DS; // Small Perturbation for Numerical Gradients (meters)
            const float yaw_eps = 1.0f*DQ;

            const float i_eps = x_eps / DS;
            const float j_eps = y_eps / DS;
            const float q_eps = yaw_eps / DQ;

            const float ic = fminf(fmaxf(i_eps, ir), (float)(IMAX-1)-i_eps); // Saturated Because of Finite Grid Size
            const float jc = fminf(fmaxf(j_eps, jr), (float)(JMAX-1)-j_eps); // Numerical Derivatives Shrink Effective Grid Size
            const float qc = q_wrap(qr);

            // Get Safety Function Rate
            const float *hgrid0 = hgrid+0*IMAX*JMAX*QMAX;
            const float *hgrid1 = hgrid+1*IMAX*JMAX*QMAX;
            const float h0 = trilinear_interpolation(hgrid0, ic, jc, qc);
            const float h1 = trilinear_interpolation(hgrid1, ic, jc, qc);
            dhdt = (h1 - h0) / DT;
            
            // Get Safety Function Value
            h = h0;
            h += dhdt * dt;
            
            // Compute Gradients
            const float ip = ic - i_eps;
            const float im = ic + i_eps;
            const float jp = jc + j_eps;
            const float jm = jc - j_eps;
            const float qp = q_wrap(qc + q_eps);
            const float qm = q_wrap(qc - q_eps);
            const float dhdx0 = (trilinear_interpolation(hgrid0, ic, jp, qc) - trilinear_interpolation(hgrid0, ic, jm, qc)) / (2.0f * x_eps);
            const float dhdy0 = (trilinear_interpolation(hgrid0, ip, jc, qc) - trilinear_interpolation(hgrid0, im, jc, qc)) / (2.0f * y_eps);
            const float dhdyaw0 = (trilinear_interpolation(hgrid0, ic, jc, qp) - trilinear_interpolation(hgrid0, ic, jc, qm)) / (2.0f * yaw_eps);
            const float dhdx1 = (trilinear_interpolation(hgrid1, ic, jp, qc) - trilinear_interpolation(hgrid1, ic, jm, qc)) / (2.0f * x_eps);
            const float dhdy1 = (trilinear_interpolation(hgrid1, ip, jc, qc) - trilinear_interpolation(hgrid1, im, jc, qc)) / (2.0f * y_eps);
            const float dhdyaw1 = (trilinear_interpolation(hgrid1, ic, jc, qp) - trilinear_interpolation(hgrid1, ic, jc, qm)) / (2.0f * yaw_eps);
            const float kgrad = dt/DT;
            dhdx = dhdx0*(1.0f-kgrad) + dhdx1* kgrad;
            dhdy = dhdy0*(1.0f-kgrad) + dhdy1* kgrad;
            dhdyaw = dhdyaw0*(1.0f-kgrad) + dhdyaw1* kgrad;

            // Single Integrator Safety Filter
            const float issf = 100.0f;

            const float b = dhdx*dhdx + dhdy*dhdy + dhdyaw*dhdyaw;
            float a = dhdx*vx + dhdy*vy + dhdyaw*vyaw;
            a += dhdt;
            a += WN*h;
            a -= 1.0f/issf * b;

            if(fabsf(b) < 1.0e-3f){
                vxs = 0.0f;
                vys = 0.0f;
                vyaws = 0.0f;
            }
            else{
                vxs = -a * dhdx / b;
                vys = -a * dhdy / b;
                vyaws = -a * dhdyaw / b;
            }

            if(a<=0.0f & filter_flag){
                vx += vxs;
                vy += vys;
                vyaw += vyaws;
            }

        };

        void mpc_callback(void){
            
            if(h_flag){

                const float xd[3] = {rxd, ryd, yawd};
                const float x[3] = {rx, ry, yaw};
                
                // Perform SQP Iterations
                const int sqp_iters = 5;
                mpc_controller.initialize_solution(xd, x);
                for(int i=0; i<sqp_iters; i++){
                    mpc_controller.update_cost_and_constraints(hgrid);
                    mpc_fail_flag = mpc_controller.solve();
                }
                mpc_age = 0.0f;
            }
            
            // Publish ROS Message
            float z[mpc_controller.nZ];
            mpc_controller.extract_solution(z);
            for(int i = 0; i < mpc_controller.nZ; i++){
                mpc_message.data[i] = z[i];
            }
            mpc_message.data[mpc_controller.nZ+0] = rxd;
            mpc_message.data[mpc_controller.nZ+1] = ryd;
            this->mpc_publisher_->publish(mpc_message);

        }

        void occ_grid_callback(std_msgs::msg::UInt8MultiArray::UniquePtr msg){
            
            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Compute Grid Timing
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;
            t_grid = std::chrono::high_resolution_clock::now();
            dt_grid = dt_grid0.count() * 1.0e-9f;
            grid_age = dt_grid;

            // Update Occupancy
            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                if(msg->data[n]) occ[n] = -1.0f;
            }

            // Store Old Solution
            memcpy(h0grid, hgrid, IMAX*JMAX*QMAX*TMAX*sizeof(float));

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            solve_poisson_safety_function(hgrid, vxgrid, vygrid, occ, occ_vi, occ_vj);
            //printf("Laplace Iterations: %u \n", vx_iters + vy_iters);
            printf("Poisson Iterations: %u \n", h_iters);
            h_flag = true;
            
            // Publish Poisson Safety Function Grid (Interpolated)
            const float qr = q_wrap(yaw/DQ);
            const float q1f = floorf(qr);
            const float q2f = ceilf(qr);
            const int q1 = (int)q_wrap(q1f);
            const int q2 = (int)q_wrap(q2f);
            //int k = TMAX-1;
            int k = 0;
            for(int n = 0; n < IMAX*JMAX; n++){
                hgrid_message.data[n] = (q2f - qr) * hgrid[k*IMAX*JMAX*QMAX+q1*IMAX*JMAX+n] + (qr - q1f) * hgrid[k*IMAX*JMAX*QMAX+q2*IMAX*JMAX+n];
            }
            this->hgrid_publisher_->publish(hgrid_message);

            solve_timer.time("Grid Solve Time: ");
            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vx << "," << vy << "," << vyaw << ">" << std::endl;
            save_flag = writeDataToFile(save_flag, hgrid, IMAX*JMAX, "poisson_safety_grid.csv");
        
        };

        void optical_flow_callback(std_msgs::msg::Float32MultiArray::UniquePtr msg){
            
            dt_flow0 = std::chrono::high_resolution_clock::now() - t_flow;
            t_flow = std::chrono::high_resolution_clock::now();
            dt_flow = dt_flow0.count() * 1.0e-9f;
 
            const float wf = 0.0f;
            const float kf = 1.0f - expf(-wf*dt_flow);

            // Store Optical Flow Data
            for(int n = 0; n < IMAX*JMAX; n++){
                occ_vi_old[n] *= 1.0f - kf;
                occ_vj_old[n] *= 1.0f - kf;
                occ_vi_old[n] += kf * msg->data[0*IMAX*JMAX+n] / dt_flow;
                occ_vj_old[n] += kf * msg->data[1*IMAX*JMAX+n] / dt_flow;
                occ_vi[n] *= 1.0f - kf;
                occ_vj[n] *= 1.0f - kf;
                occ_vi[n] += kf * occ_vi_old[n];
                occ_vj[n] += kf * occ_vj_old[n];
            }
        
        };

        void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){
            
            // Interpret State
            const float rc[2] = {1.75f, 1.75f}; // Location of OptiTrack Origin in Grid Frame
            rx = data->pose.position.x + rc[0];
            ry = data->pose.position.y + rc[1];
            float sin_yaw = 2.0f * (data->pose.orientation.w * data->pose.orientation.z); 
            float cos_yaw = 1.0f - 2.0f * data->pose.orientation.z * data->pose.orientation.z;
            yaw = atan2f(sin_yaw, cos_yaw);
            if(yaw<0.0f) yaw += 2.0f*M_PI;

            // Increment Age of Latest Grid
            dt_state0 = std::chrono::high_resolution_clock::now() - t_state;
            t_state = std::chrono::high_resolution_clock::now();
            dt_state = dt_state0.count() * 1.0e-9f;
            
            grid_age += dt_state;
            mpc_age += dt_state;

            // Apply Nominal Control & Safety Filter
            nominal_controller(rx, ry, yaw);
            if(h_flag){
                if(!mpc_fail_flag){
                    mpc_controller.set_input(&vx, &vy, &vyaw, mpc_age);
                }
                else{
                    filter_flag = mpc_fail_flag;
                }
                safety_filter(rx, ry, yaw, grid_age, filter_flag);
            }
            moveG1(&yaw, &vx, &vy, &vyaw);

            //Publish Control Action
            //u_message.data[0] = rx;
            //u_message.data[1] = ry;
            //u_message.data[2] = yaw;
            //u_message.data[3] = vx;
            //u_message.data[4] = vy;
            //u_message.data[5] = vyaw;
            //u_message.data[6] = vxs;
            //u_message.data[7] = vys;
            //u_message.data[8] = vyaws;
            //u_message.data[9] = h;
            //u_message.data[0] = yaw;
            //u_message.data[1] = vx;
            //u_message.data[2] = vy;
            //u_message.data[3] = vyaw;
            //this->u_publisher_->publish(u_message);

        };

        MPC mpc_controller;
        
        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        bool save_flag = false;
        bool h_flag = false;
        bool filter_flag = false;
        bool mpc_fail_flag = false;

        int vx_iters, vy_iters, h_iters;
        
        float rx = 1.75f;
        float ry = 1.75f;
        float yaw = 0.0f;

        //float t = -5.0f;
        std::chrono::high_resolution_clock::time_point t_grid, t_state, t_flow;
        std::chrono::duration<float, std::nano> dt_grid0, dt_state0, dt_flow0;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;
        float dt_flow = 1.0e10f;
        float grid_age = 0.0f;
        float mpc_age = 0.0f;

        float rxd, ryd, yawd;
        float vx, vy, vyaw;
        float vxs, vys, vyaws;
        float h, dhdt, dhdyaw, dhdx, dhdy;

        float i_obs, j_obs, i_obs0, j_obs0;
        float vi_obs = 0.0f;
        float vj_obs = 0.0f;
        bool v_obs_flag = false;
        
        float occ[IMAX*JMAX];
        float occ_vi[IMAX*JMAX];
        float occ_vj[IMAX*JMAX];
        float occ_vi_old[IMAX*JMAX];
        float occ_vj_old[IMAX*JMAX];
        float *f0, *hgrid, *vxgrid, *vygrid, *h0grid;
        
        std_msgs::msg::Float32MultiArray hgrid_message;
        std_msgs::msg::Float32MultiArray u_message;
        std_msgs::msg::Float32MultiArray mpc_message;

        rclcpp::TimerBase::SharedPtr mpc_timer_;
        rclcpp::TimerBase::SharedPtr ref_timer_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr u_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mpc_publisher_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr optflow_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;

        int g1_socket;
        struct sockaddr_in dest_addr;
    
};

struct termios oldt;
struct termios newt;

void setNonBlocking(bool enable) {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if(enable) fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    else fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
}

void setRawMode(bool enable) {

    if(enable){
        tcgetattr(STDIN_FILENO, &oldt);  // save old settings
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO); // disable buffering and echo
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    } 
    else{
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // restore old settings
    }
}

int main(int argc, char * argv[]){

    setRawMode(true);
    setNonBlocking(true);
    
    rclcpp::init(argc, argv);   
    rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::Node::SharedPtr poissonNode = std::make_shared<PoissonControllerNode>();
    executor.add_node(poissonNode);

    try{
        executor.spin();
        throw("Terminated");
    }
    catch(const char* msg){
        rclcpp::shutdown();
        setRawMode(false);
        setNonBlocking(false);
        std::cout << msg << std::endl;
    }

  return 0;

}