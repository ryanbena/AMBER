#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include "../inc/kernel.hpp"
#include "../inc/poisson.h"
#include "../inc/utils.h"
#include "../inc/mpc_cbf.h"
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

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

            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                occ_vi[n] = 0.0f;
                occ_vj[n] = 0.0f;
                occ_vi_old[n] = 0.0f;
                occ_vj_old[n] = 0.0f;
            }
            occGrid0 = cv::Mat::zeros(IMAX, JMAX, CV_8UC1);
            flow = cv::Mat::zeros(IMAX, JMAX, CV_32FC2);
            dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

            polar_occ(occ_r2, occ_th);

            //hgrid = (float *)malloc(IMAX*JMAX*QMAX*TMAX*sizeof(float));
            cudaMallocHost((void**)&hgrid, IMAX*JMAX*QMAX*TMAX*sizeof(float));
            cudaMallocHost((void**)&bound, IMAX*JMAX*QMAX*TMAX*sizeof(float));
            cudaMallocHost((void**)&force, IMAX*JMAX*QMAX*TMAX*sizeof(float));
            Kernel::poissonInit();
            for(int n = 0; n < IMAX*JMAX*QMAX*TMAX; n++) hgrid[n] = h0;
            robot_kernel_dim = build_robot_kernel(robot_kernel);
            
            hgrid_message.data.resize(IMAX*JMAX);
            u_message.data.resize(10);
            mpc_message.data.resize(mpc_controller.nZ+2);
            
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            
            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_grid_topic", 1);
            u_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("safety_command_topic", 1);
            mpc_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("mpc_solution_topic", 1);
            //occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occ_grid_topic", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            //pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/MacLane_pose_internal", 1, std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1), options2);
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/utlidar/robot_pose", 1, std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1), options2);
        
            // Create Timer for Reference
            const float ref_period = 0.01f;
            auto ref_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(ref_period));
            ref_timer_ = this->create_wall_timer(ref_timer_period, std::bind(&PoissonControllerNode::reference_callback, this));
            
            // Create Timer for MPC
            const float mpc_period = 0.010f;
            auto mpc_timer_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(mpc_period));
            mpc_timer_ = this->create_wall_timer(mpc_timer_period, std::bind(&PoissonControllerNode::mpc_callback, this));

            const float xd[3] = {rxd, ryd, yawd};
            const float x[3] = {rx, ry, yaw};
            mpc_controller.setup_QP(xd,x);
            mpc_controller.solve();
            qp_init_flag = true;
        
        }

    private:
  
        void reference_callback(void){
            
            char ch;
            if (read(STDIN_FILENO, &ch, 1) > 0) {

                // Define Reference Trajectory
                if(ch == KEY_UP) rxd += 0.01f;
                if(ch == KEY_DOWN) rxd -= 0.01f;
                if(ch == KEY_LEFT) ryd += 0.01f;
                if(ch == KEY_RIGHT) ryd -= 0.01f;
            
            }

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

        /* Run Optical Flow on Occupancy Grid*/
        void optical_flow(float *occ_vi, float *occ_vj, const float *occ){

            /* Convert Occupancy to cv::Mat Type */
            occGrid = cv::Mat::zeros(IMAX, JMAX, CV_8UC1);
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    if(occ[i*JMAX+j]==-1.0f) occGrid.at<u_char>(i, j) = 255;
                }
            }

            /* Run Dense Inverse Search (DIS) Optical Flow */
            dis->calc(occGrid0, occGrid, flow);
            occGrid.copyTo(occGrid0);

            /* Split Flow into i and j Components */
            float occ_vj_new[IMAX*JMAX];
            float occ_vi_new[IMAX*JMAX];
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    cv::Vec2f flow_ij = flow.at<cv::Vec2f>(i,j);
                    occ_vj_new[i*JMAX+j] = flow_ij[0] / dt_grid;
                    occ_vi_new[i*JMAX+j] = flow_ij[1] / dt_grid;
                }
            }

            /* 2nd Order Low Pass Filter */
            const float wf = 20.0f;
            const float kf = 1.0f - expf(-wf*dt_grid);
            for(int n = 0; n < IMAX*JMAX; n++){
                occ_vj_old[n] *= 1.0f - kf;
                occ_vi_old[n] *= 1.0f - kf;
                occ_vj_old[n] += kf * occ_vj_new[n];
                occ_vi_old[n] += kf * occ_vi_new[n];
                occ_vj[n] *= 1.0f - kf;
                occ_vi[n] *= 1.0f - kf;
                occ_vj[n] += kf * occ_vj_old[n];
                occ_vi[n] += kf * occ_vi_old[n];
            }

        //     /* View and Analyze Flow */
        //     cv::Mat flowMag = cv::Mat::zeros(IMAX, JMAX, CV_32FC1);
        //     float flow_max = 0.0f;
        //    for (int i = 0; i < IMAX; i++){
        //         for (int j = 0; j < JMAX; j++){
        //             cv::Vec2f flow_ij = flow.at<cv::Vec2f>(i,j);
        //             flowMag.at<float>(i,j) = sqrtf(flow_ij[0]*flow_ij[0] + flow_ij[1]*flow_ij[1]);
        //             flowMag.at<float>(i,j) *= DS / dt_grid;
        //             flow_max = std::max(flow_max, flowMag.at<float>(i,j));
        //         }
        //     }
        //     cv::Mat display_img;
        //     cv::normalize(flowMag, display_img, 0, 255, cv::NORM_MINMAX);
        //     display_img.convertTo(display_img, CV_8U);
        //     cv::Mat flipped_img;
        //     cv::flip(display_img, flipped_img, 0);
        //     cv::Mat resized_img;
        //     cv::resize(flipped_img, resized_img, cv::Size(720, 720));
        //     cv::imshow("My Image Window", resized_img);
        //     cv::waitKey(1);
        //     std::cout << flow_max << std::endl;

        };

        /* Propogate Future Occupancy Grids */
        void propogate_occupancy_grid(float *bound, const float *occ, const float *occ_vi, const float *occ_vj, const int k){
            
            for(int n = 0; n < IMAX*JMAX; n++){
                bound[n] = 1.0f;
            }
            
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(occ[i*JMAX+j]==-1.0f){
                        const float vbi = rydot / DS;
                        const float vbj = rxdot / DS;
                        const float if_new = (float)i + (occ_vi[i*JMAX+j] + vbi) * ((float)k * DT + grid_age/2.0f);
                        const float jf_new = (float)j + (occ_vj[i*JMAX+j] + vbj) * ((float)k * DT + grid_age/2.0f);
                        const int i_new = std::min(std::max((int)roundf(if_new), 0), IMAX-1);
                        const int j_new = std::min(std::max((int)roundf(jf_new), 0), JMAX-1);
                        bound[i_new*JMAX+j_new] = -1.0f;
                    }
                }
            }
        
        };

        /* Precompute Polar Indices for Occupancy Map */
        void polar_occ(int *occ_r2, float *occ_th){

            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    const int io = i - IMAX/2;
                    const int io2 = io * io;
                    const int jo = j - JMAX/2;
                    const int jo2 = jo * jo;
                    occ_r2[i*JMAX+j] = io2 + jo2;
                    occ_th[i*JMAX+j] = atan2f((float)io, (float)jo);
                }
            }

        };

        /* Fill Occupancy Map for Unseen Regions */
        void fill_to_back(float *bound){

          float b0[IMAX*JMAX];
          memcpy(b0, bound, IMAX*JMAX*sizeof(float));

          for(int i = 1; i < IMAX-1; i++){
            for(int j = 1; j < JMAX-1; j++){
              const int ro2 = occ_r2[i*JMAX+j];
              if(!b0[i*JMAX+j]){
                const float tho = occ_th[i*JMAX+j];
                int p_start = 0;
                int p_stop = IMAX;
                if(i > IMAX/2) p_start = IMAX/2;
                else p_stop = IMAX/2 + 1;
                int q_start = 0;
                int q_stop = JMAX;
                if(j > JMAX/2) q_start = JMAX/2;
                else q_stop = JMAX/2 + 1;
                for(int p = p_start; p < p_stop; p++){
                  for(int q = q_start; q < q_stop; q++){
                    if(bound[p*JMAX+q]>0.0f){
                      const float dth = occ_th[p*JMAX+q] - tho;
                      if((occ_r2[p*JMAX+q]>ro2) && (dth<0.04f) && (dth>-0.04f)) bound[p*JMAX+q] = -1.0f;
                    }
                  }
                }
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
        int build_robot_kernel(float*& kernel){
            
            const float length = 0.76f; // Go2
            const float width = 0.35f;
            //const float length = 0.35f; // G1
            //const float width = 1.02f;
            //const float length = 0.40f; // Drone
            //const float width = 0.40f;
            
            const float MOS = 1.25f;
            const float ar = MOS * length / 2.0f;
            const float br = MOS * width / 2.0f;
            const float expo = 2.0f;
            const float D = 2.0f * sqrtf(ar*ar + br*br); // Max Robot Dimension to Define Kernel Size
            const int dim = ceilf(ceilf(D / DS) / 2.0f) * 2.0f; //Make Sure Kernel Dimension is Even
            kernel = (float *)malloc(dim*dim*QMAX*sizeof(float));
            
            for(int q=0; q<QMAX; q++){
                const float yawq = q_to_yaw(q, yawc);
                for(int i = 0; i < dim; i++){
                    const float yi = (float)(i-dim/2)*DS;
                    for(int j = 0; j < dim; j++){
                        kernel[q*dim*dim+i*dim+j] = 0.0;
                        const float xi = (float)(j-dim/2)*DS;
                        const float xb = cosf(yawq)*xi + sinf(yawq)*yi;
                        const float yb = -sinf(yawq)*xi + cosf(yawq)*yi;
                        const float dist = powf(fabsf(xb/ar), expo) + powf(fabsf(yb/br), expo);
                        if(dist <= 1.0f) kernel[q*dim*dim+i*dim+j] = -1.0f;
                    }
                }
            }

            return dim;

        };

        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, float *kernel){
            
            /* Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));

            int lim = (robot_kernel_dim - 1)/2;
            for(int i = 1; i < IMAX-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 1; j < JMAX-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, JMAX);
                    if(!b0[i*JMAX+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*JMAX+q] += kernel[(p-i+lim)*robot_kernel_dim+(q-j+lim)];
                            }
                        }
                    }
                }
            }
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] < -1.0f) bound[n] = -1.0f;
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

                return Kernel::poissonSolve(grid, force, bound, relTol, w_SOR); // CUDA!

            }

        };

        /* Compute the Poisson Safety Function */
        void solve_poisson_safety_function(float *grid, float *occ_vi, float *occ_vj, const float *occ){
            
            const bool gpu_flag = true;

            optical_flow(occ_vi, occ_vj, occ);
            
            #pragma omp parallel for
            for(int k=0; k<TMAX; k++){

                float *bound_slice_q0 = (float *)malloc(IMAX*JMAX*sizeof(float));
                memcpy(bound_slice_q0, occ, IMAX*JMAX*sizeof(float));
                propogate_occupancy_grid(bound_slice_q0, occ, occ_vi, occ_vj, k);
                find_boundary(bound_slice_q0);
                
                //fill_to_back(bound_slice_q0);
                //find_boundary(bound_slice_q0);
                
                for(int q=0; q<QMAX; q++){

                    float *force_slice = force + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *bound_slice = bound + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *grid_slice = grid + k*IMAX*JMAX*QMAX + q*IMAX*JMAX;
                    float *kernel_slice = robot_kernel + q*robot_kernel_dim*robot_kernel_dim;

                    memcpy(bound_slice, bound_slice_q0, IMAX*JMAX*sizeof(float));
                    inflate_occupancy_grid(bound_slice, kernel_slice);
                    find_and_fix_boundary(grid_slice, bound_slice);
                    compute_fast_forcing_function(force_slice, bound_slice);
                    
                }
                
                free(bound_slice_q0);

            }

            const float h_RelTol = 1.0e-4f;
            // Start Solve Timer
            Timer poisson_timer(true);
            poisson_timer.start();
            h_iters = poisson(grid, force, bound, h_RelTol, 25.0f, gpu_flag);
            poisson_timer.time("Poisson Time: ");           

        };

        void safety_filter(const float rx, const float ry, const float yaw, const float dt, const bool filter_flag){

            // Fractional Index Corresponding to Current Position
            const float ir = y_to_i(ry,ryc);
            const float jr = x_to_j(rx,rxc);
            const float qr = yaw_to_q(yaw,yawc);

            // Saturated Because of Finite Grid Size
            const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
            const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
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
                        
            // If You Have Left The Grid, Use SDF to Get Back
            //if((ic!=ir) && (jc!=jr)){
            //    h -= sqrtf((ir-ic)*(ir-ic) + (jr-jc)*(jr-jc)) * DS;
            //}
            //else if(ic!=ir){
            //    h -= fabsf(ir-ic) * DS;
            //}
            //else if(jc!=jr){
            //    h -= fabsf(jr-jc) * DS;
            //}

            // Compute Gradients
            const float i_eps = 5.0f;
            const float j_eps = 5.0f;
            const float q_eps = 1.0f;
            const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
            const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
            const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
            const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));
            const float qp = q_wrap(qc + q_eps);
            const float qm = q_wrap(qc - q_eps);

            const float dhdx0 = (trilinear_interpolation(hgrid0, ic, jp, qc) - trilinear_interpolation(hgrid0, ic, jm, qc)) / ((jp-jm)*DS);
            const float dhdy0 = (trilinear_interpolation(hgrid0, ip, jc, qc) - trilinear_interpolation(hgrid0, im, jc, qc)) / ((ip-im)*DS);
            const float dhdyaw0 = (trilinear_interpolation(hgrid0, ic, jc, qp) - trilinear_interpolation(hgrid0, ic, jc, qm)) / (2.0f*q_eps*DQ);
            const float dhdx1 = (trilinear_interpolation(hgrid1, ic, jp, qc) - trilinear_interpolation(hgrid1, ic, jm, qc)) / ((jp-jm)*DS);
            const float dhdy1 = (trilinear_interpolation(hgrid1, ip, jc, qc) - trilinear_interpolation(hgrid1, im, jc, qc)) / ((ip-im)*DS);
            const float dhdyaw1 = (trilinear_interpolation(hgrid1, ic, jc, qp) - trilinear_interpolation(hgrid1, ic, jc, qm)) / (2.0f*q_eps*DQ);
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
            
            if(h_flag && qp_init_flag){

                const float xd[3] = {rxd, ryd, yawd};
                const float x[3] = {rx, ry, yaw};
                const float xc[3] = {rxc, ryc, yawc};

                // Perform SQP Iterations
                const int sqp_iters = 5;
                mpc_controller.initialize_solution(xd, x);
                for(int i=0; i<sqp_iters; i++){
                    mpc_controller.update_cost_and_constraints(hgrid, xc);
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

        void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){

            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Compute Grid Timing
            dt_grid0 = std::chrono::high_resolution_clock::now() - t_grid;
            t_grid = std::chrono::high_resolution_clock::now();
            dt_grid = dt_grid0.count() * 1.0e-9f;
            grid_age = dt_grid + 0.02f;
            
            // Update Occupancy
            const float sin_yawc = 2.0f * (msg->info.origin.orientation.w * msg->info.origin.orientation.z); 
            const float cos_yawc = 1.0f - 2.0f * msg->info.origin.orientation.z * msg->info.origin.orientation.z;
            yawc = atan2f(sin_yawc, cos_yawc);
            if(yawc<0.0f) yawc += 2.0f*M_PI;
            rxc = msg->info.origin.position.x + 0.38f * cosf(yaw);
            ryc = msg->info.origin.position.y + 0.38f * sinf(yaw);
            for(int n = 0; n < IMAX*JMAX; n++){
                occ[n] = 1.0f;
                if(msg->data[n]) occ[n] = -1.0f;
            }

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            solve_poisson_safety_function(hgrid, occ_vi, occ_vj, occ);
            printf("Poisson Iterations: %u \n", h_iters);
            solve_timer.time("Grid Solve Time: ");
            h_flag = true;
            
            // Publish Poisson Safety Function Grid (Interpolated)
            const float qr = yaw_to_q(yaw,yawc);
            const float q1f = floorf(qr);
            const float q2f = ceilf(qr);
            const int q1 = (int)q_wrap(q1f);
            const int q2 = (int)q_wrap(q2f);
            //int k = TMAX-1;
            int k = 0;
            cv::Mat poisson_img = cv::Mat::zeros(IMAX, JMAX, CV_32FC1);
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    if(q1f!=q2f) hgrid_message.data[i*JMAX+j] = (q2f - qr) * hgrid[k*IMAX*JMAX*QMAX+q1*IMAX*JMAX+i*JMAX+j] + (qr - q1f) * hgrid[k*IMAX*JMAX*QMAX+q2*IMAX*JMAX+i*JMAX+j];
                    else hgrid_message.data[i*JMAX+j] = hgrid[k*IMAX*JMAX*QMAX+q1*IMAX*JMAX+i*JMAX+j];
                    poisson_img.at<float>(i,j) = hgrid_message.data[i*JMAX+j];
                }
            }
            this->hgrid_publisher_->publish(hgrid_message);

            /* View Poisson */
            cv::Mat gray_img;
            cv::normalize(poisson_img, gray_img, 0, 255, cv::NORM_MINMAX);
            gray_img.convertTo(gray_img, CV_8U);
            cv::Mat color_img;
            cv::applyColorMap(gray_img, color_img, cv::COLORMAP_HOT);
            cv::Mat resized_img;
            const int upscale = 12;
            cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);
            cv::Point curr_pt = cv::Point(upscale*x_to_j(rx,rxc),upscale*y_to_i(ry,ryc));
            cv::Point goal_pt = cv::Point(upscale*x_to_j(rxd,rxc),upscale*y_to_i(ryd,ryc));
            cv::circle(resized_img, curr_pt, upscale, cv::Scalar(0, 0, 0), cv::FILLED); // Draw Goal Position
            cv::circle(resized_img, goal_pt, upscale, cv::Scalar(0, 127, 0), cv::FILLED); // Draw Goal Position
            for(int k = 1; k < TMAX; k++){
                cv::Point traj_pt = cv::Point(upscale*(x_to_j(mpc_message.data[3*k+0],rxc)),upscale*(y_to_i(mpc_message.data[3*k+1],ryc)));
                cv::circle(resized_img, traj_pt, upscale/2, cv::Scalar(255, 0, 0), cv::FILLED); // Draw MPC Trajectory
            }
            cv::Mat flipped_img;
            cv::flip(resized_img, flipped_img, 0);
            cv::imshow("Poisson Solution", flipped_img);
            cv::waitKey(1);

            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vx << "," << vy << "," << vyaw << ">" << std::endl;
            //save_flag = writeDataToFile(save_flag, hgrid, IMAX*JMAX, "poisson_safety_grid.csv");
        
        };

        void state_update_callback(geometry_msgs::msg::PoseStamped::UniquePtr data){
            
            // Increment Age of Latest Grid
            dt_state0 = std::chrono::high_resolution_clock::now() - t_state;
            t_state = std::chrono::high_resolution_clock::now();
            dt_state = dt_state0.count() * 1.0e-9f;
            
            grid_age += dt_state;
            mpc_age += dt_state;
            
            // Interpret State
            rxdot = (data->pose.position.x - rx) / dt_state;
            rydot = (data->pose.position.y - ry) / dt_state;
            rx = data->pose.position.x;
            ry = data->pose.position.y;
            float sin_yaw = 2.0f * (data->pose.orientation.w * data->pose.orientation.z); 
            float cos_yaw = 1.0f - 2.0f * data->pose.orientation.z * data->pose.orientation.z;
            yaw = atan2f(sin_yaw, cos_yaw);
            if(yaw<0.0f) yaw += 2.0f*M_PI;

            // Apply Nominal Control & Safety Filter
            nominal_controller(rx, ry, yaw);
            if(h_flag){
                if(qp_init_flag){
                    mpc_controller.set_input(&vx, &vy, &vyaw, mpc_age);
                }
                else{
                    filter_flag = !qp_init_flag;
                }
                safety_filter(rx, ry, yaw, grid_age, filter_flag);
            }

            // Check for Valid Control Action
            bool valid_flag = true;
            if(fabsf(vx)>100.0f || fabsf(vy)>100.0f || fabsf(vyaw)>100.0f) valid_flag = false;

            //Publish Control Action
            if(valid_flag){
                u_message.data[0] = rx;
                u_message.data[1] = ry;
                u_message.data[2] = yaw;
                u_message.data[3] = vx;
                u_message.data[4] = vy;
                u_message.data[5] = vyaw;
                u_message.data[6] = vxs;
                u_message.data[7] = vys;
                u_message.data[8] = vyaws;
                u_message.data[9] = h;
                this->u_publisher_->publish(u_message);
            }
            else{
                qp_init_flag = false;
                const float xd[3] = {rxd, ryd, yawd};
                const float x[3] = {rx, ry, yaw};
                mpc_controller.reset_QP(xd, x);
                qp_init_flag = true;
            }

        };
        
        MPC mpc_controller;
        cv::Ptr<cv::DISOpticalFlow> dis;

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        bool save_flag = false;
        bool h_flag = false;
        bool filter_flag = false;
        bool mpc_fail_flag = false;
        bool qp_init_flag = false;

        int vx_iters, vy_iters, h_iters;
        
        // Define State
        float rx = 0.0f;
        float ry = 0.0f;
        float yaw = 0.0f;
        float rxdot = 0.0f;
        float rydot = 0.0f;

        // Define Reference Trajectory
        float rxd = 0.0f;
        float ryd = 0.0f;
        float yawd = 0.0f;

        // Define Occupancy Grid Origin
        float rxc = -2.00f;
        float ryc = -2.00f;
        float yawc = 0.0f;

        std::chrono::high_resolution_clock::time_point t_grid, t_state;
        std::chrono::duration<float, std::nano> dt_grid0, dt_state0;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;
        float grid_age = 0.0f;
        float mpc_age = 0.0f;

        float vx, vy, vyaw;
        float vxs, vys, vyaws;
        float h, dhdt, dhdyaw, dhdx, dhdy;

        float i_obs, j_obs, i_obs0, j_obs0;
        float vi_obs = 0.0f;
        float vj_obs = 0.0f;
        bool v_obs_flag = false;
        
        float occ[IMAX*JMAX];
        int occ_r2[IMAX*JMAX];
        float occ_th[IMAX*JMAX];
        float occ_vi[IMAX*JMAX];
        float occ_vj[IMAX*JMAX];
        float occ_vi_old[IMAX*JMAX];
        float occ_vj_old[IMAX*JMAX];
        float *hgrid, *bound, *force;
        float *robot_kernel;
        int robot_kernel_dim;
        cv::Mat occGrid0, occGrid, flow;
        
        std_msgs::msg::Float32MultiArray hgrid_message;
        std_msgs::msg::Float32MultiArray u_message;
        std_msgs::msg::Float32MultiArray mpc_message;

        rclcpp::TimerBase::SharedPtr mpc_timer_;
        rclcpp::TimerBase::SharedPtr ref_timer_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr u_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mpc_publisher_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr optflow_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;
    
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