#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include "../inc/kernel.hpp"
#include "../inc/Types.h"
#include "../inc/utils.h"

#include <Eigen/Core>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

const int imax = 120;
const int jmax = 120;
const double ds = 0.0254; // grid resolution

const double h0 = 0.0; // Set boundary level set value
const double dh0 = 1.0; // Set dh Value

bool save_flag = false;

int vx_iters, vy_iters, h_iters;

Eigen::MatrixXd f0 = Eigen::MatrixXd::Zero(imax, jmax);

void find_boundary(Eigen::MatrixXd &bound);
void zero_boundary(Eigen::MatrixXd &grid, Eigen::MatrixXd &bound);
void inflate_occupancy_grid(Eigen::MatrixXd &bound, const double yaw);
void compute_boundary_gradients(Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &bound);
double softMin(const double x0, const double xmin, const double alpha);
double softMax(const double x0, const double xmax, const double alpha);
void compute_forcing_function(Eigen::MatrixXd &force, Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &bound);
int poisson(Eigen::MatrixXd &grid, Eigen::MatrixXd &force, Eigen::MatrixXd &bound, const double RelTol, const double N, const bool gpu_flag);
void solve_poisson_safety_function(Eigen::MatrixXd &grid, Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &occ, const double yaw, const bool gpu_flag);
void querry_safety_function(const double rx, const double ry);
bool writeDataToFile(bool flag);

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void find_boundary(Eigen::MatrixXd &bound){
    
    // Set Border
    bound.block(0, 0, imax, 1).setConstant(0.0);
    bound.block(0, 0, 1, jmax).setConstant(0.0);
    bound.block(0, jmax-1, imax, 1).setConstant(0.0);
    bound.block(imax-1, 0, 1, jmax).setConstant(0.0);

    Eigen::MatrixXd b0(imax, jmax);
    memcpy(b0.data(), bound.data(), sizeof(double)*imax*jmax);
    for(int i = 1; i < imax-1; i++){
        for(int j = 1; j < jmax-1; j++){
            if(b0(i,j)==1.0){
                const double neighbors = b0(i+1,j) + b0(i-1,j) + b0(i,j+1) + b0(i,j-1) + b0(i+1,j+1) + b0(i-1,j+1) + b0(i-1,j-1) + b0(i+1,j-1);
                if(neighbors < 8.0){
                    bound(i,j) = 0.0;
                }
            }
        }
    }

}

/* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
void zero_boundary(Eigen::MatrixXd &grid, Eigen::MatrixXd &bound){
    
    for(int i = 0; i < imax; i++){
        for(int j = 0; j < jmax; j++){
            if(!bound(i,j)) grid(i,j) = h0;
        }
    }

}

/* Buffer Occupancy Grid with 2-D Robot Shape */
void inflate_occupancy_grid(Eigen::MatrixXd &bound, const double yaw){

    /* Step 1: Create Robot Kernel */
    const double go2_length = 0.80;
    const double go2_width = 0.40;
    //const double g1_radius = 0.25;
    
    const double D = sqrt(go2_length*go2_length + go2_width*go2_width); // Max Robot Dimension to Define Template Size
    int dim = ceil((ceil(D / ds) + 1.0) / 2.0) * 2.0 - 1.0;
    Eigen::MatrixXd robot_grid = Eigen::MatrixXd::Zero(dim, dim);

    const double MOS = 1.2;
    const double ar = MOS * go2_length / 2.0;
    const double br = MOS * go2_width / 2.0;
    
    for(int i = 0; i < dim; i++){
        const double xi = (double)i * ds - D/2.0;
        for(int j = 0; j < dim; j++){
            const double yi = (double)j * ds - D/2.0;
            
            const double xb = cos(yaw)*xi + sin(yaw)*yi;
            const double yb = -sin(yaw)*xi + cos(yaw)*yi;

            const double dist = pow(xb/br, 4.0) + pow(yb/ar, 4.0);
            if(dist <= 1.0){
                robot_grid(i,j) = -1.0;
            }
        }
    }
 
    /* Step 2: Convolve Robot Kernel with Occupancy Grid, Along the Boundary */
    Eigen::MatrixXd b0(imax, jmax);
    memcpy(b0.data(), bound.data(), sizeof(double)*imax*jmax);

    int lim = (dim - 1)/2;
    for(int i = 1; i < imax-1; i++){
        int ilow = std::max(i - lim, 0);
        int itop = std::min(i + lim, imax);
        for(int j = 1; j < jmax-1; j++){
            int jlow = std::max(j - lim, 0);
            int jtop = std::min(j + lim, jmax);
            if(!b0(i,j)){
                for(int p = ilow; p < itop; p++){
                    for(int q = jlow; q < jtop; q++){
                        bound(p,q) += robot_grid(p-i+lim, q-j+lim);
                    }
                }
            }
        }
    }
    for(int i = 0; i < imax; i++){
        for(int j = 0; j < jmax; j++){
            if(bound(i,j) < -1.0) bound(i,j) = -1.0;
        }
    }

}

/* Using Occupancy Grid, Find Desired Boundary Gradients */
void compute_boundary_gradients(Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &bound){

    const int blend = 1; // How Many Pixels Will Be Used to Blend Gradients (>= 1, <= buffer)
    for(int i = blend; i < imax-blend; i++){
        for(int j = blend; j < jmax-blend; j++){
            if(!bound(i,j)){
                guidance_x(i,j) = 0.0;
                guidance_y(i,j) = 0.0;
                for(int p = -blend; p <= blend; p++){
                    for(int q = -blend; q <= blend; q++){
                        if(q > 0){
                            guidance_x(i,j) += bound(i+q,j+p);
                            guidance_y(i,j) += bound(i+p,j+q);
                        }
                        else if (q < 0){
                            guidance_x(i,j) -= bound(i+q,j+p);
                            guidance_y(i,j) -= bound(i+p,j+q);
                        }
                    }
                }
                const double V = sqrt(guidance_x(i,j)*guidance_x(i,j) + guidance_y(i,j)*guidance_y(i,j));
                if(V != 0.0){
                    guidance_x(i,j) *= dh0 / V;
                    guidance_y(i,j) *= dh0 / V;
                }
            }
        }
    }

}

double softMin(const double x0, const double xmin, const double alpha){
    double xf = xmin + log(1.0+exp(alpha*(x0-xmin))) / alpha;
    return xf;
}

double softMax(const double x0, const double xmax, const double alpha){
    double xf = xmax - log(1.0+exp(alpha*(xmax-x0))) / alpha;
    return xf;
}

/* Compute Forcing Function from Guidance Field */
void compute_forcing_function(Eigen::MatrixXd &force, Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &bound){

    const double max_div = 4.0;
    const double alpha = 2.0;

    for(int i = 1; i < (imax-1); i++){
        for(int j = 1; j < (jmax-1); j++){
            force(i,j) = (guidance_x(i+1,j) - guidance_x(i-1,j)) / (2.0*ds) + (guidance_y(i,j+1) - guidance_y(i,j-1)) / (2.0*ds);
            if(bound(i,j) > 0.0){
                force(i,j) = softMin(force(i,j), -max_div, alpha);
                force(i,j) = softMax(force(i,j), 0.0, alpha);
            }
            else if(bound(i,j) < 0.0){
                force(i,j) = softMax(force(i,j), max_div, alpha);
                force(i,j) = softMin(force(i,j), 0.0, alpha);
            }
            else{
                force(i,j) = 0.0;
            }
        }
    }

}

/* Solve Poisson's Equation -- Checkerboard Successive Overrelaxation (SOR) Method */
int poisson(Eigen::MatrixXd &grid, Eigen::MatrixXd &force, Eigen::MatrixXd &bound, const double relTol = 1.0e-4, const double N = 70.0, const bool gpu_flag = false){
    
    if(!gpu_flag){
        const double w_SOR = 2.0/(1.0+sin(M_PI/(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5

        int iters = 0;
        const int max_iters = 10000;
        double rss;
        for(int k = 0; k < max_iters; k++){

            // Checkerboard Pass
            rss = 0.0;
            for(int i = 1; i < imax-1; i++){
                for(int j = 1; j < jmax-1; j++){
                    bool red = (i%2)==(j%2);
                    if(bound(i,j) && red){
                        double dg = grid(i+1,j) + grid(i-1,j) + grid(i,j+1) + grid(i,j-1);
                        dg -= force(i,j) * ds * ds;
                        dg /= 4.0;
                        dg -= grid(i,j);
                        grid(i,j) += w_SOR * dg;
                        rss += dg * dg;
                    }
                }
            }
            for(int i = 1; i < imax-1; i++){
                for(int j = 1; j < jmax-1; j++){
                    bool black = (i%2)!=(j%2);
                    if(bound(i,j) && black){
                        double dg = grid(i+1,j) + grid(i-1,j) + grid(i,j+1) + grid(i,j-1);
                        dg -= force(i,j) * ds * ds;
                        dg /= 4.0;
                        dg -= grid(i,j);
                        grid(i,j) += w_SOR * dg;
                        rss += dg * dg;
                    }
                }
            }
            
            rss = sqrt(rss) * ds;
            iters++;
            if(rss < relTol) break;
        
        }

        return iters;
    }
    else{

        return Kernel::Poisson(grid.data(), force.data(), bound.data(), relTol, imax, jmax, ds); // CUDA!

    }

}

/* Compute the Poisson Safety Function */
void solve_poisson_safety_function(Eigen::MatrixXd &grid, Eigen::MatrixXd &guidance_x, Eigen::MatrixXd &guidance_y, Eigen::MatrixXd &occ, const double yaw, const bool gpu_flag){
    
    Eigen::MatrixXd bound(imax, jmax);
    memcpy(bound.data(), occ.data(), sizeof(double)*imax*jmax);
    
    find_boundary(bound);
    inflate_occupancy_grid(bound, yaw);
    find_boundary(bound);
    zero_boundary(grid, bound);
    compute_boundary_gradients(guidance_x, guidance_y, bound);
    
    const double v_RelTol = 1.0e-4;
    vx_iters = poisson(guidance_x, f0, bound, v_RelTol, 45.0, gpu_flag);
    vy_iters = poisson(guidance_y, f0, bound, v_RelTol, 45.0, gpu_flag);
    
    Eigen::MatrixXd force(imax, jmax);
    compute_forcing_function(force, guidance_x, guidance_y, bound);

    const double h_RelTol = 1.0e-4;
    h_iters = poisson(grid, force, bound, h_RelTol, 70.0, gpu_flag);

}

/* Perform a bilinear interpolation on a 2-D grid */
double bilinear_interpolation(Eigen::MatrixXd &grid, const double i, const double j){

    double f, f1, f2, f11, f12, f21, f22;
    
    const double i1 = floor(i);
    const double j1 = floor(j);
    const double i2 = ceil(i);
    const double j2 = ceil(j);

    if((i1 != i2) && (j1 != j2)){
        f11 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j1);
        f12 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j2);
        f21 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j1);
        f22 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j2);
        f1 = (j2 - j) / (j2 - j1) * (f11 + f21);
        f2 = (j - j1) / (j2 - j1) * (f12 + f22);
        f = f1 + f2;
    }
    else if(i1 != i2){
        f1 = (i2 - i) / (i2 - i1) * grid((int)i1,(int)j);
        f2 = (i - i1) / (i2 - i1) * grid((int)i2,(int)j);
        f = f1 + f2;
    }
    else if(j1 != j2){
        f1 = (j2 - j) / (j2 - j1) * grid((int)i,(int)j1);
        f2 = (j - j1) / (j2 - j1) * grid((int)i,(int)j2);
        f = f1 + f2;
    }
    else{
        f = grid((int)i,(int)j);
    }

    return f;

}

class PoissonSolverNode : public rclcpp::Node{

    public:
        
        PoissonSolverNode() : Node("poisson_solver"){
            
            tgrid = std::chrono::high_resolution_clock::now();
            sleep(1);
            dtgrid0 = std::chrono::high_resolution_clock::now() - tgrid;
            dtgrid = 1.0e10;

            yaw0 = 0.0;
            hgrid.setConstant(h0);
            
            hgrid_message.data.resize(imax*jmax);
            hgridy_message.data.resize(imax*jmax);
            //hdata_message.data.resize(5);
                        
            hgrid_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("safety_grid_topic", 1);
            hgridy_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("eps_safety_grid_topic", 1);
            //hdata_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("safety_data_topic", 10);
            occ_grid_suber_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>("occ_grid_topic", 1, std::bind(&PoissonSolverNode::occ_grid_callback, this, std::placeholders::_1));
            pose_suber = this->create_subscription<geometry_msgs::msg::PoseStamped>("/MacLane/pose", 1, std::bind(&PoissonSolverNode::optitrack_state_callback, this, std::placeholders::_1));

        }

    private:

        void querry_safety_function(const double rx, const double ry){

            // Fractional Index Corresponding to Current Position
            const double rc[2] = {0.0, 0.0}; // Location of OptiTrack Origin in Grid Frame
            double ir = (double)imax - (ry+rc[1]) / ds;
            double jr = (rx+rc[0]) / ds;

            const double x_eps = 1.0e-3; // Small Perturbation for Numerical Gradients (meters)
            double i_eps = x_eps / ds;

            double ic = fmin(fmax(i_eps, ir), (double)(imax-1)-i_eps); // Saturated Because Numerical Derivatives Shrink Effective Grid Size
            double jc = fmin(fmax(i_eps, jr), (double)(jmax-1)-i_eps);

            // Get Safety Function Value
            h = bilinear_interpolation(hgrid, ic, jc);
            double ht = bilinear_interpolation(hgridt, ic, jc);
            double hy = bilinear_interpolation(hgridy, ic, jc);

            // Compute Time Derivative
            double dhdt_raw = (h - ht) / dtgrid;
            const double wv = 10.0; // Low Pass Filter Cutoff
            double kv = 1.0 - exp(-wv*dtgrid);
            dhdt *= 1.0 - kv;
            dhdt += kv * dhdt_raw;

            // Compute Angular Derivative
            dhdyaw = (hy - h) / yaw_eps;

            // Compute Gradient
            double ip = ic - i_eps;
            double im = ic + i_eps;
            double jp = jc + i_eps;
            double jm = jc - i_eps;
            gradhx = (bilinear_interpolation(hgrid, ic, jp) - bilinear_interpolation(hgrid, ic, jm)) / (2.0 * x_eps);
            gradhy = (bilinear_interpolation(hgrid, ip, jc) - bilinear_interpolation(hgrid, im, jc)) / (2.0 * x_eps);

        }

        bool writeDataToFile(bool flag){

            if(!flag){
                const std::string& filename = "poisson_safety_grid.csv";
                std::ofstream outFile(filename);
                if(outFile.is_open()){
                    for(int i = 0; i < imax; i++){
                        for(int j = 0; j < jmax; j++){
                            outFile << hgrid(i,j) << std::endl;
                        }
                    }
                    outFile.close();
                } 
                else{
                    std::cerr << "Error: Could not open file " << filename << " for writing.\n";
                }
            }
            return true;

        }

        void occ_grid_callback(std_msgs::msg::UInt8MultiArray::UniquePtr msg){
            
            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Store Old Solution & Compute Loop Time
            memcpy(hgridt.data(), hgrid.data(), sizeof(double)*imax*jmax); 
            dtgrid0 = std::chrono::high_resolution_clock::now() - tgrid;
            tgrid = std::chrono::high_resolution_clock::now();
            dtgrid = dtgrid0.count() * 1.0e-9;
            std::cout << dtgrid << std::endl;
            
            // Assign Occupancy
            Eigen::MatrixXd occ = Eigen::MatrixXd::Ones(imax, jmax);
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    if(msg->data[jmax*i+j]) occ(i,j) = -1.0;
                }
            }

            // Solve Poisson Safety Function
            const bool gpu_flag = true;
            //yaw0 += 0.4;
            solve_poisson_safety_function(hgrid, vxgrid, vygrid, occ, yaw0, gpu_flag);

            // Publish Poisson Safety Function Grid
            for(int i = 0; i < imax; i++){
                for(int j = 0; j < jmax; j++){
                    hgrid_message.data[jmax*i+j] = hgrid(i,j);
                }
            }
            //memcpy(&hgrid_message.data[0], hgrid.data(), sizeof(double)*imax*jmax);
            this->hgrid_publisher_->publish(hgrid_message);

            // Print Solver Data
            int v_iters = vx_iters + vy_iters;
            printf("Laplace Iterations: %u \n", v_iters);
            printf("Poisson Iterations: %u \n", h_iters);

            // Solve Perturbed Poisson Safety Function
            //double yaw1 = yaw0 + yaw_eps;
            //memcpy(hgridy.data(), hgrid.data(), sizeof(double)*imax*jmax);
            //memcpy(vxgridy.data(), vxgrid.data(), sizeof(double)*imax*jmax);
            //memcpy(vygridy.data(), vygrid.data(), sizeof(double)*imax*jmax);
            //solve_poisson_safety_function(hgridy, vxgridy, vygridy, occ, yaw1, gpu_flag);

            // Publish Poisson Safety Function Grid
            //memcpy(&hgridy_message.data[0], hgridy.data(), sizeof(double)*imax*jmax);
            //for(int i = 0; i < imax; i++){
            //    for(int j = 0; j < jmax; j++){
            //        hgridy_message.data[jmax*i+j] = hgridy(i,j);
            //    }
            //}
            //this->hgridy_publisher_->publish(hgridy_message);

            //v_iters = vx_iters + vy_iters;
            //printf("Laplace Eps Iterations: %u \n", v_iters);
            //printf("Poisson Eps Iterations: %u \n", h_iters);

            solve_timer.time("Solve Time: ");

            //save_flag = writeDataToFile(save_flag);
        
        };

        void optitrack_state_callback(geometry_msgs::msg::PoseStamped::SharedPtr data){

            rx = data->pose.position.x;
            ry = data->pose.position.y;

            double sin_yaw = 2.0 * (data->pose.orientation.w * data->pose.orientation.z); 
            double cos_yaw = 1.0 - 2.0 * data->pose.orientation.z * data->pose.orientation.z;
            yaw0 = atan2(sin_yaw, cos_yaw);

            // Publish Poisson Safety Function Data
            //querry_safety_function(rx, ry);
            //hdata_message.data[0] = h;
            //hdata_message.data[1] = dhdt;
            //hdata_message.data[2] = dhdyaw;
            //hdata_message.data[3] = gradhx;
            //hdata_message.data[4] = gradhy;
            //this->hdata_publisher_->publish(hdata_message);

        };
        
        Eigen::MatrixXd hgrid = Eigen::MatrixXd::Zero(imax, jmax);
        Eigen::MatrixXd vxgrid = Eigen::MatrixXd::Zero(imax, jmax);
        Eigen::MatrixXd vygrid = Eigen::MatrixXd::Zero(imax, jmax);

        std::chrono::high_resolution_clock::time_point tgrid;
        std::chrono::duration<scalar_t, std::nano> dtgrid0;
        double dtgrid;

        double rx, ry, yaw0;
        double h, dhdt, dhdyaw, gradhx, gradhy;

        double yaw_eps = 1.0 * M_PI / 180.0;
        Eigen::MatrixXd hgridt = Eigen::MatrixXd::Zero(imax, jmax);
        Eigen::MatrixXd hgridy = Eigen::MatrixXd::Zero(imax, jmax);
        Eigen::MatrixXd vxgridy = Eigen::MatrixXd::Zero(imax, jmax);
        Eigen::MatrixXd vygridy = Eigen::MatrixXd::Zero(imax, jmax);
        
        std_msgs::msg::Float64MultiArray hgrid_message;
        std_msgs::msg::Float64MultiArray hgridy_message;
        //std_msgs::msg::Float64MultiArray hdata_message;

        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr hgrid_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr hgridy_publisher_;
        //rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr hdata_publisher_;
        rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_suber;
    
};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoissonSolverNode>());
    rclcpp::shutdown();

  return 0;

}