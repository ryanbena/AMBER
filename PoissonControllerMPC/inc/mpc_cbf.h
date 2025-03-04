#include <iostream>
#include "math.h"
#include "../inc/poisson.h"
#include "../inc/utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define LINEAR_STATE_LENGTH 2
#define LINEAR_INPUT_LENGTH 2

// MPC Controller Class
class MPC{
    
    public: 
        
        MPC(void){
            
            N_HORIZON = tmax - 1;
            nX = (N_HORIZON+1) * LINEAR_STATE_LENGTH;
            nU = N_HORIZON * LINEAR_INPUT_LENGTH;
            nZ = nX + nU; // Optimization Variables
            nC = nX + N_HORIZON;
           
            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            constraint_upper = Eigen::VectorXd::Zero(nC);
            sol = Eigen::VectorXd::Zero(nZ);

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH);
            Px.row(0) << 2.0, 0.0;
            Px.row(1) << 0.0, 2.0;

            Pu.setIdentity(LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH); 
            Pu.row(0) << 1.0, 0.0;
            Pu.row(1) << 0.0, 1.0;

            for(int k=0; k<N_HORIZON; k++){ 
                cost_P.block(k*LINEAR_STATE_LENGTH, k*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) = Px;
                cost_P.block(nX + k*LINEAR_INPUT_LENGTH, nX + k*LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH) = Pu;
            }
            cost_P.block(N_HORIZON*LINEAR_STATE_LENGTH, N_HORIZON*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) = 1.0*Px;
            
            // Initial Condition Constraint
            constraint_A.block<LINEAR_STATE_LENGTH,LINEAR_STATE_LENGTH>(0,0).setIdentity();

            // Dynamic constraints
            Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH);
            Eigen::MatrixXd Bd = Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH) * DT;
            for (int k=0; k<N_HORIZON; k++){
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>((k+1)*LINEAR_STATE_LENGTH, (k+1)*LINEAR_STATE_LENGTH) = -1.0f * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH); // Identity 
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>((k+1)*LINEAR_STATE_LENGTH, k*LINEAR_STATE_LENGTH) = Ad; // A Matrix
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH>((k+1)*LINEAR_STATE_LENGTH, nX + k*LINEAR_INPUT_LENGTH) = Bd; // B Matrix
            }
    
            for (int i=0; i<N_HORIZON; i++){
                constraint_upper(i + nX) = OSQP_INFTY; 
                constraint_lower(i + nX) = -OSQP_INFTY; // Turn off safety constraints
            }

        }

        int N_HORIZON;
        int nX, nU, nZ, nC;
        Eigen::MatrixXd Px, Pu;
        Eigen::MatrixXd cost_P;
        Eigen::VectorXd cost_q;
        Eigen::MatrixXd constraint_A;
        Eigen::VectorXd constraint_lower;
        Eigen::VectorXd constraint_upper;
        Eigen::VectorXd sol; 

        OsqpEigen::Solver solver;

        int setup_QP(){
            
            solver.settings()->setWarmStart(true);
            solver.settings()->setVerbosity(false); // Turn off printing
            solver.settings()->setMaxIteration(100);
            solver.data()->setNumberOfVariables(nZ); 
            solver.data()->setNumberOfConstraints(nC);
            
            Eigen::SparseMatrix<double> cost_P_sparse = cost_P.sparseView();
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();

            if(!solver.data()->setHessianMatrix(cost_P_sparse)) return 1; 
            if(!solver.data()->setGradient(cost_q)) return 1; 
            if(!solver.data()->setLinearConstraintsMatrix(constraint_A_sparse)) return 1; 
            if(!solver.data()->setLowerBound(constraint_lower)) return 1; 
            if(!solver.data()->setUpperBound(constraint_upper)) return 1; 
            if (!solver.initSolver()) return 1; 
            return 0;

        }

        int solve(){

            if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
            sol = solver.getSolution();
            return 0;
            
        }

        void extract_solution(float *z){

            for(int i = 0; i < nZ; i++){
                z[i] = sol(i);
            }

        }

        int update_constraints(const float *x0, const float *hgrid){
            
            const float wn = 0.1f;
            const float alpha = expf(-wn * DT);

            // Dynamics constraints
            constraint_lower.segment(0, LINEAR_STATE_LENGTH) << x0[0], x0[1]; 
            constraint_upper.segment(0, LINEAR_STATE_LENGTH) << x0[0], x0[1];

            // Obstacle State Constraints
            for (int k=0; k<N_HORIZON; k++){
                
                const int kp1 = k + 1;
                
                const int idx_k = k * LINEAR_STATE_LENGTH; 
                const int idx_kp1 = kp1 * LINEAR_STATE_LENGTH; 
                
                const Eigen::VectorXd x_k = sol.segment(idx_k, LINEAR_STATE_LENGTH);
                const Eigen::VectorXd x_kp1 = sol.segment(idx_kp1, LINEAR_STATE_LENGTH);

                const float *hgrid_k = hgrid + k*imax*jmax;
                const float *hgrid_kp1 = hgrid + kp1*imax*jmax;

                const float x_eps = 1.0e-3f; // Small Perturbation for Numerical Gradients (meters)
                const float y_eps = 1.0e-3f; // Small Perturbation for Numerical Gradients (meters)
                const float i_eps = x_eps / ds;
                const float j_eps = y_eps / ds;

                // Fractional Index Corresponding to Current Position
                const float ir_k = (float)imax - x_k(1) / ds;
                const float jr_k = x_k(0) / ds;
                const float ic_k = fminf(fmaxf(i_eps, ir_k), (float)(imax-1)-i_eps); // Saturated Because of Finite Grid Size
                const float jc_k = fminf(fmaxf(i_eps, jr_k), (float)(jmax-1)-i_eps); // Numerical Derivatives Shrink Effective Grid Size

                // Get Safety Function Value
                const float cbf_k = bilinear_interpolation(hgrid_k, ic_k, jc_k);

                // Compute Gradients
                const float ip_k = ic_k - i_eps;
                const float im_k = ic_k + i_eps;
                const float jp_k = jc_k + j_eps;
                const float jm_k = jc_k - j_eps;

                Eigen::VectorXd dhdx_k(LINEAR_STATE_LENGTH);
                dhdx_k(0) = (bilinear_interpolation(hgrid_k, ic_k, jp_k) - bilinear_interpolation(hgrid_k, ic_k, jm_k)) / (2.0f * x_eps);
                dhdx_k(1) = (bilinear_interpolation(hgrid_k, ip_k, jc_k) - bilinear_interpolation(hgrid_k, im_k, jc_k)) / (2.0f * y_eps);

                // Fractional Index Corresponding to Current Position
                const float ir_kp1 = (float)imax - x_kp1(1) / ds;
                const float jr_kp1 = x_kp1(0) / ds;
                const float ic_kp1 = fminf(fmaxf(i_eps, ir_kp1), (float)(imax-1)-i_eps); // Saturated Because of Finite Grid Size
                const float jc_kp1 = fminf(fmaxf(i_eps, jr_kp1), (float)(jmax-1)-i_eps); // Numerical Derivatives Shrink Effective Grid Size

                // Get Safety Function Value
                const float cbf_kp1 = bilinear_interpolation(hgrid_kp1, ic_kp1, jc_kp1);

                // Compute Gradients
                const float ip_kp1 = ic_kp1 - i_eps;
                const float im_kp1 = ic_kp1 + i_eps;
                const float jp_kp1 = jc_kp1 + j_eps;
                const float jm_kp1 = jc_kp1 - j_eps;

                Eigen::VectorXd dhdx_kp1(LINEAR_STATE_LENGTH);
                dhdx_kp1(0) = (bilinear_interpolation(hgrid_kp1, ic_kp1, jp_kp1) - bilinear_interpolation(hgrid_kp1, ic_kp1, jm_kp1)) / (2.0f * x_eps);
                dhdx_kp1(1) = (bilinear_interpolation(hgrid_kp1, ip_kp1, jc_kp1) - bilinear_interpolation(hgrid_kp1, im_kp1, jc_kp1)) / (2.0f * y_eps);
            
                // Update A matrix
                constraint_A.block<1, LINEAR_STATE_LENGTH>(nX+k, idx_kp1) = dhdx_kp1;
                constraint_A.block<1, LINEAR_STATE_LENGTH>(nX+k, idx_k)   = -alpha * dhdx_k;

                // Update Bounds
                constraint_lower(nX+k) = dhdx_kp1.dot(x_kp1);
                constraint_lower(nX+k) -= alpha * dhdx_k.dot(x_k);
                constraint_lower(nX+k) -= cbf_kp1;
                constraint_lower(nX+k) += alpha * cbf_k;
                
                constraint_upper(nX+k)  = OSQP_INFTY;
                //constraint_lower(nX+k)  = -OSQP_INFTY; // To turn off safety constraints

            }

            
            //std::cout<<"<"<<x0[0]<<","<<x0[1]<<","<<x0[2]<<">"<<std::endl;

            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();
            solver.updateLinearConstraintsMatrix(constraint_A_sparse); 
            solver.updateBounds(constraint_lower, constraint_upper); 

            return 1; 
        }


        int update_cost(const float *xd, const float *x){ 
            
            Eigen::VectorXd state_goal(LINEAR_STATE_LENGTH);
            Eigen::VectorXd state_curr(LINEAR_STATE_LENGTH);
            state_goal << xd[0] , xd[1];
            state_curr << x[0] , x[1];
            
            for (int i=0; i<=N_HORIZON; i++){
                cost_q.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) = -state_goal.transpose() * Px;
            }
            solver.updateGradient(cost_q);

            const float previous_solution_weight = 0.0;
            sol *= previous_solution_weight;
            // Interpolate Solution from current state to desired final state
            for (int i=0; i<=N_HORIZON; i++){
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - previous_solution_weight) * state_goal * (float)i / (float)N_HORIZON; 
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - previous_solution_weight) * state_curr * (float)(N_HORIZON-i) / (float)N_HORIZON;
            }


            return 0;

        }

};




