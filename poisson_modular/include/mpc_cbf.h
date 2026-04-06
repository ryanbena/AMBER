#pragma once

#include <iostream>
#include <algorithm>
#include "definitions.h"
#include "utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define STATES 3
#define INPUTS 3

// MPC Controller Class
class MPC3D{
    
    public: 
        
        MPC3D(void){

            solver.settings()->setVerbosity(false);
            solver.settings()->setWarmStart(true);
            solver.settings()->setMaxIteration(10000);
            
            nX = (N_HORIZON+1) * STATES;
            nU = N_HORIZON * INPUTS;
            nZ = nX + nU; // Optimization Variables
            nC = nZ + N_HORIZON;

            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_upper = Eigen::VectorXd::Zero(nC);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            sol = Eigen::VectorXd::Zero(nZ);
            xbar = Eigen::VectorXd::Zero(nX);
            ubar = Eigen::VectorXd::Zero(nU);
            
            Px.setIdentity(STATES, STATES);
            Px(0,0) *= 1.0;
            Px(1,1) *= 1.0;
            Px(2,2) *= 1.0;
            Pu.setIdentity(INPUTS, INPUTS);
            Pu(0,0) *= 1.0;
            Pu(1,1) *= 1.0;
            Pu(2,2) *= 1.0;
            for(int k=0; k<N_HORIZON; k++){
                cost_P.block<STATES, STATES>(k*STATES, k*STATES) = Px;
                cost_P.block<INPUTS, INPUTS>(nX + k*INPUTS, nX + k*INPUTS) = Pu;
            }
            cost_P.block<STATES, STATES>(N_HORIZON*STATES, N_HORIZON*STATES) = Px*terminal;
            
            // Build Constraints
            constraint_A.block<STATES,STATES>(0,0).setIdentity(); // Initial Condition Constraint
            for (int k=0; k<N_HORIZON; k++){
                
                const int idx = k * STATES;
                const int idxp1 = (k+1) * STATES;
                const int idu = k * INPUTS + nX;
                
                // Set Up Dynamic Constraints
                constraint_A.block<STATES, STATES>(idxp1, idxp1) = -Eigen::MatrixXd::Identity(STATES, STATES); // Identity 
                constraint_A.block<STATES, STATES>(idxp1, idx) = Eigen::MatrixXd::Identity(STATES, STATES); // A Matrix
                constraint_A.block<STATES, INPUTS>(idxp1, idu) = DT * Eigen::MatrixXd::Identity(STATES, INPUTS); // B Matrix
                
                // Set Up Saturation Constraints
                constraint_A.block<INPUTS, INPUTS>(idu, idu) = Eigen::MatrixXd::Identity(INPUTS, INPUTS);
                constraint_upper.segment(idu, INPUTS) << 1.0f, 1.0f, 1.0f; 
                constraint_lower.segment(idu, INPUTS) << -1.0f, -1.0f, -1.0f;
                
                // Set Up Safety Constraints
                const int idsf = k + nZ;
                constraint_upper(idsf) = OSQP_INFTY; 
                constraint_lower(idsf) = -OSQP_INFTY;    
                
            }

        }

        int nX, nU, nZ, nC;
        Eigen::MatrixXd Px;
        Eigen::MatrixXd Pu;
        Eigen::MatrixXd cost_P;
        Eigen::VectorXd cost_q;
        Eigen::MatrixXd constraint_A;
        Eigen::VectorXd constraint_lower, constraint_upper;
        Eigen::VectorXd sol;
        Eigen::VectorXd xbar;
        Eigen::VectorXd ubar;
        OsqpEigen::Solver solver;
        Eigen::Vector3d state_curr;
        const float terminal = 1.0f;

        int reset_QP(const Eigen::Vector3f& r){
            
            for(int k=0; k<=N_HORIZON; k++){
                xbar.segment(k*STATES, STATES) << r(0), r(1), r(2);
                if(k!=N_HORIZON) ubar.segment(k*INPUTS, INPUTS) << 0.0f, 0.0f, 0.0f;
            }

            solver.clearSolver();
            solver.data()->clearHessianMatrix();
            solver.data()->clearLinearConstraintsMatrix();
            
            solver.data()->setNumberOfVariables(nZ); 
            solver.data()->setNumberOfConstraints(nC);

            Eigen::SparseMatrix<double> cost_P_sparse = cost_P.sparseView(0.0, -1.0);
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView(0.0, -1.0);

            if(!solver.data()->setHessianMatrix(cost_P_sparse)) return 1; 
            if(!solver.data()->setGradient(cost_q)) return 1; 
            if(!solver.data()->setLinearConstraintsMatrix(constraint_A_sparse)) return 1; 
            if(!solver.data()->setLowerBound(constraint_lower)) return 1; 
            if(!solver.data()->setUpperBound(constraint_upper)) return 1; 
            if(!solver.initSolver()) return 1;

            return 0;

        }

        void shift_linearization(const Eigen::Vector3f& r){
 
            xbar.segment(0, STATES) << r(0), r(1), r(2);
            for(int k=1; k<N_HORIZON; k++){
                const int idk = k * STATES;
                const int idkp1 = (k+1) * STATES;
                xbar.segment(idk, STATES) = xbar.segment(idkp1, STATES);
                if(k==(N_HORIZON-1)) break;
                ubar.segment(idk, INPUTS) = ubar.segment(idkp1, INPUTS);
            }

        }
        
        void update_cost(const Eigen::Vector3f& xd, const Eigen::Vector3f& ud){

            Eigen::Vector3d state_goal{xd(0), xd(1), xd(2)};
            Eigen::Vector3d input_goal{0.0, 0.0, 0.0};
            for(int k=0; k<N_HORIZON; k++){
                cost_q.segment(k*STATES, STATES) = -Px * state_goal;
                cost_q.segment(k*INPUTS+nX, INPUTS) = -Pu * input_goal;
            }
            cost_q.segment(N_HORIZON*STATES, STATES) = -Px * state_goal * terminal;
            solver.updateGradient(cost_q);
            
        }

        void update_constraints(const float *h_grid, const float *dhdt_grid, const Eigen::Vector3f& r, const Eigen::Vector3f& rc, const float wn, const float issf){
   
            // Update Initial Condition Constraints
            state_curr << r(0), r(1), r(2);
            constraint_lower.segment(0, STATES) << state_curr(0), state_curr(1), state_curr(2);
            constraint_upper.segment(0, STATES) << state_curr(0), state_curr(1), state_curr(2);

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k * STATES;
                const int idu = k * STATES + nX;
                
                const float tk = k * DT;
                Eigen::Vector3f rk{(float)xbar(idx+0), (float)xbar(idx+1), (float)xbar(idx+2)};
            
                // Get Current Safety Function Value & Rate
                const float h = extract_grid_value(h_grid, rk(0), rk(1), rc(0), rc(1));
                const float dhdt = extract_grid_value(dhdt_grid, rk(0), rk(1), rc(0), rc(1));
                Eigen::Vector3f dhdr = extract_grid_gradient(h_grid, rk(0), rk(1), rc(0), rc(1));

                // Update Constraints
                const float alpha = std::exp(-wn*DT);
                const int idsf = k + nZ;
                const int idsfm1 = idsf - 1;
                if(k!=N_HORIZON){
                    constraint_A.block<1,STATES>(idsf, idx) << -alpha*dhdr(0), -alpha*dhdr(1), -alpha*dhdr(2);
                    constraint_lower(idsf) = -alpha * (dhdr.dot(rk) - h);
                    // ISSf Term
                    const float Lgh_norm = std::sqrt(dhdr.dot(dhdr));
                    float ISSf = std::pow(Lgh_norm + 0.5f, 2.0f) / issf;
                    ISSf *= k / N_HORIZON;
                    constraint_lower(idsf) += ISSf * DT;
                }
                if(k!=0){
                    constraint_A.block<1,STATES>(idsfm1, idx) << dhdr(0), dhdr(1), dhdr(2);
                    constraint_lower(idsfm1) += dhdr.dot(rk) - h;
                }

            }
            
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView(0.0, -1.0);
            solver.updateLinearConstraintsMatrix(constraint_A_sparse);
            solver.updateBounds(constraint_lower, constraint_upper);

        }

        void solve(void){

            solver.solveProblem();
            OsqpEigen::Status status = solver.getStatus();
            if(status == OsqpEigen::Status::Solved || status == OsqpEigen::Status::SolvedInaccurate || status == OsqpEigen::Status::MaxIterReached){
                sol = solver.getSolution();
            }
            else{
                std::cout << "QP Solver Error!!!!!!" << std::endl;
                reset_QP(state_curr.cast<float>());
            }
            
        }

        void line_search(void){

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k * STATES;
                const int idu = k * STATES + nX;

                const float gamma = 0.1f;
                xbar.segment(idx, STATES) = (1.0f - gamma) * xbar.segment(idx, STATES) + gamma * sol.segment(idx, STATES);
                if(k==N_HORIZON) break;
                ubar.segment(idx, INPUTS) = (1.0f - gamma) * ubar.segment(idx, INPUTS) + gamma * sol.segment(idu, INPUTS);

            }

        }

        void rollout_safety_filter(const float *h_grid, const float *dhdt_grid, const Eigen::Vector3f& rc, const float wn, const float issf){

            for(int k=0; k<N_HORIZON; k++){
                
                // Get Safety Function Rate, Value & Gradient
                const int idk = k * STATES;
                Eigen::Vector3f rk{(float)xbar(idk+0), (float)xbar(idk+1), (float)xbar(idk+2)};
                const float h = extract_grid_value(h_grid, rk(0), rk(1), rc(0), rc(1));
                const float dhdt = extract_grid_value(dhdt_grid, rk(0), rk(1), rc(0), rc(1));
                Eigen::Vector3d dhdr = extract_grid_gradient(h_grid, rk(0), rk(1), rc(0), rc(1)).cast<double>();

                // ISSf Term
                const float Lgh_norm = std::sqrt(dhdr.dot(dhdr));
                float ISSf = std::pow(Lgh_norm + 0.5f, 2.0f) / issf;
                ISSf *= k / N_HORIZON;
                
                // Single Integrator Safety Filter
                const float b = dhdr.dot(dhdr);
                float a = wn*h;
                a += dhdt; // Dynamic Environment
                a += dhdr.dot(ubar.segment(idk, INPUTS)); // Min Norm Controller
                a -= ISSf; // Input-to-State Safety (Robustness)
                
                // Analytical Safety Filter
                const float k_sontag = 1.0f;
                const float sigma_sontag = 0.1f;
                float lambda = 0.0f;
                if(b>1.0e-4f) lambda = k_sontag * (-a + std::sqrt(a*a+sigma_sontag*b*b)) / (2.0f*b); // Half Sonta

                ubar.segment(idk, INPUTS) += lambda * dhdr;
                xbar.segment(idk+STATES, STATES) = xbar.segment(idk, STATES) + ubar.segment(idk, INPUTS) * DT;

            }

        }

        void set_input(Eigen::Vector3f& u){
            
            u << (float)ubar(0), (float)ubar(1), (float)ubar(2);
                    
        }

};