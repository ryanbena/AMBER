#pragma once

#include <iostream>
#include "poisson.h"
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
            solver.settings()->setMaxIteration(2000);
            
            N_HORIZON = TMAX - 1;
            nX = (N_HORIZON+1) * STATES; // State Variables
            nU = N_HORIZON * INPUTS; // Input Variables
            nS = N_HORIZON; // Slack Variables for CBF Constraints
            nZ = nX + nU + nS; // Total Optimization Variables
            nC = nZ;

            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_upper = Eigen::VectorXd::Zero(nC);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            sol = Eigen::VectorXd::Zero(nZ);
            xbar = Eigen::VectorXd::Zero(nX);
            ubar = Eigen::VectorXd::Zero(nU);
            
            Px.setIdentity(STATES, STATES);
            Px.row(0) << 1.0, 0.0, 0.0;
            Px.row(1) << 0.0, 1.0, 0.0;
            Px.row(2) << 0.0, 0.0, 1.0;
            Pu.setIdentity(INPUTS, INPUTS);
            Pu.row(0) << 1.0, 0.0, 0.0;
            Pu.row(1) << 0.0, 1.0, 0.0;
            Pu.row(2) << 0.0, 0.0, 1.0;
            Ps = 1000.0;
            for(int k=0; k<=N_HORIZON; k++){
                cost_P.block<STATES, STATES>(k*STATES, k*STATES) = Px;
                if(k!=N_HORIZON) cost_P.block<INPUTS, INPUTS>(nX + k*INPUTS, nX + k*INPUTS) = Pu;
                if(k!=N_HORIZON) cost_P(nX + nU + k, nX + nU + k) = Ps;
            }
            cost_P.block<STATES, STATES>(N_HORIZON*STATES, N_HORIZON*STATES) *= terminal;
            
            // Build Constraints
            constraint_A.block<STATES,STATES>(0,0).setIdentity(); // Initial Condition Constraint
            for(int k=0; k<N_HORIZON; k++){
                
                const int idx = k * STATES;
                const int idxp1 = (k+1) * STATES;
                const int idu = k * INPUTS + nX;
                
                // Set Up Dynamic Constraints
                constraint_A.block<STATES, STATES>(idxp1, idxp1) = -Eigen::MatrixXd::Identity(STATES, STATES); // Identity 
                constraint_A.block<STATES, STATES>(idxp1, idx) = Eigen::MatrixXd::Identity(STATES, STATES); // A Matrix
                constraint_A.block<STATES, INPUTS>(idxp1, idu) = DT * Eigen::MatrixXd::Identity(STATES, INPUTS); // B Matrix
                
                // Set Up Saturation Constraints
                constraint_A.block<INPUTS, INPUTS>(idu, idu) = Eigen::MatrixXd::Identity(INPUTS, INPUTS);
                constraint_upper.segment(idu, INPUTS) << 0.5f, 0.5f, 1.0f; 
                constraint_lower.segment(idu, INPUTS) << -0.5f, -0.5f, -1.0f;
                
                // Set Up Safety Constraints
                const int ids = k + nX + nU;
                constraint_upper(ids) = OSQP_INFTY; 
                constraint_lower(ids) = -OSQP_INFTY;
                
            }

        }

        int N_HORIZON;
        int nX, nU, nS, nZ, nC;
        Eigen::MatrixXd Px, Pu;
        double Ps;
        Eigen::MatrixXd cost_P;
        Eigen::VectorXd cost_q;
        Eigen::MatrixXd constraint_A;
        Eigen::VectorXd constraint_lower, constraint_upper;
        Eigen::VectorXd sol, xbar, ubar;
        OsqpEigen::Solver solver;
        float cost0 = 1.0e23f;
        float cost1 = 1.0e23f;
        float resid = 1.0e23f;
        const float terminal = 1.0f;

        int reset_QP(void){
                        
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

        void reset_xbar_and_ubar(const std::vector<float> x, const std::vector<float> ud){
            
            for(int k=0; k<=N_HORIZON; k++){

                const float weight = 0.8f;
                xbar(k*STATES+0) *= weight;
                xbar(k*STATES+1) *= weight;
                xbar(k*STATES+2) *= weight;
                xbar(k*STATES+0) += (1.0f-weight) * (x[0] + ud[0] * k * DT);
                xbar(k*STATES+1) += (1.0f-weight) * (x[1] + ud[1] * k * DT);
                xbar(k*STATES+2) += (1.0f-weight) * (x[2] + ud[2] * k * DT);

                if(k!=N_HORIZON){

                    ubar(k*INPUTS+0) = ud[0];
                    ubar(k*INPUTS+1) = ud[1];
                    ubar(k*INPUTS+2) = ud[2];

                }

            }

        }
        
        void line_search(const float *h_grid, const std::vector<float> xc, const float wn, const int buff){

            float best_violation = -1.0e10f;
            float best_alpha = 0.0f;
            const int N = 10;
            const float rho = std::exp(-wn*DT);

            for(int n=0; n<=N; n++){

                float h[TMAX];
                const float alpha = std::pow((float)n / (float)N, 2.0f);
                Eigen::VectorXd xbar_test = (1.0f-alpha)*xbar + alpha*sol.segment(0, nX);
                float total_violation = 0.0f;
                for(int k=0; k<=N_HORIZON; k++){
                    const float ic = y_to_i(xbar(k*STATES+1), xc[1]);
                    const float jc = x_to_j(xbar(k*STATES+0), xc[0]);
                    h[k] = bilinear_interpolation(h_grid, ic, jc, buff);
                    if(k!=0) total_violation += std::fmin(0.0f, h[k]-rho*h[k-1]);
                }
                if(total_violation >= best_violation){
                    best_violation = total_violation;
                    best_alpha = alpha;
                }

            }

            xbar *= 1.0f - best_alpha;
            xbar += best_alpha * sol.segment(0, nX);
            ubar *= 1.0f - best_alpha;
            ubar += best_alpha * sol.segment(nX, nU);

        }
        
        void update_cost(const std::vector<float> xd_traj, const std::vector<float> ud, const int n){
            
            Eigen::VectorXd state_goal(STATES);
            Eigen::VectorXd input_goal(INPUTS);

            for(int k=0; k<=N_HORIZON; k++){
                state_goal << xd_traj[INPUTS*k+0], xd_traj[INPUTS*k+1], xd_traj[INPUTS*k+2];
                input_goal << ud[0], ud[1], ud[2];
                const int idx = k * STATES;
                const int idu = k * INPUTS + nX;
                const int ids = k + nX + nU;
                const float yawk = xbar(idx+2);
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(INPUTS, INPUTS);
                R.row(0) << std::cos(yawk), -std::sin(yawk), 0.0f;
                R.row(1) << std::sin(yawk), std::cos(yawk), 0.0f;
                R.row(2) << 0.0f, 0.0f, 1.0f;
                cost_q.segment(idx, STATES) = -Px * state_goal;
                if(k!=N_HORIZON) cost_P.block<INPUTS, INPUTS>(idu, idu) = R * Pu * R.transpose();
                if(k!=N_HORIZON) cost_q.segment(idu, INPUTS) = -R * Pu * R.transpose() * input_goal;
                if(k!=N_HORIZON) cost_P(ids, ids) = (float)(n+1) * Ps; // Tighten Slack Variable with Each Iteration

            }
            cost_q.segment(N_HORIZON*STATES, STATES) *= terminal;
            // cost_P.block<INPUTS, INPUTS>(nX + (N_HORIZON-1)*INPUTS, nX + (N_HORIZON-1)*INPUTS) *= terminal;
            
            Eigen::SparseMatrix<double> cost_P_sparse = cost_P.sparseView(0.0, -1.0);
            solver.updateHessianMatrix(cost_P_sparse);
            solver.updateGradient(cost_q);
            
        }

        void update_constraints(const float *h_grid, const std::vector<float> x, const std::vector<float> xc, const float wn, const float issf, const int buff){
   
            // Update Initial Condition Constraints
            constraint_lower.segment(0, STATES) << x[0], x[1], x[2];
            constraint_upper.segment(0, STATES) << x[0], x[1], x[2];

            // Update Obstacle State Constraints
            const float eps = 1.0f;
            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k * STATES;
                const int idu = k * STATES + nX;
                const int ids = k + nX + nU;
                
                const float rxk = xbar(idx+0);
                const float ryk = xbar(idx+1);
                const float yawk = xbar(idx+2);

                // Indices Corresponding to Current State
                const float ic = y_to_i(ryk, xc[1]);
                const float jc = x_to_j(rxk, xc[0]);
                
                // Get Current Safety Function Value & Rate
                float h = bilinear_interpolation(h_grid, ic, jc, buff);

                // Get Current Neighbor Values
                float hxp = bilinear_interpolation(h_grid, ic, jc + eps, buff);
                float hxm = bilinear_interpolation(h_grid, ic, jc - eps, buff);
                float hyp = bilinear_interpolation(h_grid, ic + eps, jc, buff);
                float hym = bilinear_interpolation(h_grid, ic - eps, jc, buff);
                float dhdx = (hxp-hxm) / (2.0f*eps*DS);
                float dhdy = (hyp-hym) / (2.0f*eps*DS);

                // Update Constraints
                const float alpha = std::exp(-wn*DT);
                const int idsm1 = ids - 1;
                if(k!=N_HORIZON){
                    constraint_A(ids, ids) = 1.0f;
                    constraint_A.block<1,STATES>(ids, idx) << -alpha*dhdx, -alpha*dhdy, 0.0f;
                    constraint_lower(ids) = -alpha * (dhdx*rxk + dhdy*ryk - h);
                    // ISSf Term
                    const float Lgh_norm = std::sqrt(dhdx*dhdx + dhdy*dhdy);
                    // float ISSf = Lgh_norm / issf + Lgh_norm * Lgh_norm / issf;
                    float ISSf = std::pow(Lgh_norm + 0.5f, 2.0f) / issf;
                    ISSf *= (float)k/(float)N_HORIZON;
                    constraint_lower(ids) += ISSf * DT;
                }
                if(k!=0){
                    constraint_A.block<1,STATES>(idsm1, idx) << dhdx, dhdy, 0.0f;
                    constraint_lower(idsm1) += dhdx*rxk + dhdy*ryk - h;
                }
        
                // Update Saturation Constraint
                Eigen::MatrixXd R = Eigen::MatrixXd::Identity(INPUTS, INPUTS);
                R.row(0) << std::cos(yawk), -std::sin(yawk), 0.0f;
                R.row(1) << std::sin(yawk), std::cos(yawk), 0.0f;
                R.row(2) << 0.0f, 0.0f, 1.0f;
                if(k!=N_HORIZON) constraint_A.block<INPUTS, INPUTS>(idu, idu) = R.transpose(); 

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
                cost1 = solver.getObjValue();
            }
            else{
                std::cout << "QP Solver Error!!!!!!" << std::endl;
                reset_QP();
                cost1 = 1.0e23f;
            }
            
        }

        float update_residual(void){

            resid = (cost1-cost0) * (cost1-cost0);
            cost0 = cost1;
            return resid;

        }

        void set_input(std::vector<float>& u){
            
            u = {(float)sol(nX+0), (float)sol(nX+1), (float)sol(nX+2)};
                    
        }

};