#include <iostream>
#include "math.h"
#include "poisson.h"
#include "utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define LINEAR_STATE_LENGTH_3D 3
#define LINEAR_INPUT_LENGTH_3D 3

// MPC Controller Class
class MPC3D{
    
    public: 
        
        MPC3D(void){
            

            N_HORIZON = TMAX - 1;
            nX = (N_HORIZON+1) * LINEAR_STATE_LENGTH_3D;
            nU = N_HORIZON * LINEAR_INPUT_LENGTH_3D;
            nZ = nX + nU; // Optimization Variables
            nC = nX + N_HORIZON;
           
            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_upper = Eigen::VectorXd::Zero(nC);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            sol = Eigen::VectorXd::Zero(nZ);
            state_goal = Eigen::VectorXd::Zero(LINEAR_STATE_LENGTH_3D);
            state_curr = Eigen::VectorXd::Zero(LINEAR_STATE_LENGTH_3D);

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D);
            Px.row(0) << 0.0, 0.0, 0.0;
            Px.row(1) << 0.0, 0.0, 0.0;
            Px.row(2) << 0.0, 0.0, 0.0;

            Pu.setIdentity(LINEAR_INPUT_LENGTH_3D, LINEAR_INPUT_LENGTH_3D); 
            Pu.row(0) << 5.0, 0.0, 0.0;
            Pu.row(1) << 0.0, 5.0, 0.0;
            Pu.row(2) << 0.0, 0.0, 1.0;

            for (int k=0; k<N_HORIZON; k++){
                cost_P.block<LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D>(k*LINEAR_STATE_LENGTH_3D, k*LINEAR_STATE_LENGTH_3D) = Px;
                cost_P.block<LINEAR_INPUT_LENGTH_3D, LINEAR_INPUT_LENGTH_3D>(nX + k*LINEAR_INPUT_LENGTH_3D, nX + k*LINEAR_INPUT_LENGTH_3D) = Pu;
                constraint_A.block<LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D>((k+1)*LINEAR_STATE_LENGTH_3D, (k+1)*LINEAR_STATE_LENGTH_3D) = -1.0f * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D); // Identity 
                constraint_A.block<LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D>((k+1)*LINEAR_STATE_LENGTH_3D, k*LINEAR_STATE_LENGTH_3D) = Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D); // A Matrix
                constraint_A.block<LINEAR_STATE_LENGTH_3D, LINEAR_INPUT_LENGTH_3D>((k+1)*LINEAR_STATE_LENGTH_3D, nX + k*LINEAR_INPUT_LENGTH_3D) = DT * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_3D, LINEAR_INPUT_LENGTH_3D); // B Matrix
                constraint_upper(nX+k) = OSQP_INFTY; 
                constraint_lower(nX+k) = -OSQP_INFTY; // Turn off safety constraints
            }
            cost_P.block<LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D>(N_HORIZON*LINEAR_STATE_LENGTH_3D, N_HORIZON*LINEAR_STATE_LENGTH_3D) = terminal*Px; // Terminal Cost
            constraint_A.block<LINEAR_STATE_LENGTH_3D,LINEAR_STATE_LENGTH_3D>(0,0).setIdentity(); // Initial Condition Constraint

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
        Eigen::VectorXd state_goal;
        Eigen::VectorXd state_curr;

        const float terminal = 5.0f;

        OsqpEigen::Solver solver;

        int setup_QP(void){
            
            solver.settings()->setVerbosity(false); // Turn off printing
            solver.settings()->setMaxIteration(1000);
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

        int update_cost(const float *xd, const float *x, const float *ud){

            state_goal << xd[0], xd[1], xd[2];
            state_curr << x[0], x[1], x[2];

            Eigen::VectorXd input_goal(LINEAR_INPUT_LENGTH_3D);
            input_goal << ud[0], ud[1], ud[2];

            while((state_goal(2)-state_curr(2)) < -M_PI) state_goal(2) += 2.0f*M_PI;
            while((state_goal(2)-state_curr(2)) >= M_PI) state_goal(2) -= 2.0f*M_PI;

            for(int k=0; k<=N_HORIZON; k++){
                const int idx = k*LINEAR_STATE_LENGTH_3D;
                const int idu = k*LINEAR_INPUT_LENGTH_3D + nX;
                cost_q.segment(idx, LINEAR_STATE_LENGTH_3D) = -state_goal.transpose() * Px;
                if(k!=N_HORIZON) cost_q.segment(idu, LINEAR_INPUT_LENGTH_3D) = -input_goal.transpose() * Pu;
            }

            cost_q.segment(N_HORIZON*LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D) *= terminal;
            solver.updateGradient(cost_q);

            return 1;
            
        }

        int update_constraints(const float *h_grid, const float *dhdt_grid, const float *xc){
   
            // Update Dynamics Constraints
            constraint_lower.segment(0, LINEAR_STATE_LENGTH_3D) = state_curr;
            constraint_upper.segment(0, LINEAR_STATE_LENGTH_3D) = state_curr;

            // Update Obstacle State Constraints
            const float i_eps = 5.0f;
            const float j_eps = 5.0f;
            const float q_eps = 1.0f;

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k*LINEAR_STATE_LENGTH_3D;
                const float rxk = sol(idx+0);
                const float ryk = sol(idx+1);
                const float yawk = sol(idx+2);

                // Indices Corresponding to Current State
                const float ir = y_to_i(ryk, xc[1]);
                const float jr = x_to_j(rxk, xc[0]);

                // Saturated Because of Finite Grid Size
                const float qc = yaw_to_q(yawk, xc[2]);
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
            
                // Get Current Safety Function Value & Rate
                const float h1 = trilinear_interpolation(h_grid, ic, jc, qc);
                const float dhdt = trilinear_interpolation(dhdt_grid, ic, jc, qc);

                // Compute Future Safety Function Value
                float h = h1 + k * dhdt * DT;

                // Get Current Neighbor Values
                const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));
                const float qp = q_wrap(qc + q_eps);
                const float qm = q_wrap(qc - q_eps);
                float hxp = trilinear_interpolation(h_grid, ic, jp, qc);
                float hxm = trilinear_interpolation(h_grid, ic, jm, qc);
                float hyp = trilinear_interpolation(h_grid, ip, jc, qc);
                float hym = trilinear_interpolation(h_grid, im, jc, qc);
                float hqp = trilinear_interpolation(h_grid, ic, jc, qp);
                float hqm = trilinear_interpolation(h_grid, ic, jc, qm);

                // Compute Future Neighbor Values
                const float dhxpdt = trilinear_interpolation(dhdt_grid, ic, jp, qc);
                const float dhxmdt = trilinear_interpolation(dhdt_grid, ic, jm, qc);
                const float dhypdt = trilinear_interpolation(dhdt_grid, ip, jc, qc);
                const float dhymdt = trilinear_interpolation(dhdt_grid, im, jc, qc);
                const float dhqpdt = trilinear_interpolation(dhdt_grid, ic, jc, qp);
                const float dhqmdt = trilinear_interpolation(dhdt_grid, ic, jc, qm);
                hxp += k * dhxpdt * DT;
                hxm += k * dhxmdt * DT;
                hyp += k * dhypdt * DT;
                hym += k * dhymdt * DT;
                hqp += k * dhqpdt * DT;
                hqm += k * dhqmdt * DT;

                float dhdx = (hxp-hxm) / ((jp-jm)*DS);
                float dhdy = (hyp-hym) / ((ip-im)*DS);
                float dhdyaw = (hqp-hqm) / (2.0f*q_eps*DQ);

                // If You Have Left The Grid, Use SDF to Get Back
                if((ir!=ic) || (jr!=jc)){
                    h = -sqrtf((ir-ic)*(ir-ic) + (jr-jc)*(jr-jc)) * DS;
                    if(jr>jc) dhdx = -1.0f;
                    if(jr<jc) dhdx = 1.0f;
                    if(ir>ic) dhdy = -1.0f;
                    if(ir<ic) dhdy = 1.0f;
                    const float norm = sqrtf(dhdx*dhdx+dhdy*dhdy);
                    dhdx /= norm;
                    dhdy /= norm;
                    dhdyaw = 0.0f;
                }

                // Update Constraints
                const float alpha = expf(-WN*DT);
                if(k!=N_HORIZON){
                    constraint_A(nX+k, idx+0) = -alpha*dhdx;
                    constraint_A(nX+k, idx+1) = -alpha*dhdy;
                    constraint_A(nX+k, idx+2) = -alpha*dhdyaw;
                    constraint_lower(nX+k) = -alpha * (dhdx*rxk + dhdy*ryk + dhdyaw*yawk - h);
                    // ISSf Term
                    const float ISSf1 = 5.0f;
                    const float ISSf2 = 20.0f;
                    const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy+dhdyaw*dhdyaw);
                    constraint_lower(nX+k) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(nX+k-1, idx+0) = dhdx;
                    constraint_A(nX+k-1, idx+1) = dhdy;
                    constraint_A(nX+k-1, idx+2) = dhdyaw;
                    constraint_lower(nX+k-1) += dhdx*rxk + dhdy*ryk + dhdyaw*yawk - h;
                }
        
            }
            
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();
            solver.updateLinearConstraintsMatrix(constraint_A_sparse); 
            solver.updateBounds(constraint_lower, constraint_upper);

            return 1; 
        }

        int solve(){

            if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
            else{
                sol = solver.getSolution();
                return 0;
            }
            
        }

        void extract_solution(float *z){
            for(int i = 0; i < nZ; i++) z[i] = sol(i);
        }

        void set_input(float *vx, float *vy, float *vyaw){
            *vx = sol(nX+0);
            *vy = sol(nX+1);
            *vyaw = sol(nX+2);
        }

};