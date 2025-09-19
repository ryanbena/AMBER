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

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D);
            Px.row(0) << 5.0, 0.0, 0.0;
            Px.row(1) << 0.0, 5.0, 0.0;
            Px.row(2) << 0.0, 0.0, 0.5;

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

        const float terminal = 1.0f;

        OsqpEigen::Solver solver;

        int setup_QP(void){
            
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

        int update_cost_and_constraints(const float *hgrid, const float *xc, const float *xd, const float *x){
   
            Eigen::VectorXd state_goal(LINEAR_STATE_LENGTH_3D);
            Eigen::VectorXd state_curr(LINEAR_STATE_LENGTH_3D);
            state_goal << xd[0], xd[1], xd[2];
            state_curr << x[0], x[1], x[2];

            while((state_goal(2)-state_curr(2)) < -M_PI) state_goal(2) += 2.0f*M_PI;
            while((state_goal(2)-state_curr(2)) >= M_PI) state_goal(2) -= 2.0f*M_PI;

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

                // Indices Corresponding to Current Position
                const float ir = y_to_i(ryk,xc[1]);
                const float jr = x_to_j(rxk,xc[0]);
                const float qr = yaw_to_q(yawk,xc[2]);

                // Saturated Because of Finite Grid Size
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
                const float qc = qr;

                // Get Safety Function Values
                float h = trilinear_interpolation(hgrid, ic, jc, qc);

                // Compute Gradients
                const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));
                const float qp = q_wrap(qc + q_eps);
                const float qm = q_wrap(qc - q_eps);

                //const float dhdx = (trilinear_interpolation(hgrid, ic, jp, qc) - trilinear_interpolation(hgrid, ic, jm, qc)) / (2.0f * x_eps);
                //const float dhdy = (trilinear_interpolation(hgrid, ip, jc, qc) - trilinear_interpolation(hgrid, im, jc, qc)) / (2.0f * y_eps);
                //const float dhdyaw = (trilinear_interpolation(hgrid, ic, jc, qp) - trilinear_interpolation(hgrid, ic, jc, qm)) / (2.0f * yaw_eps);

                const float dhdxpp = (trilinear_interpolation(hgrid, ip, jp, qp) - trilinear_interpolation(hgrid, ip, jm, qp)) / ((jp-jm)*DS);
                const float dhdxcp = (trilinear_interpolation(hgrid, ic, jp, qp) - trilinear_interpolation(hgrid, ic, jm, qp)) / ((jp-jm)*DS);
                const float dhdxmp = (trilinear_interpolation(hgrid, im, jp, qp) - trilinear_interpolation(hgrid, im, jm, qp)) / ((jp-jm)*DS);
                const float dhdxpc = (trilinear_interpolation(hgrid, ip, jp, qc) - trilinear_interpolation(hgrid, ip, jm, qc)) / ((jp-jm)*DS);
                const float dhdxcc = (trilinear_interpolation(hgrid, ic, jp, qc) - trilinear_interpolation(hgrid, ic, jm, qc)) / ((jp-jm)*DS);
                const float dhdxmc = (trilinear_interpolation(hgrid, im, jp, qc) - trilinear_interpolation(hgrid, im, jm, qc)) / ((jp-jm)*DS);
                const float dhdxpm = (trilinear_interpolation(hgrid, ip, jp, qm) - trilinear_interpolation(hgrid, ip, jm, qm)) / ((jp-jm)*DS);
                const float dhdxcm = (trilinear_interpolation(hgrid, ic, jp, qm) - trilinear_interpolation(hgrid, ic, jm, qm)) / ((jp-jm)*DS);
                const float dhdxmm = (trilinear_interpolation(hgrid, im, jp, qm) - trilinear_interpolation(hgrid, im, jm, qm)) / ((jp-jm)*DS);
                float dhdx = (dhdxpp + dhdxcp + dhdxmp + dhdxpc + dhdxcc + dhdxmc + dhdxpm + dhdxcm + dhdxmm) / 9.0f;

                const float dhdypp = (trilinear_interpolation(hgrid, ip, jp, qp) - trilinear_interpolation(hgrid, im, jp, qp)) / ((ip-im)*DS);
                const float dhdycp = (trilinear_interpolation(hgrid, ip, jc, qp) - trilinear_interpolation(hgrid, im, jc, qp)) / ((ip-im)*DS);
                const float dhdymp = (trilinear_interpolation(hgrid, ip, jm, qp) - trilinear_interpolation(hgrid, im, jm, qp)) / ((ip-im)*DS);
                const float dhdypc = (trilinear_interpolation(hgrid, ip, jp, qc) - trilinear_interpolation(hgrid, im, jp, qc)) / ((ip-im)*DS);
                const float dhdycc = (trilinear_interpolation(hgrid, ip, jc, qc) - trilinear_interpolation(hgrid, im, jc, qc)) / ((ip-im)*DS);
                const float dhdymc = (trilinear_interpolation(hgrid, ip, jm, qc) - trilinear_interpolation(hgrid, im, jm, qc)) / ((ip-im)*DS);
                const float dhdypm = (trilinear_interpolation(hgrid, ip, jp, qm) - trilinear_interpolation(hgrid, im, jp, qm)) / ((ip-im)*DS);
                const float dhdycm = (trilinear_interpolation(hgrid, ip, jc, qm) - trilinear_interpolation(hgrid, im, jc, qm)) / ((ip-im)*DS);
                const float dhdymm = (trilinear_interpolation(hgrid, ip, jm, qm) - trilinear_interpolation(hgrid, im, jm, qm)) / ((ip-im)*DS);
                float dhdy = (dhdypp + dhdycp + dhdymp + dhdypc + dhdycc + dhdymc + dhdypm + dhdycm + dhdymm) / 9.0f;

                const float dhdyawpp = (trilinear_interpolation(hgrid, ip, jp, qp) - trilinear_interpolation(hgrid, ip, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcp = (trilinear_interpolation(hgrid, ic, jp, qp) - trilinear_interpolation(hgrid, ic, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmp = (trilinear_interpolation(hgrid, im, jp, qp) - trilinear_interpolation(hgrid, im, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawpc = (trilinear_interpolation(hgrid, ip, jc, qp) - trilinear_interpolation(hgrid, ip, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcc = (trilinear_interpolation(hgrid, ic, jc, qp) - trilinear_interpolation(hgrid, ic, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmc = (trilinear_interpolation(hgrid, im, jc, qp) - trilinear_interpolation(hgrid, im, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawpm = (trilinear_interpolation(hgrid, ip, jm, qp) - trilinear_interpolation(hgrid, ip, jm, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcm = (trilinear_interpolation(hgrid, ic, jm, qp) - trilinear_interpolation(hgrid, ic, jm, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmm = (trilinear_interpolation(hgrid, im, jm, qp) - trilinear_interpolation(hgrid, im, jm, qm)) / (2.0f*q_eps*DQ);
                float dhdyaw = (dhdyawpp + dhdyawcp + dhdyawmp + dhdyawpc + dhdyawcc + dhdyawmc + dhdyawpm + dhdyawcm + dhdyawmm) / 9.0f;

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
                    const float ISSf1 = 10.0f;
                    const float ISSf2 = 10.0f;
                    const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy+dhdyaw*dhdyaw);
                    constraint_lower(nX+k) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(nX+k-1, idx+0) = dhdx;
                    constraint_A(nX+k-1, idx+1) = dhdy;
                    constraint_A(nX+k-1, idx+2) = dhdyaw;
                    constraint_lower(nX+k-1) += dhdx*rxk + dhdy*ryk + dhdyaw*yawk - h;
                }

                // Update Linear Cost
                cost_q.segment(idx, LINEAR_STATE_LENGTH_3D) = -state_goal.transpose() * Px;
        
            }
            
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();
            solver.updateLinearConstraintsMatrix(constraint_A_sparse); 
            solver.updateBounds(constraint_lower, constraint_upper);
            cost_q.segment(N_HORIZON*LINEAR_STATE_LENGTH_3D, LINEAR_STATE_LENGTH_3D) *= terminal;
            solver.updateGradient(cost_q);

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