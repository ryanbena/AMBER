#include <iostream>
#include "math.h"
#include "../inc/poisson.h"
#include "../inc/utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define LINEAR_STATE_LENGTH 3
#define LINEAR_INPUT_LENGTH 3

// MPC Controller Class
class MPC{
    
    public: 
        
        MPC(void){
            
            N_HORIZON = TMAX - 1;
            nX = (N_HORIZON+1) * LINEAR_STATE_LENGTH;
            nU = N_HORIZON * LINEAR_INPUT_LENGTH;
            nZ = nX + nU; // Optimization Variables
            nC = nX + N_HORIZON;
           
            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_upper = Eigen::VectorXd::Zero(nC);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            sol = Eigen::VectorXd::Zero(nZ);

            state_goal = Eigen::VectorXd::Zero(LINEAR_STATE_LENGTH);
            state_curr = Eigen::VectorXd::Zero(LINEAR_STATE_LENGTH);

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH);
            Px.row(0) << 5.0, 0.0, 0.0;
            Px.row(1) << 0.0, 5.0, 0.0;
            Px.row(2) << 0.0, 0.0, 0.5;

            Pu.setIdentity(LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH); 
            Pu.row(0) << 5.0, 0.0, 0.0;
            Pu.row(1) << 0.0, 5.0, 0.0;
            Pu.row(2) << 0.0, 0.0, 1.0;

            for (int k=0; k<N_HORIZON; k++){
                cost_P.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(k*LINEAR_STATE_LENGTH, k*LINEAR_STATE_LENGTH) = Px;
                cost_P.block<LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH>(nX + k*LINEAR_INPUT_LENGTH, nX + k*LINEAR_INPUT_LENGTH) = Pu;
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>((k+1)*LINEAR_STATE_LENGTH, (k+1)*LINEAR_STATE_LENGTH) = -1.0f * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH); // Identity 
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>((k+1)*LINEAR_STATE_LENGTH, k*LINEAR_STATE_LENGTH) = Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH); // A Matrix
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH>((k+1)*LINEAR_STATE_LENGTH, nX + k*LINEAR_INPUT_LENGTH) = DT * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH); // B Matrix
                constraint_upper(nX+k) = OSQP_INFTY; 
                constraint_lower(nX+k) = -OSQP_INFTY; // Turn off safety constraints
            }
            cost_P.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(N_HORIZON*LINEAR_STATE_LENGTH, N_HORIZON*LINEAR_STATE_LENGTH) = terminal*Px; // Terminal Cost
            constraint_A.block<LINEAR_STATE_LENGTH,LINEAR_STATE_LENGTH>(0,0).setIdentity(); // Initial Condition Constraint

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

        const float terminal = 1.0f;
        

        OsqpEigen::Solver solver;

        int setup_QP(const float *xd, const float *x){
            
            solver.settings()->setVerbosity(false); // Turn off printing
            solver.settings()->setMaxIteration(100);
            solver.data()->setNumberOfVariables(nZ); 
            solver.data()->setNumberOfConstraints(nC);

            state_goal << xd[0], xd[1], xd[2];
            state_curr << x[0], x[1], x[2];
    
            while((state_goal(2)-state_curr(2)) < -M_PI) state_goal(2) += 2.0f*M_PI;
            while((state_goal(2)-state_curr(2)) >= M_PI) state_goal(2) -= 2.0f*M_PI;

            for (int i=0; i<=N_HORIZON; i++){
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += state_goal * (float)i / (float)N_HORIZON; 
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += state_curr * (float)(N_HORIZON-i) / (float)N_HORIZON;
            }
            
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

        int reset_QP(const float *xd, const float *x){
            
            solver.data()->clearHessianMatrix();
            solver.data()->clearLinearConstraintsMatrix();
            solver.clearSolverVariables();
            solver.clearSolver();

            setup_QP(xd, x);
            solve();
            return 0;

        }

        int initialize_solution(const float *xd, const float *x){
            
            state_goal << xd[0], xd[1], xd[2];
            state_curr << x[0], x[1], x[2];

            while((state_goal(2)-state_curr(2)) < -M_PI) state_goal(2) += 2.0f*M_PI;
            while((state_goal(2)-state_curr(2)) >= M_PI) state_goal(2) -= 2.0f*M_PI;
            
            // Interpolate Solution from current state to desired final state
            const float previous_solution_weight = 1.0f;
            sol *= previous_solution_weight;
            for (int i=0; i<=N_HORIZON; i++){
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - previous_solution_weight) * state_goal * (float)i / (float)N_HORIZON; 
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - previous_solution_weight) * state_curr * (float)(N_HORIZON-i) / (float)N_HORIZON;
            }

            return 0;

        }

        int update_cost_and_constraints(const float *hgrid){
   
            // Update Dynamics Constraints
            constraint_lower.segment(0, LINEAR_STATE_LENGTH) = state_curr;
            constraint_upper.segment(0, LINEAR_STATE_LENGTH) = state_curr;

            // Update Obstacle State Constraints
            const float i_eps = 5.0f;
            const float j_eps = 5.0f;
            const float q_eps = 1.0f;

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k*LINEAR_STATE_LENGTH; 
                const float xk = sol(idx+0);
                const float yk = sol(idx+1);
                const float yawk = sol(idx+2);

                const float *hgridk = hgrid + k*IMAX*JMAX*QMAX;

                // Indices Corresponding to Current Position
                const float ir = yk / DS;
                const float jr = xk / DS;
                const float qr = yawk / DQ;

                // Saturated Because of Finite Grid Size
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size
                const float qc = q_wrap(qr);

                // Get Safety Function Values
                float h = trilinear_interpolation(hgridk, ic, jc, qc);
                
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
                const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));
                const float qp = q_wrap(qc + q_eps);
                const float qm = q_wrap(qc - q_eps);

                //const float dhdx = (trilinear_interpolation(hgridk, ic, jp, qc) - trilinear_interpolation(hgridk, ic, jm, qc)) / (2.0f * x_eps);
                //const float dhdy = (trilinear_interpolation(hgridk, ip, jc, qc) - trilinear_interpolation(hgridk, im, jc, qc)) / (2.0f * y_eps);
                //const float dhdyaw = (trilinear_interpolation(hgridk, ic, jc, qp) - trilinear_interpolation(hgridk, ic, jc, qm)) / (2.0f * yaw_eps);

                const float dhdxpp = (trilinear_interpolation(hgridk, ip, jp, qp) - trilinear_interpolation(hgridk, ip, jm, qp)) / ((jp-jm)*DS);
                const float dhdxcp = (trilinear_interpolation(hgridk, ic, jp, qp) - trilinear_interpolation(hgridk, ic, jm, qp)) / ((jp-jm)*DS);
                const float dhdxmp = (trilinear_interpolation(hgridk, im, jp, qp) - trilinear_interpolation(hgridk, im, jm, qp)) / ((jp-jm)*DS);
                const float dhdxpc = (trilinear_interpolation(hgridk, ip, jp, qc) - trilinear_interpolation(hgridk, ip, jm, qc)) / ((jp-jm)*DS);
                const float dhdxcc = (trilinear_interpolation(hgridk, ic, jp, qc) - trilinear_interpolation(hgridk, ic, jm, qc)) / ((jp-jm)*DS);
                const float dhdxmc = (trilinear_interpolation(hgridk, im, jp, qc) - trilinear_interpolation(hgridk, im, jm, qc)) / ((jp-jm)*DS);
                const float dhdxpm = (trilinear_interpolation(hgridk, ip, jp, qm) - trilinear_interpolation(hgridk, ip, jm, qm)) / ((jp-jm)*DS);
                const float dhdxcm = (trilinear_interpolation(hgridk, ic, jp, qm) - trilinear_interpolation(hgridk, ic, jm, qm)) / ((jp-jm)*DS);
                const float dhdxmm = (trilinear_interpolation(hgridk, im, jp, qm) - trilinear_interpolation(hgridk, im, jm, qm)) / ((jp-jm)*DS);
                const float dhdx = (dhdxpp + dhdxcp + dhdxmp + dhdxpc + dhdxcc + dhdxmc + dhdxpm + dhdxcm + dhdxmm) / 9.0f;

                const float dhdypp = (trilinear_interpolation(hgridk, ip, jp, qp) - trilinear_interpolation(hgridk, im, jp, qp)) / ((ip-im)*DS);
                const float dhdycp = (trilinear_interpolation(hgridk, ip, jc, qp) - trilinear_interpolation(hgridk, im, jc, qp)) / ((ip-im)*DS);
                const float dhdymp = (trilinear_interpolation(hgridk, ip, jm, qp) - trilinear_interpolation(hgridk, im, jm, qp)) / ((ip-im)*DS);
                const float dhdypc = (trilinear_interpolation(hgridk, ip, jp, qc) - trilinear_interpolation(hgridk, im, jp, qc)) / ((ip-im)*DS);
                const float dhdycc = (trilinear_interpolation(hgridk, ip, jc, qc) - trilinear_interpolation(hgridk, im, jc, qc)) / ((ip-im)*DS);
                const float dhdymc = (trilinear_interpolation(hgridk, ip, jm, qc) - trilinear_interpolation(hgridk, im, jm, qc)) / ((ip-im)*DS);
                const float dhdypm = (trilinear_interpolation(hgridk, ip, jp, qm) - trilinear_interpolation(hgridk, im, jp, qm)) / ((ip-im)*DS);
                const float dhdycm = (trilinear_interpolation(hgridk, ip, jc, qm) - trilinear_interpolation(hgridk, im, jc, qm)) / ((ip-im)*DS);
                const float dhdymm = (trilinear_interpolation(hgridk, ip, jm, qm) - trilinear_interpolation(hgridk, im, jm, qm)) / ((ip-im)*DS);
                const float dhdy = (dhdypp + dhdycp + dhdymp + dhdypc + dhdycc + dhdymc + dhdypm + dhdycm + dhdymm) / 9.0f;

                const float dhdyawpp = (trilinear_interpolation(hgridk, ip, jp, qp) - trilinear_interpolation(hgridk, ip, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcp = (trilinear_interpolation(hgridk, ic, jp, qp) - trilinear_interpolation(hgridk, ic, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmp = (trilinear_interpolation(hgridk, im, jp, qp) - trilinear_interpolation(hgridk, im, jp, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawpc = (trilinear_interpolation(hgridk, ip, jc, qp) - trilinear_interpolation(hgridk, ip, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcc = (trilinear_interpolation(hgridk, ic, jc, qp) - trilinear_interpolation(hgridk, ic, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmc = (trilinear_interpolation(hgridk, im, jc, qp) - trilinear_interpolation(hgridk, im, jc, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawpm = (trilinear_interpolation(hgridk, ip, jm, qp) - trilinear_interpolation(hgridk, ip, jm, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawcm = (trilinear_interpolation(hgridk, ic, jm, qp) - trilinear_interpolation(hgridk, ic, jm, qm)) / (2.0f*q_eps*DQ);
                const float dhdyawmm = (trilinear_interpolation(hgridk, im, jm, qp) - trilinear_interpolation(hgridk, im, jm, qm)) / (2.0f*q_eps*DQ);
                const float dhdyaw = (dhdyawpp + dhdyawcp + dhdyawmp + dhdyawpc + dhdyawcc + dhdyawmc + dhdyawpm + dhdyawcm + dhdyawmm) / 9.0f;

                // Update Constraints
                const float alpha = expf(-WN*DT);
                //const float alpha = 0.0f;
                if(k!=N_HORIZON){
                    constraint_A(nX+k, idx+0) = -alpha*dhdx;
                    constraint_A(nX+k, idx+1) = -alpha*dhdy;
                    constraint_A(nX+k, idx+2) = -alpha*dhdyaw;
                    constraint_lower(nX+k) = -alpha * (dhdx*xk + dhdy*yk + dhdyaw*yawk - h);
                    // ISSf Term
                    const float ISSf1 = 4.0f;
                    const float ISSf2 = 4.0f;
                    const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy+dhdyaw*dhdyaw);
                    constraint_lower(nX+k) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(nX+k-1, idx+0) = dhdx;
                    constraint_A(nX+k-1, idx+1) = dhdy;
                    constraint_A(nX+k-1, idx+2) = dhdyaw;
                    constraint_lower(nX+k-1) += dhdx*xk + dhdy*yk + dhdyaw*yawk - h;
                }

                // Update Linear Cost
                cost_q.segment(idx, LINEAR_STATE_LENGTH) = -state_goal.transpose() * Px;
        
            }
            
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();
            solver.updateLinearConstraintsMatrix(constraint_A_sparse); 
            solver.updateBounds(constraint_lower, constraint_upper);
            cost_q.segment(N_HORIZON*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) *= terminal;
            solver.updateGradient(cost_q);

            return 1; 
        }

        int solve(){

            if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError){
                return 1;
            }
            else{
                sol = solver.getSolution();
                return 0;
            }
            
        }

        void extract_solution(float *z){

            for(int i = 0; i < nZ; i++){
                z[i] = sol(i);
            }

        }

        void set_input(float *vx, float *vy, float *vyaw, float age){

            const float vx0 = sol(nX+0);
            const float vy0 = sol(nX+1);
            const float vyaw0 = sol(nX+2);

            const float vx1 = sol(nX+3);
            const float vy1 = sol(nX+4);
            const float vyaw1 = sol(nX+5);

            const float dvxdt = (vx1-vx0) / DT;
            const float dvydt = (vy1-vy0) / DT;
            const float dvyawdt = (vyaw1-vyaw0) / DT;

            *vx = vx0 + dvxdt * age;
            *vy = vy0 + dvydt * age;
            *vyaw = vyaw0 + dvyawdt * age;

        }

};




