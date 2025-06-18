#include <iostream>
#include "math.h"
#include "../inc/poisson.h"
#include "../inc/utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

/*
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
            Px.row(2) << 0.0, 0.0, 5.0;

            Pu.setIdentity(LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH); 
            Pu.row(0) << 1.0, 0.0, 0.0;
            Pu.row(1) << 0.0, 1.0, 0.0;
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
            solver.settings()->setMaxIteration(1000);
            solver.data()->setNumberOfVariables(nZ); 
            solver.data()->setNumberOfConstraints(nC);

            state_goal << xd[0], xd[1], xd[2];
            state_curr << x[0], x[1], x[2];
    
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

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k*LINEAR_STATE_LENGTH; 
                const float xk = sol(idx+0);
                const float yk = sol(idx+1);

                const float *hgridk = hgrid + k*IMAX*JMAX;

                // Indices Corresponding to Current Position
                const float ir = yk / DS;
                const float jr = xk / DS;

                // Saturated Because of Finite Grid Size
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size

                // Get Safety Function Values
                float h = bilinear_interpolation(hgridk, ic, jc);
                
                // Compute Gradients
                const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));

                const float dhdxp = (bilinear_interpolation(hgridk, ip, jp) - bilinear_interpolation(hgridk, ip, jm)) / ((jp-jm)*DS);
                const float dhdxc = (bilinear_interpolation(hgridk, ic, jp) - bilinear_interpolation(hgridk, ic, jm)) / ((jp-jm)*DS);
                const float dhdxm = (bilinear_interpolation(hgridk, im, jp) - bilinear_interpolation(hgridk, im, jm)) / ((jp-jm)*DS);
                const float dhdx = (dhdxp + dhdxc + dhdxm) / 3.0f;

                const float dhdyp = (bilinear_interpolation(hgridk, ip, jp) - bilinear_interpolation(hgridk, im, jp)) / ((ip-im)*DS);
                const float dhdyc = (bilinear_interpolation(hgridk, ip, jc) - bilinear_interpolation(hgridk, im, jc)) / ((ip-im)*DS);
                const float dhdym = (bilinear_interpolation(hgridk, ip, jm) - bilinear_interpolation(hgridk, im, jm)) / ((ip-im)*DS);
                const float dhdy = (dhdyp + dhdyc + dhdym) / 3.0f;

                // Update Constraints
                const float alpha = expf(-WN*DT);
                //const float alpha = 0.0f;
                if(k!=N_HORIZON){
                    constraint_A(nX+k, idx+0) = -alpha*dhdx;
                    constraint_A(nX+k, idx+1) = -alpha*dhdy;
                    constraint_A(nX+k, idx+2) = 0.0f;
                    constraint_lower(nX+k) = -alpha * (dhdx*xk + dhdy*yk - h);
                    // ISSf Term
                    const float ISSf1 = 4.0f;
                    const float ISSf2 = 4.0f;
                    const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy);
                    constraint_lower(nX+k) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(nX+k-1, idx+0) = dhdx;
                    constraint_A(nX+k-1, idx+1) = dhdy;
                    constraint_A(nX+k-1, idx+2) = 0.0f;
                    constraint_lower(nX+k-1) += dhdx*xk + dhdy*yk - h;
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

        void set_vd(float *vx, float *vy, float *vz, float age){

            const float vx0 = sol(nX+0);
            const float vy0 = sol(nX+1);
            const float vz0 = sol(nX+2);

            const float vx1 = sol(nX+3);
            const float vy1 = sol(nX+4);
            const float vz1 = sol(nX+5);

            const float dvxdt = (vx1-vx0) / DT;
            const float dvydt = (vy1-vy0) / DT;
            const float dvzdt = (vz1-vz0) / DT;

            *vx = vx0 + dvxdt * age;
            *vy = vy0 + dvydt * age;
            *vz = vz0 + dvzdt * age;

        }

};
*/


#define LINEAR_STATE_LENGTH 6
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

            // Initialize dynamics matrices
            A_dyn.setIdentity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH);
            A_dyn.row(0) << 1.0, 0.0, 0.0, DT, 0.0, 0.0;
            A_dyn.row(1) << 0.0, 1.0, 0.0, 0.0, DT, 0.0;
            A_dyn.row(2) << 0.0, 0.0, 1.0, 0.0, 0.0, DT;

            B_dyn = Eigen::MatrixXd::Zero(LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH);
            B_dyn.row(0) << DT*DT/2.0, 0.0, 0.0;
            B_dyn.row(1) << 0.0, DT*DT/2.0, 0.0;
            B_dyn.row(2) << 0.0, 0.0, DT*DT/2.0;
            B_dyn.row(3) << DT, 0.0, 0.0;
            B_dyn.row(4) << 0.0, DT, 0.0;
            B_dyn.row(5) << 0.0, 0.0, DT;

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH);
            Px.row(0) << 50.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            Px.row(1) << 0.0, 50.0, 0.0, 0.0, 0.0, 0.0;
            Px.row(2) << 0.0, 0.0, 50.0, 0.0, 0.0, 0.0;
            Px.row(3) << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
            Px.row(4) << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
            Px.row(5) << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

            Pu.setIdentity(LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH); 
            Pu.row(0) << 0.1, 0.0, 0.0;
            Pu.row(1) << 0.0, 0.1, 0.0;
            Pu.row(2) << 0.0, 0.0, 0.1;

            //const float G = 9.81;
            //Eigen::Vector3d asat(1.0f*G, 1.0f*G, 0.7f*G); // accl saturation

            cost_P.setIdentity();
            for (int k=0; k<N_HORIZON; k++){

                int idx = k * LINEAR_STATE_LENGTH;
                int idxp1 = (k+1) * LINEAR_STATE_LENGTH;
                int idu = k * LINEAR_INPUT_LENGTH + nX; 

                cost_P.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(idx, idx) = Px;
                cost_P.block<LINEAR_INPUT_LENGTH, LINEAR_INPUT_LENGTH>(idu, idu) = Pu;
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(idxp1, idxp1) = -1.0f * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH); // Identity 
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(idxp1, idx) = A_dyn; // A Matrix
                constraint_A.block<LINEAR_STATE_LENGTH, LINEAR_INPUT_LENGTH>(idxp1, idu) = B_dyn; // B Matrix
                
                //constraint_A.block<LINEAR_INPUT_LENGTH,LINEAR_INPUT_LENGTH>(idu, idu) = Eigen::Matrix3d::Identity();
                //constraint_upper.segment(idu,LINEAR_INPUT_LENGTH) = asat;
                //constraint_lower.segment(idu,LINEAR_INPUT_LENGTH) = -asat;
                
                constraint_upper(nX+k) = OSQP_INFTY; 
                constraint_lower(nX+k) = -OSQP_INFTY; // Turn off safety constraints
            
            }
            cost_P.block<LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH>(N_HORIZON*LINEAR_STATE_LENGTH, N_HORIZON*LINEAR_STATE_LENGTH) = terminal * Px; // Terminal Cost
            constraint_A.block<LINEAR_STATE_LENGTH,LINEAR_STATE_LENGTH>(0,0).setIdentity(); // Initial Condition Constraint
        
        }

        int N_HORIZON;
        int nX, nU, nZ, nC;
        Eigen::MatrixXd A_dyn, B_dyn; 
        Eigen::MatrixXd Px, Pu;
        Eigen::MatrixXd cost_P;
        Eigen::VectorXd cost_q;
        Eigen::MatrixXd constraint_A;
        Eigen::VectorXd constraint_lower, constraint_upper;
        Eigen::VectorXd sol;
        Eigen::VectorXd state_goal, state_curr;

        const float terminal = 1.0f;
        
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

        int initialize_solution(const float *xd, const float *x, const float weight){
            
            state_goal << xd[0], xd[1], xd[2], xd[3], xd[4], xd[5];
            state_curr << x[0], x[1], x[2], x[3], x[4], x[5];
            
            // Interpolate Solution from current state to desired final state
            sol *= weight;
            for (int i=0; i<=N_HORIZON; i++){
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - weight) * state_goal * (float)i / (float)N_HORIZON; 
                sol.segment(i*LINEAR_STATE_LENGTH, LINEAR_STATE_LENGTH) += (1.0f - weight) * state_curr * (float)(N_HORIZON-i) / (float)N_HORIZON;
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

            for(int k=0; k<=N_HORIZON; k++){
                
                const int row = nX + k;
                const int idx = k * LINEAR_STATE_LENGTH; 
                const float rk[3] = {(float)sol(idx+0), (float)sol(idx+1), (float)sol(idx+2)};
                const float vk[3] = {(float)sol(idx+3), (float)sol(idx+4), (float)sol(idx+5)};

                const float *hgridk = hgrid + k*IMAX*JMAX;

                // Indices Corresponding to Current Position
                const float ir = rk[1] / DS;
                const float jr = rk[0] / DS;

                // Saturated Because of Finite Grid Size
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size

                // Get Safety Function Values
                float h = bilinear_interpolation(hgridk, ic, jc);
                
                // Compute Gradients
                const float ip = fminf(fmaxf(0.0f, ic + i_eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic - i_eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc + j_eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc - j_eps), (float)(JMAX-1));

                const float dhdxp = (bilinear_interpolation(hgridk, ip, jp) - bilinear_interpolation(hgridk, ip, jm)) / ((jp-jm)*DS);
                const float dhdxc = (bilinear_interpolation(hgridk, ic, jp) - bilinear_interpolation(hgridk, ic, jm)) / ((jp-jm)*DS);
                const float dhdxm = (bilinear_interpolation(hgridk, im, jp) - bilinear_interpolation(hgridk, im, jm)) / ((jp-jm)*DS);
                const float dhdx = (dhdxp + dhdxc + dhdxm) / 3.0f;

                const float dhdyp = (bilinear_interpolation(hgridk, ip, jp) - bilinear_interpolation(hgridk, im, jp)) / ((ip-im)*DS);
                const float dhdyc = (bilinear_interpolation(hgridk, ip, jc) - bilinear_interpolation(hgridk, im, jc)) / ((ip-im)*DS);
                const float dhdym = (bilinear_interpolation(hgridk, ip, jm) - bilinear_interpolation(hgridk, im, jm)) / ((ip-im)*DS);
                const float dhdy = (dhdyp + dhdyc + dhdym) / 3.0f;

                // CBF Parameters
                const float alpha = expf(-WN*DT);
                const float ISSf1 = 8.0f;
                const float ISSf2 = 8.0f;
                const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy);

                // Update Constraints
                if(k!=N_HORIZON){
                    constraint_A(row, idx+0) = -alpha*dhdx;
                    constraint_A(row, idx+1) = -alpha*dhdy;
                    constraint_A(row, idx+2) = 0.0f;
                    constraint_A(row, idx+3) = 0.0f;
                    constraint_A(row, idx+4) = 0.0f;
                    constraint_A(row, idx+5) = 0.0f;
                    constraint_lower(row) = -alpha * (dhdx*rk[0] + dhdy*rk[1] - h);
                    constraint_lower(row) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(row-1, idx+0) = dhdx;
                    constraint_A(row-1, idx+1) = dhdy;
                    constraint_A(row-1, idx+2) = 0.0f;
                    constraint_A(row-1, idx+3) = 0.0f;
                    constraint_A(row-1, idx+4) = 0.0f;
                    constraint_A(row-1, idx+5) = 0.0f;
                    constraint_lower(row-1) += dhdx*rk[0] + dhdy*rk[1] - h;
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

            if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return 1;
            sol = solver.getSolution();
            return 0;
            
        }

        void extract_solution(float *z){

            for(int i = 0; i < nZ; i++){
                z[i] = sol(i);
            }

        }

        void set_ad(float *ax, float *ay, float *az, float age){

            const float ax0 = sol(nX+0);
            const float ay0 = sol(nX+1);
            const float az0 = sol(nX+2);

            const float ax1 = sol(nX+3);
            const float ay1 = sol(nX+4);
            const float az1 = sol(nX+5);

            const float jx = (ax1-ax0) / DT;
            const float jy = (ay1-ay0) / DT;
            const float jz = (az1-az0) / DT;

            *ax = ax0 + jx * age;
            *ay = ay0 + jy * age;
            *az = az0 + jz * age;

        }

};