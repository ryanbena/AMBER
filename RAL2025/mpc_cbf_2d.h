#include <iostream>
#include "math.h"
#include "poisson.h"
#include "utils.h"
#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Sparse>

#define LINEAR_STATE_LENGTH_2D 2
#define LINEAR_INPUT_LENGTH_2D 2

// MPC Controller Class
class MPC2D{
    
    public: 
        
        MPC2D(void){
            
            N_HORIZON = TMAX - 1;
            nX = (N_HORIZON+1) * LINEAR_STATE_LENGTH_2D;
            nU = N_HORIZON * LINEAR_INPUT_LENGTH_2D;
            nZ = nX + nU; // Optimization Variables
            nC = nX + N_HORIZON;
           
            cost_P = Eigen::MatrixXd::Zero(nZ, nZ);
            cost_q = Eigen::VectorXd::Zero(nZ);
            constraint_A = Eigen::MatrixXd::Zero(nC, nZ);
            constraint_upper = Eigen::VectorXd::Zero(nC);
            constraint_lower = Eigen::VectorXd::Zero(nC); 
            sol = Eigen::VectorXd::Zero(nZ);

            // Build Cost Function
            Px.setIdentity(LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D);
            Px.row(0) << 1.0, 0.0;
            Px.row(1) << 0.0, 1.0;

            Pu.setIdentity(LINEAR_INPUT_LENGTH_2D, LINEAR_INPUT_LENGTH_2D); 
            Pu.row(0) << 1.0, 0.0;
            Pu.row(1) << 0.0, 1.0;

            for (int k=0; k<N_HORIZON; k++){
                cost_P.block<LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D>(k*LINEAR_STATE_LENGTH_2D, k*LINEAR_STATE_LENGTH_2D) = Px;
                cost_P.block<LINEAR_INPUT_LENGTH_2D, LINEAR_INPUT_LENGTH_2D>(nX + k*LINEAR_INPUT_LENGTH_2D, nX + k*LINEAR_INPUT_LENGTH_2D) = Pu;
                constraint_A.block<LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D>((k+1)*LINEAR_STATE_LENGTH_2D, (k+1)*LINEAR_STATE_LENGTH_2D) = -1.0f * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D); // Identity 
                constraint_A.block<LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D>((k+1)*LINEAR_STATE_LENGTH_2D, k*LINEAR_STATE_LENGTH_2D) = Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D); // A Matrix
                constraint_A.block<LINEAR_STATE_LENGTH_2D, LINEAR_INPUT_LENGTH_2D>((k+1)*LINEAR_STATE_LENGTH_2D, nX + k*LINEAR_INPUT_LENGTH_2D) = DT * Eigen::MatrixXd::Identity(LINEAR_STATE_LENGTH_2D, LINEAR_INPUT_LENGTH_2D); // B Matrix
                constraint_upper(nX+k) = OSQP_INFTY; 
                constraint_lower(nX+k) = -OSQP_INFTY; // Turn off safety constraints
            }
            cost_P.block<LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D>(N_HORIZON*LINEAR_STATE_LENGTH_2D, N_HORIZON*LINEAR_STATE_LENGTH_2D) = terminal*Px; // Terminal Cost
            constraint_A.block<LINEAR_STATE_LENGTH_2D,LINEAR_STATE_LENGTH_2D>(0,0).setIdentity(); // Initial Condition Constraint

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
   
            // Update Dynamics Constraints
            constraint_lower.segment(0, LINEAR_STATE_LENGTH_2D) << x[0], x[1];
            constraint_upper.segment(0, LINEAR_STATE_LENGTH_2D) << x[0], x[1];

            // Update Obstacle State Constraints
            const float eps = 3.0f;

            for(int k=0; k<=N_HORIZON; k++){
                
                const int idx = k*LINEAR_STATE_LENGTH_2D;
                const float rxk = sol(idx+0);
                const float ryk = sol(idx+1);

                // Indices Corresponding to Current Position
                const float ir = y_to_i(ryk,xc[1]);
                const float jr = x_to_j(rxk,xc[0]);

                // Saturated Because of Finite Grid Size
                const float ic = fminf(fmaxf(0.0f, ir), (float)(IMAX-1)); // Saturated Because of Finite Grid Size
                const float jc = fminf(fmaxf(0.0f, jr), (float)(JMAX-1)); // Numerical Derivatives Shrink Effective Grid Size

                // Get Safety Function Values
                float h = bilinear_interpolation(hgrid, ic, jc);

                // Compute Gradients
                const float ip = fminf(fmaxf(0.0f, ic+eps), (float)(IMAX-1));
                const float im = fminf(fmaxf(0.0f, ic-eps), (float)(IMAX-1));
                const float jp = fminf(fmaxf(0.0f, jc+eps), (float)(JMAX-1));
                const float jm = fminf(fmaxf(0.0f, jc-eps), (float)(JMAX-1));

                const float dhdxp = (bilinear_interpolation(hgrid, ip, jp) - bilinear_interpolation(hgrid, ip, jm)) / ((jp-jm)*DS);
                const float dhdxc = (bilinear_interpolation(hgrid, ic, jp) - bilinear_interpolation(hgrid, ic, jm)) / ((jp-jm)*DS);
                const float dhdxm = (bilinear_interpolation(hgrid, im, jp) - bilinear_interpolation(hgrid, im, jm)) / ((jp-jm)*DS);
                float dhdx = (dhdxp + dhdxc + dhdxm) / 3.0f;
                const float dhdyp = (bilinear_interpolation(hgrid, ip, jp) - bilinear_interpolation(hgrid, im, jp)) / ((ip-im)*DS);
                const float dhdyc = (bilinear_interpolation(hgrid, ip, jc) - bilinear_interpolation(hgrid, im, jc)) / ((ip-im)*DS);
                const float dhdym = (bilinear_interpolation(hgrid, ip, jm) - bilinear_interpolation(hgrid, im, jm)) / ((ip-im)*DS);
                float dhdy = (dhdyp + dhdyc + dhdym) / 3.0f;

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
                }

                // Update Constraints
                const float alpha = expf(-WN*DT);
                if(k!=N_HORIZON){
                    constraint_A(nX+k, idx+0) = -alpha*dhdx;
                    constraint_A(nX+k, idx+1) = -alpha*dhdy;
                    constraint_lower(nX+k) = -alpha * (dhdx*rxk + dhdy*ryk - h);
                    // ISSf Term
                    const float ISSf1 = 10.0f;
                    const float ISSf2 = 10.0f;
                    const float Lgh_norm = sqrtf(dhdx*dhdx+dhdy*dhdy);
                    constraint_lower(nX+k) += (Lgh_norm/ISSf1 + Lgh_norm*Lgh_norm/ISSf2) * DT;
                }
                if(k!=0){
                    constraint_A(nX+k-1, idx+0) = dhdx;
                    constraint_A(nX+k-1, idx+1) = dhdy;
                    constraint_lower(nX+k-1) += dhdx*rxk + dhdy*ryk - h;
                }

                // Update Linear Cost
                Eigen::VectorXd state_goal(LINEAR_STATE_LENGTH_2D);
                state_goal << xd[0], xd[1];
                cost_q.segment(idx, LINEAR_STATE_LENGTH_2D) = -state_goal.transpose() * Px;
        
            }
            
            Eigen::SparseMatrix<double> constraint_A_sparse = constraint_A.sparseView();
            solver.updateLinearConstraintsMatrix(constraint_A_sparse); 
            solver.updateBounds(constraint_lower, constraint_upper);
            cost_q.segment(N_HORIZON*LINEAR_STATE_LENGTH_2D, LINEAR_STATE_LENGTH_2D) *= terminal;
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

            for(int i = 0; i < nZ; i++) z[i] = sol(i);

        }

        void set_input(float *vx, float *vy){

            *vx = sol(nX+0);
            *vy = sol(nX+1);

        }

};




