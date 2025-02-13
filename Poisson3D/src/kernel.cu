#include <kernel.hpp>
#include <iostream>
#include <stdio.h>

#define w_SOR 1.9

// CUDA Version
namespace Kernel{
    
    __global__ void updateRedGrid(double *grid, const double *force, const double *boundary, const int nrows, const int ncols, const double cell_size){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if((i >= (nrows*ncols)) || (boundary[i] == 0.0)) return; // Should I Update It?

        int row = i / ncols;
        int col = i % ncols;
        if((row%2)!=(col%2)) return; // Is This Cell Red?

        double dg = grid[i+1] + grid[i-1] + grid[i+nrows] + grid[i-nrows]; // Update The Cell!
        dg -= force[i] * cell_size * cell_size;
        dg /= 4.0;
        dg -= grid[i];
        atomicAdd(grid+i, w_SOR*dg); // SOR Factor
        return;

    }

    __global__ void updateBlackGrid(double *grid, const double *force, const double *boundary, const int nrows, const int ncols, const double cell_size){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if((i >= (nrows*ncols)) || (boundary[i] == 0.0)) return; // Should I Update It?

        int row = i / ncols;
        int col = i % ncols;
        if((row%2)==(col%2)) return; //Is This Cell Black?

        double dg = grid[i+1] + grid[i-1] + grid[i+nrows] + grid[i-nrows]; // Update The Cell!
        dg -= force[i] * cell_size * cell_size;
        dg /= 4.0;
        dg -= grid[i];
        atomicAdd(grid+i, w_SOR*dg); // SOR Factor
        return;

    }

    __global__ void updateResidual(double *grid, const double *force, const double *boundary, double *rss, const int nrows, const int ncols, const double cell_size){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if((i >= (nrows*ncols)) || (boundary[i] == 0.0)) return; // Should I Update It?

        double dg = grid[i+1] + grid[i-1] + grid[i+nrows] + grid[i-nrows]; // Update The Cell!
        dg -= force[i] * cell_size * cell_size;
        dg /= 4.0;
        dg -= grid[i];
        atomicAdd(grid+i, dg);
        atomicAdd(rss, dg*dg);
        return;

    }

    int Poisson(double *h, const double *f, const double *b, const double& relTol, const int& rows, const int& cols, const double& cell_size){

        // Memory sizes
        size_t grid_size = rows * cols * sizeof(double);

        // Allocate memory on the device
        double *b_grid;        
        double *h_grid;
        double *f_grid;
        cudaMalloc((void **)&b_grid, grid_size);
        cudaMalloc((void **)&h_grid, grid_size);
        cudaMalloc((void **)&f_grid, grid_size);
        
        // Copy data to device
        cudaMemcpy(b_grid, b, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_grid, h, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(f_grid, f, grid_size, cudaMemcpyHostToDevice);

        // Launch the kernel
        const int threadsPerBlock = 1024; // CAUTION: THIS CANNOT BE MORE THAN YOUR TENSOR CORE COUNT
        const int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate memory for residual calculation
        double *rss;
        cudaMalloc((void **)&rss, sizeof(double));
        double rss0[1];

        int epoch, iter;
        const int max_iters = 1000;
        const int iter_per_epoch = 80;
        int max_epoch = max_iters / iter_per_epoch;

        for(epoch = 1; epoch <= max_epoch; epoch++){
            for(iter = 0; iter < iter_per_epoch-1; iter++){

                // Update Cells in Checkerboard Pattern to Enable Parallel SOR
                updateRedGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rows, cols, cell_size); // Update All "Red" Cells
                updateBlackGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rows, cols, cell_size); // Update All "Black" Cells
            
            }
            
            rss0[0] = 0.0;
            cudaMemcpy(rss, rss0, sizeof(double), cudaMemcpyHostToDevice);
            updateResidual<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rss, rows, cols, cell_size); // One Gauss-Seidel Iteration to Compute Residual
            cudaMemcpy(rss0, rss, sizeof(double), cudaMemcpyDeviceToHost);
            rss0[0] = sqrt(rss0[0]) * cell_size;
            if(rss0[0] < relTol) break;
        
        }
        
        cudaMemcpy(h, h_grid, grid_size, cudaMemcpyDeviceToHost); // Copy the result back to the host

        // Free device memory
        cudaFree(b_grid);
        cudaFree(h_grid);
        cudaFree(f_grid);
        cudaFree(rss);
        
        return epoch * iter_per_epoch;

    }

}