#include <kernel.hpp>
#include <iostream>
#include <stdio.h>

//#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// CUDA Version
namespace Kernel{
    
    __global__ void updateGrid(double *grid, const double *force, const double *boundary, const int nrows, const int ncols, const double cell_size){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        
        if((i >= (nrows*ncols)) || (boundary[i] == 0.0)) return; // Should I Update This Cell?

        double dg = grid[i+1] + grid[i-1] + grid[i+nrows] + grid[i-nrows]; // Update The Cell!
        dg -= force[i] * cell_size * cell_size;
        dg /= 4.0;
        dg -= grid[i];
        atomicAdd(grid+i, dg);

        return;

    }

    __global__ void updateResidual(double *grid, const double *force, const double *boundary, double *rss, const int nrows, const int ncols, const double cell_size){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        
        if((i >= (nrows*ncols)) || (boundary[i] == 0.0)) return; // Should I Update This Cell?

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
        const int max_iters = 10000;
        const int iter_per_epoch = 100;
        int max_epoch = max_iters / iter_per_epoch;

        for(epoch = 1; epoch <= max_epoch; epoch++){
            for(iter = 0; iter < iter_per_epoch-1; iter++){
                updateGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rows, cols, cell_size);
            }
            
            rss0[0] = 0.0;
            cudaMemcpy(rss, rss0, sizeof(double), cudaMemcpyHostToDevice);
            updateResidual<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rss, rows, cols, cell_size);
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