#include <kernel.hpp>
#include <iostream>
#include <stdio.h>
#include <poisson.h>

// CUDA Version
namespace Kernel{
    
    __constant__ int nElem = IMAX * JMAX * QMAX * TMAX;
    
    __global__ void updateRedGrid(float *grid, const float *force, const float *boundary, const float w_SOR){

        const int i = (blockIdx.x * blockDim.x) + threadIdx.x; // What Cell Is This?
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        const int row = i / JMAX;
        const int col = i % JMAX;
        if((row%2)!=(col%2)){ // Is This Cell Red?
            float dg = (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) / 4.0f - grid[i];
            grid[i] += w_SOR * dg; // SOR Factor
        }
        return;

    }

    __global__ void updateBlackGrid(float *grid, const float *force, const float *boundary, const float w_SOR){

        const int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        const int row = i / JMAX;
        const int col = i % JMAX;
        if((row%2)==(col%2)){ //Is This Cell Black?
            float dg = (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) / 4.0f - grid[i];
            grid[i] += w_SOR * dg; // SOR Factor
        }
        return;

    }

    __global__ void updateResidual(float *grid, const float *force, const float *boundary, float *rss){

        const int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        float dg = (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) / 4.0f - grid[i];
        grid[i] += dg;
        atomicAdd(rss, dg*dg);
        return;

    }

    int Poisson(float *h, const float *f, const float *b, const float relTol, const float w_SOR){

        // Memory sizes
        size_t grid_size = IMAX * JMAX * QMAX * TMAX * sizeof(float);

        // Allocate memory on the device
        float *b_grid;        
        float *h_grid;
        float *f_grid;
        float *rss_pt;
        cudaMalloc((void **)&b_grid, grid_size);
        cudaMalloc((void **)&h_grid, grid_size);
        cudaMalloc((void **)&f_grid, grid_size);
        cudaMalloc((void **)&rss_pt, sizeof(float));
        
        // Copy data to device
        cudaMemcpy(b_grid, b, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_grid, h, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(f_grid, f, grid_size, cudaMemcpyHostToDevice);

        // Launch the kernel
        const int threadsPerBlock = 1024; // CAUTION: THIS CANNOT BE MORE THAN YOUR TENSOR CORE COUNT
        const int blocksPerGrid = (IMAX * JMAX * QMAX * TMAX + threadsPerBlock - 1) / threadsPerBlock;

        int epoch, iter;
        const int max_iters = 1000;
        const int iter_per_epoch = 20;
        int max_epoch = max_iters / iter_per_epoch;

        for(epoch = 1; epoch <= max_epoch; epoch++){
            // Update Cells in Checkerboard Pattern to Enable Parallel SOR
            for(iter = 0; iter < iter_per_epoch-1; iter++){
                updateRedGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, w_SOR); // Update All "Red" Cells
                updateBlackGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, w_SOR); // Update All "Black" Cells
            }
            
            float rss = 0.0f;
            cudaMemcpy(rss_pt, &rss, sizeof(float), cudaMemcpyHostToDevice);
            updateResidual<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rss_pt); // One Gauss-Seidel Iteration to Compute Residual
            cudaMemcpy(&rss, rss_pt, sizeof(float), cudaMemcpyDeviceToHost);
            rss = sqrtf(rss) * DS / (float)(QMAX*TMAX);
            if(rss < relTol) break;
        
        }
        
        cudaMemcpy(h, h_grid, grid_size, cudaMemcpyDeviceToHost); // Copy the result back to the host

        // Free device memory
        cudaFree(b_grid);
        cudaFree(h_grid);
        cudaFree(f_grid);
        cudaFree(rss_pt);
        
        return epoch * iter_per_epoch;

    }

}