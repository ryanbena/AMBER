#include <kernel.hpp>
#include <iostream>
#include <stdio.h>
#include <poisson.h>

#define w_SOR 1.9f

// CUDA Version
namespace Kernel{
    
    __global__ void updateRedGrid(float *grid, const float *force, const float *boundary){

        int i = (blockIdx.x * blockDim.x) + threadIdx.x; // What Cell Is This?
        static const int nElem = tmax*imax*jmax;
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        int row = i / jmax;
        int col = i % jmax;
        if((row%2)!=(col%2)) return; // Is This Cell Red?

        float dg = grid[i+1] + grid[i-1] + grid[i+jmax] + grid[i-jmax]; // Update The Cell!
        dg -= force[i] * ds * ds;
        dg /= 4.0f;
        dg -= grid[i];
        grid[i] += w_SOR * dg; // SOR Factor
        //return;

    }

    __global__ void updateBlackGrid(float *grid, const float *force, const float *boundary){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        static const int nElem = tmax*imax*jmax;
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        int row = i / jmax;
        int col = i % jmax;
        if((row%2)==(col%2)) return; //Is This Cell Black?

        float dg = grid[i+1] + grid[i-1] + grid[i+jmax] + grid[i-jmax]; // Update The Cell!
        dg -= force[i] * ds * ds;
        dg /= 4.0f;
        dg -= grid[i];
        grid[i] += w_SOR * dg; // SOR Factor
        //return;

    }

    __global__ void updateResidual(float *grid, const float *force, const float *boundary, float *rss){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        static const int nElem = tmax*imax*jmax;
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        float dg = grid[i+1] + grid[i-1] + grid[i+jmax] + grid[i-jmax]; // Update The Cell!
        dg -= force[i] * ds * ds;
        dg /= 4.0f;
        dg -= grid[i];
        grid[i] += w_SOR * dg;
        *rss += dg * dg;
        //return;

    }

    int Poisson(float *h, const float *f, const float *b, const float relTol){

        // Memory sizes
        size_t grid_size = tmax * imax * jmax * sizeof(float);

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
        const int blocksPerGrid = (tmax * imax * jmax + threadsPerBlock - 1) / threadsPerBlock;

        int epoch, iter;
        const int max_iters = 10000;
        const int iter_per_epoch = 40;
        int max_epoch = max_iters / iter_per_epoch;

        
        for(epoch = 1; epoch <= max_epoch; epoch++){
            for(iter = 0; iter < iter_per_epoch-1; iter++){

                // Update Cells in Checkerboard Pattern to Enable Parallel SOR
                updateRedGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid); // Update All "Red" Cells
                updateBlackGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid); // Update All "Black" Cells
            
            }
            
            float rss = 0.0f;
            cudaMemcpy(rss_pt, &rss, sizeof(float), cudaMemcpyHostToDevice);
            updateResidual<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, rss_pt); // One Gauss-Seidel Iteration to Compute Residual
            cudaMemcpy(&rss, rss_pt, sizeof(float), cudaMemcpyDeviceToHost);
            rss = sqrtf(rss) * ds;
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