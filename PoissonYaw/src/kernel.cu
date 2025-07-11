#include <kernel.hpp>
#include <iostream>
#include <stdio.h>
#include <poisson.h>

// CUDA Version
namespace Kernel{

    // Memory sizes
    const int nElem = IMAX * JMAX * QMAX * TMAX;
    size_t grid_size = nElem * sizeof(float);

    // Kernel Parameters
    const int threadsPerBlock = 1024; // CAUTION: THIS CANNOT BE MORE THAN YOUR TENSOR CORE COUNT
    const int blocksPerGrid = (nElem + threadsPerBlock - 1) / threadsPerBlock;

    // Poisson Solver Parameters
    const int max_iters = 1000;
    const int iter_per_epoch = 10;
    int max_epoch = max_iters / iter_per_epoch;

    // Allocate memory on the device
    float *b_grid;        
    float *h_grid;
    float *f_grid;
    float *rss_pt;
    
    __global__ void updateRedGrid(float *grid, const float *force, const float *boundary, const float w_SOR, const int nElem){

        const int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        const int row = i / JMAX;
        const int col = i % JMAX;

        if((row%2)!=(col%2)){ // Is This Cell Red?
            float dg = 0.25f * (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) - grid[i];
            grid[i] += w_SOR * dg; // SOR Factor
        }
        __syncthreads();
        return;

    }

    __global__ void updateBlackGrid(float *grid, const float *force, const float *boundary, const float w_SOR, const int nElem){

        const int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        const int row = i / JMAX;
        const int col = i % JMAX;
        if((row%2)==(col%2)){ //Is This Cell Black?
            float dg = 0.25f * (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) - grid[i];
            grid[i] += w_SOR * dg; // SOR Factor
        }
        __syncthreads();
        return;

    }

    __global__ void updateResidual(float *grid, const float *force, const float *boundary, float *rss, const int nElem){

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= nElem) return;
        if(boundary[i] == 0.0f) return; // Should I Update It?

        float dg = 0.25f * (grid[i+1] + grid[i-1] + grid[i+JMAX] + grid[i-JMAX] - force[i]) - grid[i];
        grid[i] += dg;

        // Each thread stores its value in shared memory
        extern __shared__ float block_rss[];
        block_rss[threadIdx.x] = dg * dg;
        __syncthreads();

        // Parallel reduction in shared memory
        for(int s = blockDim.x / 2; s > 0; s >>= 1){
            if(threadIdx.x < s) block_rss[threadIdx.x] += block_rss[threadIdx.x + s];
            __syncthreads();
        }

        if(threadIdx.x == 0) atomicAdd(rss, block_rss[0]); // One thread does the atomicAdd per block
        return;

    }

    void poissonInit(void){

        // Allocate memory on the device
        cudaMalloc((void **)&b_grid, grid_size);
        cudaMalloc((void **)&h_grid, grid_size);
        cudaMalloc((void **)&f_grid, grid_size);
        cudaMalloc((void **)&rss_pt, sizeof(float));

    }
    
    int poissonSolve(float *h, const float *f, const float *b, const float relTol, const float w_SOR){
        
        // Copy data to device
        cudaMemcpy(b_grid, b, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_grid, h, grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(f_grid, f, grid_size, cudaMemcpyHostToDevice);

        int epoch;
        for(epoch = 1; epoch <= max_epoch; epoch++){
            // Update Cells in Checkerboard Pattern to Enable Parallel SOR
            for(int iter = 0; iter < iter_per_epoch-1; iter++){
                updateRedGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, w_SOR, nElem); // Update All "Red" Cells
                updateBlackGrid<<<blocksPerGrid, threadsPerBlock>>>(h_grid, f_grid, b_grid, w_SOR, nElem); // Update All "Black" Cells
            }
            
            float rss = 0.0f;
            cudaMemcpy(rss_pt, &rss, sizeof(float), cudaMemcpyHostToDevice);
            updateResidual<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(h_grid, f_grid, b_grid, rss_pt, nElem); // One Gauss-Seidel Iteration to Compute Residual
            cudaMemcpy(&rss, rss_pt, sizeof(float), cudaMemcpyDeviceToHost);
            rss = sqrtf(rss) * DS / (float)(QMAX*TMAX);
            if(rss < relTol) break;
        
        }
        
        cudaMemcpy(h, h_grid, grid_size, cudaMemcpyDeviceToHost); // Copy the result back to the host
        return epoch * iter_per_epoch;

    }

}