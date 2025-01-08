#include <kernel.hpp>
#include <Eigen/Core>

#include <iostream>
#include <stdio.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// CUDA Version
namespace Kernel{
    
    __global__ void updateGrid(double *grid, double *force, double *boundary, const int nrows, const int ncols, const int num_gridpoints){

        int i = blockIdx.x * blockDim.x + threadIdx.x; // What Cell Is This?
        if((i >= num_gridpoints) || (boundary[i] == 0.0)) return; // Should I Update This Cell?
        grid[i] = (grid[i+1] + grid[i-1] + grid[i+nrows] + grid[i-nrows] - force[i]) / 4.0; // Update The Cell!
        return;

    }

    void Poisson(Eigen::MatrixXd &h, Eigen::MatrixXd &f, Eigen::MatrixXd &b, const int& num_iter){

        int num_gridpoints = h.size();
        int rows = h.rows();
        int cols = h.cols();

        // Memory sizes
        size_t b_size = num_gridpoints * sizeof(double);
        size_t h_size = num_gridpoints * sizeof(double);
        size_t f_size = num_gridpoints * sizeof(double);

        // Allocate memory on the device
        double *b_grid;        
        double *h_grid;
        double *f_grid;

        cudaMalloc((void **)&b_grid, b_size);
        cudaMalloc((void **)&h_grid, h_size);
        cudaMalloc((void **)&f_grid, f_size);

        // Copy data to device
        cudaMemcpy(b_grid, b.data(), b_size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_grid, h.data(), h_size, cudaMemcpyHostToDevice);
        cudaMemcpy(f_grid, f.data(), f_size, cudaMemcpyHostToDevice);

        // Launch the kernel
        // CAUTION: THIS CANNOT BE MORE THAN YOUR TENSOR CORE COUNT
        int threadsPerBlock = 128;
        int numBlocks = (num_gridpoints + threadsPerBlock - 1) / threadsPerBlock;

        for (int i = 0; i < num_iter; i++){
            updateGrid<<<numBlocks, threadsPerBlock>>>(h_grid, f_grid, b_grid, rows, cols, num_gridpoints);
        }

        // Copy the result back to the host
        cudaMemcpy(b.data(), b_grid, b_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h.data(), h_grid, h_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(f.data(), f_grid, f_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(b_grid);
        cudaFree(h_grid);
        cudaFree(f_grid);

    }
}
