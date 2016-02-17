/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * This file contains CUDA kernels for comparing two PRNU noise patterns
 * using Peak To Correlation Energy.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */

// Should be a power of two!!
#define LARGETB 1024
//function interfaces to prevent C++ garbling the kernel names
extern "C" {
    __global__ void sumSquared(double *output, float *x, int n);
    __global__ void computeNCC(double *output, float *x, double sumsq_x, float* y, double sumsq_y, int n);
}



/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 * 
 * This function performs a sum of an array of doubles
 *
 * This function is to be called with only a single thread block
 */
__global__ void sumSquared(double *output, float *x, int n) {
    int _x = blockIdx.x * LARGETB + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * LARGETB;

    __shared__ double shmem[LARGETB];

    if (ti < n) {

    	//compute thread-local sums
    	double sumsq = 0.0;
    	for (int i=_x; i < n; i+=step_size) {
            sumsq += x[i] * x[i];
        }
  
        //store local sums in shared memory
        shmem[ti] = sumsq;
        __syncthreads();

        //reduce local sums
        for (unsigned int s=LARGETB/2; s>0; s>>=1) {
            if (ti < s) {
            shmem[ti] += shmem[ti + s];
            }
            __syncthreads();
        }

        //write result
        if (ti == 0) {
            output[blockIdx.x] = shmem[0];
        }
    }
}
 
__global__ void computeNCC(double *output, float *x, float *y,  int n) {
    int _x = blockIdx.x * LARGETB + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * LARGETB;

    __shared__ double shmem[LARGETB];

    if (ti < n) {

        //compute thread-local sums
        double sumxy = 0.0;        
        for (int i=_x; i < n; i+=step_size) {
            sumxy += x[i] * y[i];
        }

        //store local sums in shared memory
        shmem[ti] = sumxy;
        __syncthreads();
        
        //reduce local sums
        for (unsigned int s=LARGETB/2; s>0; s>>=1) {
            if (ti < s) {
                shmem[ti] += shmem[ti + s];
            }
            __syncthreads();
        }
        
        //write result
        if (ti == 0) {
            output[blockIdx.x] = shmem[0];
        }
    }
}   


