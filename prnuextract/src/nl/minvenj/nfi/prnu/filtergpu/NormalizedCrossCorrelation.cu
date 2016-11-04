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
#ifndef block_size_x
#define block_size_x 1024
#endif

#ifndef vector
#define vector 1
#endif

#if (vector==1)
#define floatvector float
#elif (vector == 2)
#define floatvector float2
#elif (vector == 4)
#define floatvector float4
#endif

//function interfaces to prevent C++ garbling the kernel names
extern "C" {
    __global__ void sumSquared(double *output, float *x, int n);
    __global__ void computeNCC(double *output, float *x, float *y,  int n);
}

# 

/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 * 
 * This function performs a sum of an array of doubles
 *
 * This function is to be called with only a single thread block
 */
__global__ void sumSquared(double *output, float *x, int n) {
    int _x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * block_size_x;

    __shared__ double shmem[block_size_x];

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
        for (unsigned int s=block_size_x/2; s>0; s>>=1) {
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
    int _x = blockIdx.x * block_size_x + threadIdx.x;
    int ti = threadIdx.x;
    int step_size = gridDim.x * block_size_x;

    __shared__ double shmem[block_size_x];

    if (ti < n) {

        //compute thread-local sums
        double sumxy = 0.0;        
        for (int i=_x; i < n/vector; i+=step_size) {
            // sumxy += x[i] * y[i];
      
            floatvector v = x[i];
            floatvector w = y[i];
            #if vector == 1
            sumxy += v * w;
            #elif vector == 2
            sumxy += v.x * w.x + v.y * w.y;
            #elif vector == 4
            sumxy += v.x * w.x + v.y * w.y + v.z * w.z + v.w * w.w;
            #endif
        }
        //store local sums in shared memory
        shmem[ti] = sumxy;
        __syncthreads();
        
        //reduce local sums
        for (unsigned int s=block_size_x/2; s>0; s>>=1) {
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


