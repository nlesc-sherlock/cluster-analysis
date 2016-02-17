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
    __global__ void sumSquared(double *output, double *x, int n);
    __global__ void computeNCC(double *output, float* x, double sumsq_x, float* y, double sumsq_y, int n);
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
    int ti = threadIdx.x;
    __shared__ double shmem[LARGETB];

    //compute thread-local sums
    double sumsq = 0.0;
    for (int i=ti; i < n; i+=LARGETB) {
        sumsq += x[i] * x[i];
    }
    shmem[ti] = sumsq;
    __syncthreads();

    for (unsigned int s=LARGETB/2; s>0; s>>=1) {
        if (ti < s) {
            shmem[ti] += shmem[ti + s];
        }
        __syncthreads();
    }

    //write result
    if (ti == 0) {
        output[0] = shmem[0];
    }
}
 
__global__ void computeNCC(double *output, float *x, double sumsq_x, float *y, double sumsq_y, int n) {
    int ti = threadIdx.x;
    __shared__ double shmem[LARGETB];

    //compute thread-local sums
    double sumxy = 0.0;        
    for (int i=ti; i < n; i+=LARGETB) {
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
        output[0] = shmem[0]/sqrtf(sumsq_x * sumsq_y);
    }
}


/*
 * Simple CUDA helper functions to reduce the output of a reducing kernel with multiple
 * thread blocks to a single value
 *
 * This function performs a reduction for the max and the location of the max
 *
 * This function is to be called with only one thread block
 */
__global__ void maxlocFloats(int *output_loc, float *output_float, int *input_loc, float *input_float, int n) {

    int ti = threadIdx.x;
    __shared__ float shmax[LARGETB];
    __shared__ int shind[LARGETB];

    //compute thread-local variables
    float max = -1.0f;
    float val = 0.0f;
    int loc = -1;
    for (int i=ti; i < n; i+=LARGETB) {
         val = input_float[i];
         if (val > max) {
             max = val;
             loc = input_loc[i];
         }
    }
        
    //store local variables in shared memory
    shmax[ti] = max;
    shind[ti] = loc;
    __syncthreads();
        
    //reduce local variables
    for (unsigned int s=LARGETB/2; s>0; s>>=1) {
        if (ti < s) {
            float v1 = shmax[ti];
            float v2 = shmax[ti + s];
            if (v1 < v2) {
                shmax[ti] = v2;
                shind[ti] = shind[ti + s];
            }
        }
        __syncthreads();
    }
        
    //write result
    if (ti == 0) {
        output_float[0] = shmax[0]; 
        output_loc[0] = shind[0]; 
    }

}

