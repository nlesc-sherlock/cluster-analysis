/*
 * Copyright (c) 2012-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu.compare;

import nl.minvenj.nfi.prnu.compare.PatternComparator;
import nl.minvenj.nfi.prnu.Util;
import nl.minvenj.nfi.cuba.cudaapi.*;
import jcuda.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;
import jcuda.driver.*;



/**
 * This class is performs a Normalized Cross Correlation on the GPU
 */
public class NormalizedCrossCorrelation implements PatternComparator {

    //cuda handles
    protected CudaContext _context;
    protected CudaStream _stream;

    //references to GPU memory
    protected CudaMemFloat _d_input1;
    protected CudaMemFloat _d_input2;
    protected CudaMemDouble _d_output;

    protected CudaMemFloat _d_x_patterns[];
    protected CudaMemFloat _d_y_patterns[];

    //handles to CUDA kernels
    protected CudaFunction _sumSquared;
    protected CudaFunction _computeNCC;
    protected CudaFunction _sumDoubles;

    //parameterlists for kernel invocations
    protected Pointer sumSquared;
    protected Pointer sumDoubles;
    protected Pointer computeNCC;

    //PRNU pattern dimensions
    int h;
    int w;
    int n;

    public int num_patterns;

    //gpu performance accounting
    double total_kernel_time = 0.0;
    double total_bandwidth = 0.0;
    int total_called = 0;

    /**
     * Constructor for the Normalized Cross Correlation GPU implementation
     *
     * @param h - the image height in pixels
     * @param w - the image width in pixels
     * @param context   - the CudaContext as created by the factory
     * @param stream    - the CudaStream as created by the factory
     * @param module    - the CudaModule containing the kernels compiled by the PRNUFilterFactory
     * @param lib       - another CudaModule containing the kernels compiled by the PRNUFilterFactory
     */
    public NormalizedCrossCorrelation(int h, int w, CudaContext context, CudaModule module, CudaModule lib) {
        _context = context;
        _stream = new CudaStream();
        this.h = h;
        this.w = w;
        this.n = h*w;

        //setup CUDA functions
        JCudaDriver.setExceptionsEnabled(true);
        int threads = 256;
        int reducing_thread_blocks = 1024; //optimally this equals the number of SMs in the GPU

        //get number of SMs
        //int num_sm =_context.getDevice().getComputeModules();
        //System.out.println("detected " + num_sm + " SMs on GPU");
        //reducing_thread_blocks = num_sm;
    

        _sumSquared = module.getFunction("sumSquared");
        _sumSquared.setDim(   reducing_thread_blocks, 1, 1,
                threads, 1, 1);

        _computeNCC = module.getFunction("computeNCC");
        _computeNCC.setDim(   reducing_thread_blocks, 1, 1,
                threads, 1, 1);

        _sumDoubles = lib.getFunction("sumDoubles");
        _sumDoubles.setDim(   1, 1, 1,
                threads, 1, 1);

        //allocate local variables in GPU memory
        _d_input1 = _context.allocFloats(h*w);
        _d_input2 = _context.allocFloats(h*w);
        _d_output = _context.allocDoubles(reducing_thread_blocks);

        long free[] = new long[1];
        long total[] = new long[1];
        JCuda.cudaMemGetInfo(free, total);
        long patternSize = h*w*4; //size of pattern on the GPU
        int fit_patterns = (int)(free[0] / patternSize);
        num_patterns = fit_patterns/2 ; //removed +1
        System.out.println("Allocating space for in total " + 2*num_patterns + " patterns on the GPU");

        _d_x_patterns = new CudaMemFloat[num_patterns];
        _d_y_patterns = new CudaMemFloat[num_patterns];

        _d_x_patterns[0] = _d_input1;
        _d_y_patterns[0] = _d_input2;
        for (int i=1; i<num_patterns; i++) {
            _d_x_patterns[i] = _context.allocFloats(h*w);
            _d_y_patterns[i] = _context.allocFloats(h*w);
        }

        //construct parameter lists for the CUDA kernels
        sumSquared = Pointer.to(
                Pointer.to(_d_output.getDevicePointer()),
                Pointer.to(_d_input1.getDevicePointer()),
                Pointer.to(new int[]{n})
                );

        sumDoubles = Pointer.to(
                Pointer.to(_d_output.getDevicePointer()),
                Pointer.to(_d_output.getDevicePointer()),
                Pointer.to(new int[]{reducing_thread_blocks})
                );

        computeNCC = Pointer.to(
                Pointer.to(_d_output.getDevicePointer()),
                Pointer.to(_d_input1.getDevicePointer()),
                Pointer.to(_d_input2.getDevicePointer()),
                Pointer.to(new int[]{n})
        );


    }

    /**
     * This method performs an array of comparisons between patterns
     * It computes the NCC scores between all patterns in xPatterns and those in yPatterns
     *
     * @param xPatterns     array of PRNU patterns stored as float arrays
     * @param yPatterns     array of PRNU patterns stored as float arrays
     * @param predicate     a boolean matrix denoting which comparsions are to be made and which not
     * @returns             a double matrix with all NCC scores from comparing all patterns in x with all in y
     */
    public double[][] compareGPU(Pointer[] xPatterns, Pointer[] yPatterns, boolean[][] predicate) {

        double result[][] = new double[num_patterns][num_patterns];
        int pattern_length = 0;
        int called = 0;

        //copy inputs to the GPU (host to device)
        for (int i=0; i < num_patterns; i++) {
            if (xPatterns[i] != null) {
                _d_x_patterns[i].copyHostToDeviceAsync(xPatterns[i], h*w*Sizeof.FLOAT, _stream);
            }
            if (yPatterns[i] != null) {
                _d_y_patterns[i].copyHostToDeviceAsync(yPatterns[i], h*w*Sizeof.FLOAT, _stream);
            }
        }

        //call the kernel
        _stream.synchronize();
        long start = System.nanoTime();

        for (int i=0; i < num_patterns; i++) {
            for (int j=0; j < num_patterns; j++) {
                if (predicate[i][j]) {

                    //call the kernel
                    computeNCC = Pointer.to(
                            Pointer.to(_d_output.getDevicePointer()),
                            Pointer.to(_d_x_patterns[i].getDevicePointer()),
                            Pointer.to(_d_y_patterns[j].getDevicePointer()),
                            Pointer.to(new int[]{n})
                    );
                    _computeNCC.launch(_stream, computeNCC);
                    _sumDoubles.launch(_stream, sumDoubles);

                    //copy output (device to host)
                    double out[] = new double[1];
                    _d_output.copyDeviceToHostAsync(out, 1, _stream);
                    _stream.synchronize();
                    result[i][j] = out[0];
                    called++;

                } else {
                    result[i][j] = 0.0;
                }
            }
        }

        _stream.synchronize();

        double gpu_time = (System.nanoTime() - start)/1e6;
        total_kernel_time += gpu_time; // ms
        total_bandwidth += ((long)called*(long)w*h*(long)4*(long)2)/1e9 ; //GB
        total_called += called;

        return result;
    }


    /**
     * This method performs an array of comparisons between patterns
     * It computes the NCC scores between pattern x and y
     *
     * @param x         a PRNU patterns stored as a float array
     * @param sumsq_x   the sum squared of pattern x
     * @param y         a PRNU patterns stored as a float array
     * @param sumsq_y   the sum squared of pattern y
     * @returns         the NCC scores from comparing patterns x and y
     */
    public double compareGPU(float[] x, double sumsq_x, float[] y, double sumsq_y) {
        //copy inputs to the GPU (host to device)
        _d_input1.copyHostToDeviceAsync(x, _stream);
        _d_input2.copyHostToDeviceAsync(y, _stream);

        //call the kernel
        _stream.synchronize();
        long start = System.nanoTime();
        _computeNCC.launch(_stream, computeNCC);
        _sumDoubles.launch(_stream, sumDoubles);

        //copy output (device to host)
        double result[] = new double[1];
        _d_output.copyDeviceToHostAsync(result, 1, _stream);
        _stream.synchronize();
        double res = result[0] / Math.sqrt(sumsq_x * sumsq_y);

        double gpu_time = (System.nanoTime() - start)/1e6;
        total_kernel_time += gpu_time;
        total_bandwidth += ((x.length*4*2)/1e9 ) / (gpu_time/1e3);  // GB/s
        total_called++;
        System.out.println("computeNCC GPU: " + res + " took: " + gpu_time + " ms.");

        return res;
    }

    /**
     * This method computes the sum of squares of a PRNU pattern
     *
     * @param pattern   a float array containing the pattern
     * @returns         the sum of squares as a double
     */
    public double sumSquaredGPU(Pointer pattern) {
        //copy input to GPU
        _d_input1.copyHostToDeviceAsync(pattern, h*w*Sizeof.FLOAT, _stream);

        //_stream.synchronize();
        //long start = System.nanoTime();

        //call the kernel
        _sumSquared.launch(_stream, sumSquared);
        _sumDoubles.launch(_stream, sumDoubles);

        //copy output (device to host)
        double result[] = new double[1];
        _d_output.copyDeviceToHostAsync(result, 1, _stream);
        _stream.synchronize();

        //double gpu_time = (System.nanoTime() - start)/1e6;
        //System.out.println("sumSquared GPU: " + result[0] + " took: " + gpu_time + " ms.");

        return result[0];
    }

    /**
     * This method prints some GPU performance stats
     */
    public void printTime() {
        System.out.println("total GPU kernel time: " + total_kernel_time + " ms. on average: " + (total_kernel_time / (double)total_called) + " ms.");
        System.out.println("Average bandwidth per kernel: " + (total_bandwidth / (total_kernel_time/1000.0)) + " GB/s.");
    }

    /**
     * This method computes the sum of squares of a PRNU pattern on the CPU
     * 
     * @param pattern   a float array containing a PRNU pattern
     * @returns         a double containing the sum of squares
     */
    public static double sumSquared(final float[] pattern) {
	    double sumsq = 0.0;
	    for (int i=0; i<pattern.length; i++) {
	        sumsq += pattern[i] * pattern[i];
	    }
	    return sumsq;
    }

    /**
     * This method performs an array of comparisons between patterns on the CPU
     * It computes the NCC scores between pattern x and y
     *
     * @param x         a PRNU patterns stored as a float array
     * @param sumsq_x   the sum squared of pattern x
     * @param y         a PRNU patterns stored as a float array
     * @param sumsq_y   the sum squared of pattern y
     * @returns         the NCC scores from comparing patterns x and y
     */
    public static double compare(final float[] x, double sumsq_x, final float[] y, double sumsq_y) {
	    double sum_xy = 0.0;
            for (int i=0; i<x.length; i++) {
                sum_xy += x[i] * y[i];
       	    }
    	return (sum_xy / Math.sqrt(sumsq_x * sumsq_y));
    }


    /**
     * Cleans up GPU memory 
     */
    public void cleanup() {
        _d_input1.free();
        _d_input2.free();
        _d_output.free();
        for (int i=1; i<num_patterns; i++) {
            _d_x_patterns[i].free();
            _d_y_patterns[i].free();
        }
    }


}
