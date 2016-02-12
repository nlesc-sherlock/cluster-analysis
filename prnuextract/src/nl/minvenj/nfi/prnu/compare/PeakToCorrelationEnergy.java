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
package nl.minvenj.nfi.prnu.compare;

import nl.minvenj.nfi.prnu.Util;
import nl.minvenj.nfi.cuba.cudaapi.*;
import jcuda.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;
import jcuda.driver.*;
import jcuda.jcufft.*;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;

/**
 * Class for applying a series of Wiener Filters to a PRNU pattern 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PeakToCorrelationEnergy {

	protected CudaContext _context;
	protected CudaStream _stream1;
	protected CudaStream _stream2;
	protected CudaMemFloat _d_input;
    protected CudaEvent _event;

	//handles to CUDA kernels
	protected CudaFunction _tocomplex;
	protected CudaFunction _tocomplexandflip;
	protected CudaFunction _computeEnergy;
	protected CudaFunction _sumDoubles;
	protected CudaFunction _computeCrossCorr;
	protected CudaFunction _findPeak;
	protected CudaFunction _maxlocFloats;

	//handle for CUFFT plan
	protected cufftHandle _plan1;
	protected cufftHandle _plan2;

	//handles to device memory arrays
	protected CudaMemFloat _d_inputx;
	protected CudaMemFloat _d_inputy;
	protected CudaMemFloat _d_x;
	protected CudaMemFloat _d_y;
	protected CudaMemFloat _d_c;
	protected CudaMemInt _d_peakIndex;
	protected CudaMemFloat _d_peakValue;
	protected CudaMemFloat _d_peakValues;
	protected CudaMemDouble _d_energy;

    protected CudaMemFloat _d_x_patterns[];
    protected CudaMemFloat _d_y_patterns[];

	//parameterlists for kernel invocations
	protected Pointer toComplex;
	protected Pointer toComplexAndFlip;
	protected Pointer computeEnergy;
	protected Pointer sumDoubles;
	protected Pointer computeCrossCorr;
	protected Pointer findPeak;
	protected Pointer maxlocFloats;

	protected int h;
	protected int w;
    protected boolean useRealPeak;

    public int _rows;
    public int _columns;
    public int _squareSize;
    public float[] _rowBuffer1;
    public float[] _rowBuffer2;
    public float[] _c;
    public float[] _x;
    public float[] _y;

    public int num_patterns;

    public FloatFFT_2D _fft;

	/**
	 * Constructor for the Wiener Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public PeakToCorrelationEnergy(int h, int w, CudaContext context, CudaModule module, boolean usePeak) {
		_context = context;
		_stream1 = new CudaStream();
		_stream2 = new CudaStream();
        _event = new CudaEvent();
		int n = h*w;
		this.h = h;
		this.w = w;
        this.useRealPeak = usePeak;

        _rows = h;
        _columns = w;
        int squareSize = 11;        //constant compiled into GPU code
        _squareSize = squareSize;
        _fft = new FloatFFT_2D(_rows, _columns);
        _rowBuffer1 = new float[_columns];
        _rowBuffer2 = new float[_columns];
        _c = new float[_rows * _columns * 2];
        _x = new float[_rows * _columns * 2];
        _y = new float[_rows * _columns * 2];

		//initialize CUFFT
		JCufft.initialize();
		JCufft.setExceptionsEnabled(true);
		_plan1 = new cufftHandle();
		_plan2 = new cufftHandle();

		//setup CUDA functions
        JCudaDriver.setExceptionsEnabled(true);
		int threads_x = 32;
		int threads_y = 16;
		_tocomplex = module.getFunction("toComplex");
		_tocomplex.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_tocomplexandflip = module.getFunction("toComplexAndFlip");
		_tocomplexandflip.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		_computeCrossCorr = module.getFunction("computeCrossCorr");
		_computeCrossCorr.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

        //dimensions for reducing kernels		
		int threads = 1024;
        int reducing_thread_blocks = 15; //optimally this equals the number of SMs in the GPU

        //nicer way of accesssing the number of SMs through the Cuba API
        int num_sm =_context.getDevice().getComputeModules();
        System.out.println("detected " + num_sm + " SMs on GPU");
        reducing_thread_blocks = num_sm;

		_findPeak = module.getFunction("findPeak");
		_findPeak.setDim(	reducing_thread_blocks, 1, 1,
				threads, 1, 1);

		_maxlocFloats = module.getFunction("maxlocFloats");
		_maxlocFloats.setDim( 1, 1, 1,
                threads, 1, 1);	

		_computeEnergy = module.getFunction("computeEnergy");
		_computeEnergy.setDim( reducing_thread_blocks, 1, 1,
                threads, 1, 1);

		_sumDoubles = module.getFunction("sumDoubles");
		_sumDoubles.setDim( 1, 1, 1,
                threads, 1, 1);	

        long free[] = new long[1];
        long total[] = new long[1];

        //JCuda.cudaMemGetInfo(free, total);
        //System.out.println("Before allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");

		//allocate local variables in GPU memory
		_d_inputx = _context.allocFloats(h*w);
		_d_inputy = _context.allocFloats(h*w);
		_d_x = _context.allocFloats(h*w*2);
		_d_y = _context.allocFloats(h*w*2);
		_d_c = _context.allocFloats(h*w*2);
		_d_peakIndex = _context.allocInts(reducing_thread_blocks);
		_d_peakValue = _context.allocFloats(1);
		_d_peakValues = _context.allocFloats(reducing_thread_blocks);
		_d_energy = _context.allocDoubles(reducing_thread_blocks);

        //JCuda.cudaMemGetInfo(free, total);
        //System.out.println("After allocations in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");

		//create CUFFT plan and associate with stream
		int res;
		res = JCufft.cufftPlan2d(_plan1, h, w, cufftType.CUFFT_C2C);
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while creating CUFFT plan 2D 1");
		}
		res = JCufft.cufftPlan2d(_plan2, h, w, cufftType.CUFFT_C2C);
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while creating CUFFT plan 2D 2");
		}
		res = JCufft.cufftSetStream(_plan1, new cudaStream_t(_stream1.cuStream()));
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while associating plan with stream");
		}
		res = JCufft.cufftSetStream(_plan2, new cudaStream_t(_stream2.cuStream()));
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while associating plan with stream");
		}

        JCuda.cudaMemGetInfo(free, total);
        //System.out.println("After FFT plans in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");

        long patternSize = h*w*2*4; //size of the FFT transformed pattern on the GPU

        int fit_patterns = (int)(free[0] / patternSize); //debugging
        //System.out.println("There is still room for " + fit_patterns + " patterns on the GPU");
        num_patterns = fit_patterns/2 + 1;
        System.out.println("Allocating space for in total " + 2*num_patterns + " patterns on the GPU");
                
        _d_x_patterns = new CudaMemFloat[num_patterns];
        _d_y_patterns = new CudaMemFloat[num_patterns];

        _d_x_patterns[0] = _d_x;
        _d_y_patterns[0] = _d_y;
        for (int i=1; i<num_patterns; i++) {
            _d_x_patterns[i] = _context.allocFloats(h*w*2);
            _d_y_patterns[i] = _context.allocFloats(h*w*2);
        }

        JCuda.cudaMemGetInfo(free, total);
        System.out.println("After allocating patterns in PCE free GPU mem: " + free[0]/1024/1024 + " MB total: " + total[0]/1024/1024 + " MB ");


		//construct parameter lists for the CUDA kernels
		toComplex = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_x.getDevicePointer()),
				Pointer.to(_d_inputx.getDevicePointer())
				);
		toComplexAndFlip = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_y.getDevicePointer()),
				Pointer.to(_d_inputy.getDevicePointer())
                );
		computeEnergy = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_energy.getDevicePointer()),
				Pointer.to(_d_peakIndex.getDevicePointer()),
				Pointer.to(_d_c.getDevicePointer())
				);
		sumDoubles = Pointer.to(
				Pointer.to(_d_energy.getDevicePointer()),
				Pointer.to(_d_energy.getDevicePointer()),
                Pointer.to(new int[]{reducing_thread_blocks})
				);
		computeCrossCorr = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_c.getDevicePointer()),
				Pointer.to(_d_x.getDevicePointer()),
				Pointer.to(_d_y.getDevicePointer())
				);
		findPeak = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_peakValue.getDevicePointer()),
				Pointer.to(_d_peakValues.getDevicePointer()),
				Pointer.to(_d_peakIndex.getDevicePointer()),
				Pointer.to(_d_c.getDevicePointer())
				);
		maxlocFloats = Pointer.to(
				Pointer.to(_d_peakIndex.getDevicePointer()),
				Pointer.to(_d_peakValues.getDevicePointer()),
				Pointer.to(_d_peakIndex.getDevicePointer()),
				Pointer.to(_d_peakValues.getDevicePointer()),
                Pointer.to(new int[]{reducing_thread_blocks})
				);

	}


    /**
     * This method performs an array of comparisons between patterns
     * Arrays xPatterns and yPatterns should be of size num_patterns
     *
     * Predicate is a boolean matrix that denotes if any of the comparisons are not to be computed
     *
     * Returns a matrix with all PCE scores comparing all in x with all in y
     */
    public double[][] compareGPU(float[][] xPatterns, float[][] yPatterns, boolean[][] predicate) {

        double result[][] = new double[num_patterns][num_patterns];

        //ship all patterns to the GPU and FFT Transform
        for (int i=0; i < num_patterns; i++) {
            if (xPatterns[i] != null) {
                xTransform(_d_x_patterns[i], xPatterns[i]);
            } else {
                //System.out.println("GPU: xPatterns["+i+"]=null");
            }
            if (yPatterns[i] != null) {
                yTransform(_d_y_patterns[i], yPatterns[i]);
            } else {
                //System.out.println("GPU: yPatterns["+i+"]=null");
            }
        }

        //make sure all FFT transforms are done
        syncStreams();

        //compare all x with all y
        for (int i=0; i < num_patterns; i++) {
            for (int j=0; j < num_patterns; j++) {
                if (predicate[i][j]) {
                    result[i][j] = compareGPUInPlace(_d_x_patterns[i], _d_y_patterns[j]);
                } else {
                    result[i][j] = 0.0;
                }
            }
        }

        return result;
    }

    public void xTransform(CudaMemFloat d_x, float[] x) {
        _d_inputx.copyHostToDeviceAsync(x, _stream1);
		Pointer toComplexParams = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(d_x.getDevicePointer()),
				Pointer.to(_d_inputx.getDevicePointer())
				);
        _tocomplex.launch(_stream1, toComplexParams);
        JCufft.cufftExecC2C(_plan1, d_x.getDevicePointer(), d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);
    }
    public void yTransform(CudaMemFloat d_y, float[] y) {
        _d_inputy.copyHostToDeviceAsync(y, _stream2);
		Pointer toComplexAndFlipParams = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(d_y.getDevicePointer()),
				Pointer.to(_d_inputy.getDevicePointer())
                );
        _tocomplexandflip.launch(_stream2, toComplexAndFlipParams);
        JCufft.cufftExecC2C(_plan2, d_y.getDevicePointer(), d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);
    }
    public void syncStreams() {
        _event.record(_stream2);
        _stream1.waitEvent(_event);
    }


    /**
     * Performs a PCE comparison between two PRNU patterns that are already in GPU memory and FFT transformed
     */
    public double compareGPUInPlace(CudaMemFloat d_x, CudaMemFloat d_y) {

		Pointer computeCrossCorrParams = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_c.getDevicePointer()),
				Pointer.to(d_x.getDevicePointer()),
				Pointer.to(d_y.getDevicePointer())
				);
        _computeCrossCorr.launch(_stream1, computeCrossCorrParams);

        JCufft.cufftExecC2C(_plan1, _d_c.getDevicePointer(), _d_c.getDevicePointer(), JCufft.CUFFT_INVERSE);

        _findPeak.launch(_stream1, findPeak);
        _maxlocFloats.launch(_stream1, maxlocFloats);

        _computeEnergy.launch(_stream1, computeEnergy);
        _sumDoubles.launch(_stream1, sumDoubles);

        float peak[] = new float[1];
        double energy[] = new double[1];
        if (useRealPeak) {
            _d_peakValue.copyDeviceToHostAsync(peak, 1, _stream1);
        } else {
            _d_peakValues.copyDeviceToHostAsync(peak, 1, _stream1);
        }
        _d_energy.copyDeviceToHostAsync(energy, 1, _stream1);
        _stream1.synchronize();

        //System.out.println("peak=" + peak[0] + " energy=" + energy[0]);
        double absPce = (double)(peak[0] * peak[0]) / energy[0];

        return absPce;
    }


    public double compareGPU(float[] x, float[] y) {

        //copy and process the two inputs in separate streams as long as we can
        _d_inputy.copyHostToDeviceAsync(y, _stream2);
        _tocomplexandflip.launch(_stream2, toComplexAndFlip);
        JCufft.cufftExecC2C(_plan2, _d_y.getDevicePointer(), _d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);

        _d_inputx.copyHostToDeviceAsync(x, _stream1);
        _tocomplex.launch(_stream1, toComplex);
        JCufft.cufftExecC2C(_plan1, _d_x.getDevicePointer(), _d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);

        //record event in stream2 and let stream1 wait on this event to synchronize streams
        _event.record(_stream2);
        _stream1.waitEvent(_event);

        //now continue in stream1
        _computeCrossCorr.launch(_stream1, computeCrossCorr);

        JCufft.cufftExecC2C(_plan1, _d_c.getDevicePointer(), _d_c.getDevicePointer(), JCufft.CUFFT_INVERSE);

        _findPeak.launch(_stream1, findPeak);
        _maxlocFloats.launch(_stream1, maxlocFloats);

        _computeEnergy.launch(_stream1, computeEnergy);
        _sumDoubles.launch(_stream1, sumDoubles);

        float peak[] = new float[1];
        double energy[] = new double[1];
        if (useRealPeak) {
            _d_peakValue.copyDeviceToHostAsync(peak, 1, _stream1);
        } else {
            _d_peakValues.copyDeviceToHostAsync(peak, 1, _stream1);
        }
        _d_energy.copyDeviceToHostAsync(energy, 1, _stream1);
        _stream1.synchronize();
        double absPce = (double)(peak[0] * peak[0]) / energy[0];

        System.out.println("peak=" + peak[0] + " energy=" + energy[0]);

		JCudaDriver.cuCtxSynchronize();
        int err = JCuda.cudaGetLastError();
        if (err != cudaError.cudaSuccess) {
            System.err.println("CUDA Error: " + JCuda.cudaGetErrorString(err));
        } else {
            //System.out.println("CUDA: No Errors!");
        }

        return absPce;
    }
    
    public double compareGPUTiming(float[] x, float[] y) {
        long start = System.nanoTime();
          _d_inputx.copyHostToDeviceAsync(x, _stream1);
          _d_inputy.copyHostToDeviceAsync(y, _stream1);
        _stream1.synchronize();
        long end = System.nanoTime();
        System.out.println("Memcpy Host to Device took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          _tocomplex.launch(_stream1, toComplex);
          _tocomplexandflip.launch(_stream1, toComplexAndFlip);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("toComplexAndFlip took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          JCufft.cufftExecC2C(_plan1, _d_x.getDevicePointer(), _d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);
          JCufft.cufftExecC2C(_plan1, _d_y.getDevicePointer(), _d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("2 Fourrier transforms took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          _computeCrossCorr.launch(_stream1, computeCrossCorr);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("ComputeCrossCorr took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          JCufft.cufftExecC2C(_plan1, _d_c.getDevicePointer(), _d_c.getDevicePointer(), JCufft.CUFFT_INVERSE);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("Inverse FFT took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          _findPeak.launch(_stream1, findPeak);
          _maxlocFloats.launch(_stream1, maxlocFloats);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("findPeak took: " + (end-start)/1e6f + " ms.");

        start = System.nanoTime();
          _computeEnergy.launch(_stream1, computeEnergy);
          _sumDoubles.launch(_stream1, sumDoubles);
        _stream1.synchronize();
        end = System.nanoTime();
        System.out.println("computeEnergy took: " + (end-start)/1e6f + " ms.");

		JCudaDriver.cuCtxSynchronize();
        int err = JCuda.cudaGetLastError();
        if (err != cudaError.cudaSuccess) {
            System.out.println("CUDA Error: " + JCuda.cudaGetErrorString(err));
        } else {
            System.out.println("CUDA: No Errors!");
        }

        float peak[] = new float[1];
        double energy[] = new double[1];
        _d_peakValue.copyDeviceToHostAsync(peak, 1, _stream1);
        _d_energy.copyDeviceToHostAsync(energy, 1, _stream1);
        _stream1.synchronize();

        System.out.println("peak=" + peak[0] + " energy=" + energy[0]);
        double absPce = (double)(peak[0] * peak[0]) / energy[0];

        return absPce;
    }
	

    public void forwardTransform(float[] x) {
        _fft.complexForward(x);
    }
    public void inverseTransform(float[] x) {
        _fft.complexInverse(x, true);
    }


    public double compare(float[] x, float[] y) {
        toComplexAndFlip(x,y);

        forwardTransform(_x);
        forwardTransform(_y);

        compute_crosscorr();

        inverseTransform(_c);

        int peakIndex = findPeak();

        double peak = _c[((_rows * _columns) - 1) << 1];
        int indexY = peakIndex / _columns;
        int indexX = peakIndex - (indexY * _columns);

        double energy = energyFixed(_squareSize, indexX, indexY);

        double absPce = (peak * peak) / energy;

        return absPce;
    }

    public int findPeak() { 
        float max = 0.0f;
        int res = 0;
        for (int i=0; i<_c.length; i+=2) {
            if (Math.abs(_c[i]) > max) {
                max = Math.abs(_c[i]);
                res = i;
            }
        }
        return res / 2; //divided by 2, because we want the index of the complex number in the complex array
    }

    public double energyFixed(int squareSize, int peakIndexX, int peakIndexY) {
        int radius = (squareSize - 1) / 2;
        int n = (_rows * _columns) - (squareSize * squareSize);

        // Determine the energy, i.e. the sample variance of circular cross-correlations, minus an area around the peak
        double energy = 0.0;
        for (int row = 0; row < _rows; row++) {
            boolean peakRow = row > peakIndexY - radius && row < peakIndexY + radius;
            for (int col = 0; col < _columns; col++) {
                if (peakRow && col > peakIndexX - radius && col < peakIndexX + radius) {
                    continue;
                }
                else {
                    float f = _c[row*_columns*2 + col*2];
                    energy += (f * f);
                }
            }
        }
        return (energy / n);
    }

    public void toComplexAndFlip(float[] x, float[] y) {
        for (int row = 0; row < _rows; row++) {
            for (int i=0; i<_columns; i++) {
                    _rowBuffer1[i] = x[row * _columns + i];
                    _rowBuffer2[i] = y[row * _columns + i];
            }            

            int xOffset = (row * _columns) * 2;
            int yOffset = (((_rows - row) * _columns) - 1) * 2;

            //this is a toComplex operation that copies the contents of rowbuffer 1 and 2 into _x and _y
            //however it appears to vertically and horizontally flip _y
            for (int col = 0; col < _columns; col++) {
                _x[xOffset + (col + col)] = _rowBuffer1[col];
                _x[xOffset + (col + col) + 1] = 0.0f;
                _y[yOffset - (col + col)] = _rowBuffer2[col];
                _y[xOffset + (col + col) + 1] = 0.0f;
            }
        }
    }

    void compute_crosscorr() {
        for (int i = 0; i < _x.length; i += 2) {
            float xRe = _x[i];
            float xIm = _x[i + 1];
            float yRe = _y[i];
            float yIm = _y[i + 1];
            _c[i] = (xRe * yRe) - (xIm * yIm);
            _c[i + 1] = (xRe * yIm) + (xIm * yRe);
        }
    }








	/**
	 * Applies the Wiener Filter to the input pattern already in GPU memory
	 */
/*
	public void applyGPU() {

		//convert values from real to complex
		_tocomplex.launch(_stream, toComplex);

		//apply complex to complex forward Fourier transform
		JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);

		//square the complex frequency values and store as real values
		_computeSquaredMagnitudes.launch(_stream, sqmag);

		//estimate local variances for four filter sizes, store minimum 
		_computeVarianceEstimates.launch(_stream, varest);

		//compute global variance
		_computeVarianceZeroMean.launch(_stream, variancep);

		//scale the frequencies using global and local variance
		_scaleWithVariances.launch(_stream, scale);

		//inverse fourier transform using CUFFT
		JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_INVERSE);

		//CUFFT does not normalize the values after inverse transform, as such all values are scaled with N=(h*w)
		//normalize the values and convert from complex to real
		_normalizeToReal.launch(_stream, normalize);

		//for measuring time
		JCudaDriver.cuCtxSynchronize();
	}
*/
	/**
	 * Applies the Wiener Filter to the input pattern already in GPU memory
	 * and measures the time spent on each step. This function is used for
	 * benchmarking only.
	 */
	//public void applyGPUTiming() {
		//JCudaDriver.cuCtxSynchronize();
		//long start = System.nanoTime();
		//_tocomplex.launch(_stream, toComplex); //from _d_input to _d_complex
		//JCudaDriver.cuCtxSynchronize();
		//long end = System.nanoTime();
		//System.out.println("tocomplex: " + (double)(end-start)/1e6 + " ms.");

		//start = System.nanoTime();
		//JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);
		//JCudaDriver.cuCtxSynchronize();
		//end = System.nanoTime();
		//System.out.println("fftforward: " + (double)(end-start)/1e6 + " ms.");

        //...
	//}

	/**
	 * Cleans up GPU memory and destroys FFT plan
	 */
	public void cleanup() {
        _d_inputx.free();
        _d_inputy.free();
		_d_x.free();
		_d_y.free();
        for (int i=1; i<num_patterns; i++) {
            _d_x_patterns[i].free();
            _d_y_patterns[i].free();
        }
		_d_c.free();
		_d_peakIndex.free();
		_d_peakValue.free();
		_d_energy.free();
		JCufft.cufftDestroy(_plan1);
		JCufft.cufftDestroy(_plan2);
	}



}
