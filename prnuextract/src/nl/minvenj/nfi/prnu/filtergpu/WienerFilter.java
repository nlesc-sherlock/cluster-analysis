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
package nl.minvenj.nfi.prnu.filtergpu;

import nl.minvenj.nfi.prnu.Util;
import nl.minvenj.nfi.cuba.cudaapi.*;
import jcuda.*;
import jcuda.runtime.cudaStream_t;
import jcuda.driver.*;
import jcuda.jcufft.*;

/**
 * Class for applying a series of Wiener Filters to a PRNU pattern 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class WienerFilter {

	protected CudaContext _context;
	protected CudaStream _stream;
	protected CudaMemFloat _d_input;

	protected static final int[] FILTER_SIZES = { 3, 5, 7, 9 };

	//handles to CUDA kernels
	protected CudaFunction _tocomplex;
	protected CudaFunction _toreal;
	protected CudaFunction _computeSquaredMagnitudes;
	protected CudaFunction _computeVarianceEstimates;
	protected CudaFunction _computeVarianceZeroMean;
	protected CudaFunction _sumFloats;
	protected CudaFunction _scaleWithVariances;
	protected CudaFunction _normalizeToReal;
	protected CudaFunction _normalizeComplex;

	//handle for CUFFT plan
	protected cufftHandle _planc2c;

	//handles to device memory arrays
	protected CudaMemFloat _d_comp;
	protected CudaMemFloat _d_sqmag;
	protected CudaMemFloat _d_varest;
	protected CudaMemFloat _d_variance;

	//parameterlists for kernel invocations
	protected Pointer toComplex;
	protected Pointer toReal;
	protected Pointer sqmag;
	protected Pointer varest;
	protected Pointer variancep;
	protected Pointer sumfloats;
	protected Pointer scale;
	protected Pointer normalize;
	protected Pointer normalizeComplex;

	protected int h;
	protected int w;
	
	/**
	 * Constructor for the Wiener Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param input - the GPU memory for storing the pattern
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public WienerFilter(int h, int w, CudaContext context, CudaStream stream, CudaMemFloat input, CudaModule module) {
		_context = context;
		_stream = stream;
		_d_input = input;
		int n = h*w;
		this.h = h;
		this.w = w;

		//initialize CUFFT
		JCufft.initialize();
		JCufft.setExceptionsEnabled(true);
		_planc2c = new cufftHandle();

		//setup CUDA functions
		final int threads_x = 32;
		final int threads_y = 16;
		_tocomplex = module.getFunction("toComplex");
		_tocomplex.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		_toreal = module.getFunction("toReal");
		_toreal.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		_computeSquaredMagnitudes = module.getFunction("computeSquaredMagnitudes");
		_computeSquaredMagnitudes.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_computeVarianceEstimates = module.getFunction("computeVarianceEstimates");
		_computeVarianceEstimates.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);
		
		final int threads = 128;
        final int nblocks = 1024;
		_computeVarianceZeroMean = module.getFunction("computeVarianceZeroMean");
		_computeVarianceZeroMean.setDim(	nblocks, 1, 1,
				threads, 1, 1);
		_sumFloats = module.getFunction("sumFloats");
		_sumFloats.setDim(	1, 1, 1,
				threads, 1, 1);

		_scaleWithVariances = module.getFunction("scaleWithVariances");
		_scaleWithVariances.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_normalizeToReal = module.getFunction("normalizeToReal");
		_normalizeToReal.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		_normalizeComplex = module.getFunction("normalize");
		_normalizeComplex.setDim(	(int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		//allocate local variables in GPU memory
		_d_comp = _context.allocFloats(h*w*2);
		_d_sqmag = _context.allocFloats(h*w);
		_d_varest = _context.allocFloats(h*w);
		_d_variance = _context.allocFloats(nblocks);

		//create CUFFT plan and associate with stream
		int res;
		res = JCufft.cufftPlan2d(_planc2c, h, w, cufftType.CUFFT_C2C);
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while creating CUFFT plan 2D");
		}
		res = JCufft.cufftSetStream(_planc2c, new cudaStream_t(_stream.cuStream()));
		if (res != cufftResult.CUFFT_SUCCESS) {
			System.err.println("Error while associating plan with stream");
		}

		//construct parameter lists for the CUDA kernels
		toComplex = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_comp.getDevicePointer()),
				Pointer.to(_d_input.getDevicePointer())
				);
		toReal = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_input.getDevicePointer()),
				Pointer.to(_d_comp.getDevicePointer())
				);
		sqmag = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_sqmag.getDevicePointer()),
				Pointer.to(_d_comp.getDevicePointer())
				);
		varest = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_varest.getDevicePointer()),
				Pointer.to(_d_sqmag.getDevicePointer())
				);
		variancep = Pointer.to(
				Pointer.to(new int[]{n}),
				Pointer.to(_d_variance.getDevicePointer()),
				Pointer.to(_d_input.getDevicePointer())
				);
		sumfloats = Pointer.to(
				Pointer.to(_d_variance.getDevicePointer()),
				Pointer.to(_d_variance.getDevicePointer()),
				Pointer.to(new int[]{nblocks})
				);
		scale = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_comp.getDevicePointer()),
				Pointer.to(_d_comp.getDevicePointer()),
				Pointer.to(_d_varest.getDevicePointer()),
				Pointer.to(_d_variance.getDevicePointer())
				);
		normalize = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_input.getDevicePointer()),
				Pointer.to(_d_comp.getDevicePointer())
				);
		normalizeComplex = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_comp.getDevicePointer()),
				Pointer.to(_d_comp.getDevicePointer())
				);

	}
	
	/**
	 * Computes the square of each frequency and stores the result as a real.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param frequencies - the frequencies as the result of Fourier transform
	 * @return - a float array containing the frequencies squared as real values
	 */
	public static float[] computeSquaredMagnitudes(int h, int w, float[] frequencies) {
		float[] result = new float[h*w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float re = frequencies[i*2*w+(2 * j)];
				float im = frequencies[i*2*w+(2 * j + 1)];
				result[i*w+j] = (re * re) + (im * im);
			}
		}

		return result;
	}

	/**
	 * This function scales the frequencies in input with a combination of the global variance and an estimate
	 * for the local variance at that position. Effectively this cleans the input pattern from low frequency
	 * noise.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param input - a float array of complex values that contain the frequencies
	 * @param varianceEstimates - an array containing the estimated local variance
	 * @param variance - the global variance of the input
	 * @return - a complex array in which the frequencies have been scaled
	 */
	static float[] scaleWithVariances(int h, int w, float[] input, float[] varianceEstimates, float variance) {
		float[] output = new float[h * w * 2];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float scale = variance / Math.max(variance, varianceEstimates[i*w+j]);
				output[i*(2*w)+(j * 2)] = input[i*(2*w)+(j * 2)] * scale;
				output[i*(2*w)+(j * 2 + 1)] = input[i*(2*w)+(j * 2 + 1)] * scale;
			}
		}

		return output;
	}

	/**
	 * Estimates the minimum local variances by applying all filters.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param squaredMagnitudes - the input array containing the squared frequency values as reals
	 * @return - a float array containing the estimated minimum local variance
	 */
	static float[] computeVarianceEstimates (int h, int w, float[] squaredMagnitudes) {

		float[] varianceEstimates = Util.from2DTo1D(h, w, Util.initializeArray(h, w, Float.MAX_VALUE));
		for (final int filterSize : FILTER_SIZES) {
			float[] squaredMagnitudesWithBorder = Util.addBorder(h, w, squaredMagnitudes, filterSize / 2);
			float[] output = Util.convolve(h, w, filterSize, squaredMagnitudesWithBorder);
			varianceEstimates = Util.minimum(varianceEstimates, output);
		}

		return varianceEstimates;
	}


	/**
	 * Applies the Wiener Filter to the input pattern on the CPU.
	 * This function is mainly used to check the GPU result.
	 * 
	 * @param input - the input pattern stored as an 1D float array
	 * @return - a float array containing the filtered pattern
	 */
	public float[] apply1D(float[] input) {
		
		//convert input to complex values
		float[] complex = Util.toComplex(h, w, input);
		
		//forward Fourier transform using JTransforms
		Util.fft(h, w, complex);
		
		//compute frequencies squared and store as real
		float[] squaredMagnitudes = computeSquaredMagnitudes(h, w, complex);

		//estimate local variances and keep the mimimum
		float[] varianceEstimates = computeVarianceEstimates(h, w, squaredMagnitudes);

		//compute global variance, assuming zero mean
		int n = w * h;
		float variance = (float) ((Util.sum(Util.multiply(input, input)) * n) / (n - 1));
		
		//scale the frequencies with the global and local variance
		float[] frequenciesScaled = scaleWithVariances(h, w, complex, varianceEstimates, variance);

		//inverse Fourier transform
		Util.ifft(h, w, frequenciesScaled);

		//convert values to real and return result
		return Util.assign(input, Util.toReal(frequenciesScaled));
	}

	/**
	 * Applies the Wiener Filter to the input pattern already in GPU memory
	 */
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
		_sumFloats.launch(_stream, sumfloats);

		//scale the frequencies using global and local variance
		_scaleWithVariances.launch(_stream, scale);

		//inverse fourier transform using CUFFT
		JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_INVERSE);

		//CUFFT does not normalize the values after inverse transform, as such all values are scaled with N=(h*w)
		//normalize the values and convert from complex to real
		_normalizeToReal.launch(_stream, normalize);

	}

	/**
	 * Applies the Wiener Filter to the input pattern already in GPU memory
	 * and measures the time spent on each step. This function is used for
	 * benchmarking only.
	 */
	public void applyGPUTiming() {
		JCudaDriver.cuCtxSynchronize();
		long start = System.nanoTime();
		_tocomplex.launch(_stream, toComplex); //from _d_input to _d_complex
		JCudaDriver.cuCtxSynchronize();
		long end = System.nanoTime();
		System.out.println("tocomplex: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("fftforward: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		_computeSquaredMagnitudes.launch(_stream, sqmag);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("sqmag: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		_computeVarianceEstimates.launch(_stream, varest);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("varest: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		_computeVarianceZeroMean.launch(_stream, variancep);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("variance: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		_scaleWithVariances.launch(_stream, scale);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("scale: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		JCufft.cufftExecC2C(_planc2c, _d_comp.getDevicePointer(), _d_comp.getDevicePointer(), JCufft.CUFFT_INVERSE);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("fft inverse: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		_normalizeToReal.launch(_stream, normalize);
		JCudaDriver.cuCtxSynchronize();
		end = System.nanoTime();
		System.out.println("normalize: " + (double)(end-start)/1e6 + " ms.");

		JCudaDriver.cuCtxSynchronize();
	}

	/**
	 * Applies the Wiener Filter to the input pattern on the CPU 
	 * and measures the time spent on each step. This function is used for
	 * benchmarking only.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param input - the input pattern stored as an 1D float array
	 * @return - a float array containing the filtered pattern
	 */
	public static float[] applyCPUTiming(int h, int w, float[] input) {
		long start = System.nanoTime();
		float[] complex = Util.toComplex(h, w, input);
		long end = System.nanoTime();
		System.out.println("tocomplex: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		Util.fft(h, w, complex);
		end = System.nanoTime();
		System.out.println("fft forward: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		float[] squaredMagnitudes = computeSquaredMagnitudes(h, w, complex);
		end = System.nanoTime();
		System.out.println("sqmag: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		float[] varianceEstimates = Util.from2DTo1D(h, w, Util.initializeArray(h, w, Float.MAX_VALUE));
		for (final int filterSize : FILTER_SIZES) {
			float[] squaredMagnitudesWithBorder = Util.addBorder(h, w, squaredMagnitudes, filterSize / 2);
			float[] output = Util.convolve(h, w, filterSize, squaredMagnitudesWithBorder);
			varianceEstimates = Util.minimum(varianceEstimates, output);
		}
		end = System.nanoTime();
		System.out.println("varest: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		int n = w * h;
		float variance = (float) ((Util.sum(Util.multiply(input, input)) * n) / (n - 1));
		end = System.nanoTime();
		System.out.println("variance: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		float[] frequenciesScaled = scaleWithVariances(h, w, complex, varianceEstimates, variance);
		end = System.nanoTime();
		System.out.println("scale: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		Util.ifft(h, w, frequenciesScaled);
		end = System.nanoTime();
		System.out.println("fft inverse: " + (double)(end-start)/1e6 + " ms.");

		start = System.nanoTime();
		input = Util.toReal(frequenciesScaled);
		end = System.nanoTime();
		System.out.println("toreal: " + (double)(end-start)/1e6 + " ms.");

		return input;
	}

	/**
	 * Cleans up GPU memory and destroys FFT plan
	 */
	public void cleanup() {
		_d_comp.free();
		_d_sqmag.free();
		_d_varest.free();
		_d_variance.free();
		JCufft.cufftDestroy(_planc2c);
	}



}
