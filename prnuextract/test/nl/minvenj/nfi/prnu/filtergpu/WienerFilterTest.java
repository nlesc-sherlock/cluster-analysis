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

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;

import jcuda.jcufft.JCufft;
import jcuda.driver.*;

import nl.minvenj.nfi.prnu.PrnuExtract;
import nl.minvenj.nfi.prnu.Util;
import nl.minvenj.nfi.prnu.filtergpu.WienerFilter;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class WienerFilterTest {

	static WienerFilter wienerFilter; 
	static float[] pixels;
	static PRNUFilter filter;
	static PRNUFilterFactory filterFactory;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		//instantiate the PRNUFilterFactory to compile CUDA source files
		filterFactory = new PRNUFilterFactory(false);

		//load test image and measure time
		long start = System.nanoTime();
		long end = 0;
		BufferedImage image = Util.readImage(PrnuExtract.INPUT_FILE);
		end = System.nanoTime();
		System.out.println("Load image: " + (end-start)/1e6 + " ms. size: " + image.getHeight()+"x"+image.getWidth());

		//construct a PRNUFilter for this image size
		filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());
		wienerFilter = filter.getWienerFilter();

		//perform grayscale filter on the GPU
		filter.getGrayscaleFilter().applyGPU(image);
		filter.getFastNoiseFilter().applyGPU();
		filter.getZeroMeanTotalFilter().applyGPU();

		//copy GPU result to host memory
		pixels = new float[image.getHeight()*image.getWidth()];
        JCudaDriver.cuCtxSynchronize();
		filter.getZeroMeanTotalFilter()._d_input.copyDeviceToHost(pixels, pixels.length);
        JCudaDriver.cuCtxSynchronize();

	}

	@Ignore
	public void applyGPUTest() {
		//perform Wiener filter on the CPU
		float[] pixelsCPU = wienerFilter.apply1D(pixels);

		//perform Wiener filter on the GPU
		wienerFilter.applyGPU();

		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixelsCPU.length];
		wienerFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);

		//compare CPU and GPU result
		boolean result = Util.compareArray(pixelsCPU, pixelsGPU, 1f/256f);
		assertTrue(result);
	}

	@Test
	public void tocomplexTest() {
		//copy input to GPU
		wienerFilter._d_input.copyHostToDevice(pixels, pixels.length);
		
		//convert input to complex values on CPU
		float[] pixelsCPU = Util.toComplex(wienerFilter.h, wienerFilter.w, pixels);
		
		//convert values from real to complex on GPU
		wienerFilter._tocomplex.launch(wienerFilter._stream, wienerFilter.toComplex);
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixelsCPU.length];
		wienerFilter._d_comp.copyDeviceToHost(pixelsGPU, pixelsGPU.length);

		//compare CPU and GPU result
		boolean result = Util.compareArray(pixelsCPU, pixelsGPU, 0.0001f);
		assertTrue(result);
	}
	
	@Test
	public void computeSquaredMagnitudesTest() {
        boolean testNaN = Util.compareArray(pixels, pixels, 10f);
        if (!testNaN) { System.err.println("Input array before we did anything contains NaNs"); }
        assertTrue(testNaN);

        //copy input to GPU
        wienerFilter._d_input.copyHostToDevice(pixels, pixels.length);

		//convert values from real to complex
		wienerFilter._tocomplex.launch(wienerFilter._stream, wienerFilter.toComplex);
		
		//apply complex to complex forward Fourier transform
        JCudaDriver.cuCtxSynchronize();
		JCufft.cufftExecC2C(wienerFilter._planc2c, wienerFilter._d_comp.getDevicePointer(), wienerFilter._d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);
        JCudaDriver.cuCtxSynchronize();

		//copy FFT output to host memory
		float[] complex = new float[pixels.length*2];
		wienerFilter._d_comp.copyDeviceToHost(complex, complex.length);

        testNaN = Util.compareArray(complex, complex, 10f);
        if (!testNaN) { System.err.println("GPU result after cuFFT contains NaNs"); }
        assertTrue(testNaN);

		//compute frequencies squared and store as real on the GPU
		wienerFilter._computeSquaredMagnitudes.launch(wienerFilter._stream, wienerFilter.sqmag);

		//compute frequencies squared and store as real on the CPU
		float[] squaredMagnitudes = WienerFilter.computeSquaredMagnitudes(wienerFilter.h, wienerFilter.w, complex);
		
		//copy GPU result to host memory
		float[] squaredMagnitudesGPU = new float[squaredMagnitudes.length];
		wienerFilter._d_sqmag.copyDeviceToHost(squaredMagnitudesGPU, squaredMagnitudesGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(squaredMagnitudes, squaredMagnitudesGPU, 0.0001f);
		assertTrue(result);
	}
	
	@Test
	public void computeVarianceEstimatesTest() { 
		//convert values from real to complex
		wienerFilter._tocomplex.launch(wienerFilter._stream, wienerFilter.toComplex);
		
		//apply complex to complex forward Fourier transform
        JCudaDriver.cuCtxSynchronize();
		JCufft.cufftExecC2C(wienerFilter._planc2c, wienerFilter._d_comp.getDevicePointer(), wienerFilter._d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);
        JCudaDriver.cuCtxSynchronize();

		//compute frequencies squared and store as real on the GPU
		wienerFilter._computeSquaredMagnitudes.launch(wienerFilter._stream, wienerFilter.sqmag);
		
		//copy GPU result to host memory
		float[] squaredMagnitudes = new float[pixels.length];
		wienerFilter._d_sqmag.copyDeviceToHost(squaredMagnitudes, squaredMagnitudes.length);

        //test to see if squaredMagnitues does not contain NaN values
		boolean testNaN = Util.compareArray(squaredMagnitudes, squaredMagnitudes, 0.0001f);
        assertTrue (testNaN);
		
		//estimate local variances and keep the mimimum
		float[] varianceEstimates = WienerFilter.computeVarianceEstimates(wienerFilter.h, wienerFilter.w, squaredMagnitudes);
		
		//estimate local variances for four filter sizes, store minimum 
		wienerFilter._computeVarianceEstimates.launch(wienerFilter._stream, wienerFilter.varest);
		
		//copy GPU result to host memory
		float[] varianceEstimatesGPU = new float[pixels.length];
		wienerFilter._d_varest.copyDeviceToHost(varianceEstimatesGPU, varianceEstimatesGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(varianceEstimates, varianceEstimatesGPU, 0.0001f);
		assertTrue(result);
		
	}
	
	@Test
	public void computeVarianceZeroMeanTest() {
		//copy input to GPU
		wienerFilter._d_input.copyHostToDevice(pixels, pixels.length);
		
		//compute global variance, assuming zero mean
		int n = wienerFilter.w * wienerFilter.h;
		float[] varianceCPU = new float[1];
		varianceCPU[0] = (float) ((Util.sum(Util.multiply(pixels, pixels)) * n) / (n - 1));
		
		//compute global variance
		wienerFilter._computeVarianceZeroMean.launch(wienerFilter._stream, wienerFilter.variancep);
		
		//copy GPU result to host memory
		float[] varianceGPU = new float[1];
		wienerFilter._d_variance.copyDeviceToHost(varianceGPU, varianceGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(varianceCPU, varianceGPU, 0.0001f);
		assertTrue(result);	
	}
	
	@Test
	public void scaleWithVariancesTest() { 
		//convert values from real to complex
		wienerFilter._tocomplex.launch(wienerFilter._stream, wienerFilter.toComplex);

		//apply complex to complex forward Fourier transform
		JCufft.cufftExecC2C(wienerFilter._planc2c, wienerFilter._d_comp.getDevicePointer(), wienerFilter._d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);

		//square the complex frequency values and store as real values
		wienerFilter._computeSquaredMagnitudes.launch(wienerFilter._stream, wienerFilter.sqmag);

		//estimate local variances for four filter sizes, store minimum 
		wienerFilter._computeVarianceEstimates.launch(wienerFilter._stream, wienerFilter.varest);

		//compute global variance
		wienerFilter._computeVarianceZeroMean.launch(wienerFilter._stream, wienerFilter.variancep);

		//copy GPU result to host memory
		float[] variance = new float[1];
		wienerFilter._d_variance.copyDeviceToHost(variance, variance.length);
		float[] complex = new float[pixels.length*2];
		wienerFilter._d_comp.copyDeviceToHost(complex, complex.length);
		float[] varianceEstimates = new float[pixels.length];
		wienerFilter._d_varest.copyDeviceToHost(varianceEstimates, varianceEstimates.length);
		
		//scale the frequencies using global and local variance
		wienerFilter._scaleWithVariances.launch(wienerFilter._stream, wienerFilter.scale);
		
		//scale the frequencies with the global and local variance
		float[] frequenciesScaled = WienerFilter.scaleWithVariances(wienerFilter.h, wienerFilter.w, complex, varianceEstimates, variance[0]);
		
		//copy GPU result to host memory
		float[] frequenciesScaledGPU = new float[pixels.length*2];
		wienerFilter._d_comp.copyDeviceToHost(frequenciesScaledGPU, frequenciesScaledGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(frequenciesScaled, frequenciesScaledGPU, 0.0001f);
		assertTrue(result);	
	}
	
	@Test
	public void normalizeToRealTest() { 
		//convert values from real to complex on GPU
		wienerFilter._tocomplex.launch(wienerFilter._stream, wienerFilter.toComplex);

		//apply complex to complex forward Fourier transform
		JCufft.cufftExecC2C(wienerFilter._planc2c, wienerFilter._d_comp.getDevicePointer(), wienerFilter._d_comp.getDevicePointer(), JCufft.CUFFT_FORWARD);

		//inverse fourier transform using CUFFT
		JCufft.cufftExecC2C(wienerFilter._planc2c, wienerFilter._d_comp.getDevicePointer(), wienerFilter._d_comp.getDevicePointer(), JCufft.CUFFT_INVERSE);

		//convert values to real
		wienerFilter._toreal.launch(wienerFilter._stream, wienerFilter.toReal);
		
		//copy GPU result to host memory
		float[] pixelsCPU = new float[pixels.length];
		
		//normalize real values on the CPU
		wienerFilter._d_input.copyDeviceToHost(pixelsCPU, pixelsCPU.length);
		for (int i =0; i<pixelsCPU.length; i++) {
			pixelsCPU[i] /= pixelsCPU.length;
		}
		
		//CUFFT does not normalize the values after inverse transform, as such all values are scaled with N=(h*w)
		//normalize the values and convert from complex to real
		wienerFilter._normalizeToReal.launch(wienerFilter._stream, wienerFilter.normalize);
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixels.length];
		wienerFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(pixelsCPU, pixelsGPU, 0.0001f);
		assertTrue(result);
	}

	/**
	 * @throws java.lang.Exception
	 */
	@AfterClass
	public static void tearDownAfterClass() throws Exception {
		filter.cleanup();
		filterFactory.cleanup();
	}
	
}
