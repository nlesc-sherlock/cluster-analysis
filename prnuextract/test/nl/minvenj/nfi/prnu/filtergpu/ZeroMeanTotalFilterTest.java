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

import nl.minvenj.nfi.prnu.PrnuExtract;
import nl.minvenj.nfi.prnu.Util;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class ZeroMeanTotalFilterTest {

	static ZeroMeanTotalFilter zeroMeanTotalFilter; 
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
		zeroMeanTotalFilter = filter.getZeroMeanTotalFilter();
		
		//perform grayscale filter on the GPU
		filter.getGrayscaleFilter().applyGPU(image);
		filter.getFastNoiseFilter().applyGPU();
		
		//copy GPU result to host memory
		pixels = new float[image.getHeight()*image.getWidth()];
		filter.getFastNoiseFilter()._d_input.copyDeviceToHost(pixels, image.getHeight()*image.getWidth());
	}

	@Ignore
	public void applyGPUTest() {
		//perform zero mean total filter on the CPU
		float[] pixelsCPU = zeroMeanTotalFilter.apply1D(pixels);
		
		//perform zero mean total filter on the GPU
		zeroMeanTotalFilter.applyGPU();
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixelsCPU.length];
		zeroMeanTotalFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(pixelsCPU, pixelsGPU, 1f/256f);
		assertTrue(result);
	}
	
	
	/**
	 * The Zero Mean implementation on the GPU contains a parallelized reduction,
	 * for which is practically impossible to the same numerical result. Therefore,
	 * we test on a small data set.
	 */
	@Test
	public void computeMeanVerticallyTest() {
		int h = 47;
		int w = 43;

		PRNUFilter filter = filterFactory.createPRNUFilter(h, w);
		ZeroMeanTotalFilter zeroMeanTotalFilter = filter.getZeroMeanTotalFilter();
		
		float[] pixels = new float[h*w];
		for (int i=0; i<pixels.length; i++) {
			pixels[i] = (float)Math.random();
		}
		//copy input to GPU
		zeroMeanTotalFilter._d_input.copyHostToDevice(pixels, pixels.length);
		
		//apply zero mean filter vertically on the CPU
		ZeroMeanTotalFilter.computeMeanVertically(zeroMeanTotalFilter.h, zeroMeanTotalFilter.w, pixels);
		
		//apply zero mean filter vertically on GPU
		zeroMeanTotalFilter._computeMeanVertically.launch(zeroMeanTotalFilter._stream, zeroMeanTotalFilter.computeMeanVerticallyCol);
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixels.length];
		zeroMeanTotalFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(pixels, pixelsGPU, 0.0001f);
		assertTrue(result);
		
		filter.cleanup();
	}
	
	@Test
	public void transposeTest() {
		//copy input to GPU
		zeroMeanTotalFilter._d_input.copyHostToDevice(pixels, pixels.length);
		
		//transpose on CPU
		float[] inputTransposed = Util.transpose(zeroMeanTotalFilter.h, zeroMeanTotalFilter.w, pixels);
		
		//transpose on GPU
		zeroMeanTotalFilter._transpose.launch(zeroMeanTotalFilter._stream, zeroMeanTotalFilter.transposeForward);
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixels.length];
		zeroMeanTotalFilter._d_output.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(inputTransposed, pixelsGPU, 0.0001f);
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
