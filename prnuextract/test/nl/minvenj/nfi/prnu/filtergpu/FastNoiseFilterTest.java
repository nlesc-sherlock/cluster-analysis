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
import org.junit.Test;

/**
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class FastNoiseFilterTest {

	static FastNoiseFilter fastNoiseFilter; 
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
		fastNoiseFilter = filter.getFastNoiseFilter();
		
		//perform grayscale filter on the GPU
		filter.getGrayscaleFilter().applyGPU(image);
		
		//copy GPU result to host memory
		pixels = new float[image.getHeight()*image.getWidth()];
		filter.getGrayscaleFilter()._d_input.copyDeviceToHost(pixels, image.getHeight()*image.getWidth());
	}

	@Test
	public void applyGPUTest() {
		//perform fast noise filter on the CPU
		float[] pixelsCPU = fastNoiseFilter.apply1D(pixels);
		
		//perform fast noise filter on the GPU
		fastNoiseFilter.applyGPU();
		
		//copy GPU result to host memory
		float[] pixelsGPU = new float[pixelsCPU.length];
		fastNoiseFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
		//compare CPU and GPU result
		boolean result = Util.compareArray(pixelsCPU, pixelsGPU, 0.001f);
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
