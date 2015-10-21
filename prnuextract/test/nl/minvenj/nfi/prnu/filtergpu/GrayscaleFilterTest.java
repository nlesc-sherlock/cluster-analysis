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
import java.io.IOException;

import nl.minvenj.nfi.prnu.PrnuExtract;
import nl.minvenj.nfi.prnu.Util;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class GrayscaleFilterTest {

	static GrayscaleFilter grayscaleFilter;
	static BufferedImage imageGPU;
	static BufferedImage image;
	static PRNUFilter filter;
	static PRNUFilterFactory filterFactory;
	
	/**
	 * @throws java.lang.Exception
	 */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		//instantiate the PRNUFilterFactory to compile CUDA source files
		filterFactory = new PRNUFilterFactory(false);

		//load test image
		try {
			image = Util.readImage(PrnuExtract.INPUT_FILE);
			imageGPU = Util.readImage(PrnuExtract.INPUT_FILE);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Load image: " + image.getHeight()+"x"+image.getWidth());

		//construct a PRNUFilter for this image size
		filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());
		grayscaleFilter = filter.getGrayscaleFilter();
	}

	@Test
	public void applyGPUTest() {

		float[] pixelsGPU = new float[image.getHeight()*image.getWidth()];

		//perform grayscale filter on the GPU
		grayscaleFilter.applyGPU(imageGPU);

		//perform grayscale filter on the CPU
		float[] pixelsCPU = grayscaleFilter.apply1D(image);
				
		//copy GPU result to host memory
		grayscaleFilter._d_input.copyDeviceToHost(pixelsGPU, pixelsGPU.length);
		
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
