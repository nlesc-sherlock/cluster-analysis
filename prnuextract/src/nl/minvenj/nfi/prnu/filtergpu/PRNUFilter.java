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

import java.awt.image.BufferedImage;

import nl.minvenj.nfi.cuba.cudaapi.CudaContext;
import nl.minvenj.nfi.cuba.cudaapi.CudaMemFloat;
import nl.minvenj.nfi.cuba.cudaapi.CudaModule;
import nl.minvenj.nfi.cuba.cudaapi.CudaStream;

/**
 * PRNUFilter is created for a specific image size. The CUDA source files
 * have been compiled by PRNUFilterFactory. Therefore, this object should
 * only be created using the Factory.
 * 
 * This class is used to instantiate the individual filter objects,
 * allocate GPU memory, etc.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PRNUFilter {

	GrayscaleFilter grayscaleFilter;
	FastNoiseFilter fastNoiseFilter;
	ZeroMeanTotalFilter zeroMeanTotalFilter;
	WienerFilter wienerFilter;
	
	protected int h;
	protected int w;
	
	protected CudaMemFloat d_input;
	
	protected CudaStream stream;
	
	/**
	 * This constructor creates a CUDA stream for this filter, and
	 * instantiates the individual filters.
	 * 
	 * @param height - the image height
	 * @param width - the image width
	 * @param context - CudaContext object as created by PRNUFilterFactory
	 * @param modules - array of CudaModules as compiled by PRNUFilterFactory
	 */
	public PRNUFilter(int height, int width, CudaContext context, CudaModule[] modules) {
		this.h = height;
		this.w = width;
		
		//setup GPU memory
		//note that the filters also allocate memory for local variables
		d_input = context.allocFloats(height*width);
		d_input.memset(0f, height*width);
		
		//setup stream
		stream = new CudaStream();
		
        //instantiate individual filters
		grayscaleFilter		= new GrayscaleFilter(height, width, context, stream, d_input, modules[0]);
		fastNoiseFilter		= new FastNoiseFilter(height, width, context, stream, d_input, modules[1]);
		zeroMeanTotalFilter = new ZeroMeanTotalFilter(height, width, context, stream, d_input, modules[2]);
		wienerFilter		= new WienerFilter(height, width, context, stream, d_input, modules[3]);
		
	}
	
	/**
	 * This method applies all individual filters in order. 
	 * 
	 * @param image - a BufferedImage containing the input image from which the PRNU pattern is to be extracted
	 * @return - a 1D float array containing the PRNU pattern of the input image 
	 */
	public float[] apply(BufferedImage image) {
		grayscaleFilter.applyGPU(image);
		fastNoiseFilter.applyGPU();
		zeroMeanTotalFilter.applyGPU();
		wienerFilter.applyGPU();
		
		float[] pixelsFloat = new float[w*h];
		d_input.copyDeviceToHostAsync(pixelsFloat, pixelsFloat.length, stream);

		stream.synchronize();
		
		return pixelsFloat;
	}
	
	/*
	 * Getters for the individual filters
	 */
	public GrayscaleFilter getGrayscaleFilter() {
		return grayscaleFilter;
	}

	public FastNoiseFilter getFastNoiseFilter() {
		return fastNoiseFilter;
	}

	public ZeroMeanTotalFilter getZeroMeanTotalFilter() {
		return zeroMeanTotalFilter;
	}

	public WienerFilter getWienerFilter() {
		return wienerFilter;
	}

	/**
	 * cleans up allocated GPU memory and other resources
	 */
	public void cleanup() {
		//call clean up methods of the filters
		grayscaleFilter.cleanup();
		fastNoiseFilter.cleanup();
		zeroMeanTotalFilter.cleanup();
		wienerFilter.cleanup();
		
		//free GPU memory
		d_input.free();
		
		//destroy stream
		stream.destroy();
	}
}
